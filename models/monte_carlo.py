"""
models/monte_carlo.py
=====================
Monte Carlo simulation for option pricing via Geometric Brownian Motion.

WHY MONTE CARLO?
----------------
Black-Scholes gives closed-form solutions only for European options.
For path-dependent options (Asian, barrier, lookback) or multi-asset
derivatives, no formula exists — simulation is the only feasible approach.

Core idea:
  1. Simulate thousands of price paths under risk-neutral measure.
  2. Compute payoff at expiry for each path.
  3. Average payoffs, discount to today.
By Law of Large Numbers, this converges to the true price as n_sims → ∞.

GBM EXACT SOLUTION
------------------
Under risk-neutral measure, GBM has exact solution (no discretisation error):

    S_T = S_0 · exp( (r - ½σ²)T + σ√T · Z ),  Z ~ N(0,1)

Why exp() not addition? Markets compound — a 10% move on $100 is $10,
but a 10% move on the resulting $110 is $11. Multiplicative, not additive.
The ½σ² drift correction is Itô's correction (same as in Black-Scholes).

VARIANCE REDUCTION — ANTITHETIC VARIATES
-----------------------------------------
For every Z drawn, also simulate -Z. Positive and negative shocks are
always paired → sample mean is exactly 0 → estimation error cancels.
Halves variance at zero extra computation cost.

ERROR CONVERGENCE
-----------------
MC error ∝ 1/√n_sims. To halve error, need 4× more simulations.
Confidence intervals quantify remaining uncertainty.

EXOTIC OPTIONS COVERED
----------------------
- Asian    : payoff on average price (cheaper, used in commodities)
- Barrier  : activated/killed if price crosses a level (cheaper, path risk)
- Lookback : payoff on max/min price (most expensive exotic)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]


@dataclass
class MonteCarlo:
    """
    Monte Carlo pricer for European and path-dependent options.

    Parameters
    ----------
    S       : Current stock price.
    K       : Strike price.
    T       : Time to maturity (years).
    r       : Risk-free rate (continuously compounded).
    sigma   : Annualised volatility.
    n_sims  : Number of simulation paths.
    n_steps : Time steps per path (1 = exact for European options).
    seed    : Random seed for reproducibility.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    n_sims: int = 100_000
    n_steps: int = 1
    seed: int = 42

    def _simulate_paths(self, antithetic: bool = True) -> np.ndarray:
        """
        Simulate GBM price paths using exact solution.

        Returns array of shape (n_sims, n_steps+1).
        Row 0 = t=0 (all equal to S), last column = terminal prices S_T.

        Antithetic variates: simulate Z and -Z pairs.
        Pairs always cancel each other's bias → lower variance, free speedup.
        """
        rng = np.random.default_rng(self.seed)
        dt = self.T / self.n_steps
        drift     = (self.r - 0.5 * self.sigma ** 2) * dt   # Ito-corrected drift
        diffusion = self.sigma * np.sqrt(dt)

        n = self.n_sims // 2 if antithetic else self.n_sims
        Z = rng.standard_normal((n, self.n_steps))
        if antithetic:
            Z = np.vstack([Z, -Z])                   # pair every path with mirror

        log_returns = drift + diffusion * Z          # (n_sims, n_steps)
        log_paths   = np.cumsum(log_returns, axis=1)
        log_paths   = np.hstack([np.zeros((self.n_sims, 1)), log_paths])  # prepend t=0
        return self.S * np.exp(log_paths)            # convert log-returns to prices

    def _result(self, payoffs: np.ndarray) -> dict[str, float]:
        """Discount payoffs and return price + 95% confidence interval."""
        discounted = np.exp(-self.r * self.T) * payoffs
        price    = discounted.mean()
        std_err  = discounted.std() / np.sqrt(self.n_sims)
        return {
            "price":    price,
            "std_error": std_err,
            "ci_lower": price - 1.96 * std_err,
            "ci_upper": price + 1.96 * std_err,
        }

    # ------------------------------------------------------------------
    # European options
    # ------------------------------------------------------------------

    def price_european(self, option_type: OptionType = "call") -> dict[str, float]:
        """
        Price a European option. Compare with BlackScholes.price() —
        results converge as n_sims increases.
        """
        S_T = self._simulate_paths()[:, -1]
        if option_type == "call":
            payoffs = np.maximum(S_T - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_T, 0)
        return self._result(payoffs)

    # ------------------------------------------------------------------
    # Path-dependent — no BS closed form exists for these
    # ------------------------------------------------------------------

    def price_asian(
        self,
        option_type: OptionType = "call",
        averaging: Literal["arithmetic", "geometric"] = "arithmetic",
    ) -> dict[str, float]:
        """
        Asian option — payoff based on average price over the path.

        Cheaper than European because averaging smooths extremes.
        Widely used in commodity markets to reduce manipulation at expiry.

        arithmetic: S_avg = mean(S_t)
        geometric:  S_avg = exp(mean(log(S_t)))
        """
        paths = self._simulate_paths()
        if averaging == "arithmetic":
            S_avg = paths[:, 1:].mean(axis=1)
        else:
            S_avg = np.exp(np.log(paths[:, 1:]).mean(axis=1))

        if option_type == "call":
            payoffs = np.maximum(S_avg - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_avg, 0)
        return {**self._result(payoffs), "averaging": averaging}

    def price_barrier(
        self,
        barrier: float,
        barrier_type: Literal["up-and-out", "down-and-out", "up-and-in", "down-and-in"] = "down-and-out",
        option_type: OptionType = "call",
    ) -> dict[str, float]:
        """
        Barrier option — activated or killed if price crosses a level.

        Cheaper than vanilla because you carry extra path risk:
        the option can vanish even if you'd otherwise be ITM at expiry.

        knock-out: starts alive, dies if barrier crossed
        knock-in:  starts dead, activates if barrier crossed
        """
        paths = self._simulate_paths()
        S_T   = paths[:, -1]
        path_max = paths[:, 1:].max(axis=1)
        path_min = paths[:, 1:].min(axis=1)

        if   barrier_type == "up-and-out":   active = path_max < barrier
        elif barrier_type == "down-and-out":  active = path_min > barrier
        elif barrier_type == "up-and-in":     active = path_max >= barrier
        else:                                 active = path_min <= barrier   # down-and-in

        if option_type == "call":
            payoffs = np.maximum(S_T - self.K, 0) * active
        else:
            payoffs = np.maximum(self.K - S_T, 0) * active
        return {**self._result(payoffs), "barrier_type": barrier_type, "barrier": barrier}

    def price_lookback(self, option_type: OptionType = "call") -> dict[str, float]:
        """
        Lookback option — payoff based on best price seen over the path.

        Call: max(S_max - K, 0)  — always captures the highest point
        Put:  max(K - S_min, 0)  — always captures the lowest point

        Most expensive exotic because it has perfect hindsight.
        No BS formula — simulation only.
        """
        paths = self._simulate_paths()
        if option_type == "call":
            payoffs = np.maximum(paths[:, 1:].max(axis=1) - self.K, 0)
        else:
            payoffs = np.maximum(self.K - paths[:, 1:].min(axis=1), 0)
        return self._result(payoffs)

    def simulate_paths(self, antithetic: bool = True) -> np.ndarray:
        """Public interface — returns raw price paths for plotting/research."""
        return self._simulate_paths(antithetic=antithetic)
