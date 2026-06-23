"""
models/merton.py
================
Merton (1976) jump-diffusion model — closed-form European option pricing.

WHERE BLACK-SCHOLES BREAKS: NO JUMPS
------------------------------------
Black-Scholes assumes the stock diffuses continuously: prices never gap.  Real
markets jump — earnings surprises, macro shocks, takeovers — and those gaps put
fat tails on the return distribution that a single Gaussian cannot match.  The
visible symptom is the short-dated vol smile: deep out-of-the-money options are
far more expensive than BS says, because the market prices in the chance of a
sudden gap.  BS gives a flat smile and is structurally blind to this.

THE MERTON SDE
--------------
Merton adds a compound-Poisson jump term to geometric Brownian motion:

    dS_t / S_t = (r − λk) dt + σ dW_t + (Y − 1) dN_t

    σ        — diffusive volatility (the ordinary BS vol)
    N_t      — Poisson process with intensity λ (expected jumps per year)
    Y        — jump multiplier, log-normal: ln Y ~ N(μ_J, δ_J²)
    k        — mean proportional jump size, k = E[Y − 1] = e^{μ_J + ½δ_J²} − 1
    −λk dt   — drift compensator so the discounted stock stays a martingale

Negative μ_J (down-gaps more likely / larger) tilts the distribution left and
produces the downward-sloping equity skew; δ_J controls how fat the tails are.

CLOSED FORM: A POISSON-WEIGHTED SUM OF BLACK-SCHOLES PRICES
-----------------------------------------------------------
Condition on the number of jumps n before expiry.  Given exactly n jumps, the
log-price is again Gaussian — just with a shifted variance and drift — so the
option is a plain Black-Scholes price.  Averaging over the Poisson law of n:

    V = Σ_{n=0}^∞  e^{−λ'T} (λ'T)^n / n!  ·  V_BS(S, K, T, r_n, σ_n)

    λ'    = λ(1 + k)                     (jump intensity under the pricing drift)
    σ_n²  = σ² + n·δ_J² / T              (each jump adds variance)
    r_n   = r − λk + n·ln(1 + k) / T     (each jump shifts the effective rate)

The series converges fast: λ'T is usually O(1), so the Poisson weights die off
within a few tens of terms.  Two consequences worth testing:

  - **λ = 0 collapses to Black-Scholes.**  Only the n=0 term survives (weight 1,
    σ_0 = σ, r_0 = r), so Merton must reproduce BS exactly.  Primary anchor.
  - **Put-call parity holds exactly.**  Σ weights = 1 and Σ wₙ e^{−rₙT} = e^{−rT},
    so C − P = S − K e^{−rT} survives the jumps.

WHERE THIS BREAKS DOWN
----------------------
- Jumps are i.i.d. and independent of the diffusion: no vol clustering, no
  leverage feedback between jumps and vol (Heston captures the latter).
- The smile it produces is strongest at short maturity and flattens too fast
  with T relative to some markets.
- Calibrating five effective parameters (σ, λ, μ_J, δ_J and the implied skew)
  to a single surface is under-determined without care.

REFERENCES
----------
  Merton (1976). Option pricing when underlying stock returns are discontinuous.
  J. Financial Economics 3(1-2), 125-144.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Literal

import numpy as np

from config import SEED
from models.black_scholes import BlackScholes

OptionType = Literal["call", "put"]


@dataclass
class MertonJumpDiffusion:
    """
    Merton (1976) jump-diffusion European option pricer.

    Parameters
    ----------
    S       : Current stock price.
    K       : Strike price.
    T       : Time to maturity (years).
    r       : Risk-free rate (continuously compounded).
    sigma   : Diffusive volatility (the continuous-path BS vol).
    lam     : Jump intensity λ — expected number of jumps per year.
    mu_j    : Mean of the log jump size, μ_J  (negative → downward skew).
    delta_j : Std of the log jump size, δ_J  (larger → fatter tails).
    n_terms : Number of Poisson terms summed in the closed form (default 50;
              the weights are negligible well before this for typical λ'T).
    n_sims  : Monte Carlo paths used by `price_mc` (validation / path demos).
    seed    : RNG seed for `price_mc`.

    Examples
    --------
    >>> m = MertonJumpDiffusion(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
    ...                         lam=1.0, mu_j=-0.1, delta_j=0.15)
    >>> round(m.price("call"), 4)
    12.7613
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    lam: float
    mu_j: float
    delta_j: float
    n_terms: int = 50
    n_sims: int = 200_000
    seed: int = SEED

    @property
    def _k(self) -> float:
        """Mean proportional jump size k = E[Y − 1] = e^{μ_J + ½δ_J²} − 1."""
        return np.exp(self.mu_j + 0.5 * self.delta_j ** 2) - 1.0

    def price(self, option_type: OptionType = "call") -> float:
        """
        Closed-form Merton price as a Poisson-weighted sum of Black-Scholes prices.

        Each term n conditions on exactly n jumps having occurred before expiry:
        the conditional option is Black-Scholes with variance σ² + n·δ_J²/T and
        rate r − λk + n·ln(1+k)/T, weighted by the Poisson probability of n jumps
        under the jump-adjusted intensity λ' = λ(1+k).

        Log-space Poisson weights (via lgamma) keep the n! denominator stable for
        large n.
        """
        if self.T <= 0.0:
            intrinsic = self.S - self.K if option_type == "call" else self.K - self.S
            return float(max(intrinsic, 0.0))
        k = self._k
        lam_p = self.lam * (1.0 + k)          # λ' = λ(1+k)
        log_lam_p_T = np.log(lam_p * self.T) if lam_p > 0 else -np.inf

        total = 0.0
        for n in range(self.n_terms):
            # Poisson weight e^{-λ'T} (λ'T)^n / n!  computed in log space
            if lam_p > 0:
                log_w = -lam_p * self.T + n * log_lam_p_T - lgamma(n + 1)
                weight = np.exp(log_w)
            else:
                weight = 1.0 if n == 0 else 0.0   # no jumps → only the BS term

            sigma_n = np.sqrt(self.sigma ** 2 + n * self.delta_j ** 2 / self.T)
            r_n = self.r - self.lam * k + n * np.log1p(k) / self.T
            bs_n = BlackScholes(self.S, self.K, self.T, r_n, sigma_n).price(option_type)
            total += weight * bs_n

        return float(total)

    def _simulate_terminal(self) -> np.ndarray:
        """
        Exact-in-distribution terminal prices S_T under the Merton SDE.

        Over [0, T] the total jump count M ~ Poisson(λT), and given M the summed
        jump log-size is N(M·μ_J, M·δ_J²); the diffusion contributes an exact
        Gaussian.  So a single draw per path is exact — no time-discretisation
        bias, only Monte Carlo noise.  Antithetic variates pair each diffusive
        shock with its mirror to cut variance.
        """
        rng = np.random.default_rng(self.seed)
        k = self._k
        n_half = self.n_sims // 2

        # Diffusion log-return: (r − λk − ½σ²)T + σ√T·Z, antithetic in Z
        Z = rng.standard_normal(n_half)
        Z = np.concatenate([Z, -Z])
        n = Z.size
        drift = (self.r - self.lam * k - 0.5 * self.sigma ** 2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z

        # Compound-Poisson jump log-size: M ~ Poisson(λT), sum ~ N(M·μ_J, M·δ_J²)
        counts = rng.poisson(self.lam * self.T, size=n)
        jump = rng.normal(counts * self.mu_j, np.sqrt(counts) * self.delta_j)

        return self.S * np.exp(drift + diffusion + jump)

    def delta(self, option_type: OptionType = "call") -> float:
        """
        Closed-form Merton delta = Poisson-weighted sum of Black-Scholes deltas.

        Since the per-term weights, variances σ_n and rates r_n do not depend on S,
        ∂price/∂S = Σ_n w_n · ∂BS_n/∂S = Σ_n w_n · Δ_BS(S, K, T, r_n, σ_n).  This is
        the jump-adjusted hedge ratio — the model-based benchmark a Merton hedger
        would use instead of the naive constant-σ Black-Scholes delta.
        """
        k = self._k
        lam_p = self.lam * (1.0 + k)
        log_lam_p_T = np.log(lam_p * self.T) if lam_p > 0 else -np.inf

        total = 0.0
        for n in range(self.n_terms):
            if lam_p > 0:
                weight = np.exp(-lam_p * self.T + n * log_lam_p_T - lgamma(n + 1))
            else:
                weight = 1.0 if n == 0 else 0.0
            sigma_n = np.sqrt(self.sigma ** 2 + n * self.delta_j ** 2 / self.T)
            r_n = self.r - self.lam * k + n * np.log1p(k) / self.T
            total += weight * BlackScholes(self.S, self.K, self.T, r_n, sigma_n).delta(option_type)
        return float(total)

    def simulate_paths(self, n_paths: int, n_steps: int, seed: int | None = None) -> np.ndarray:
        """
        Full jump-diffusion price paths; shape (n_paths, n_steps+1), column 0 = S0.

        Each step adds the exact GBM diffusion increment plus a compound-Poisson
        jump (a Poisson count of jumps, each log-normal), so the construction is
        exact-in-distribution for any n_steps — the discretisation only sets the
        rebalancing grid, not a bias.  Antithetic on the diffusion shocks only;
        the Poisson jumps are drawn independently.  Used as a "jumpy world" path
        source for deep hedging.
        """
        rng = np.random.default_rng(self.seed if seed is None else seed)
        k = self._k
        dt = self.T / n_steps
        n_half = n_paths // 2

        Z = rng.standard_normal((n_half, n_steps))
        Z = np.vstack([Z, -Z])                                   # antithetic diffusion
        n = Z.shape[0]
        drift = (self.r - self.lam * k - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z

        counts = rng.poisson(self.lam * dt, size=(n, n_steps))
        jumps = rng.normal(counts * self.mu_j, np.sqrt(counts) * self.delta_j)

        log_ret = drift + diffusion + jumps
        log_path = np.cumsum(log_ret, axis=1)
        log_path = np.hstack([np.zeros((n, 1)), log_path])
        return self.S * np.exp(log_path)

    def price_mc(self, option_type: OptionType = "call") -> dict[str, float]:
        """
        Monte Carlo price — an independent cross-check on the closed form and the
        source of jump-fattened terminal samples for the "BS breaks" notebook.

        Returns the standard {price, std_error, ci_lower, ci_upper} schema.
        """
        S_T = self._simulate_terminal()
        if option_type == "call":
            payoffs = np.maximum(S_T - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - S_T, 0.0)

        discounted = np.exp(-self.r * self.T) * payoffs
        price = discounted.mean()
        std_err = discounted.std() / np.sqrt(self.n_sims)
        return {
            "price":     price,
            "std_error": std_err,
            "ci_lower":  price - 1.96 * std_err,
            "ci_upper":  price + 1.96 * std_err,
        }
