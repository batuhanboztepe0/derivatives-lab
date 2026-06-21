"""
models/black_scholes.py
=======================
Black-Scholes option pricing model.

FINANCIAL INTUITION
-------------------
A call option's price = cost of replicating its payoff via a dynamic hedge.
You borrow money (at risk-free rate) to buy shares — continuously adjusting
the hedge ratio as the stock price moves. Under no-arbitrage, this cost
is unique: that's the Black-Scholes price.

    C = S·N(d1) - K·e^{-rT}·N(d2)

    S·N(d1)          — shares you need to hold (hedge cost)
    K·e^{-rT}·N(d2)  — amount you borrow (discounted strike x exercise prob)

THE ½σ² CORRECTION (Itô's correction)
--------------------------------------
In stochastic calculus, dW² = dt (doesn't vanish like in regular calculus).
Volatility is asymmetric: losing 10% then gaining 10% != break even
(0.9 × 1.1 = 0.99). The ½σ² term corrects for this compounding asymmetry.

WHERE THIS BREAKS DOWN
----------------------
  1. σ constant over time      → stochastic vol (see models/heston.py)
  2. Normal log-returns        → fat tails exist in reality
  3. σ constant across strikes → vol smile/skew (see ml/vol_surface_nn.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize, stats

OptionType = Literal["call", "put"]


@dataclass
class BlackScholes:
    """
    European option pricer under the Black-Scholes model.

    Parameters
    ----------
    S     : Current price of the underlying asset.
    K     : Strike price of the option.
    T     : Time to maturity in years (e.g. 0.5 = 6 months).
    r     : Continuously compounded risk-free rate (e.g. 0.05 = 5%).
    sigma : Annualised volatility of the underlying (e.g. 0.2 = 20%).

    Examples
    --------
    >>> bs = BlackScholes(S=150, K=155, T=0.5, r=0.05, sigma=0.2)
    >>> round(bs.price("call"), 2)
    7.92
    >>> round(bs.delta("call"), 4)
    0.5062
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float

    def _d1_d2(self) -> tuple[float, float]:
        """
        d1: Z-score for the hedge ratio (delta).
            log-moneyness adjusted for drift (r + ½σ²), scaled by σ√T.

        d2: Z-score for ITM probability under risk-neutral measure.
            d2 = d1 - σ√T

        N(d2) = P(S_T > K) under Q (risk-neutral ITM probability).
        N(d1) = delta, always > N(d2) — weights by conditional stock price.
        """
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self, option_type: OptionType = "call") -> float:
        """
        Closed-form Black-Scholes price.
        Call: C = S·N(d1) - K·e^{-rT}·N(d2)
        Put:  P = K·e^{-rT}·N(-d2) - S·N(-d1)
        """
        d1, d2 = self._d1_d2()
        disc = np.exp(-self.r * self.T)
        if option_type == "call":
            return self.S * stats.norm.cdf(d1) - self.K * disc * stats.norm.cdf(d2)
        else:
            return self.K * disc * stats.norm.cdf(-d2) - self.S * stats.norm.cdf(-d1)

    def delta(self, option_type: OptionType = "call") -> float:
        """
        Delta = dV/dS — hedge ratio.
        Call: N(d1)     → [0,1],  deep ITM→1, deep OTM→0
        Put:  N(d1) - 1 → [-1,0], from put-call parity: delta_call - delta_put = 1
        """
        d1, _ = self._d1_d2()
        if option_type == "call":
            return stats.norm.cdf(d1)
        else:
            return stats.norm.cdf(d1) - 1

    def gamma(self) -> float:
        """
        Gamma = d(Delta)/dS — rate of change of delta.
        gamma = PDF(d1) / (S·σ·√T)
        Why PDF? Gamma = derivative of delta = d/dS[N(d1)]. CDF' = PDF.
        Peaks ATM, explodes near expiry. Same for calls and puts.
        """
        d1, _ = self._d1_d2()
        return stats.norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def theta(self, option_type: OptionType = "call") -> float:
        """
        Theta = dV/dt — time decay per calendar day.
        Short-dated ATM options: must decide ITM/OTM fast → high theta.
        Deep OTM: already near zero, decay is slow.
        Almost always negative for long options.
        """
        d1, d2 = self._d1_d2()
        term1 = -(self.S * stats.norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == "call":
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)
        return (term1 + term2) / 365

    def vega(self) -> float:
        """
        Vega = dV/dσ per 1% vol move.
        Higher vol → fatter tails → more prob of large payoffs.
        Both calls AND puts gain (vol has no direction). x0.01 = convention.
        """
        d1, _ = self._d1_d2()
        return self.S * stats.norm.pdf(d1) * np.sqrt(self.T) * 0.01

    def rho(self, option_type: OptionType = "call") -> float:
        """
        Rho = dV/dr per 1% rate move.
        Higher r → K·e^{-rT} shrinks → cheaper to finance hedge → call up, put down.
        """
        d1, d2 = self._d1_d2()
        if option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(d2) * 0.01
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(-d2) * 0.01

    def all_greeks(self, option_type: OptionType = "call") -> dict[str, float]:
        return {
            "delta": self.delta(option_type),
            "gamma": self.gamma(),
            "theta": self.theta(option_type),
            "vega":  self.vega(),
            "rho":   self.rho(option_type),
        }

    # ── Digital / binary (cash-or-nothing) ────────────────────────────────────────
    # A cash-or-nothing option pays a fixed $1 if the terminal stock price lands
    # on the right side of the strike, and $0 otherwise.  Under risk-neutral
    # measure, the probability of S_T > K is N(d2), so:
    #
    #   Digital call = e^{-rT} · N(d2)
    #   Digital put  = e^{-rT} · N(-d2)
    #
    # Together they exhaust the probability space: call + put = e^{-rT} (parity).

    def digital_call_price(self) -> float:
        """
        Cash-or-nothing call: pays $1 if S_T > K, else $0.

        Formula: e^{-rT} · N(d2)

        N(d2) is the risk-neutral probability of finishing in the money.
        The discount factor e^{-rT} converts that probability-dollar into today's
        present value — analogous to the K·e^{-rT}·N(d2) term in Black-Scholes,
        but with the strike replaced by a fixed $1 notional.

        Where this breaks down: near expiry and ATM, the price jumps sharply from
        0 to e^{-rT} over a tiny range of S — the gamma blows up.  Dealers
        typically trade these as tight vanilla call-spreads to avoid the discontinuity.
        """
        _, d2 = self._d1_d2()
        disc = np.exp(-self.r * self.T)
        return disc * stats.norm.cdf(d2)

    def digital_put_price(self) -> float:
        """
        Cash-or-nothing put: pays $1 if S_T < K, else $0.

        Formula: e^{-rT} · N(-d2)

        By risk-neutral parity, digital_call + digital_put = e^{-rT}.
        """
        _, d2 = self._d1_d2()
        disc = np.exp(-self.r * self.T)
        return disc * stats.norm.cdf(-d2)

    def digital_delta(self, option_type: OptionType = "call") -> float:
        """
        Delta of the cash-or-nothing option: dV/dS.

        Derivation:
            V_call = e^{-rT} · N(d2)
            dV/dS  = e^{-rT} · φ(d2) · d(d2)/dS
            d2     = [ln(S/K) + (r - ½σ²)T] / (σ√T)
            d(d2)/dS = 1 / (S · σ · √T)

        Call:  e^{-rT} · φ(d2) / (S · σ · √T)
        Put:  -e^{-rT} · φ(d2) / (S · σ · √T)  (symmetric: put = 1 - call, so same magnitude)

        The "binary is hard to hedge" point: as T→0 with S near K, φ(d2) spikes to a
        Dirac-like peak — the delta explodes and no continuous hedge can replicate
        the discontinuous payoff without infinite trading cost.  This is why dealers
        prefer call-spread approximations in practice.
        """
        _, d2 = self._d1_d2()
        disc = np.exp(-self.r * self.T)
        call_delta = disc * stats.norm.pdf(d2) / (self.S * self.sigma * np.sqrt(self.T))
        if option_type == "call":
            return call_delta
        else:
            return -call_delta

    def implied_vol(self, market_price: float, option_type: OptionType = "call") -> float:
        """
        Recover IV from market price via Brent's method.
        IV > realised vol → options expensive → sell premium
        IV < realised vol → options cheap    → buy gamma
        Returns np.nan if no solution found.
        """
        def objective(sigma):
            return BlackScholes(self.S, self.K, self.T, self.r, sigma).price(option_type) - market_price
        try:
            if objective(1e-6) * objective(5.0) >= 0:
                return np.nan
            return optimize.brentq(objective, 1e-6, 5.0, xtol=1e-6, maxiter=200)
        except (ValueError, RuntimeError):
            return np.nan

    def put_call_parity_check(self) -> dict[str, float]:
        """C - P = S - K·e^{-rT}. Difference should be ~0."""
        c, p = self.price("call"), self.price("put")
        rhs = self.S - self.K * np.exp(-self.r * self.T)
        return {"call": c, "put": p, "C-P": c - p, "S-Ke^-rT": rhs, "diff": abs(c - p - rhs)}
