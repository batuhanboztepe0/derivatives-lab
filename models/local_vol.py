"""
models/local_vol.py
===================
Local volatility models: CEV and Dupire local vol from a smooth IV surface.

THE VOLATILITY SMILE PROBLEM
-----------------------------
Black-Scholes assumes σ is constant across strikes and maturities. Real markets
disagree: OTM puts trade at higher IV than ATM options (the "skew"), and short-dated
options can be more volatile than long-dated ones (the "term structure"). Plotting
IV against strike/maturity gives a surface, not a flat plane.

Local volatility is one class of solution: replace the constant σ with a
deterministic function σ_LV(S,t) chosen so that the model prices today's entire
surface exactly. The key insight is that any smooth IV surface implies a *unique*
risk-neutral distribution for the terminal stock price — and therefore a unique
local vol function. Dupire (1994) derived the formula from this no-arbitrage argument.

TWO MODELS HERE
---------------
1. CEV  — a parametric SDE with a simple structural lever (β) for the skew.
2. DupireLocalVol — a non-parametric approach: read σ_LV off the observed surface
   via finite differences of market call prices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from config import SEED, SIGMA_MAX, SIGMA_MIN, T_MIN
from models.black_scholes import BlackScholes

OptionType = Literal["call", "put"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. CEV  (Constant-Elasticity-of-Variance)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CEV:
    """
    Constant-Elasticity-of-Variance (CEV) model for European options.

    The SDE under the risk-neutral measure is:

        dS = r·S·dt + σ·S^β·dW

    The key parameter is β (the *elasticity* of variance with respect to price):

    - **β = 1** recovers exactly Geometric Brownian Motion / Black-Scholes
      (σ·S^1·dW = σ·S·dW, so percentage vol is constant at σ).
    - **β < 1** gives the *leverage effect*: when S falls, the local vol σ·S^(β-1)
      rises (β-1 < 0 → S^(β-1) grows as S shrinks). This produces a left-skewed
      distribution and a downward-sloping IV skew — exactly what equity markets show.
      Intuitively: a falling stock price raises the firm's financial leverage, which
      raises equity risk. CEV bakes this in structurally.
    - **β > 1** gives an *inverse* leverage effect (rare; more common in commodity
      markets where supply constraints create right skew).

    **β = 1 sanity check:** Setting beta=1 turns CEV into exactly GBM.  The MC
    price should therefore match `BlackScholes.price()` within simulation noise.
    This is the primary unit-test anchor for this class.

    **Where this breaks down:** β is a single scalar — it sets a *uniform* power-law
    relationship between S and vol.  Real vol surfaces have richer shapes (curvature,
    changing skew across maturities, stochastic shifts) that a single β cannot capture.
    CEV also has no stochastic vol component, so it cannot reproduce vol-of-vol
    dynamics (butterfly spreads on vol).

    Parameters
    ----------
    S      : Current stock price.
    K      : Strike price.
    T      : Time to maturity (years).
    r      : Risk-free rate (continuously compounded).
    sigma  : Volatility scale factor (not directly comparable to BS σ when β≠1).
    beta   : Elasticity parameter. β=1 → GBM; β<1 → equity-like skew.
    n_sims : Number of Monte Carlo simulation paths.
    n_steps: Euler-Maruyama time steps (more → smaller discretisation bias).
    seed   : RNG seed for reproducibility.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    beta: float
    n_sims: int = 100_000
    n_steps: int = 100
    seed: int = SEED

    def price_mc(self, option_type: OptionType = "call") -> dict[str, float]:
        """
        Price a European call or put via Euler-Maruyama simulation of the CEV SDE.

        Discretisation:
            S_{t+dt} = S_t + r·S_t·dt + σ·S_t^β·√dt·Z

        Absorption at zero: once a path hits S≤0 (possible for β<1 when noise is
        large), we clamp S=0 permanently.  This reflects the real absorbing barrier
        at bankruptcy; it also keeps S^β well-defined (avoids complex numbers for
        non-integer β).

        Variance reduction via antithetic variates: we draw n_sims/2 standard
        normals Z and also simulate the mirror paths with –Z.  This pairs each
        upward shock with a matching downward shock, reducing estimator variance
        at zero extra random-number cost.
        """
        rng = np.random.default_rng(self.seed)
        dt = self.T / self.n_steps
        n_half = self.n_sims // 2

        # Draw noise for half the paths; mirror for antithetic pairs
        Z = rng.standard_normal((n_half, self.n_steps))
        Z = np.vstack([Z, -Z])  # shape: (n_sims, n_steps)

        S = np.full(self.n_sims, float(self.S))

        for step in range(self.n_steps):
            dW = Z[:, step] * np.sqrt(dt)
            # CEV diffusion: σ·S^β; absorb first so power of negative S is safe
            S_pow = np.maximum(S, 0.0) ** self.beta
            dS = self.r * S * dt + self.sigma * S_pow * dW
            S = np.maximum(S + dS, 0.0)  # absorbing barrier at 0

        if option_type == "call":
            payoffs = np.maximum(S - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - S, 0.0)

        discounted = np.exp(-self.r * self.T) * payoffs
        price = discounted.mean()
        std_err = discounted.std() / np.sqrt(self.n_sims)
        return {
            "price":     price,
            "std_error": std_err,
            "ci_lower":  price - 1.96 * std_err,
            "ci_upper":  price + 1.96 * std_err,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dupire Local Volatility
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DupireLocalVol:
    """
    Non-parametric local vol extracted from a smooth IV surface via Dupire's formula.

    DUPIRE'S FORMULA (no dividends, q=0)
    -------------------------------------
    Given market call prices C(K,T) as a function of strike and maturity, the
    local vol that exactly reproduces those prices satisfies:

        σ_LV²(K,T) = [∂C/∂T + r·K·∂C/∂K] / [½·K²·∂²C/∂K²]

    Derivation sketch: the Kolmogorov forward (Fokker-Planck) equation relates the
    risk-neutral density p(S,t) to the local vol.  Differentiating the Breeden-
    Litzenberger identity (∂²C/∂K² = e^{-rT}·p(K,T)) with respect to T and
    matching coefficients yields Dupire's formula.  It is remarkable: *one formula,
    any shape surface, exact fit guaranteed* — as long as C(K,T) is arbitrage-free.

    FINITE-DIFFERENCE IMPLEMENTATION
    ---------------------------------
    We do not have an analytic surface; we have a callable iv_surface(K,T) → IV.
    We compute call prices C(K,T) = BlackScholes(S,K,T,r,IV).price("call") and
    then approximate the three partial derivatives by central differences:

        ∂C/∂T  ≈ [C(K, T+dT) - C(K, T-dT)] / (2·dT)
        ∂C/∂K  ≈ [C(K+dK, T) - C(K-dK, T)] / (2·dK)
        ∂²C/∂K²≈ [C(K+dK, T) - 2·C(K,T) + C(K-dK, T)] / dK²

    Near T=0, the backward time step T-dT could violate T>0, so we fall back to a
    one-sided forward difference for ∂C/∂T there.

    **This breaks down** when the surface is not smooth.  Raw market quotes are noisy,
    and ∂²C/∂K² amplifies noise (second derivative of noisy data → numerical garbage).
    The approach works only with parametric or ML-smoothed surfaces (SVI, VolSurfaceNN,
    etc.).  Never apply Dupire directly to bid/ask quotes.

    Parameters
    ----------
    S          : Current underlying price (needed for BS call computation).
    r          : Risk-free rate.
    iv_surface : Callable (K, T) → implied vol (e.g. a fitted VolSurfaceNN or lambda).
    dK_rel     : Relative strike bump for FD (default 0.5% of K).
    dT         : Absolute time bump for FD (default 0.01 yr ≈ 2.5 trading days).
    """

    S: float
    r: float
    iv_surface: Callable[[float, float], float]
    dK_rel: float = 0.005   # 0.5% of K
    dT: float = 0.01        # ~2.5 trading days in years

    def _call_price(self, K: float, T: float) -> float:
        """BS call price using the IV surface to supply vol."""
        T_safe = max(T, T_MIN)
        iv = self.iv_surface(K, T_safe)
        iv = float(np.clip(iv, SIGMA_MIN, SIGMA_MAX))
        return BlackScholes(self.S, K, T_safe, self.r, iv).price("call")

    # Minimum remaining maturity below which Dupire FD is numerically degenerate.
    # As T→0, ∂²C/∂K² → ∞ faster than ∂C/∂T, making the ratio explode.  For
    # τ below this threshold we simply return the IV surface value directly.
    _TAU_MIN: float = 0.05   # ~12 trading days

    def local_vol(self, K: float, T: float) -> float:
        """
        Compute Dupire local vol at (K, T) via finite-difference approximation.

        Returns σ_LV clipped to [SIGMA_MIN, SIGMA_MAX].
        """
        T_safe = max(T, T_MIN)

        # Near expiry the FD of the second K-derivative blows up numerically.
        # Fall back to the IV surface directly — for smooth parametric surfaces
        # the difference between IV and local vol is small close to maturity.
        if T_safe < self._TAU_MIN:
            return float(np.clip(self.iv_surface(K, T_safe), SIGMA_MIN, SIGMA_MAX))

        dK = K * self.dK_rel

        # ── ∂C/∂K and ∂²C/∂K² via central differences in strike ──────────────
        C_mid = self._call_price(K, T)
        C_up  = self._call_price(K + dK, T)
        C_dn  = self._call_price(K - dK, T)

        dC_dK   = (C_up - C_dn) / (2.0 * dK)
        d2C_dK2 = (C_up - 2.0 * C_mid + C_dn) / (dK ** 2)

        # ── ∂C/∂T: central if room, forward one-sided near T=0 ────────────────
        if T - self.dT > T_MIN:
            dC_dT = (
                self._call_price(K, T + self.dT) - self._call_price(K, T - self.dT)
            ) / (2.0 * self.dT)
        else:
            dC_dT = (self._call_price(K, T + self.dT) - C_mid) / self.dT

        # ── Dupire formula ────────────────────────────────────────────────────
        numerator   = dC_dT + self.r * K * dC_dK
        denominator = 0.5 * K ** 2 * d2C_dK2

        if abs(denominator) < 1e-12 or denominator < 0:
            # Degenerate: fall back to the IV surface value
            return float(np.clip(self.iv_surface(K, max(T, T_MIN)), SIGMA_MIN, SIGMA_MAX))

        var = numerator / denominator
        if var < 0:
            return float(np.clip(self.iv_surface(K, max(T, T_MIN)), SIGMA_MIN, SIGMA_MAX))

        return float(np.clip(np.sqrt(var), SIGMA_MIN, SIGMA_MAX))

    def price_mc(
        self,
        K: float,
        T: float,
        option_type: OptionType = "call",
        n_sims: int = 100_000,
        n_steps: int = 100,
        seed: int = SEED,
    ) -> dict[str, float]:
        """
        Price a European option via local-vol Monte Carlo.

        At each Euler step the local vol is evaluated at the *current spot* S_t
        and *remaining maturity* τ = T - t_elapsed.  This is the defining feature
        of local vol simulation: σ_LV changes along each path as S and τ evolve.

        SDE step:
            S_{t+dt} = S_t + r·S_t·dt + σ_LV(S_t, τ)·S_t·√dt·Z

        Note: σ_LV(K=S_t, T=τ) treats the current spot as the "strike" argument
        in Dupire's formula — this is the standard path-simulation convention for
        local vol models (evaluated on the diagonal of the (K,T) surface).

        Antithetic variates and absorbing barrier at zero are applied as in CEV.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        n_half = n_sims // 2

        Z = rng.standard_normal((n_half, n_steps))
        Z = np.vstack([Z, -Z])  # antithetic pairs

        S_paths = np.full(n_sims, float(self.S))

        for step in range(n_steps):
            tau = T - step * dt  # remaining maturity at this step
            tau = max(tau, T_MIN)

            # Paths that have hit zero carry zero vol (absorbed); guard K>0 for BS
            lv = np.array([
                self.local_vol(float(s), tau) if s > 0.0 else 0.0
                for s in S_paths
            ])

            dW = Z[:, step] * np.sqrt(dt)
            dS = self.r * S_paths * dt + lv * S_paths * dW
            S_paths = np.maximum(S_paths + dS, 0.0)

        if option_type == "call":
            payoffs = np.maximum(S_paths - K, 0.0)
        else:
            payoffs = np.maximum(K - S_paths, 0.0)

        discounted = np.exp(-self.r * T) * payoffs
        price = discounted.mean()
        std_err = discounted.std() / np.sqrt(n_sims)
        return {
            "price":     price,
            "std_error": std_err,
            "ci_lower":  price - 1.96 * std_err,
            "ci_upper":  price + 1.96 * std_err,
        }
