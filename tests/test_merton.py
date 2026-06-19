"""
tests/test_merton.py
=====================
Analytic-controlled tests for the Merton jump-diffusion model (models/merton.py).

Anchors, from strongest to softest:
  1. λ=0 collapses the Poisson series to a single term → exact Black-Scholes.
     (Validates the n=0 machinery.)
  2. Put-call parity C - P = S - K·e^{-rT} survives the jumps exactly.
     (Validates the Poisson weights sum to 1 and Σ wₙ e^{-rₙT} = e^{-rT}.)
  3. Closed form vs independent Monte Carlo within 3·std_error.
     (Validates the n≥1 terms' shifted variance σ_n and rate r_n — neither (1)
     nor (2) exercises those.)
  4. Fat-tail monotonicity: larger jump dispersion δ_J raises an OTM call.
"""

from __future__ import annotations

import math

import pytest

from config import DEFAULT_RISK_FREE_RATE, SEED
from models.black_scholes import BlackScholes
from models.merton import MertonJumpDiffusion

# ── Shared parameters ─────────────────────────────────────────────────────────
S = 100.0
T = 0.5
r = DEFAULT_RISK_FREE_RATE
sigma = 0.2
LAM = 1.0
MU_J = -0.1
DELTA_J = 0.15


def _merton(K: float, lam: float = LAM, **kw) -> MertonJumpDiffusion:
    return MertonJumpDiffusion(
        S=S, K=K, T=T, r=r, sigma=sigma, lam=lam, mu_j=MU_J, delta_j=DELTA_J, seed=SEED, **kw
    )


# ── (1) λ=0 reduces to Black-Scholes ──────────────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_no_jumps_equals_black_scholes(K: float, option_type: str) -> None:
    """With λ=0 only the n=0 term survives, so Merton must equal Black-Scholes."""
    merton = _merton(K, lam=0.0).price(option_type)
    bs = BlackScholes(S, K, T, r, sigma).price(option_type)
    assert abs(merton - bs) < 1e-12, f"K={K} {option_type}: Merton={merton}, BS={bs}"


# ── (2) Put-call parity survives jumps ────────────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_put_call_parity(K: float) -> None:
    """C - P = S - K·e^{-rT} holds exactly under the jump-diffusion pricing drift."""
    m = _merton(K)
    c, p = m.price("call"), m.price("put")
    rhs = S - K * math.exp(-r * T)
    assert abs(c - p - rhs) < 1e-10, f"K={K}: C-P={c - p:.8f}, S-Ke^-rT={rhs:.8f}"


# ── (3) Closed form vs independent Monte Carlo ────────────────────────────────

@pytest.mark.parametrize("K", [95.0, 105.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_closed_form_matches_monte_carlo(K: float, option_type: str) -> None:
    """
    The terminal Merton MC is exact-in-distribution, so the closed form must lie
    within 3·std_error of it.  This is the only anchor that exercises the n≥1
    terms' σ_n and r_n.
    """
    m = _merton(K)
    cf = m.price(option_type)
    mc = m.price_mc(option_type)
    tol = 3.0 * mc["std_error"]
    assert abs(cf - mc["price"]) < tol, (
        f"K={K} {option_type}: closed={cf:.4f}, MC={mc['price']:.4f}, tol={tol:.4f}"
    )


def test_mc_result_keys() -> None:
    assert set(_merton(100.0).price_mc("call").keys()) == {
        "price", "std_error", "ci_lower", "ci_upper"
    }


# ── (4) Fat tails: bigger jumps lift an OTM call ──────────────────────────────

def test_jump_dispersion_raises_otm_call() -> None:
    """Wider jump dispersion δ_J fattens the right tail → OTM call is worth more."""
    small = MertonJumpDiffusion(S, 130.0, 0.25, r, sigma, lam=1.0, mu_j=0.0, delta_j=0.05)
    big = MertonJumpDiffusion(S, 130.0, 0.25, r, sigma, lam=1.0, mu_j=0.0, delta_j=0.30)
    assert big.price("call") > small.price("call")
