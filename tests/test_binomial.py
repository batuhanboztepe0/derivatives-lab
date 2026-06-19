"""
tests/test_binomial.py
=======================
Analytic-controlled tests for the CRR binomial tree (models/binomial.py).

Every check is anchored to a closed form or a model-internal identity, so the
suite is deterministic and needs no external data.

TOLERANCE NOTES
---------------
- European → Black-Scholes: abs tol = 2e-2.  CRR error is O(1/N) and
  oscillatory; at N=2000 the measured error is ≤ 4e-4 for standard moneyness,
  comfortably inside 2e-2.
- Put-call parity: abs tol = 1e-9.  Parity holds in the discrete CRR model
  *exactly* for any N (up to floating-point round-off), independent of σ or N.
- American call premium: exactly 0 for a non-dividend call (early exercise is
  never optimal when r > 0), so American and European prices are bit-identical.
"""

from __future__ import annotations

import math

import pytest

from config import DEFAULT_RISK_FREE_RATE
from models.binomial import BinomialTree
from models.black_scholes import BlackScholes

# ── Shared parameters ─────────────────────────────────────────────────────────
S = 100.0
T = 1.0
r = DEFAULT_RISK_FREE_RATE
sigma = 0.2

BS_TOL = 2e-2      # European CRR vs Black-Scholes at N=2000
PARITY_TOL = 1e-9  # discrete put-call parity, any N


# ── European → Black-Scholes convergence ──────────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_matches_black_scholes(K: float, option_type: str) -> None:
    """CRR European price converges to the Black-Scholes closed form."""
    tree = BinomialTree(S=S, K=K, T=T, r=r, sigma=sigma, N=2000)
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    crr_price = tree.price(option_type, "european")
    bs_price = bs.price(option_type)
    assert abs(crr_price - bs_price) < BS_TOL, (
        f"K={K} {option_type}: CRR={crr_price:.5f}, BS={bs_price:.5f}, "
        f"diff={abs(crr_price - bs_price):.5f}"
    )


def test_convergence_improves_with_N() -> None:
    """Refining the tree shrinks the Black-Scholes error (coarse N is worse)."""
    bs = BlackScholes(S=S, K=105.0, T=0.5, r=r, sigma=sigma).price("call")
    err_coarse = abs(BinomialTree(S, 105.0, 0.5, r, sigma, N=50).price("call") - bs)
    err_fine = abs(BinomialTree(S, 105.0, 0.5, r, sigma, N=2000).price("call") - bs)
    assert err_fine < err_coarse, f"err(N=2000)={err_fine:.5f} !< err(N=50)={err_coarse:.5f}"


# ── Discrete put-call parity (exact, any N) ───────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("N", [3, 50, 500])
def test_european_put_call_parity(K: float, N: int) -> None:
    """C - P = S - K·e^{-rT} holds exactly in the discrete CRR model, any N."""
    tree = BinomialTree(S=S, K=K, T=T, r=r, sigma=sigma, N=N)
    c = tree.price("call", "european")
    p = tree.price("put", "european")
    rhs = S - K * math.exp(-r * T)
    assert abs(c - p - rhs) < PARITY_TOL, (
        f"K={K}, N={N}: C-P={c - p:.6f}, S-Ke^-rT={rhs:.6f}"
    )


# ── American exercise ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_american_call_equals_european_call(K: float) -> None:
    """Non-dividend American call = European call (early exercise never optimal)."""
    tree = BinomialTree(S=S, K=K, T=T, r=r, sigma=sigma, N=500)
    assert tree.price("call", "american") == tree.price("call", "european")
    assert tree.early_exercise_premium("call") == 0.0


def test_american_put_premium_positive() -> None:
    """In-the-money American put with r > 0 carries a strictly positive premium."""
    tree = BinomialTree(S=S, K=110.0, T=T, r=r, sigma=0.3, N=1000)
    premium = tree.early_exercise_premium("put")
    assert premium > 0.0, f"American put premium {premium:.5f} should be > 0"
    # American value must also dominate immediate intrinsic value.
    assert tree.price("put", "american") >= max(110.0 - S, 0.0)
