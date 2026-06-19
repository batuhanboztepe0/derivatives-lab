"""
tests/test_pde_solver.py
========================
Analytic-controlled tests for the Crank-Nicolson PDE pricer.

Each test compares the PDE result to the closed-form Black-Scholes formula.
The grid is M=N=300 (fine enough that dS ~ 1.3 and dt ~ 0.003 for K=100).

TOLERANCE NOTES
---------------
- Vanilla call/put: abs tol = 1e-2 ($0.01).
  CN is O(dS² + dt²) accurate.  With M=N=300, dS ≈ S_max/300 and
  dt = 1/300 ≈ 0.003.  Empirically the error is well below $0.01 for
  standard moneyness and T=1.

- Digital call/put: abs tol = 1e-3.
  Rannacher smoothing (θ=1 for first 2 steps) plus a half-jump initialisation
  at the strike node damp the Gibbs-like oscillations that an undamped CN
  scheme would otherwise show on the discontinuous payoff.  With M=N=300 and
  rannacher=True the measured error is ≤ 1.4e-4 across K∈{90,100,110} — ATM is
  the most accurate, since the half-jump sits exactly on the discontinuity.
"""

from __future__ import annotations

import pytest

from config import DEFAULT_RISK_FREE_RATE
from models.black_scholes import BlackScholes
from models.pde_solver import CrankNicolsonBS

# ── Shared test parameters ────────────────────────────────────────────────────
S = 100.0
T = 1.0
r = DEFAULT_RISK_FREE_RATE
sigma = 0.2
M = N = 300

VANILLA_TOL = 1e-2   # $0.01 — see module docstring
DIGITAL_TOL = 1e-3   # measured error ≤ 1.4e-4 with M=N=300 — see module docstring


# ── Vanilla call ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_vanilla_call(K: float) -> None:
    """Crank-Nicolson vanilla call matches Black-Scholes closed form."""
    pde = CrankNicolsonBS(S=S, K=K, T=T, r=r, sigma=sigma, M=M, N=N)
    bs  = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    pde_price = pde.price("call", "vanilla")
    bs_price  = bs.price("call")
    assert abs(pde_price - bs_price) < VANILLA_TOL, (
        f"K={K}: PDE={pde_price:.4f}, BS={bs_price:.4f}, diff={abs(pde_price-bs_price):.4f}"
    )


# ── Vanilla put ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_vanilla_put(K: float) -> None:
    """Crank-Nicolson vanilla put matches Black-Scholes closed form."""
    pde = CrankNicolsonBS(S=S, K=K, T=T, r=r, sigma=sigma, M=M, N=N)
    bs  = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    pde_price = pde.price("put", "vanilla")
    bs_price  = bs.price("put")
    assert abs(pde_price - bs_price) < VANILLA_TOL, (
        f"K={K}: PDE={pde_price:.4f}, BS={bs_price:.4f}, diff={abs(pde_price-bs_price):.4f}"
    )


# ── Digital call ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_digital_call(K: float) -> None:
    """Crank-Nicolson digital call matches e^{-rT}·N(d2) from BlackScholes."""
    pde = CrankNicolsonBS(S=S, K=K, T=T, r=r, sigma=sigma, M=M, N=N, rannacher=True)
    bs  = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    pde_price = pde.price("call", "digital")
    bs_price  = bs.digital_call_price()
    assert abs(pde_price - bs_price) < DIGITAL_TOL, (
        f"K={K}: PDE={pde_price:.4f}, BS={bs_price:.4f}, diff={abs(pde_price-bs_price):.4f}"
    )


# ── Digital put ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_digital_put(K: float) -> None:
    """Crank-Nicolson digital put matches e^{-rT}·N(-d2) from BlackScholes."""
    pde = CrankNicolsonBS(S=S, K=K, T=T, r=r, sigma=sigma, M=M, N=N, rannacher=True)
    bs  = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    pde_price = pde.price("put", "digital")
    bs_price  = bs.digital_put_price()
    assert abs(pde_price - bs_price) < DIGITAL_TOL, (
        f"K={K}: PDE={pde_price:.4f}, BS={bs_price:.4f}, diff={abs(pde_price-bs_price):.4f}"
    )


# ── Digital call + put parity ─────────────────────────────────────────────────

@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_digital_parity(K: float) -> None:
    """Digital call + digital put = e^{-rT} (risk-neutral parity)."""
    import math
    pde = CrankNicolsonBS(S=S, K=K, T=T, r=r, sigma=sigma, M=M, N=N, rannacher=True)
    dc = pde.price("call", "digital")
    dp = pde.price("put",  "digital")
    expected = math.exp(-r * T)
    assert abs(dc + dp - expected) < DIGITAL_TOL, (
        f"K={K}: dc+dp={dc+dp:.4f}, e^(-rT)={expected:.4f}, diff={abs(dc+dp-expected):.4f}"
    )
