from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from config import DEFAULT_RISK_FREE_RATE
from models.black_scholes import BlackScholes

# ── Param sets ────────────────────────────────────────────────────────────────
ATM = {"S": 100.0, "K": 100.0, "T": 1.0, "r": DEFAULT_RISK_FREE_RATE, "sigma": 0.2}
OTM = {"S": 100.0, "K": 120.0, "T": 1.0, "r": DEFAULT_RISK_FREE_RATE, "sigma": 0.2}

CASES = [ATM, OTM]


def _independent_d2(S, K, T, r, sigma) -> float:
    """Inline recomputation of d2 independent of the class."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1 - sigma * np.sqrt(T)


# ── (a) digital_call_price matches independent analytic formula ───────────────
@pytest.mark.parametrize("p", CASES)
def test_digital_call_price_analytic(p):
    bs = BlackScholes(**p)
    d2 = _independent_d2(**p)
    expected = np.exp(-p["r"] * p["T"]) * norm.cdf(d2)
    assert abs(bs.digital_call_price() - expected) < 1e-12


# ── (b) digital parity: call + put = e^{-rT} ─────────────────────────────────
@pytest.mark.parametrize("p", CASES)
def test_digital_parity(p):
    bs = BlackScholes(**p)
    disc = np.exp(-p["r"] * p["T"])
    assert abs(bs.digital_call_price() + bs.digital_put_price() - disc) < 1e-12


# ── (c) call-spread limit: digital call ≈ -∂C/∂K (central difference) ────────
@pytest.mark.parametrize("p", CASES)
def test_digital_call_spread_limit(p):
    h = 1e-3 * p["K"]
    bs = BlackScholes(**p)
    lo = BlackScholes(p["S"], p["K"] - h, p["T"], p["r"], p["sigma"])
    hi = BlackScholes(p["S"], p["K"] + h, p["T"], p["r"], p["sigma"])
    finite_diff = (lo.price("call") - hi.price("call")) / (2 * h)
    assert abs(bs.digital_call_price() - finite_diff) < 1e-3


# ── (d) digital_delta matches finite difference of digital price wrt S ─────────
@pytest.mark.parametrize("p", CASES)
def test_digital_delta_finite_difference(p):
    h = 1e-3 * p["S"]
    bs = BlackScholes(**p)

    lo = BlackScholes(p["S"] - h, p["K"], p["T"], p["r"], p["sigma"])
    hi = BlackScholes(p["S"] + h, p["K"], p["T"], p["r"], p["sigma"])
    fd_call_delta = (hi.digital_call_price() - lo.digital_call_price()) / (2 * h)

    assert abs(bs.digital_delta("call") - fd_call_delta) < 1e-4

    # put delta = -call delta
    assert bs.digital_delta("put") == -bs.digital_delta("call")
