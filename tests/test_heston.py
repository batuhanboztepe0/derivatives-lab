"""
tests/test_heston.py
=====================
Tests for the `Heston.delta` model-based hedge ratio (models/heston.py).

Heston has no closed-form delta; `Heston.delta` central-differences the FFT
(Carr-Madan) price.  The anchor here is an *independent* pricing engine: a
central finite difference of the quadrature reference price `price_quad` must
reproduce the same delta.  Agreement validates that the FFT-based delta is
differentiating a correct price (not just self-consistent with itself), and a
call delta must sit in (0, 1).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import DEFAULT_RISK_FREE_RATE as r
from models.black_scholes import BlackScholes
from models.heston import (
    Heston,
    HestonCalibrator,
    HestonParams,
    heston_implied_vol,
    heston_mc,
    heston_price_fft,
)

S0 = 100.0
T = 1.0
PARAMS = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)


def _heston(K: float) -> Heston:
    return Heston(S=S0, K=K, T=T, r=r, params=PARAMS)


def test_delta_matches_quadrature_finite_difference() -> None:
    """FFT-differenced delta agrees with a finite difference of the quad price."""
    h = 1.0  # absolute spot bump (= h_rel·S0 with h_rel = 1e-2)
    for K in (90.0, 100.0, 110.0):
        opt = _heston(K)
        d_fft = opt.delta(h_rel=h / S0)
        d_quad = (replace(opt, S=S0 + h).price_quad() - replace(opt, S=S0 - h).price_quad()) / (2.0 * h)
        assert abs(d_fft - d_quad) < 1e-2, f"K={K}: d_fft={d_fft:.6f}, d_quad={d_quad:.6f}"


def test_call_delta_in_unit_interval() -> None:
    """A European call delta is a probability-like quantity in (0, 1)."""
    for K in (90.0, 100.0, 110.0):
        d = _heston(K).delta()
        assert 0.0 < d < 1.0, f"K={K}: delta={d}"


def test_xi_to_zero_recovers_black_scholes() -> None:
    """With v₀=θ and ξ→0 the variance is frozen at θ, so the FFT price collapses to BS(√θ)."""
    p = HestonParams(kappa=10.0, theta=0.04, xi=0.01, rho=0.0, v0=0.04)   # √θ = 0.2
    for K in (90.0, 100.0, 110.0):
        fft = float(heston_price_fft(S0, np.array([K]), T, r, p)[0])
        bs = BlackScholes(S0, K, T, r, 0.2).price("call")
        assert abs(fft - bs) < 2e-3, f"K={K}: Heston(ξ→0)={fft:.5f}, BS={bs:.5f}"


def test_qe_monte_carlo_responds_to_rho() -> None:
    """QE prices must move with ρ — guards the bug where stock and variance were uncorrelated."""
    neg = heston_mc(S0, S0, T, r, HestonParams(2.0, 0.04, 0.5, -0.7, 0.04), scheme="qe", n_sims=60_000)["price"]
    pos = heston_mc(S0, S0, T, r, HestonParams(2.0, 0.04, 0.5, +0.7, 0.04), scheme="qe", n_sims=60_000)["price"]
    assert abs(neg - pos) > 0.1, f"QE ignores ρ: neg={neg:.4f}, pos={pos:.4f}"


def test_calibrator_round_trip() -> None:
    """Calibrate to a surface generated from known params; the fit RMSE is tight and the
    recovered params are in the right ballpark (Heston's κ is famously weakly identified, so
    we anchor on RMSE + the equity-relevant ρ<0 and long-run vol)."""
    true = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
    strikes = np.array([90.0, 100.0, 110.0])
    K = np.concatenate([strikes, strikes])
    Tn = np.concatenate([np.full(3, 0.5), np.full(3, 1.0)])
    mivs = np.array([heston_implied_vol(float(heston_price_fft(S0, np.array([k]), t, r, true)[0]),
                                        S0, k, t, r) for k, t in zip(K, Tn, strict=False)])
    cal = HestonCalibrator(r=r, de_maxiter=60, de_popsize=10)
    rec = cal.calibrate(S0, K, Tn, mivs, verbose=False)
    assert cal.fit_quality(S0, K, Tn, mivs)["rmse"] < 1e-2          # recovers a tight fit
    assert rec.rho < -0.3                                           # equity skew sign
    assert abs(np.sqrt(rec.theta) - np.sqrt(true.theta)) < 0.07     # long-run vol ballpark
