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

from config import DEFAULT_RISK_FREE_RATE as r
from models.heston import Heston, HestonParams

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
