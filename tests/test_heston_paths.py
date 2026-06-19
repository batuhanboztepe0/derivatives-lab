"""
tests/test_heston_paths.py
===========================
Tests for the additive `heston_paths` full-path simulator (models/heston.py).

The path simulator feeds the deep hedger a "stochastic-vol world"; these checks
keep it honest against the model-internal martingale property and the FFT price.
"""

from __future__ import annotations

import numpy as np

from config import DEFAULT_RISK_FREE_RATE as r
from models.heston import HestonParams, heston_paths, heston_price_fft

S0 = 100.0
T = 1.0
PARAMS = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)


def test_paths_shape_and_martingale() -> None:
    paths = heston_paths(S0, T, r, PARAMS, n_paths=80_000, n_steps=50, seed=42)
    assert paths.shape == (80_000, 51)
    assert np.allclose(paths[:, 0], S0)
    # Discounted terminal is a martingale under Q: E[e^{-rT} S_T] = S0.
    assert abs(np.exp(-r * T) * paths[:, -1].mean() - S0) < 0.4


def test_paths_price_near_fft() -> None:
    """ATM call from log-Euler paths is within Euler-bias range of the FFT price."""
    paths = heston_paths(S0, T, r, PARAMS, n_paths=120_000, n_steps=50, seed=42)
    mc_call = np.exp(-r * T) * np.maximum(paths[:, -1] - S0, 0.0).mean()
    fft = float(heston_price_fft(S0, np.array([S0]), T, r, PARAMS)[0])
    assert abs(mc_call - fft) < 0.5   # n_steps=50 Euler vs FFT
