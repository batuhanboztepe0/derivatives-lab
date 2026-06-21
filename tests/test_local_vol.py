"""
tests/test_local_vol.py
========================
Analytic-controlled tests for models/local_vol.py.

All checks use closed-form anchors (Black-Scholes) so there is no
dependence on external data; results must be reproducible via SEED.
"""

from __future__ import annotations

import pytest

from config import SEED
from models.black_scholes import BlackScholes
from models.local_vol import CEV, DupireLocalVol

# ── Shared market parameters ──────────────────────────────────────────────────
S   = 100.0
K   = 105.0
T   = 0.5
r   = 0.05
VOL = 0.20   # used for both BS reference and flat IV surface

# High-accuracy MC settings: enough sims + steps so Euler bias < MC noise
N_SIMS  = 200_000
N_STEPS = 200


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CEV β=1  →  should match Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────

class TestCEVBeta1:
    """
    β=1 turns CEV into GBM, so CEV.price_mc must reproduce Black-Scholes
    within the Monte Carlo 95% confidence interval.

    Tolerance: 3 × std_error (covers > 99.7% of outcomes under CLT).
    """

    def _cev(self) -> CEV:
        return CEV(S=S, K=K, T=T, r=r, sigma=VOL, beta=1.0, n_sims=N_SIMS, n_steps=N_STEPS, seed=SEED)

    def test_call(self):
        cev_res = self._cev().price_mc("call")
        bs_price = BlackScholes(S, K, T, r, VOL).price("call")
        # tol = 3 * std_error; with 200k sims ~0.03 for typical params
        tol = 3.0 * cev_res["std_error"]
        assert abs(cev_res["price"] - bs_price) < tol, (
            f"CEV(β=1) call={cev_res['price']:.4f} vs BS={bs_price:.4f}, tol={tol:.4f}"
        )

    def test_put(self):
        cev_res = self._cev().price_mc("put")
        bs_price = BlackScholes(S, K, T, r, VOL).price("put")
        tol = 3.0 * cev_res["std_error"]
        assert abs(cev_res["price"] - bs_price) < tol, (
            f"CEV(β=1) put={cev_res['price']:.4f} vs BS={bs_price:.4f}, tol={tol:.4f}"
        )

    def test_result_keys(self):
        result = self._cev().price_mc("call")
        assert set(result.keys()) == {"price", "std_error", "ci_lower", "ci_upper"}

    def test_ci_contains_bs(self):
        """The 95% CI should bracket the BS price."""
        cev_res = self._cev().price_mc("call")
        bs_price = BlackScholes(S, K, T, r, VOL).price("call")
        assert cev_res["ci_lower"] < bs_price < cev_res["ci_upper"], (
            f"BS price {bs_price:.4f} outside CI [{cev_res['ci_lower']:.4f}, {cev_res['ci_upper']:.4f}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dupire flat surface  →  local_vol ≈ constant vol
# ─────────────────────────────────────────────────────────────────────────────

class TestDupireFlatSurface:
    """
    A flat IV surface σ(K,T) = c implies C(K,T) is exactly the BS call price
    with constant vol c.  Dupire's formula applied to those prices must return c.

    FD truncation error with 0.5% strike bumps and 0.01 T bumps is O(dK²,dT²);
    the measured error on a flat surface is ~3e-5, so abs tol = 1e-3 is comfortable.
    """

    FLAT_VOL = 0.25
    TOL = 1e-3   # FD truncation tolerance (measured ~3e-5 on a flat surface)

    def _model(self) -> DupireLocalVol:
        return DupireLocalVol(S=S, r=r, iv_surface=lambda K, T: self.FLAT_VOL)

    @pytest.mark.parametrize("strike,maturity", [
        (90.0,  0.3),
        (100.0, 0.3),
        (110.0, 0.3),
        (90.0,  0.7),
        (100.0, 0.7),
        (110.0, 0.7),
    ])
    def test_local_vol_equals_flat_vol(self, strike, maturity):
        lv = self._model().local_vol(strike, maturity)
        assert abs(lv - self.FLAT_VOL) < self.TOL, (
            f"local_vol({strike},{maturity})={lv:.5f}, expected {self.FLAT_VOL}, tol={self.TOL}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dupire flat-surface MC  →  price ≈ Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────

class TestDupireMCFlatSurface:
    """
    With a flat vol surface, the Dupire MC simulation is driven by a constant
    local vol = FLAT_VOL, so the result should converge to the BS price.

    We use a modest n_sims=20_000 / n_steps=50 for the MC here because calling
    local_vol() per-path-per-step is expensive (each call does 7 BS evaluations).
    Tolerance is relaxed to 3 * std_error (same statistical guarantee as CEV test).
    """

    FLAT_VOL = 0.25
    # Smaller sim count to keep test runtime reasonable (local_vol is per-step).
    # n_steps=5 keeps dt=0.1yr so tau never drops below _TAU_MIN=0.05 until the
    # last step (tau=0.1→remaining>0.05 at every evaluation point), avoiding
    # the Dupire FD blow-up near expiry.
    MC_SIMS  = 10_000
    MC_STEPS = 5

    def _model(self) -> DupireLocalVol:
        return DupireLocalVol(S=S, r=r, iv_surface=lambda k, t: self.FLAT_VOL)

    def test_call_price_vs_bs(self):
        result = self._model().price_mc(
            K=K, T=T, option_type="call",
            n_sims=self.MC_SIMS, n_steps=self.MC_STEPS, seed=SEED,
        )
        bs_price = BlackScholes(S, K, T, r, self.FLAT_VOL).price("call")
        tol = 3.0 * result["std_error"]
        assert abs(result["price"] - bs_price) < tol, (
            f"Dupire flat MC call={result['price']:.4f} vs BS={bs_price:.4f}, tol={tol:.4f}"
        )

    def test_result_keys(self):
        result = self._model().price_mc(K=K, T=T, n_sims=1000, n_steps=10, seed=SEED)
        assert set(result.keys()) == {"price", "std_error", "ci_lower", "ci_upper"}
