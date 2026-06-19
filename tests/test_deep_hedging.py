"""
tests/test_deep_hedging.py
===========================
Tests for ml/deep_hedging.py.

torch is NOT installed in CI (only the [dev] extra runs there), so the whole
module is skipped when torch is unavailable — matching the repo convention for
PyTorch code.  When torch IS present the tests run end-to-end.

Anchors:
  1. Premium = Black-Scholes price (the sale price we hedge against).
  2. Simulated GBM is a Q-martingale: E[e^{-rT} S_T] ≈ S0.
  3. P&L engine (no training): the BS-delta hedge collapses the frictionless
     P&L to a tight spike at zero, far tighter than the unhedged position.
  4. Training reduces the risk measure and the learned policy tracks BS delta.
  5. Under transaction costs the cost-aware policy beats the BS-delta benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import DEFAULT_RISK_FREE_RATE, SEED
from models.black_scholes import BlackScholes

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (CI has no torch)")

from ml.deep_hedging import DeepHedger  # noqa: E402

# ── Shared parameters ─────────────────────────────────────────────────────────
S0, K, T = 100.0, 100.0, 1.0
r, sigma = DEFAULT_RISK_FREE_RATE, 0.2


def _hedger(**kw) -> DeepHedger:
    base = {"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma, "n_steps": 50, "seed": SEED}
    base.update(kw)
    return DeepHedger(**base)


# ── (1) premium = Black-Scholes ───────────────────────────────────────────────

def test_premium_matches_black_scholes() -> None:
    h = _hedger()
    assert abs(h.premium - BlackScholes(S0, K, T, r, sigma).price("call")) < 1e-12


# ── (2) simulated paths are a discounted martingale ───────────────────────────

def test_simulated_paths_martingale() -> None:
    h = _hedger()
    paths = h.simulate_paths(100_000, seed=SEED)
    assert tuple(paths.shape) == (100_000, h.n_steps + 1)
    disc_terminal = np.exp(-r * T) * paths[:, -1].numpy()
    # SE of the mean ~ S0·σ·√T/√n ≈ 0.06; 4·SE ≈ 0.25 tolerance.
    assert abs(disc_terminal.mean() - S0) < 0.25


# ── (3) P&L engine: BS-delta hedge tightens P&L (no training needed) ───────────

def test_bs_delta_hedge_reduces_variance() -> None:
    """Frictionless BS-delta hedge → near-zero mean, far smaller std than unhedged."""
    h = _hedger(tc=0.0, risk="mean_var")
    paths = h.simulate_paths(40_000, seed=SEED)

    bs_pnl = h.bs_delta_pnl(paths=paths)
    zero_holdings = torch.zeros(paths.shape[0], h.n_steps)
    with torch.no_grad():
        unhedged = h._pnl(torch, paths, zero_holdings).numpy()

    assert abs(bs_pnl.mean()) < 0.05                 # replication → mean ≈ 0
    assert bs_pnl.std() < 0.1 * unhedged.std()       # hedge kills most of the risk


# ── (4) training converges toward BS delta ────────────────────────────────────

def test_training_converges_to_bs_delta() -> None:
    h = _hedger(tc=0.0, risk="mean_var")
    h.fit(epochs=150, batch_size=4096, lr=1e-3)

    assert h.train_losses[-1] < h.train_losses[0]    # risk went down

    spot = np.linspace(90, 110, 21)
    learned = h.policy_holdings(spot, t_step=0)
    bs_delta = h.bs_delta(spot, tau=T)
    # Shape match: high correlation and small pointwise gap in the liquid region.
    corr = np.corrcoef(learned, bs_delta)[0, 1]
    assert corr > 0.98, f"learned-vs-BS-delta corr={corr:.3f}"
    assert np.max(np.abs(learned - bs_delta)) < 0.08


# ── (5) cost-aware policy beats BS delta under transaction costs ───────────────

def test_cost_aware_policy_beats_bs_delta() -> None:
    """With costs, minimising risk should do strictly better than naive BS delta."""
    h = _hedger(tc=0.01, risk="mean_var", risk_aversion=1.0)
    h.fit(epochs=200, batch_size=4096, lr=1e-3)

    paths = h.simulate_paths(40_000, seed=SEED)
    pol = h.policy_pnl(paths=paths)
    bsd = h.bs_delta_pnl(paths=paths)

    def mean_var_risk(pnl):
        return -pnl.mean() + 0.5 * h.risk_aversion * pnl.var()

    assert mean_var_risk(pol) < mean_var_risk(bsd), (
        f"policy risk {mean_var_risk(pol):.4f} !< BS-delta risk {mean_var_risk(bsd):.4f}"
    )


# ── risk-measure smoke: all three train without error ─────────────────────────

@pytest.mark.parametrize("risk", ["mean_var", "entropic", "cvar"])
def test_all_risk_measures_train(risk: str) -> None:
    h = _hedger(tc=0.0005, risk=risk)
    h.fit(epochs=40, batch_size=2048, lr=1e-3)
    assert h.fitted
    assert np.isfinite(h.train_losses[-1])
    assert h.policy_pnl(n_paths=5000, seed=SEED).shape == (5000,)


# ── training on an external (zoo-world) path source ───────────────────────────

def test_fit_with_external_paths_fn() -> None:
    """fit() accepts a paths_fn so the policy can train in a non-GBM world."""
    h = _hedger(tc=0.0, risk="mean_var")
    calls = {"n": 0}

    def paths_fn(n):
        calls["n"] += 1
        return h.simulate_paths(n)        # plumbing check: any (n, n_steps+1) tensor

    h.fit(epochs=30, batch_size=2048, lr=1e-3, paths_fn=paths_fn)
    assert calls["n"] == 30                # paths_fn was used every epoch
    assert h.fitted and np.isfinite(h.train_losses[-1])
    assert h.policy_pnl(n_paths=4000, seed=SEED).shape == (4000,)


# ── static option overlay (gamma hedge) plumbing ──────────────────────────────

def test_option_overlay_plumbing() -> None:
    """
    hedge_options adds a static, fairly-priced option overlay whose quantities are
    learned jointly with the policy and folded into the P&L.  Plumbing only — no
    risk-reduction claim — checks fair pricing, exposed quantities and P&L wiring.
    """
    assert _hedger().option_quantities() == {}        # no overlay → empty

    h = _hedger(tc=0.0, risk="mean_var", hedge_options=[("put", 90.0), ("call", 110.0)])
    h.fit(epochs=40, batch_size=2048, lr=1e-3)

    prices = h._hedge_prices.numpy()                   # fair value priced per leg
    assert prices.shape == (2,)
    assert np.all(prices > 0) and np.all(np.isfinite(prices))

    q = h.option_quantities()                          # learned quantities exposed
    assert set(q) == {"put@90", "call@110"}
    assert all(np.isfinite(v) for v in q.values())

    pnl = h.policy_pnl(n_paths=4000, seed=SEED)         # overlay wired into the P&L
    assert pnl.shape == (4000,) and np.all(np.isfinite(pnl))
