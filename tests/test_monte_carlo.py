"""
tests/test_monte_carlo.py
=========================
Tests for models/monte_carlo.py (GBM Monte Carlo, antithetic variates).

Anchors: the European MC price lies within 3·SE of the Black-Scholes closed form,
and a down-and-in + down-and-out barrier pair reconstructs the vanilla — the two
are complementary on every path, so with the same seed they sum exactly.
"""

from __future__ import annotations

from config import DEFAULT_RISK_FREE_RATE as r
from models.black_scholes import BlackScholes
from models.monte_carlo import MonteCarlo

S, K, T, sigma = 100.0, 100.0, 1.0, 0.2


def test_european_within_3se_of_black_scholes() -> None:
    mc = MonteCarlo(S=S, K=K, T=T, r=r, sigma=sigma, n_sims=200_000)
    for ot in ("call", "put"):
        res = mc.price_european(ot)
        bs = BlackScholes(S, K, T, r, sigma).price(ot)
        assert abs(res["price"] - bs) < 3.0 * res["std_error"], (
            f"{ot}: MC={res['price']:.4f} BS={bs:.4f} 3·SE={3 * res['std_error']:.4f}")


def test_barrier_in_plus_out_equals_vanilla() -> None:
    """down-and-in + down-and-out (same barrier, same seed → same paths) reconstruct the vanilla.

    Monitored intraday (n_steps=252) so the knock-in is non-trivial: with n_steps=1 the
    barrier is only seen at expiry, where a down-and-in call (needs S_T > K > barrier) can
    never activate and the identity would hold vacuously."""
    mc = MonteCarlo(S=S, K=K, T=T, r=r, sigma=sigma, n_sims=100_000, n_steps=252)
    vanilla = mc.price_european("call")["price"]
    ki = mc.price_barrier(80.0, "down-and-in", "call")["price"]
    ko = mc.price_barrier(80.0, "down-and-out", "call")["price"]
    assert ki > 0.0, f"knock-in never activates (KI={ki}); barrier monitoring is trivial"
    assert abs((ki + ko) - vanilla) < 1e-9, f"KI+KO={ki + ko:.6f} vanilla={vanilla:.6f}"
