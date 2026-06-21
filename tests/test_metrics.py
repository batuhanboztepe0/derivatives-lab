"""
tests/test_metrics.py
=====================
Hand-computed anchors and edge guards for backtesting/metrics.py.
"""

from __future__ import annotations

import numpy as np

from backtesting.metrics import (
    calmar_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def test_max_drawdown_hand_value() -> None:
    # cum = [1.1, 0.88, 0.924]; peak idx 0, trough idx 1, dd = 0.88/1.1 − 1 = −0.2
    d = max_drawdown(np.array([0.1, -0.2, 0.05]))
    assert abs(d["max_drawdown"] - (-0.2)) < 1e-12
    assert d["peak_idx"] == 0 and d["trough_idx"] == 1


def test_max_drawdown_empty_is_nan() -> None:
    assert np.isnan(max_drawdown(np.array([]))["max_drawdown"])


def test_sharpe_constant_returns_nan() -> None:
    # zero variance ⇒ undefined Sharpe (regression guard for the float `== 0` bug)
    assert np.isnan(sharpe_ratio(np.full(252, 0.001)))


def test_sharpe_zero_mean_is_zero() -> None:
    assert abs(sharpe_ratio(np.array([0.01, -0.01]))) < 1e-12


def test_sortino_hand_value() -> None:
    # downside = [−0.01, −0.03]; std(ddof0)=0.01; mean=0.005 ⇒ sortino = 0.5·√252
    val = sortino_ratio(np.array([0.02, -0.01, -0.03, 0.04]))
    assert abs(val - 0.5 * np.sqrt(252)) < 1e-6


def test_calmar_uses_geometric_cagr() -> None:
    # mean(r) = 0 here, so an arithmetic-annualised Calmar would be 0; the geometric
    # CAGR is negative — this pins the CAGR definition.
    r = np.array([0.05, -0.10, 0.03, 0.02])
    cagr = np.prod(1 + r) ** (252 / len(r)) - 1
    mdd = abs(max_drawdown(r)["max_drawdown"])
    assert abs(calmar_ratio(r) - cagr / mdd) < 1e-9
    assert calmar_ratio(r) < -1.0


def test_win_rate_and_profit_factor() -> None:
    r = np.array([0.02, -0.01, 0.03, -0.04])
    assert abs(win_rate(r) - 0.5) < 1e-12
    assert abs(profit_factor(r) - (0.05 / 0.05)) < 1e-12
