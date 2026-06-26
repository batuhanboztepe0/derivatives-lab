"""
backtesting/metrics.py
======================
Performance and risk metrics for evaluating options strategies.

PHILOSOPHY
----------
Raw P&L is meaningless without context. A strategy making $10k/year
with $1M drawdown is far worse than one making $10k with $10k drawdown.
Every metric here normalises returns by some measure of risk.

KEY INSIGHT FOR OPTIONS STRATEGIES
-----------------------------------
Options have asymmetric P&L profiles — OTM expiry = 100% loss on premium,
occasional large wins. This asymmetry means:

  - Sharpe penalises upside vol (unfair — upside is good)
  - Sortino only penalises downside (more appropriate)
  - Max Drawdown captures tail risk Sharpe hides ("steamroller risk")
  - Calmar asks: "for each unit of worst-case pain, how much did I earn?"

SHORT VOL TRAP (Sharpe vs Drawdown)
-------------------------------------
Short vol strategies collect small premiums daily → high Sharpe.
But one crisis (2008, March 2020) wipes out years of gains → huge drawdown.
Sharpe looks great, max drawdown tells the real story.
This is called "picking up pennies in front of a steamroller."
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from config import TRADING_DAYS

ArrayLike = Union[np.ndarray, pd.Series, list]


def sharpe_ratio(returns: ArrayLike, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe = (mean return - rf) / std(returns), annualised.

    Penalises ALL volatility — up and down. Use with caution for
    options strategies where upside vol is a feature, not a bug.

    Rule of thumb: <0.5 poor | 0.5-1 ok | 1-2 good | >2 suspicious
    """
    returns = np.asarray(returns, dtype=float)
    rf = risk_free_rate / TRADING_DAYS
    excess = returns - rf
    if excess.std() < 1e-14:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: ArrayLike, risk_free_rate: float = 0.0) -> float:
    """
    Sortino = (mean return - rf) / downside_deviation, annualised.

    Downside deviation is the root-mean-square shortfall below the target (here
    the risk-free rate), averaged over ALL periods. This is the canonical
    target-downside-deviation, not the std of the negative-return subset.
    Only penalises returns below target. More appropriate for options:
    a strategy with large gains and small losses should score well here,
    even if total vol is high.
    """
    returns = np.asarray(returns, dtype=float)
    rf = risk_free_rate / TRADING_DAYS
    excess = returns - rf
    shortfall = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(shortfall**2))
    if downside_dev < 1e-14:
        return np.nan
    return (excess.mean() / downside_dev) * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: ArrayLike) -> dict[str, float]:
    """
    Max Drawdown = largest peak-to-trough decline in cumulative returns.

    The worst-case scenario for someone who bought at the peak and sold
    at the trough. Captures tail/crisis risk that Sharpe ratio hides.

    Returns dict with max_drawdown (negative fraction), peak and trough indices.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return {"max_drawdown": np.nan, "peak_idx": -1, "trough_idx": -1}
    cum = np.concatenate([[1.0], np.cumprod(1 + returns)])   # prepend the t=0 unit NAV
    rolling_max = np.maximum.accumulate(cum)
    drawdowns   = (cum - rolling_max) / rolling_max      # always <= 0

    max_dd     = drawdowns.min()
    trough_idx = int(drawdowns.argmin())
    peak_idx   = int(np.argmax(cum[:trough_idx + 1]))

    # shift indices back to the returns axis (-1 = the pre-first-return starting NAV)
    return {
        "max_drawdown": max_dd,
        "peak_idx":     peak_idx - 1,
        "trough_idx":   trough_idx - 1,
    }


def calmar_ratio(returns: ArrayLike) -> float:
    """
    Calmar = annualised return / abs(max drawdown).

    "For each unit of worst-case pain, how much did I earn?"
    Preferred over Sharpe when tail risk matters — investors care more
    about "how much can I lose?" than "what was the average volatility?"
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.nan
    ann_return = np.prod(1 + returns) ** (TRADING_DAYS / len(returns)) - 1   # CAGR (geometric)
    mdd = max_drawdown(returns)["max_drawdown"]
    if mdd == 0:
        return np.nan
    return ann_return / abs(mdd)


def win_rate(returns: ArrayLike) -> float:
    """Fraction of periods with positive returns. Meaningless alone —
    30% win rate + 3:1 reward/risk can be excellent."""
    returns = np.asarray(returns, dtype=float)
    return float((returns > 0).mean())


def profit_factor(returns: ArrayLike) -> float:
    """
    Profit Factor = sum(gains) / abs(sum(losses)).

    >1 = profitable. Pairs with win rate to tell the full story:
    Short vol: high win rate, low profit factor (many small wins, rare large loss)
    Long vol:  low win rate, high profit factor (rare large wins, many small losses)
    """
    returns = np.asarray(returns, dtype=float)
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf
    return gains / losses


def summary(returns: ArrayLike, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Full performance summary as a tidy DataFrame.

    Use to compare strategies side-by-side:
        pd.concat([
            summary(bs_returns),
            summary(ml_returns),
        ], axis=1)
    """
    returns = np.asarray(returns, dtype=float)
    mdd_info = max_drawdown(returns)

    data = {
        "Annualised Return": f"{np.mean(returns) * TRADING_DAYS:.2%}",
        "Annualised Vol":    f"{np.std(returns) * np.sqrt(TRADING_DAYS):.2%}",
        "Sharpe Ratio":      f"{sharpe_ratio(returns, risk_free_rate):.3f}",
        "Sortino Ratio":     f"{sortino_ratio(returns, risk_free_rate):.3f}",
        "Calmar Ratio":      f"{calmar_ratio(returns):.3f}",
        "Max Drawdown":      f"{mdd_info['max_drawdown']:.2%}",
        "Win Rate":          f"{win_rate(returns):.2%}",
        "Profit Factor":     f"{profit_factor(returns):.3f}",
        "Num Periods":       len(returns),
    }
    return pd.DataFrame(data, index=["Value"]).T
