"""
reports/make_figures.py
========================
Regenerate the figures embedded in reports/MODEL_ZOO_FINDINGS.md and README.md.

Deterministic and reproducible: synthetic figures need only the models; the real-data
figures read the dated parquet caches under data/cache/ (gitignored — see data/fetcher.py).
Run from the repo root:  python reports/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_RISK_FREE_RATE as R  # noqa: E402
from config import TRADING_DAYS
from data.fetcher import fetch_and_cache  # noqa: E402
from models.black_scholes import BlackScholes  # noqa: E402
from models.heston import HestonParams, heston_implied_vol, heston_price_fft  # noqa: E402
from models.merton import MertonJumpDiffusion  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
GREEN, AMBER, GREY = "#2e7d32", "#ef6c00", "#9e9e9e"


def _miss():
    raise RuntimeError("cache missing — run the V-notebooks first")


def _top5_convexity_share() -> int:
    """Return the top-5% convexity-P&L share as an integer percent (used in V4 figures)."""
    S, IV = _spy_vix()
    tau = 21 / TRADING_DAYS
    dS = np.diff(S)
    gamma = np.array([BlackScholes(S[t], S[t], tau, R, IV[t]).gamma() for t in range(len(S) - 1)])
    gpnl = 0.5 * gamma * dS ** 2
    order = np.argsort(gpnl)[::-1]
    cum = np.cumsum(gpnl[order]) / gpnl.sum()
    return int(round(cum[int(0.05 * len(cum))] * 100))


def fig_evidence_map() -> None:
    """Synthetic claim → real-data verdict, one row per verification (the project anchor)."""
    v4_share = _top5_convexity_share()
    rows = [
        ("V1  smile / skew", "Merton (jumps) fits short skew 0.52vp;\nHeston underfits 1.7vp; flat BS 4.5vp", GREEN),
        ("V2  fat tails", "excess kurtosis 15.2, GBM rejected\n(Student-t/jump-mix beats Normal by AIC)", GREEN),
        ("V3  MV-delta", "positive vs 0% GBM null; magnitude\ninflated by VIX-as-IV (~88% leakage)", AMBER),
        ("V4  gamma P&L", f"concentrates on big moves ({v4_share}%) —\n= fat-tail null, illustrative not jumps", AMBER),
        ("V5  deep hedge OOS", "−42% turnover, CI [0.56,0.62] (clean);\nCVaR gain not robust (drift-aided)", GREEN),
        ("V6  longshot bias", "83k markets: slope 1.08 > 1 across tiers/years\n(cluster-robust); longshot side measure-sensitive", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    ax.grid(False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows) + 0.5)
    ax.text(2.0, len(rows) + 0.15, "Synthetic model-zoo claim", ha="center", fontweight="bold")
    ax.text(7.2, len(rows) + 0.15, "Real-data verdict", ha="center", fontweight="bold")
    for i, (claim, verdict, col) in enumerate(rows):
        y = len(rows) - i - 0.5
        ax.add_patch(plt.Rectangle((0.2, y - 0.34), 3.6, 0.68, fc="#eceff1", ec="#90a4ae"))
        ax.text(2.0, y, claim, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.annotate("", xy=(5.0, y), xytext=(3.9, y),
                    arrowprops={"arrowstyle": "-|>", "color": col, "lw": 2})
        ax.add_patch(plt.Rectangle((5.05, y - 0.40), 4.75, 0.80, fc=col, ec="none", alpha=0.12))
        ax.text(5.2, y, verdict, ha="left", va="center", fontsize=8.6)
        ax.scatter([9.6], [y], s=90, color=col, zorder=5)
    ax.text(5, -0.15, "green = confirmed   ·   amber = direction confirmed, magnitude caveated",
            ha="center", fontsize=8.5, color=GREY)
    fig.suptitle("derivatives-lab — synthetic claims vs real-data verification (V1–V6)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "evidence_map.png", bbox_inches="tight")
    plt.close(fig)


def fig_v1_term_structure() -> None:
    """Why Merton beats Heston on the short skew: Heston's skew builds with maturity."""
    S = 100.0
    mny = np.linspace(0.85, 1.15, 25)
    K = mny * S
    maturities = [(1 / 52, "1w"), (1 / 12, "1m"), (0.25, "3m"), (1.0, "1y")]
    hp = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.04)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(maturities)))
    for (T, lbl), c in zip(maturities, cmap, strict=False):
        # Price every strike in one FFT, then invert. Deep-wing 1w options have
        # no invertible time value (FFT price = intrinsic), so their implied vol
        # comes back ~0; mask those rather than draw a misleading cliff to zero.
        hpx = heston_price_fft(S, K, T, R, hp)
        hiv = np.array([heston_implied_vol(float(px), S, k, T, R) or np.nan
                        for px, k in zip(hpx, K, strict=False)], dtype=float)
        miv = np.array([BlackScholes(S, k, T, R, 0.2).implied_vol(
            MertonJumpDiffusion(S, k, T, R, 0.13, 1.0, -0.18, 0.15).price("call"), "call") or np.nan
            for k in K], dtype=float)
        hiv[hiv < 1e-4] = np.nan
        miv[miv < 1e-4] = np.nan
        a1.plot(mny, hiv * 100, color=c, lw=2, label=lbl)
        a2.plot(mny, miv * 100, color=c, lw=2, label=lbl)
    a1.set_title("Heston: skew BUILDS with maturity\n(flat at short T → underfits 1-month skew)")
    a2.set_title("Merton (jumps): skew STRONGEST at short T\n(matches the real 1-month skew)")
    for a in (a1, a2):
        a.set_xlabel("moneyness  K / S")
        a.legend(title="maturity", fontsize=8)
    a1.set_ylabel("model implied vol (%)")
    fig.suptitle("V1 — short-dated skew: jumps vs diffusive stochastic vol (synthetic)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "v1_term_structure.png", bbox_inches="tight")
    plt.close(fig)


def _spy_vix():
    spy = fetch_and_cache("SPY", "prices_10y", "2026-06-20", _miss)
    vix = fetch_and_cache("VIX", "close_10y", "2026-06-20", _miss)
    df = spy.join(vix, how="inner").dropna()
    return df["close"].to_numpy(float), df["vix"].to_numpy(float) / 100.0


def fig_v4_concentration() -> None:
    """Short-gamma P&L concentrates on big-move days — vs a fat-tailed null."""
    S, IV = _spy_vix()
    tau = 21 / TRADING_DAYS
    dS = np.diff(S)
    gamma = np.array([BlackScholes(S[t], S[t], tau, R, IV[t]).gamma() for t in range(len(S) - 1)])
    gpnl = 0.5 * gamma * dS ** 2
    order = np.argsort(gpnl)[::-1]
    frac = np.arange(1, len(gpnl) + 1) / len(gpnl)
    cum = np.cumsum(gpnl[order]) / gpnl.sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(frac * 100, cum * 100, color=GREEN, lw=2.4, label="observed (SPY)")
    a1.plot([0, 100], [0, 100], color=GREY, ls=":", label="uniform (no concentration)")
    a1.axvline(5, color=AMBER, ls="--", lw=1)
    a1.text(6, 20, f"top 5% of days\n→ {cum[int(0.05 * len(cum))] * 100:.0f}% of convexity P&L", fontsize=8.5)
    a1.set_xlabel("% of days (largest convexity-P&L first)")
    a1.set_ylabel("cumulative % of convexity P&L")
    a1.set_title("Convexity P&L is concentrated…")
    rng = np.random.default_rng(42)
    n = len(dS)
    top5 = lambda x: np.sort(x ** 2)[-int(0.05 * n):].sum() / (x ** 2).sum()
    gauss = np.mean([top5(rng.standard_normal(n)) for _ in range(800)])
    t6 = np.mean([top5(rng.standard_t(6, n)) for _ in range(800)])
    obs = cum[int(0.05 * len(cum))]
    a2.bar(["Gaussian\nnull", "Student-t(6)\nnull", "observed\n(SPY)"],
           [gauss * 100, t6 * 100, obs * 100], color=[GREY, AMBER, GREEN])
    a2.set_ylabel("top-5% share of squared moves (%)")
    a2.set_title("…but that is what fat tails imply\n(observed ≈ t(6) null, not jumps)")
    for i, v in enumerate([gauss, t6, obs]):
        a2.text(i, v * 100 + 0.6, f"{v * 100:.0f}%", ha="center", fontsize=9)
    fig.suptitle("V4 — gamma-P&L concentration is a fat-tail consequence (real SPY)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "v4_concentration.png", bbox_inches="tight")
    plt.close(fig)


def fig_v6_calibration() -> None:
    """The favorite–longshot calibration curve over 83k resolved Polymarket markets."""
    pm = fetch_and_cache("polymarket", "resolved_trades", "2026-06-20", _miss)
    _yr_min = int(pm["end_date"].dt.year.min())
    _yr_max = int(pm["end_date"].dt.year.max())
    p = pm["prob"].to_numpy()
    y = pm["y"].to_numpy()
    edges = np.array([0, .05, .1, .2, .35, .5, .65, .8, .95, 1.0])
    mp, fr, se = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() < 5:
            continue
        f = y[m].mean()
        mp.append(p[m].mean())
        fr.append(f)
        se.append(np.sqrt(max(f * (1 - f), 1e-9) / m.sum()))
    mp, fr, se = np.array(mp), np.array(fr), np.array(se)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", label="calibrated (45°)")
    ax.errorbar(mp, fr, yerr=se, fmt="o-", color="#c62828", lw=2, capsize=3,
                label=f"Polymarket ({len(pm):,} markets)")
    ax.text(0.30, 0.07, "longshots overpriced\n(below the line)", fontsize=8.5, color=AMBER)
    ax.text(0.52, 0.92, "favorites underpriced\n(above the line)", fontsize=8.5, color=GREEN)
    ax.set_xlabel("market price (implied YES probability)")
    ax.set_ylabel("realised YES frequency")
    ax.set_title(f"V6 — favorite–longshot bias on Polymarket ({_yr_min}–{_yr_max})\n"
                 "slope 1.08 > 1 across tiers and years (cluster-robust)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "v6_calibration.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_evidence_map()
    print("wrote evidence_map.png")
    fig_v1_term_structure()
    print("wrote v1_term_structure.png")
    fig_v4_concentration()
    print("wrote v4_concentration.png")
    fig_v6_calibration()
    print("wrote v6_calibration.png")
