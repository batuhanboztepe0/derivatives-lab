# derivatives-lab

A quantitative derivatives research lab: a zoo of option-pricing and hedging models, built
from scratch and tested with analytic anchors, then **stress-tested against real market
data**. Every load-bearing claim made in the synthetic world is verified (or honestly
qualified) on live SPY options, 10 years of SPY/VIX history, and 83k resolved Polymarket
markets.

![Synthetic claims vs real-data verification](reports/figures/evidence_map.png)

## Highlights

The full write-up with numbers and caveats is in
**[`reports/MODEL_ZOO_FINDINGS.md`](reports/MODEL_ZOO_FINDINGS.md)** (with embedded figures).
In brief:

| # | Real-data verification | Result |
|---|---|---|
| V1 | Implied-vol smile (SPY chain) | Merton (jumps) fits the steep short-dated skew to **0.52 vol pts**; Heston underfits (**1.67**, structural, degenerate even at full DE budget); flat BS misses the wings (**4.49**), all on 77 strikes. An illustration, not a calibrated benchmark. |
| V2 | Fat tails & jumps (10y daily SPY) | excess kurtosis **15.2**, left skew **−0.61**, Jarque–Bera rejects the Normal, **GBM rejected**; a fat-tailed model (Student-t / jump-mixture) wins by AIC. |
| V3 | Minimum-variance delta (SPY + VIX) | Positive OOS hedging-variance reduction vs a **0% GBM null**, but ~88% of it is VIX-as-IV leakage, so the **direction** is the result, not the headline 49% (cf. Hull–White 2017 ~26% on real quotes). |
| V4 | Gamma P&L attribution | Short-gamma bleed concentrates on big moves (top 5% of days = **38%** of convexity P&L), but a Student-t(6) null gives ~38% too, so this *illustrates* the fat-tail consequence (V2), not jumps. |
| V5 | Deep hedger out-of-sample | On block-bootstrapped real returns, the cost-aware policy **trades ~42% less** than BS-delta (turnover ratio 0.58, 95% CI [0.56, 0.62], the clean result); the CVaR₅ gain is drift-aided and not robust (positive in under half of OOS-block bootstraps). |
| V6 | Favorite–longshot bias (83k Polymarket markets, 2023–2028) | Longshots (p<0.10) priced **2.3%** resolve **1.6%**; favorites (p>0.90) **96.4%→98.8%**; slope **1.08** (cluster-robust, 95% CI [1.07, 1.10]) > 1 across tiers and years; longshot side measure-sensitive. |

## Models

- **`models/`**: `black_scholes` (greeks, IV, digital options), `merton` (jump-diffusion,
  closed form + paths), `heston` (Carr–Madan FFT + quadrature + QE Monte Carlo + DE
  calibrator), `binomial` (CRR, American early exercise), `monte_carlo` (GBM, antithetic,
  exotics), `local_vol` (CEV + Dupire), `pde_solver` (Crank–Nicolson).
- **`ml/`**: `vol_surface_nn` (small MLP fit to a hand-specified skew/term-structure surface + yfinance fetcher),
  `deep_hedging` (PyTorch policy, mean-variance / entropic / CVaR risk, transaction costs,
  external path sources, option overlay).
- **`backtesting/`**: Sharpe, Sortino, max drawdown, summary metrics.
- **`data/`**: `fetcher` (cache-first market-data pulls; cache is gitignored).
- **`research/`**: executed notebooks `04`–`09` (the V1–V6 verifications above).
- **`tests/`**: analytic-anchor unit tests.

## Setup

```bash
make install          # pip install -e ".[all]"   (numpy/scipy/pandas/matplotlib/yfinance + torch + duckdb)
make ci               # ruff + pytest  (mirrors GitHub Actions)
```

Python ≥ 3.10. PyTorch lives in the `[ml]` extra and `duckdb` in the `[research]` extra (V6
re-reads the Polymarket trade dump via remote DuckDB on a cache miss); both are folded into
`[all]`. CI installs `[dev]` only, so the torch-dependent tests skip there automatically.

## Tests & reproducibility

`pytest`: **112 tests** (102 run in CI; 10 torch-dependent deep-hedging tests need the `ml`
extra and skip in CI), most anchored on analytic identities (λ=0 Merton collapses to
Black–Scholes, put–call parity, CRR → BS convergence, finite-difference vs closed-form deltas,
ξ→0 Heston → BS, discounted-martingale checks). Real-data fetches are pinned to dated parquet
caches so the notebooks re-run offline and deterministically; CI runs only `tests/`, never the
data notebooks. The caches are gitignored, so a fresh clone regenerates them from the network
(yfinance, plus remote DuckDB for V6) on the first notebook run, then runs offline thereafter.

## Scope

This is a **verification lab**, not a discovery or a trading strategy. The phenomena tested are
well-established; the point is to re-derive the models from scratch, then stress-test their
claims on real data and show honestly where each holds, where its magnitude is construction-
inflated (V3), and where it is just a fat-tail consequence (V4). It demonstrates engineering +
research hygiene, not live edge. See the **Limitations & future work** section of
[`reports/MODEL_ZOO_FINDINGS.md`](reports/MODEL_ZOO_FINDINGS.md) for what more data/time would add
(multi-date calibration with CIs and real option quotes for V3).
