# derivatives-lab

**Eight option-pricing and hedging models, built from scratch and anchored with 112 analytic
tests, then stress-tested against real data: a live SPY option chain, 10 years of SPY/VIX
history, and 83,304 resolved Polymarket markets. Four claims held; two I qualified honestly
after arguing against my own headline.**

This is a verification lab, not a discovery and not a trading strategy. The interesting part is
the discipline: build the models, take six load-bearing claims out of the synthetic world, and
report honestly where each survives real data, where its magnitude is construction-inflated
(V3), and where it is just a fat-tail consequence (V4). The two amber results, where the number
shrinks once you compute the right null, are the ones I find most worth reading.

![Synthetic claims vs real-data verification](reports/figures/evidence_map.png)

## How to read this

The repo is one story in two acts (the notebooks are numbered to be read in order):

- **Build the zoo (`research/01`–`03`).** `01` prices a single reference contract eight ways and
  shows the engines converge (Monte Carlo at 1/√n, the CRR tree at 1/N, all to the Black–Scholes
  closed form), compares the smiles each model produces, and benchmarks their speed. `02` is the
  PyTorch deep hedger, sanity-checked to reproduce the BS delta at zero cost before frictions are
  added. `03` frames a prediction market as a binary option and solves it with a Crank–Nicolson PDE.
- **Verify on real data (`research/04`–`09`).** The six stress tests, V1 through V6, summarised below.
  Each notebook states its claim, the test, and the verdict, and keeps its caveats in plain sight.

The fastest read is [`research/00_overview.ipynb`](research/00_overview.ipynb), which walks the whole
project top to bottom with the figures. A one-page text map is in [`research/README.md`](research/README.md).

## The six verifications (V1–V6)

The full write-up with every number and caveat is in
**[`reports/MODEL_ZOO_FINDINGS.md`](reports/MODEL_ZOO_FINDINGS.md)** (with embedded figures). In brief:

| # | Real-data verification | Result |
|---|---|---|
| V1 | Implied-vol smile (SPY chain) | Merton (jumps) fits the steep short-dated skew to **0.52 vol pts**; Heston underfits (**1.67**, structural, degenerate even at full DE budget); flat BS misses the wings (**4.49**), all on 77 strikes. An illustration, not a calibrated benchmark. |
| V2 | Fat tails & jumps (10y daily SPY) | excess kurtosis **15.2**, left skew **−0.61**, Jarque–Bera rejects the Normal, **GBM rejected**; a fat-tailed model (Student-t / jump-mixture) wins by AIC. |
| V3 | Minimum-variance delta (SPY + VIX) | Positive OOS hedging-variance reduction vs a **0% GBM null**, but ~88% of it is VIX-as-IV leakage, so the **direction** is the result, not the headline 49% (cf. Hull–White 2017 ~26% on real quotes). |
| V4 | Gamma P&L attribution | Short-gamma bleed concentrates on big moves (top 5% of days = **38%** of convexity P&L), but a Student-t(6) null gives ~38% too, so this *illustrates* the fat-tail consequence (V2), not jumps. |
| V5 | Deep hedger out-of-sample | On block-bootstrapped real returns, the cost-aware policy **trades ~42% less** than BS-delta (turnover ratio 0.58, 95% CI [0.56, 0.62], the clean result); the CVaR₅ gain is drift-aided and not robust (positive in under half of OOS-block bootstraps). |
| V6 | Favorite–longshot bias (83k Polymarket markets, 2023–2028) | Longshots (p<0.10) priced **2.3%** resolve **1.6%**; favorites (p>0.90) **96.4%→98.8%**; slope **1.08** (cluster-robust, 95% CI [1.07, 1.10]) > 1 across tiers and years; longshot side measure-sensitive. |

## The model zoo

Each model relaxes one Black–Scholes assumption, which is the thread the six verifications pull on:

- **Black–Scholes** (`models/black_scholes`) is the baseline: greeks, implied vol, digital options.
- **Merton** (`models/merton`) adds jumps, which is what produces the steep short-dated skew in V1.
- **Heston** (`models/heston`) adds stochastic vol (Carr–Madan FFT, semi-analytic quadrature, a QE
  Monte Carlo scheme, and a differential-evolution calibrator) and competes with Merton on the smile.
- **CRR binomial** (`models/binomial`) prices American early exercise; **Monte Carlo**
  (`models/monte_carlo`) and the **Crank–Nicolson PDE** (`models/pde_solver`) are the numerical backbones.
- **Local vol** (`models/local_vol`, CEV and Dupire) reads the skew straight off a surface.
- The **deep hedger** (`ml/deep_hedging`, PyTorch) replaces the formula delta with a learned policy
  under transaction costs, which is V5. `ml/vol_surface_nn` fits a small MLP to a vol surface.

Supporting code: `backtesting/` (Sharpe, Sortino, max drawdown), `data/fetcher` (cache-first market
data), `tests/` (analytic-anchor unit tests).

## Setup

```bash
make install          # pip install -e ".[all]"   (numpy/scipy/pandas/matplotlib/yfinance + torch + duckdb)
make ci               # ruff + pytest  (mirrors GitHub Actions)
```

Python ≥ 3.10. PyTorch lives in the `[ml]` extra and `duckdb` in the `[research]` extra (V6
re-reads the Polymarket trade dump via remote DuckDB on a cache miss); both are folded into `[all]`.
CI installs `[dev]` only, so the torch-dependent tests skip there automatically.

## Tests & reproducibility

`pytest`: **112 tests** (102 run in CI; 10 torch-dependent deep-hedging tests need the `ml` extra and
skip in CI), most anchored on analytic identities (λ=0 Merton collapses to Black–Scholes, put–call
parity, CRR → BS convergence, finite-difference vs closed-form deltas, ξ→0 Heston → BS,
discounted-martingale checks). Real-data fetches are pinned to dated parquet caches so the notebooks
re-run offline and deterministically; CI runs only `tests/`, never the data notebooks. The caches are
gitignored, so a fresh clone regenerates them from the network (yfinance, plus remote DuckDB for V6)
on the first notebook run, then runs offline thereafter.

## What this is

A demonstration of engineering and research hygiene: models re-derived from scratch, claims taken to
real data, null hypotheses computed against my own results, and magnitudes reported with their
construction caveats rather than the prettiest version. Start with the
[write-up](reports/MODEL_ZOO_FINDINGS.md), then the notebooks in `research/`. What more data and time
would add (multi-date calibration with confidence intervals, real option quotes for V3) is in the
write-up's Limitations section.

## AI tools

This project was built with AI coding assistance (Claude). The use was limited to code
implementation and refactoring, figure generation, and drafting. At all stages the outputs were
reviewed, checked against the analytic unit tests and the source code, and revised by me. The
responsibility for the final content, analysis, and conclusions rests entirely with me.
