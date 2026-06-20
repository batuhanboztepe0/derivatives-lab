# Real-Data Verification of the Model Zoo

Every pricing and hedging claim in this lab was first established in **synthetic worlds**
(GBM, Merton jump-diffusion, Heston). This report records what happens when the
load-bearing claims are tested against **real market data**. The goal is not to re-derive the
models but to check that their qualitative predictions survive contact with reality — and to
be honest about where they don't.

## Method & reproducibility

- **Data:** `yfinance` only (a declared dependency). One dated snapshot per series, taken
  **2026-06-20**: SPY option chain (V1), 10y daily SPY (V2, V5), 10y daily SPY + `^VIX` (V3).
- **Reproducibility:** every fetch is pinned to a dated parquet under `data/cache/`
  (gitignored) via `data/fetcher.fetch_and_cache`. After the first fetch the notebooks run
  **offline and deterministically**. CI runs only `tests/`, never these notebooks.
- **Notebooks:** [`research/04_real_smile_calibration.ipynb`](../research/04_real_smile_calibration.ipynb)
  (V1), [`research/05_real_returns_jumps.ipynb`](../research/05_real_returns_jumps.ipynb) (V2),
  [`research/06_mv_delta_hedging.ipynb`](../research/06_mv_delta_hedging.ipynb) (V3),
  [`research/07_real_gamma_attribution.ipynb`](../research/07_real_gamma_attribution.ipynb) (V4),
  [`research/08_deep_hedge_oos.ipynb`](../research/08_deep_hedge_oos.ipynb) (V5),
  [`research/09_pm_longshot.ipynb`](../research/09_pm_longshot.ipynb) (V6).
- **V6** uses the **SII-WANGZJ/Polymarket_data** HuggingFace dump (`markets.parquet` for outcomes,
  `trades.parquet` for prices via remote DuckDB), not yfinance.

## Summary

| # | Synthetic claim | Real-data result | Verdict |
|---|---|---|---|
| **V1** | Real option smiles are skewed/fat-tailed; jump & stoch-vol models bend, flat BS can't | Merton IV-RMSE **0.38 vp**, Heston **2.76 vp** (underfits short skew), BS flat **6.16 vp** | ✅ jumps win the short skew |
| **V2** | Real returns are non-Gaussian (fat tails, left skew); Merton fits | excess kurtosis **15.2**, skew **−0.61**, Jarque–Bera rejects Normal (p≈0), Merton beats Normal by AIC | ✅ GBM rejected |
| **V3** ★ | Minimum-variance delta should help **more** on real data than in the synthetic `v₀=θ` world (−4%) | OOS variance reduction **≈49%** (in-sample 54%) | ✅ hypothesis confirmed |
| **V4** | Delta-only hedge is short gamma; jumps hit it — losses concentrate on big moves | top 5% move days carry **38%** of convexity P&L; |ret|>3% days (2.1%) carry **30%** of Σ(HE²) | ✅ short-gamma bleed real |
| **V5** | The deep hedger's cost/turnover edge isn't a synthetic artefact | OOS: **−42% turnover**, **+11% CVaR₅** vs BS-delta under 10 bps costs | ✅ cost channel real |
| **V6** | Prediction-market prices are biased vs realised frequency (Q-vs-P) | **83k markets**: longshots (p<0.10) priced **2.3%** resolve **1.6%**; favorites (p>0.90) priced **96%** resolve **99%**; slope **1.08**, robust across volume tiers & years | ✅ favorite-longshot bias |

---

## V1 — Implied-volatility smile

**Setup.** SPY call chain (S≈746.74), liquid quotes only, representative ~33-day expiry,
moneyness ∈ [0.85, 1.15]. Fit Merton `(σ,λ,μ_J,δ_J)` by least squares on IV; calibrate Heston
`(κ,θ,ξ,ρ,v₀)` with `HestonCalibrator`; flat BS pinned at ATM IV.

**Result.** The real 33-day smile is steep and downward-skewed (slope ≈ −0.92). **Merton fits
it tightly (0.38 vol pts)** with λ≈0.89/yr and a mean down-jump ≈ −15%. **Heston bends the
right way (ρ≈−0.80) but underfits (2.76 vp):** diffusive stochastic-vol skew builds with
maturity, so matching a one-month skew this steep forces a near-degenerate high θ. **Flat BS
(6.16 vp)** misses the wings entirely. On real data the **jump** mechanism explains the
short-dated equity skew better than diffusive stochastic vol.

**Caveats.** SPY options are American and the index pays dividends (small IV bias; we drop the
deep wings where it bites). Heston's miss is a structural short-maturity limitation, not an
optimiser failure. Single dated snapshot.

## V2 — Fat tails & jumps in returns

**Setup.** 10y of daily SPY log-returns (2513 obs). Moments + Jarque–Bera; Merton fit by
maximum likelihood (Poisson-weighted Normal density) vs a plain Normal, compared by AIC.

**Result.** **Excess kurtosis 15.2** and **skew −0.61**; Jarque–Bera rejects normality at any
level (p≈0). The Merton MLE is preferred over the Normal by AIC (−16224 vs −15373). **GBM
(Normal log-returns) is rejected**; a jump component is present and material.

**Caveats.** This is the **physical (P) measure**, not the risk-neutral (Q) smile fit of V1:
MLE on daily returns favours *many small* jumps (λ≈81/yr), whereas the smile favours *rarer,
larger* down-jumps — the jump/diffusion split is not sharply identified at daily frequency,
and real volatility clustering (not in Merton) also contributes to the kurtosis. The robust
conclusions are the stylised facts and the decisive AIC gap, not the exact λ.

## V3 ★ — Minimum-variance delta hedging

**Setup (reproducible, no option panel).** Daily SPY + `^VIX` as the ATM ~1-month IV proxy.
Each day strike a fresh ATM 1-month call at `S_t`, `IV=VIX_t`; one-day delta-hedged P&L
`HE = ΔC − δ·ΔS`. BS delta `N(d₁)`; Hull–White MV delta
`δ_MV = δ_BS + (vega/(S√τ))(a + b·δ_BS + c·δ_BS²)`, with `(a,b,c)` fit by least squares.
Gain `= 1 − Var(HE_MV)/Var(HE_BS)`, reported in-sample and out-of-sample (fit first half).

**Result.** Strong leverage effect (ΔVIX vs return correlation ≈ −0.79). MV-delta cuts
hedging-error variance by **≈49% out-of-sample** (54% in-sample) — versus the synthetic
Heston-world baseline of **−4%** (where `v₀=θ` left no IV/realised-vol gap to exploit). **The
plan's hypothesis — that real spot-vol dynamics make the MV delta pay off far more than in the
synthetic world — is confirmed.**

**Caveat (important).** The magnitude is construction-dependent and likely overstated: using
VIX as the option's literal IV pushes the full daily VIX move into the option P&L, so the MV
correction (which predicts ΔVIX from ΔS) removes a large share — well above the ~26% Hull–White
(2017) report on **actual** quotes. The robust result is the **large positive sign vs the
synthetic −4%**, not the exact percentage.

## V4 — Gamma P&L attribution

**Setup.** The V3 rolling ATM 1-month SPY hedge. The delta-hedged residual decomposes as
`HE = ΔC − δ·ΔS ≈ ½Γ(ΔS)² + θ·dt + vega·Δσ`; attribute it to the convexity term and flag the
big-move days.

**Result.** The short-gamma P&L is sharply concentrated on the largest moves: the top 5% of
move-days carry **38%** of the total convexity P&L, and days with |return| > 3% (just **2.1%**
of days) carry **30%** of the total squared hedging error. The realised-minus-implied gamma
P&L averages **+1.15** on big-move days versus **−0.09** on calm days. This is the real-data
counterpart of the synthetic "jumps hit short gamma" result — and why a delta-only hedge needs
an option (gamma) overlay, not just a better delta.

**Caveats.** The convexity term explains only part of `HE` (corr ≈ 0.5; the rest is the
vol-move/vega channel that V3 targets) — the robust result is the *concentration* on big-move
days, not a full decomposition. `|return| > 3%` is a large-move flag, not a formal jump test
(that is V2). Same VIX-as-IV construction as V3.

## V5 — Deep hedger out-of-sample

**Setup.** Block-bootstrap of real SPY daily returns (preserving fat tails / clustering),
split into in-sample (train) and out-of-sample (eval) halves. Train a CVaR(5%) cost-aware
deep hedger on in-sample blocks via `DeepHedger.fit(paths_fn=...)`; evaluate OOS against
BS-delta on identical accounting, frictionless and at 10 bps.

**Result.** OOS with costs, the cost-aware policy trades **~42% less** (turnover ratio 0.58)
and has a **~11% better tail (CVaR₅)** than static BS-delta. The cost/turnover channel from
the synthetic deep-hedging work **survives out-of-sample on real return dynamics**.

**Caveats.** The CVaR objective is not variance: the policy shows higher *central* std while
improving the tail, and part of the mean-P&L gap rides the bootstrap's upward drift — so the
clean, drift-independent finding is the **turnover** advantage. Block bootstrap assumes
stationarity across the split; single horizon, normalised contract, no option bid/ask.

## V6 — Favorite–longshot bias in prediction markets

**Setup.** **83,304 resolved binary Yes/No markets (2023–2026)** from the public
**SII-WANGZJ/Polymarket_data** HuggingFace dump: `markets.parquet` (538k markets) for the
universe + resolution outcomes, and `trades.parquet` (418M trades) for a pre-resolution YES
price per market — the median YES-token trade price over the first 90% of each market's trading
life (dropping the final convergence to 0/1). Prices are pulled by **remote DuckDB** with
column projection (range reads, no 28 GB download). The live CLOB API was unusable for history:
it only retains the last ~weeks, so every older market returns nothing.

**Result.** A clean, monotonic favorite–longshot bias across 83k markets: longshots
(price < 0.10) priced at **2.3%** resolve YES only **1.6%**; favorites (price > 0.90) priced at
**96.4%** resolve **98.8%**. A regression of outcome on price gives slope **1.08**, intercept
**−0.007** — the realised curve is steeper than the 45° line (longshots below, favorites above).
**Robustness:** the slope stays ≈1.07–1.13 across volume tiers ($10k/$50k/$250k) and across every
resolution year, so it is not an artefact of liquidity or period. This is the real-data face of
the Q-vs-P wedge: a market price is not an unbiased physical probability — demand for cheap,
high-payout longshots shades it.

**Caveats.** The representative price is the median YES-token trade over the first 90% of each
market's life; alternative horizons shift the numbers but not the slope>1 signature (§ robustness).
The dump is a snapshot through ~May 2026; the derived dataset (~9 MB) is cached and reproducible
from the public dump + DuckDB. Selection: liquid (>$10k), cleanly-resolved binary Yes/No with ≥5
YES-token trades. Markets within an event/period are correlated, so binomial error bars understate
true uncertainty — though the effect is large and monotonic regardless.

---

## Overall

The synthetic model zoo's qualitative story holds up on real data: real equity returns and
option smiles are skewed and fat-tailed (GBM/flat-BS rejected), jumps explain the short-dated
skew better than diffusive stochastic vol, and both the minimum-variance delta and the
cost-aware deep hedger deliver real, out-of-sample edges that the synthetic worlds had
understated or could not show. The convexity (gamma) risk a delta hedge cannot touch
concentrates on the big-move days (V4), and across 83k prediction markets the prices carry the
textbook favorite–longshot bias (V6) — the real-data face of the Q-vs-P wedge. Where the real
numbers are construction-sensitive (V3's magnitude, V2's λ), the direction of the effect is
robust — and V6's bias holds across volume tiers and years — with the caveats stated rather
than hidden.
