# research/: the project in nine notebooks

Start with [`00_overview.ipynb`](00_overview.ipynb): it walks the whole project top to bottom with
the figures, in one read. The nine working notebooks below have the detail.

Read in order. The first three build the model zoo; the last six stress-test six claims from
that zoo against real data (V1–V6). Each verdict is either confirmed (green in the evidence map) or qualified (amber: direction confirmed, magnitude caveated).

## Act 1: build the zoo (synthetic)

| Notebook | What it shows |
|---|---|
| [`01_model_zoo_pricing`](01_model_zoo_pricing.ipynb) | One reference contract priced eight ways. Monte Carlo converges at 1/√n, the CRR tree at 1/N, all to the Black–Scholes closed form. Smile shapes by model, plus a speed benchmark. |
| [`02_deep_hedging_rl`](02_deep_hedging_rl.ipynb) | A PyTorch deep hedger. Sanity check first: at zero cost it reproduces the BS delta. Then transaction costs make a no-trade band emerge, and mean-variance vs CVaR objectives trade off tail risk. |
| [`03_prediction_markets_pde`](03_prediction_markets_pde.ipynb) | A prediction-market contract is a binary (cash-or-nothing) option. Binary put–call parity, a Rannacher-smoothed Crank–Nicolson PDE checked against the closed form, and the bounded-martingale price dynamics. Theoretical, all synthetic. |

## Act 2: verify on real data (V1–V6)

| # | Notebook | Claim → verdict |
|---|---|---|
| V1 | [`04_real_smile_calibration`](04_real_smile_calibration.ipynb) | Jumps, not stochastic vol, own the short-dated skew. Merton 0.52 vp vs Heston 1.67 vp vs flat BS 4.49 vp on 77 SPY strikes. (confirmed) |
| V2 | [`05_real_returns_jumps`](05_real_returns_jumps.ipynb) | The Gaussian assumption fails: excess kurtosis 15.2, left skew −0.61, Normal rejected by AIC. The literal "81 jumps/year" is not the conclusion; fat-tailed-and-left-skewed is. (confirmed) |
| V3 | [`06_mv_delta_hedging`](06_mv_delta_hedging.ipynb) | A minimum-variance delta helps out-of-sample, but ~88% of the gain is VIX-as-IV leakage. The direction survives; the headline 49% does not. (qualified) |
| V4 | [`07_real_gamma_attribution`](07_real_gamma_attribution.ipynb) | Short-gamma bleed concentrates on big-move days (top 5% carry 38% of convexity P&L), but a Student-t(6) null already gives ~38%. A fat-tail consequence, not new evidence of jumps. (qualified) |
| V5 | [`08_deep_hedge_oos`](08_deep_hedge_oos.ipynb) | Out-of-sample, the cost-aware deep hedger trades ~42% less than BS-delta (turnover ratio 0.58, 95% CI [0.56, 0.62]). The turnover edge is clean; the CVaR gain is drift-aided. (confirmed) |
| V6 | [`09_pm_longshot`](09_pm_longshot.ipynb) | Across 83,304 Polymarket markets, the calibration slope is 1.08 > 1 (cluster-robust), stable across tiers and years. A favorite–longshot bias; the longshot side is measure-sensitive. (confirmed) |

V6 is the same Q-vs-P wedge as risk-neutral pricing: the market price is the risk-neutral
probability, the realised resolution frequency is the physical one, and the gap between them is
the bias. That is why a prediction-market study closes an options-pricing project.

The one-image summary is [`reports/figures/evidence_map.png`](../reports/figures/evidence_map.png);
the full write-up is [`reports/MODEL_ZOO_FINDINGS.md`](../reports/MODEL_ZOO_FINDINGS.md).
