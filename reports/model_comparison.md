# Bayesian model comparison (PSIS-LOO)

Leave-one-out expected log predictive density for the naive Binomial vs the DiD-corrected Binomial on on-time delivery. Higher `elpd_loo` is better. `p_loo` is the effective number of parameters used by the fit.

## Per-model LOO summary

| Model | elpd_loo | SE | p_loo | n_obs | elpd_loo per cell | Pareto-k worst |
|---|---|---|---|---|---|---|
| naive | -11435.90 | 123.25 | 114.90 | 16,082 | -0.7111 | 0.58 |
| DiD | -6277.71 | 86.35 | 91.86 | 6,689 | -0.9385 | 0.66 |

## Why no pairwise az.compare() table

`az.compare()` requires the two models to have the same observation count. The two binomial fits here aggregate orders into different panel grains: the naive model uses 16,082 cells (category x seller_tier x state x month x treatment) while the DiD model uses 6,689 cells (category x seller_tier x state x eligible x post), since the DiD design collapses month into a single `post` indicator. The two LOO sums are therefore not directly comparable.

The defensible cross-comparison is **per-cell elpd**: naive = -0.7111, DiD = -0.9385. But this still isn't strictly apples-to-apples because per-cell log-likelihood depends on cell size (a Binomial(n, p) has higher per-cell entropy when n is larger, so coarser cells get larger |elpd| per cell). The honest interpretation is that **both models predict their respective held-out cells well** (Pareto-k all under 0.7 on each), and the choice between them rests on causal-identification grounds rather than on LOO numbers.

## Substantive takeaway

The DiD model's advantage is *causal identification* — it cleanly separates the policy effect from basket-size structural effects and the marketplace-wide time trend (as documented in §4.1 and §6 of the main report). Out-of-sample predictive accuracy is a separate question, and one that LOO can answer well only when the two models are fit on the same observation grain. To do that properly we would need to re-fit both at a single shared aggregation (e.g., the DiD's 6,689 cells with month dummies added to the naive specification). That re-fit is queued as future work; the current LOO output suffices to confirm both posterior fits are well-behaved (Pareto-k diagnostic clean), which is the main diagnostic value of LOO here.
