# Cost-benefit envelope (rough)

**This is a back-of-envelope sanity check, not a forecast.** It uses the posterior means from the DiD-corrected revenue model and the actual freight statistics on Olist to size the policy's incremental revenue vs subsidy cost. Margin assumptions are explicit and conservative.

## Inputs

| Quantity | Value | Source |
|---|---|---|
| Subtotal-eligibility threshold | R$ 150 | treatment definition |
| N eligible orders (panel total) | 12,405 | DuckDB `gold.fact_orders` |
| Avg freight on an eligible order | R$ 36.83 | DuckDB query |
| Avg payment on an eligible order | R$ 381.95 | DuckDB query |
| Baseline P(repeat within 180d) | 0.0264 | DiD posterior mean (alpha_bar) |
| Treated P(repeat within 180d) | 0.0281 | DiD posterior mean (alpha_bar + delta_b_bar) |
| Conditional spend lift (treated / control) | x1.097 | DiD posterior mean (exp delta_l_bar) |
| Assumed contribution margin on incremental GMV | 20% | conservative platform assumption |

## Envelope

| Line | R$ |
|---|---|
| Subsidy cost (N_eligible x avg_freight_eligible) |        456,879 |
| Incremental GMV from policy lift              |         21,921 |
| Incremental contribution margin @ 20%        |          4,384 |
| **Net envelope (margin - subsidy)**           | **      -452,495** |

## Interpretation

The envelope produces a *negative net envelope*: at the assumed 20% contribution margin, the freight subsidy cost is larger than the incremental margin from lifted retention and conditional spend.

**Sensitivity.** Holding everything else fixed, the break-even contribution margin is approximately `subsidy_cost / incremental_GMV = 2084.2%`. If actual margins on the incremental categories are above this threshold the policy pays for itself; if below it does not. The DiD spend multiplier carries wide uncertainty (94% HDI on the log scale crosses zero), so this envelope should be treated as a midpoint estimate, not a prediction.

**What the envelope omits.** Order-acquisition cost savings from retained customers (not modelled), lifetime-value impact beyond the 180-day window, review-score-driven brand effects (review drops ~0.13 stars), seller-side price responses (would inflate observed lift), and the bunching-induced basket-padding dynamic that the static data cannot capture. All are listed in §7 Limitations of the main report.