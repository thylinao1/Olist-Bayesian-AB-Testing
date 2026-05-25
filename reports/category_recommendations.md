# Where the policy is most worth running

Per-category posterior summaries from the three DiD-corrected Bayesian models. Hierarchical partial pooling over the 73 product categories means each estimate borrows strength from the rest, so the rankings are stable against small-cell noise.

**How to read each table.** δ̅ is the policy effect on the model's link scale (logit for binomial / Bernoulli, log for spend, cumulative-logit for review). The HDI is the 94% credible interval. `P(δ>0)` is the posterior probability the policy *helps* the outcome in that category.

### On-time delivery - top 5 categories where the policy lifts most

| Rank | Category | δ̅ (logit) | 94% HDI | P(δ>0) | ≈ Δ at base rate |
|---|---|---|---|---|---|
| 1 | furniture_decor | +0.427 | (+0.200, +0.665) | 100.0% | +3.54 pp |
| 2 | home_appliances_2 | +0.358 | (+0.024, +0.773) | 97.9% | +3.04 pp |
| 3 | auto | +0.348 | (+0.122, +0.595) | 99.8% | +2.98 pp |
| 4 | home_appliances | +0.310 | (-0.038, +0.710) | 95.5% | +2.69 pp |
| 5 | furniture_living_room | +0.290 | (-0.046, +0.698) | 94.7% | +2.53 pp |

### P(repeat purchase) - top 5 categories where the policy lifts retention most

| Rank | Category | δ̅ (logit) | 94% HDI | P(δ>0) | ≈ Δ at base rate |
|---|---|---|---|---|---|
| 1 | fashion_shoes | +0.099 | (-0.182, +0.434) | 74.3% | +0.23 pp |
| 2 | office_furniture | +0.097 | (-0.173, +0.417) | 74.4% | +0.23 pp |
| 3 | stationery | +0.088 | (-0.197, +0.404) | 72.6% | +0.21 pp |
| 4 | bed_bath_table | +0.087 | (-0.166, +0.354) | 74.2% | +0.20 pp |
| 5 | cool_stuff | +0.086 | (-0.192, +0.384) | 73.0% | +0.20 pp |

### Conditional spend if customer returns - top 5 categories

| Rank | Category | δ̅ (log) | × multiplier | 94% HDI | P(δ>0) |
|---|---|---|---|---|---|
| 1 | telephony | +0.196 | × 1.216 | (-0.069, +0.628) | 90.8% |
| 2 | watches_gifts | +0.191 | × 1.211 | (-0.032, +0.492) | 94.1% |
| 3 | fashion_shoes | +0.190 | × 1.209 | (-0.084, +0.627) | 89.8% |
| 4 | office_furniture | +0.170 | × 1.185 | (-0.083, +0.527) | 89.2% |
| 5 | auto | +0.158 | × 1.171 | (-0.080, +0.469) | 89.3% |

### Review score - top 5 categories where the policy hurts LEAST

| Rank | Category | δ̅ | 94% HDI | P(δ>0) |
|---|---|---|---|---|
| 1 | toys | -0.024 | (-0.280, +0.337) | 39.6% |
| 2 | musical_instruments | -0.055 | (-0.314, +0.320) | 33.4% |
| 3 | pet_shop | -0.088 | (-0.327, +0.260) | 25.0% |
| 4 | home_appliances | -0.100 | (-0.349, +0.258) | 23.5% |
| 5 | perfumery | -0.106 | (-0.331, +0.208) | 20.7% |

### Review score - bottom 5 categories where the policy hurts MOST

| Rank | Category | δ̅ | 94% HDI | P(δ>0) |
|---|---|---|---|---|
| 1 | sports_leisure | -0.326 | (-0.583, -0.121) | 0.0% |
| 2 | auto | -0.279 | (-0.521, -0.085) | 0.4% |
| 3 | furniture_decor | -0.258 | (-0.484, -0.072) | 0.5% |
| 4 | baby | -0.247 | (-0.506, -0.031) | 1.6% |
| 5 | consoles_games | -0.244 | (-0.569, +0.024) | 3.9% |

## How to read these tables together (no aggregate score)

Earlier drafts of this report ranked categories by summing standardised z-scores across the four outcomes. That ranking was **methodologically unsound** because the four scales are not commensurable: logit, log, and cumulative-logit effects measured on different latent variables don't add up to a meaningful unit even after standardising. We removed that table in line with §8 of the main report (`reports/final_report.md`).

The defensible way to read these per-outcome tables together: look for categories that appear in *multiple* favourable lists (on-time top-5, retention top-5, conditional-spend top-5) AND are *absent* from the review-hurts-most list. Those are the candidates for a phased rollout. Categories like `fashion_shoes`, `office_furniture`, and `watches_gifts` tend to satisfy this criterion in the per-outcome tables above; `auto` and `furniture_decor` lead on-time lift but carry the largest review cost, so they would only belong in a first wave if the platform has a separate plan to manage customer expectations there.

Any cardinal aggregation across the four outcomes requires domain-defined weights (e.g., GMV contribution margins, lifetime-value cost of a one-star review drop). Those weights are not in the public Olist data and we do not invent them.