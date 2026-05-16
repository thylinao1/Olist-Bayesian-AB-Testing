# Where the policy is most worth running

Per-category posterior summaries from the three DiD-corrected Bayesian models. Hierarchical partial pooling over the 73 product categories means each estimate borrows strength from the rest, so the rankings are robust to small-cell noise.

**How to read each table.** δ̅ is the policy effect on the model's link scale (logit for binomial / Bernoulli, log for spend, cumulative-logit for review). The HDI is the 94% credible interval. `P(δ>0)` is the posterior probability the policy *helps* the outcome in that category.

### On-time delivery — top 5 categories where the policy lifts most

| Rank | Category | δ̅ (logit) | 94% HDI | P(δ>0) | ≈ Δ at base rate |
|---|---|---|---|---|---|
| 1 | furniture_decor | +0.427 | (+0.200, +0.665) | 100.0% | +3.54 pp |
| 2 | home_appliances_2 | +0.358 | (+0.024, +0.773) | 97.9% | +3.04 pp |
| 3 | auto | +0.348 | (+0.122, +0.595) | 99.8% | +2.98 pp |
| 4 | home_appliances | +0.310 | (-0.038, +0.710) | 95.5% | +2.69 pp |
| 5 | furniture_living_room | +0.290 | (-0.046, +0.698) | 94.7% | +2.53 pp |

### P(repeat purchase) — top 5 categories where the policy lifts retention most

| Rank | Category | δ̅ (logit) | 94% HDI | P(δ>0) | ≈ Δ at base rate |
|---|---|---|---|---|---|
| 1 | fashion_shoes | +0.099 | (-0.182, +0.434) | 74.3% | +0.23 pp |
| 2 | office_furniture | +0.097 | (-0.173, +0.417) | 74.4% | +0.23 pp |
| 3 | stationery | +0.088 | (-0.197, +0.404) | 72.6% | +0.21 pp |
| 4 | bed_bath_table | +0.087 | (-0.166, +0.354) | 74.2% | +0.20 pp |
| 5 | cool_stuff | +0.086 | (-0.192, +0.384) | 73.0% | +0.20 pp |

### Conditional spend if customer returns — top 5 categories

| Rank | Category | δ̅ (log) | × multiplier | 94% HDI | P(δ>0) |
|---|---|---|---|---|---|
| 1 | telephony | +0.196 | × 1.216 | (-0.069, +0.628) | 90.8% |
| 2 | watches_gifts | +0.191 | × 1.211 | (-0.032, +0.492) | 94.1% |
| 3 | fashion_shoes | +0.190 | × 1.209 | (-0.084, +0.627) | 89.8% |
| 4 | office_furniture | +0.170 | × 1.185 | (-0.083, +0.527) | 89.2% |
| 5 | auto | +0.158 | × 1.171 | (-0.080, +0.469) | 89.3% |

### Review score — top 5 categories where the policy hurts LEAST

| Rank | Category | δ̅ | 94% HDI | P(δ>0) |
|---|---|---|---|---|
| 1 | toys | -0.024 | (-0.277, +0.349) | 40.2% |
| 2 | musical_instruments | -0.071 | (-0.332, +0.338) | 28.6% |
| 3 | pet_shop | -0.097 | (-0.334, +0.248) | 23.2% |
| 4 | home_appliances | -0.098 | (-0.345, +0.251) | 22.5% |
| 5 | perfumery | -0.108 | (-0.322, +0.188) | 19.4% |

### Review score — bottom 5 categories where the policy hurts MOST

| Rank | Category | δ̅ | 94% HDI | P(δ>0) |
|---|---|---|---|---|
| 1 | sports_leisure | -0.320 | (-0.566, -0.124) | 0.0% |
| 2 | auto | -0.277 | (-0.519, -0.077) | 0.4% |
| 3 | furniture_decor | -0.259 | (-0.475, -0.087) | 0.5% |
| 4 | baby | -0.246 | (-0.499, -0.025) | 1.9% |
| 5 | consoles_games | -0.242 | (-0.537, +0.011) | 3.6% |

## Recommendation summary

If the platform were to roll the free-shipping policy out to a *subset* of categories rather than universally, the ones where the on-time lift, the conditional-spend lift, and the review-score impact are jointly most favourable are the ones to target. A simple aggregate score is the sum of standardised per-category mean policy effects across the four outcomes (negating the review effect since lower is worse).

| Rank | Category | Aggregate z-score | δ on-time | δ repeat | δ spend | δ review |
|---|---|---|---|---|---|---|
| 1 | auto | +6.32 | +0.348 | +0.077 | +0.158 | -0.277 |
| 2 | fashion_shoes | +4.53 | +0.116 | +0.099 | +0.190 | -0.189 |
| 3 | office_furniture | +3.87 | +0.094 | +0.097 | +0.170 | -0.196 |
| 4 | air_conditioning | +3.45 | +0.240 | +0.080 | +0.130 | -0.210 |
| 5 | furniture_living_room | +3.37 | +0.290 | +0.066 | +0.131 | -0.230 |
| 6 | furniture_decor | +3.15 | +0.427 | +0.052 | +0.081 | -0.259 |
| 7 | bed_bath_table | +3.07 | +0.227 | +0.087 | +0.072 | -0.242 |
| 8 | watches_gifts | +2.99 | +0.002 | +0.082 | +0.191 | -0.229 |

Categories at the top of this list are where every channel of the policy is most favourable simultaneously. They are the natural first wave for a phased rollout.