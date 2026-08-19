# Olist Bayesian A/B Testing

A DuckDB feature pipeline and a set of hierarchical Bayesian models (PyMC) built on the public
Olist Brazilian E-commerce dataset: 99,441 orders across nine relational tables. The analysis
estimates what a hypothetical free-shipping-above-R$ 150 policy would do to on-time delivery,
repeat-purchase revenue, and review scores, and whether the answer depends on which product
category you ask about.

The headline result is that the difference-in-differences design flips the sign of the naive
on-time-delivery result. A flat treated-versus-everything-else comparison reports a 2 pp drop.
Once basket size and the marketplace-wide time trend are given their own coefficients, the policy
effect is +1.5 pp: delta_bar = +0.169 on the logit scale, 94% HDI (+0.048, +0.289),
P(delta_bar > 0) = 99.7%. A frequentist DiD logistic on the same panel gives +1.35 pp, p = 0.013.

The treatment is synthetic. Olist never ran this policy, so the exposure variable is constructed
from columns that already exist (`purchase_week >= cutover_week AND items_subtotal >= 150`). What
the project demonstrates is the inference pipeline that would be used if the assignment were real.

## Install

Python 3.11.

```bash
git clone <repo-url>
cd Olist-Bayesian-AB-Testing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is not committed (about 45 MB of CSVs). Download it once into `data/raw/`:

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce --unzip -p data/raw/
```

`data/README.md` covers the manual download route and the expected filenames.

## Run

Building the warehouse and fitting the on-time DiD model is the shortest path to the headline
number:

```bash
python -m src.etl                                 # bronze, silver, gold, analytics (~5 s)
python scripts/fit_binomial_did.py --use-nutpie   # 47 s wall time on 4 cores
```

The other two DiD fits are considerably slower, so run them when you need the full picture. Wall
times below are from the runs quoted in the report:

```bash
python scripts/fit_revenue_did.py                 # 472 s, standard NUTS
python scripts/fit_review_did.py                  # 4121 s, standard NUTS, 30k subsample
```

The naive (non-DiD) specifications are kept because the report contrasts them with the DiD:

```bash
python scripts/fit_binomial.py --use-nutpie       # 107 s
python scripts/fit_revenue.py                     # 388 s, ADVI init
python scripts/fit_review.py                      # 219 s, 30k subsample
```

Everything downstream reads the saved traces and finishes in seconds:

```bash
python -m src.dag                             # render the DAG figure
python scripts/run_baselines.py               # two-proportion z, Welch t, Mann-Whitney, chi-square
python scripts/parallel_trends.py             # pre-trend test for the DiD
python scripts/bunching_test.py               # McCrary-style density check at R$ 150
python scripts/prior_sensitivity.py --use-nutpie   # three small refits
python scripts/posterior_predictive_checks.py
python scripts/model_comparison.py            # PSIS-LOO
python scripts/cost_benefit_envelope.py
python scripts/category_recommendations.py
```

Tests:

```bash
pytest tests/ -v
```

Eleven unit tests run anywhere. Six integration tests auto-skip when the DuckDB warehouse has not
been built. `.github/workflows/ci.yml` runs the unit tests plus `scripts/smoke_test.py` on every
push to `main`.

## Method

The SQL layer is a medallion pipeline on DuckDB. Bronze loads the nine raw CSVs with types pinned
and no transforms. Silver cleans them, including the `customer_id` grain trap (Olist issues a new
`customer_id` per order, so the stable person key is `customer_unique_id`) and the 1M-row
geolocation table, which is deduplicated to one centroid per zip prefix. Gold builds
`fact_orders`, `dim_customer`, and `dim_seller` with window functions for first-purchase ranking,
repeat intervals, and dominant category. The analytics layer produces the cohort retention matrix,
a weekly funnel, gap-and-island session reconstruction, the modelling panels, and a
`quality_diagnostics` table that runs five integrity checks at the end of every ETL pass.

The causal assumptions live in `src/dag.py`, which builds the graph with NetworkX and derives the
adjustment set computationally. The four elemental confounds (fork, pipe, collider, descendant)
give `{C, S, G, M}`: product category, seller volume tier, customer state, calendar month.
Variables downstream of the treatment (freight charged, basket size, delivery days, whether a
review was left) are deliberately excluded, because conditioning on a mediator introduces
post-treatment bias. Deriving the set by hand, by the confound recipe, and computationally all
agree. `docs/01_treatment_and_dag.md` has the path-by-path backdoor table.

Three likelihoods, each with partial pooling over the 73 product categories and a category-varying
treatment slope, all non-centered:

- Binomial on on-time delivery, at the cell grain (category, seller tier, state, month, treatment).
- A two-stage hurdle for per-customer repeat revenue: Bernoulli on whether the customer returns
  within 180 days, LogNormal on how much they spend if they do.
- Ordered logit on the 1-5 review score, with cumulative-link cutpoints.

The interesting part is what the naive specification got wrong. Defining `treated = post-cutover
AND subtotal >= R$ 150` makes the control group everything else, so treated and control differ on
three dimensions at once: the policy, basket size, and calendar time. Each model was refit as a
hierarchical Bayesian difference-in-differences with `eligible` and `post` carrying their own main
effects and the interaction identifying the policy. On on-time delivery the decomposition is
beta_eligible = -0.19 logit (large baskets ship slower), beta_post = -0.34 logit (a common
marketplace time trend), and delta_bar = +0.17 logit. The naive -2 pp was those first two terms
swamping a real +1.5 pp lift. Retention behaved similarly: the naive -0.5 pp turned out to be
right-censoring in the 180-day repeat window, which the DiD absorbs into beta_post = -0.47, and
the policy effect on returning is null. The review-score effect was the one case where the naive
answer was already correct, because there was almost no confounding to remove.

Two things did not work as first written. The ordered-logit cutpoints share an additive ridge with
the linear-predictor intercept, and early fits showed it: cutpoint ESS around 17 to 41 with r-hat
drifting to 1.18. The fix in `src/models/review.py` anchors `kappa[0] = 0` and parameterises the
remaining cutpoints as a cumulative sum of HalfNormal gaps; cutpoint ESS went to roughly 941 to
2195 and `delta_bar` barely moved, which confirms the policy effect was always orthogonal to the
ridge. Separately, an earlier draft ranked categories by summing standardised z-scores across the
four outcomes. That is not a defensible aggregation, since logit, log, and cumulative-logit
effects on different latent scales do not add up to a meaningful unit even after standardising.
The table was removed and the per-outcome rankings are now presented side by side.

PSIS-LOO is computed for both binomial specifications but cannot be turned into an `az.compare`
table, because the two fits aggregate orders into different panel grains (16,082 cells versus
6,689). Both posteriors are well behaved, with every Pareto-k below 0.7. The choice between the
two models rests on causal identification, not on out-of-sample fit.

The Bayesian methodology here (DAGs and the four elemental confounds, partial pooling,
non-centered parameterisation, hurdle and ordered-logit likelihoods) follows Richard McElreath's
*Statistical Rethinking* and its accompanying lecture course.

## Repository layout

| Path | Contents |
|---|---|
| `sql/bronze`, `sql/silver`, `sql/gold` | medallion ETL layers |
| `sql/analytics` | cohort retention, funnel, sessionisation, modelling panels, diagnostics |
| `src/` | treatment definition, feature builders, DAG, classical baselines, paths |
| `src/models/` | PyMC factories: binomial, revenue hurdle, ordered logit, and a DiD variant of each |
| `scripts/` | one entry point per fit or diagnostic |
| `reports/` | generated writeups and figures; `final_report.md` is the full methodology |
| `docs/` | treatment definition and DAG derivation |
| `tests/` | pytest suite (unit plus DuckDB integration) |
| `data/` | gitignored; raw CSVs and the DuckDB file land here |

## Results

Policy effects on the 97k-order panel, DiD-corrected. The review model is fit on a stratified 30k
subsample of orders that carry a review score.

| Outcome | Naive Bayesian | DiD-corrected | 94% HDI | P(effect > 0) |
|---|---|---|---|---|
| On-time delivery | -2 pp | +1.5 pp (delta_bar +0.169 logit) | (+0.048, +0.289) | 99.7% |
| P(repeat within 180 d) | -0.5 pp | null (delta_b_bar +0.065 logit) | (-0.14, +0.26) | 73% |
| Conditional spend given repeat | +13.5% | +10% (delta_l_bar +0.093 log, x1.097) | (-0.07, +0.25) | 86% |
| Review score | -0.16 cum-logit | -0.17 cum-logit | (-0.29, -0.06) | 0.3% |

Classical baselines on the same slices, for comparison:

| Test | Outcome | Result |
|---|---|---|
| Two-proportion z | on-time delivery | -2.36 pp, 95% CI (-2.97, -1.74), p < 0.0001 |
| Welch t | per-customer repeat revenue, 180 d | +R$ 1.90, 95% CI (+0.44, +3.37), p = 0.011 |
| Mann-Whitney U | per-customer repeat revenue | p = 0.0002, both medians R$ 0 |
| Chi-square | review score by treatment | chi2 = 201, df = 4, p < 0.0001 |

Identification checks: the pre-trend regression on the 46,774 pre-cutover orders gives a slope
difference of -0.003 logit per week (p = 0.33), so parallel trends is consistent with the data.
The McCrary-style density test at R$ 150 finds a deficit above the threshold (Z = -13.9), not the
excess that policy bunching would produce, which is consistent with retail price points clustering
just below round numbers. Prior sensitivity moves `delta_bar` by 0.0004 logit across an
`Exponential(1)`, `Exponential(2)`, and `HalfNormal(1)` hyperprior on the across-category scale.

Sized in R$ using the observed freight data and the DiD posterior means, the policy does not pay
for itself at this design point: R$ 456,879 of freight subsidy on the 12,405 post-cutover eligible
orders against R$ 4,384 of incremental contribution margin at an assumed 20% rate, for a net of
about -R$ 452,495. Break-even would need a contribution margin around 2084%.

`reports/final_report.md` carries the full methodology, the per-category forest plots, and the
limitations.

## License

Code is MIT. The Olist Brazilian E-commerce Public Dataset is Olist Store's, released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), non-commercial use only.
