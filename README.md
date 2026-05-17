# Hierarchical Bayesian A/B Testing on a Marketplace

A DuckDB SQL feature pipeline and a hierarchical Bayesian inference stack (PyMC) applied to the public **Olist Brazilian E-commerce** dataset - ≈100k orders across 9 relational tables.

> The Bayesian methods used here are inspired by Richard McElreath's book and accompanying YouTube course.

The headline question - *"would a hypothetical free-shipping-above-R\$ 150 policy lift on-time delivery, repeat-purchase revenue, and customer reviews, and does the answer depend on which product category we ask about?"* - is answered with three hierarchical Bayesian models running on the same DAG-justified adjustment set. Classical A/B-test baselines (two-proportion z, Welch t, Mann-Whitney U, chi-square) are run side-by-side, so the gap between flat and hierarchical analyses is visible in the same place as each headline number.

> The full methodology and results write-up is in [`reports/final_report.md`](reports/final_report.md).

---

## Scope

The SQL layer is structured as a medallion pipeline (bronze → silver → gold → analytics) on DuckDB: multi-table joins, CTEs, window functions for cohort matrices and per-customer order ranking, gap-and-island session reconstruction, and an automated quality-diagnostics table.

The causal layer encodes the hypothetical treatment in code, draws the DAG programmatically with NetworkX (`src/dag.py`), and derives the adjustment set three ways (by hand, by the four-elemental-confounds recipe, and computationally) - all three agree. Conditional independencies are tested as falsification checks.

The Bayesian layer fits three hierarchical models (Binomial conversion, hurdle-LogNormal revenue, ordered-logit review). All three use non-centered priors and carry varying treatment slopes by product category, so the treatment effect itself is a posterior distribution per category rather than a single number.

Classical baselines (two-proportion z, Welch t, Mann-Whitney, chi-square) are run on the same data slices for comparison.

---

## Tech stack

DuckDB (storage + SQL engine, full window-function support) · Python 3.11 · PyMC 5 + ArviZ + nutpie (Rust-backed NUTS sampler) · NetworkX (DAG) · matplotlib, seaborn (figures) · scipy + statsmodels (classical baselines).

**On scalability.** NUTS MCMC at this scale (97k orders × ~120 free parameters, ~5 min per fit on 4 cores with `nutpie`) gives full posterior uncertainty, but would not support daily refreshes at Shopee- or TikTok-Shop-style data volumes. The realistic production alternatives are: (1) variational inference (`pm.fit(method="advi")`), which trades some posterior-tail fidelity for ~100× speedup; (2) Laplace approximation around the posterior mode, sufficient when the likelihood is well-behaved; (3) a frequentist DiD-logistic, which runs in milliseconds and (as the cross-method triangulation in §4.1 of the report shows) returns essentially the same point estimate as the Bayesian fit. Full MCMC is used here because the analysis is about credible intervals over latent decomposition channels, not real-time scoring.

**On the geolocation data.** Olist ships a 1M-row geolocation table (one row per zip-prefix × lat/lng observation). The silver layer deduplicates it to one centroid per zip prefix, then the seller dimension joins on `seller_zip_code_prefix` to attach lat/lng to each seller. Customer state and seller state are used as adjustment-set variables in the models (per the DAG). The raw lat/lng coordinates themselves are not used in the current models - a natural extension would be a distance-to-seller covariate as a delivery-time confound, or a Gaussian-process state-level effect, but neither was needed for the headline analysis.

---

## Repository layout

```
.
├── data/                       # gitignored: raw CSVs + DuckDB file
├── docs/
│   └── 01_treatment_and_dag.md # treatment definition + DAG + adjustment set
├── reports/
│   ├── final_report.md         # full methodology and results
│   ├── baselines.md            # classical-test results (auto-generated)
│   └── figures/
├── sql/
│   ├── bronze/  silver/  gold/ # ETL layers
│   └── analytics/              # cohorts, funnels, retention, panels, diagnostics
├── src/
│   ├── etl.py     features.py  # data pipeline + treatment / panel builders
│   ├── dag.py     baselines.py # causal graph + classical tests
│   └── models/                 # PyMC factories: binomial, revenue, review
└── scripts/                    # fit_binomial[_did].py, fit_revenue[_did].py,
                                # fit_review[_did].py, run_baselines.py,
                                # parallel_trends.py, bunching_test.py,
                                # prior_sensitivity.py,
                                # posterior_predictive_checks.py,
                                # model_comparison.py, cost_benefit_envelope.py,
                                # category_recommendations.py, smoke_test.py,
                                # regenerate_forest_plots.py
```

---

## Reproduce in under 10 minutes

```
git clone <repo>; cd "Bayesian Project"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1.  Acquire the Olist data - see data/README.md
kaggle datasets download -d olistbr/brazilian-ecommerce --unzip -p data/raw/

# 2.  Build the warehouse: bronze → silver → gold → analytics  (~5s)
python -m src.etl

# 3.  Render the DAG figure
python -m src.dag

# 4.  Fit the naive Bayesian models  (~10 min total on 4 cores with nutpie)
python scripts/fit_binomial.py --use-nutpie
python scripts/fit_revenue.py  --use-nutpie
python scripts/fit_review.py   --use-nutpie

# 5.  Fit the DiD-corrected Bayesian models  (the canonical answers in the
#     report come from these, not from step 4; ~15 min total)
python scripts/fit_binomial_did.py --use-nutpie
python scripts/fit_revenue_did.py
python scripts/fit_review_did.py

# 6.  Classical baselines (instant)
python scripts/run_baselines.py

# 7.  Causal-identification checks and downstream summaries (instant; on saved fits)
python scripts/parallel_trends.py            # formal pre-trend test for the DiD
python scripts/bunching_test.py              # McCrary-style density check at R$ 150
python scripts/prior_sensitivity.py --use-nutpie    # ~90 s, three small fits
python scripts/posterior_predictive_checks.py
python scripts/model_comparison.py
python scripts/cost_benefit_envelope.py
python scripts/category_recommendations.py
```

The `analytics.quality_diagnostics` table runs five integrity checks at the end of every ETL pass. The fit scripts each run a prior-predictive sanity check before sampling and dump R̂, ESS, and divergence counts after.

## Tests

A pytest suite under `tests/` covers treatment-assignment correctness, PyMC model construction (including the κ[0]-anchoring invariant), and DuckDB integration. Run with:

```
pytest tests/ -v
```

11 unit tests run anywhere (no Kaggle data needed). 6 integration tests auto-skip if the DuckDB warehouse isn't built. A GitHub Actions workflow at `.github/workflows/ci.yml` runs the unit tests + smoke test on every push.

---

## License

Code: MIT. Data: Olist Brazilian E-commerce Public Dataset © Olist Store, released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) - non-commercial use only.
