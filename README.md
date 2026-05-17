# Hierarchical Bayesian A/B Testing on a Marketplace

A portfolio project that combines a **production-style DuckDB SQL feature pipeline** with **hierarchical Bayesian inference (PyMC)**, applied to the public **Olist Brazilian E-commerce** dataset (≈100k orders across 9 relational tables).

> The Bayesian methods used here are inspired by Richard McElreath's book and accompanying YouTube course.

The headline question - *"would a hypothetical free-shipping-above-R\$ 150 policy lift on-time delivery, repeat-purchase revenue, and customer reviews - and does the answer depend on which product category we ask about?"* - is answered with three hierarchical Bayesian models running on the same DAG-justified adjustment set. Classical A/B-test baselines (two-proportion z, Welch t, Mann-Whitney U, chi-square) are run side-by-side so the gap between flat and hierarchical analysis is the storytelling hook.

> The full methodology and results write-up is in [`reports/final_report.md`](reports/final_report.md).

---

## What this project demonstrates

**SQL at the level marketplace teams actually use it.**  Multi-table joins, CTEs across bronze → silver → gold → analytics layers, window functions for cohort matrices and per-customer order ranking, gap-and-island session reconstruction, automated quality-diagnostics table. Not `SELECT … WHERE … GROUP BY`.

**Causal-inference rigour, not just regression.**  The hypothetical treatment is encoded in code, the DAG is drawn programmatically with NetworkX (`src/dag.py`) and the adjustment set is derived three ways (by hand, by the four-elemental-confounds recipe, and computationally) - they all agree. Conditional independencies are tested as falsification checks.

**Hierarchical Bayesian modelling, not toy.**  Three models (Binomial conversion, hurdle-LogNormal revenue, ordered-logit review) all use **non-centered priors** for stability, and use **varying treatment slopes by category** (a partial-pooling generalisation of a varying-intercepts setup) so the treatment effect itself is a posterior distribution per category - not a single number.

**Comparison to the methods a hiring team would default to.**  Two-proportion z, Welch t, Mann-Whitney, chi-square - run on the same data slices. The Bayesian story is told *next to* them, not instead of.

---

## Tech stack

DuckDB (storage + SQL engine, full window-function support) · Python 3.11 · PyMC 5 + ArviZ + nutpie (Rust-backed NUTS sampler) · NetworkX (DAG) · matplotlib, seaborn (figures) · scipy + statsmodels (classical baselines).

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
└── scripts/                    # fit_binomial.py, fit_revenue.py, fit_review.py,
                                # run_baselines.py
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

# 4.  Fit the three Bayesian models  (~5-8 min total on 4 cores with nutpie)
python scripts/fit_binomial.py --use-nutpie
python scripts/fit_revenue.py  --use-nutpie
python scripts/fit_review.py   --use-nutpie

# 5.  Classical baselines (instant)
python scripts/run_baselines.py
```

The `analytics.quality_diagnostics` table runs five integrity checks at the end of every ETL pass. The fit scripts each run a prior-predictive sanity check before sampling and dump R̂, ESS, and divergence counts after.

---

## License

Code: MIT. Data: Olist Brazilian E-commerce Public Dataset © Olist Store, released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) - non-commercial portfolio use is fine.
