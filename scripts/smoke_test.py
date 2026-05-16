"""End-to-end smoke test on synthetic data.

Runs every public API in the project against tiny synthetic data and asserts
the basic shape contracts hold. Not a substitute for fitting on real data,
but it catches regressions in seconds and runs in any environment that has
the project's pip dependencies installed.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Make `src` importable when the script is invoked directly from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pymc as pm

warnings.filterwarnings("ignore")


def _pass(label: str) -> None:
    print(f"  PASS  {label}")


def _section(title: str) -> None:
    bar = "-" * 60
    print(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------------
# 1.  DAG module
# ---------------------------------------------------------------------------
def test_dag() -> None:
    _section("DAG module")
    from src import dag

    g = dag.build_graph()
    assert "T" in g.nodes and "Y" in g.nodes, "T and Y must be in the DAG"
    _pass("graph builds")

    adj = dag.adjustment_set(g)
    assert adj == {"C", "S", "G", "M"}, f"unexpected adjustment set: {adj}"
    _pass(f"adjustment set is {sorted(adj)}")

    indeps = dag.implied_independencies(g)
    assert len(indeps) > 0, "DAG should imply some conditional independencies"
    _pass(f"{len(indeps)} implied conditional independencies derived")


# ---------------------------------------------------------------------------
# 2.  Binomial model
# ---------------------------------------------------------------------------
def test_binomial() -> None:
    _section("Hierarchical Binomial model")
    from src.models.binomial import (
        BinomialModelData,
        build_hierarchical_binomial,
        prior_predictive_check,
    )

    rng = np.random.default_rng(0)
    N = 100
    mdata = BinomialModelData(
        n_trials=rng.integers(5, 30, size=N).astype(np.int64),
        n_successes=rng.integers(0, 5, size=N).astype(np.int64),
        treatment=rng.integers(0, 2, size=N).astype(np.int64),
        category_idx=rng.integers(0, 4, size=N).astype(np.int64),
        seller_tier_idx=rng.integers(0, 3, size=N).astype(np.int64),
        state_idx=rng.integers(0, 5, size=N).astype(np.int64),
        month_idx=rng.integers(0, 6, size=N).astype(np.int64),
        n_categories=4, n_seller_tiers=3, n_states=5, n_months=6,
        outcome_name="y_smoke",
    )
    # Successes can't exceed trials
    mdata = BinomialModelData(
        **{**mdata.__dict__,
           "n_successes": np.minimum(mdata.n_successes, mdata.n_trials)},
    )
    model = build_hierarchical_binomial(mdata)
    assert len(model.free_RVs) == 12, f"expected 12 free RVs, got {len(model.free_RVs)}"
    _pass(f"model builds with {len(model.free_RVs)} free RVs")

    idata = prior_predictive_check(mdata, samples=20, random_seed=0)
    pp = idata.prior_predictive["y_smoke"].values
    assert pp.shape == (1, 20, N)
    assert (pp <= mdata.n_trials).all(), "prior predictive exceeds trials"
    _pass(f"prior predictive shape OK: {pp.shape}")


# ---------------------------------------------------------------------------
# 3.  Revenue model (hurdle-LogNormal)
# ---------------------------------------------------------------------------
def test_revenue() -> None:
    _section("Hurdle-LogNormal revenue model")
    from src.models.revenue import RevenueModelData, build_hurdle_lognormal

    rng = np.random.default_rng(1)
    N = 200
    has_repeat = rng.binomial(1, 0.3, size=N)
    revenue = np.where(
        has_repeat == 1,
        rng.lognormal(mean=4.0, sigma=0.5, size=N),
        0.0,
    )
    mdata = RevenueModelData(
        has_repeat=has_repeat.astype(np.int64),
        repeat_revenue=revenue,
        log_first_subtotal=np.log(rng.uniform(50, 500, size=N)),
        treatment=rng.integers(0, 2, size=N).astype(np.int64),
        category_idx=rng.integers(0, 5, size=N).astype(np.int64),
        n_categories=5,
        category_labels=tuple(f"cat_{i}" for i in range(5)),
    )
    model = build_hurdle_lognormal(mdata)
    assert len(model.free_RVs) > 0, "model has no free RVs"
    _pass(f"model builds with {len(model.free_RVs)} free RVs")

    with model:
        prior = pm.sample_prior_predictive(samples=20, random_seed=0)
    assert "y_repeat" in prior.prior_predictive
    assert "y_revenue" in prior.prior_predictive
    _pass("prior predictive draws both stages (Bernoulli + LogNormal)")


# ---------------------------------------------------------------------------
# 4.  Review model (ordered logit)
# ---------------------------------------------------------------------------
def test_review() -> None:
    _section("Ordered-logit review model")
    from src.models.review import ReviewModelData, build_ordered_logit

    rng = np.random.default_rng(2)
    N = 200
    K = 5
    mdata = ReviewModelData(
        review_score=rng.integers(0, K, size=N).astype(np.int64),
        treatment=rng.integers(0, 2, size=N).astype(np.int64),
        category_idx=rng.integers(0, 5, size=N).astype(np.int64),
        seller_tier_idx=rng.integers(0, 4, size=N).astype(np.int64),
        month_idx=rng.integers(0, 6, size=N).astype(np.int64),
        n_categories=5, n_seller_tiers=4, n_months=6, K=K,
        category_labels=tuple(f"cat_{i}" for i in range(5)),
    )
    model = build_ordered_logit(mdata)
    assert len(model.free_RVs) == 11, f"expected 11 free RVs, got {len(model.free_RVs)}"
    _pass(f"model builds with {len(model.free_RVs)} free RVs")

    with model:
        prior = pm.sample_prior_predictive(samples=20, random_seed=0)
    pp = prior.prior_predictive["y_review"].values
    assert pp.min() >= 0 and pp.max() <= K - 1, \
        f"y_review must be in [0, {K-1}], got [{pp.min()}, {pp.max()}]"
    _pass(f"prior predictive in [0, {K-1}], shape {pp.shape}")


# ---------------------------------------------------------------------------
# 5.  Classical baselines
# ---------------------------------------------------------------------------
def test_baselines() -> None:
    _section("Classical baselines")
    from src.baselines import (
        chi_square_review,
        mann_whitney,
        two_proportion_z,
        welch_t_test,
    )
    import pandas as pd

    rng = np.random.default_rng(3)
    res = two_proportion_z(80, 100, 70, 100)
    assert 0 <= res.p_value <= 1
    _pass(f"two_proportion_z: diff={res.diff:+.3f}, p={res.p_value:.3f}")

    welch = welch_t_test(rng.normal(50, 10, 200), rng.normal(48, 10, 200))
    assert 0 <= welch.p_value <= 1
    _pass(f"welch_t_test: diff={welch.diff:+.3f}, p={welch.p_value:.3f}")

    mw = mann_whitney(rng.normal(50, 10, 200), rng.normal(48, 10, 200))
    assert 0 <= mw.p_value <= 1
    _pass(f"mann_whitney: U={mw.u_statistic:.0f}, p={mw.p_value:.3f}")

    df = pd.DataFrame({
        "treatment":    rng.integers(0, 2, 500),
        "review_score": rng.integers(1, 6, 500),
    })
    chi = chi_square_review(df)
    assert chi.contingency.shape == (2, 5)
    _pass(f"chi_square_review: chi2={chi.chi2:.2f}, df={chi.df}, p={chi.p_value:.3f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> int:
    print("Running smoke tests...")
    failures = 0
    for fn in [test_dag, test_binomial, test_revenue, test_review, test_baselines]:
        try:
            fn()
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc!r}")
            failures += 1
    print()
    if failures:
        print(f"{failures} test(s) FAILED.")
        return 1
    print("All smoke tests PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
