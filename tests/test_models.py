"""Unit tests for the PyMC model factories.

Exercises model construction on small synthetic inputs. These tests
verify the API contract of each factory (correct number of free
random variables, correct prior-predictive support) without sampling
on real data, so they run fast and don't need the DuckDB pipeline.
"""

import warnings

import numpy as np
import pymc as pm
import pytest

warnings.filterwarnings("ignore")


def test_binomial_did_builds_with_expected_rvs():
    from src.models.binomial_did import BinomialDiDData, build_did_binomial

    rng = np.random.default_rng(0)
    N = 80
    mdata = BinomialDiDData(
        n_trials=rng.integers(5, 30, size=N).astype(np.int64),
        n_successes=np.zeros(N, dtype=np.int64),  # fixed below
        eligible=rng.integers(0, 2, size=N).astype(np.int64),
        post=rng.integers(0, 2, size=N).astype(np.int64),
        category_idx=rng.integers(0, 4, size=N).astype(np.int64),
        seller_tier_idx=rng.integers(0, 3, size=N).astype(np.int64),
        state_idx=rng.integers(0, 5, size=N).astype(np.int64),
        n_categories=4, n_seller_tiers=3, n_states=5,
        outcome_name="y_did_smoke",
    )
    # Successes can't exceed trials
    mdata = BinomialDiDData(
        **{**mdata.__dict__,
           "n_successes": np.minimum(
               rng.integers(0, 30, size=N).astype(np.int64), mdata.n_trials),
           },
    )
    model = build_did_binomial(mdata)
    assert len(model.free_RVs) == 12, (
        f"expected 12 free RVs, got {len(model.free_RVs)}"
    )


def test_review_did_anchors_kappa_zero():
    """The DiD review model anchors kappa[0] = 0 by construction."""
    from src.models.review_did import ReviewDiDData, build_did_ordered_logit

    rng = np.random.default_rng(0)
    N = 150
    K = 5
    mdata = ReviewDiDData(
        review_score=rng.integers(0, K, size=N).astype(np.int64),
        eligible=rng.integers(0, 2, size=N).astype(np.int64),
        post=rng.integers(0, 2, size=N).astype(np.int64),
        category_idx=rng.integers(0, 4, size=N).astype(np.int64),
        seller_tier_idx=rng.integers(0, 3, size=N).astype(np.int64),
        n_categories=4, n_seller_tiers=3, K=K,
    )
    model = build_did_ordered_logit(mdata)
    # Prior predictive on kappa should always have kappa[0] == 0 exactly.
    with model:
        prior = pm.sample_prior_predictive(samples=20, random_seed=0)
    k0 = prior.prior["kappa"].values[..., 0]
    assert np.allclose(k0, 0.0), (
        f"kappa[0] should be anchored at 0 by construction, "
        f"got min={k0.min()}, max={k0.max()}"
    )


def test_revenue_did_two_stage_prior_predictive():
    """Revenue hurdle should produce both y_repeat (Bernoulli) and y_revenue."""
    from src.models.revenue_did import RevenueDiDData, build_did_hurdle_lognormal

    rng = np.random.default_rng(0)
    N = 200
    has_repeat = rng.binomial(1, 0.2, size=N).astype(np.int64)
    revenue = np.where(
        has_repeat == 1,
        rng.lognormal(mean=4.0, sigma=0.5, size=N),
        0.0,
    )
    mdata = RevenueDiDData(
        has_repeat=has_repeat,
        repeat_revenue=revenue,
        log_first_subtotal=np.log(rng.uniform(50, 500, size=N)),
        eligible=rng.integers(0, 2, size=N).astype(np.int64),
        post=rng.integers(0, 2, size=N).astype(np.int64),
        category_idx=rng.integers(0, 5, size=N).astype(np.int64),
        n_categories=5,
    )
    model = build_did_hurdle_lognormal(mdata)
    with model:
        prior = pm.sample_prior_predictive(samples=20, random_seed=0)
    assert "y_repeat"  in prior.prior_predictive
    assert "y_revenue" in prior.prior_predictive


def test_dag_adjustment_set_unchanged():
    """The adjustment set must remain {C, S, G, M} after any DAG edit."""
    from src import dag as dag_mod

    g = dag_mod.build_graph()
    adj = dag_mod.adjustment_set(g)
    assert adj == {"C", "S", "G", "M"}, (
        f"adjustment set drifted: got {sorted(adj)}, expected ['C', 'G', 'M', 'S']"
    )


def test_baselines_two_proportion_z_sign_consistency():
    """If treated > control, the z-statistic must be positive."""
    from src.baselines import two_proportion_z

    res = two_proportion_z(80, 100, 50, 100)
    assert res.diff > 0
    assert res.z > 0
    assert 0.0 <= res.p_value <= 1.0


def test_baselines_welch_handles_tiny_inputs():
    """Welch test should not crash on minimum viable inputs."""
    from src.baselines import welch_t_test

    res = welch_t_test(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
    assert 0.0 <= res.p_value <= 1.0
    assert np.isfinite(res.t)
