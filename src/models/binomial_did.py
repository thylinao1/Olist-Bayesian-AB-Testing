"""Hierarchical Bayesian difference-in-differences for on-time delivery.

Why this exists
---------------
The naive binomial model in `binomial.py` compares treated (post-cutover AND
eligible-subtotal) against everything-else as the control. Under that
construction, three things differ between the two groups:

    1. The policy itself.
    2. Basket size - eligible orders are *defined* as large baskets, which
       are structurally slower to ship. (DAG: T -> B -> D -> Y mediator.)
    3. A common time trend across the whole marketplace.

The simple 2x2 logistic of `is_on_time ~ eligible + post + eligible*post`
decomposes these cleanly:

    intercept  ≈ +2.38   →  small baskets pre-cutover, 91.5% on-time
    eligible   ≈ -0.26   →  basket-size structural effect (~-2 pp)
    post       ≈ -0.24   →  common time trend (~-2 pp marketplace-wide)
    E*P (DiD)  ≈ +0.12   →  THE POLICY EFFECT (~+1.3 pp)

The naive model attributed all three to "treatment" and reported -2 pp. The
DiD identifies the +1.3 pp policy effect under a parallel-trends assumption.

Hierarchical version
--------------------
This module fits a Bayesian DiD with the same partial-pooling structure as
the original binomial model. The headline parameter is `delta_C[c]` - the
policy effect *per product category*, with a global mean `delta_bar` and
across-category variance `sigma_delta`.

    n_on_time_i ~ Binomial(n_orders_i, p_i)
    logit(p_i) = α_C[c] + β_E[e] · 1{eligible}
                        + β_P[p] · 1{post}
                        + δ_C[c] · 1{eligible AND post}
                        + γ_S[s] + γ_G[g]    (seller-tier, state)

    α_C[c] ~ Normal(α_bar, σ_α)
    δ_C[c] ~ Normal(δ_bar, σ_δ)        ← policy effect varies by category

    α_bar, δ_bar           ~ Normal(0, 1.5)
    β_E (single param)     ~ Normal(0, 1.0)   # basket-size structural shift
    β_P (single param)     ~ Normal(0, 1.0)   # common time trend
    γ_S[s], γ_G[g]         ~ Normal(0, σ_·)
    σ_*                    ~ Exponential(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pymc as pm


@dataclass(frozen=True)
class BinomialDiDData:
    """Tensor inputs for the Bayesian DiD model."""
    n_trials:        np.ndarray   # (N,)
    n_successes:     np.ndarray   # (N,)
    eligible:        np.ndarray   # (N,) {0,1}  -- subtotal >= R$ 150
    post:            np.ndarray   # (N,) {0,1}  -- purchase_week >= W*
    category_idx:    np.ndarray   # (N,)
    seller_tier_idx: np.ndarray   # (N,)
    state_idx:       np.ndarray   # (N,)
    n_categories:    int
    n_seller_tiers:  int
    n_states:        int
    category_labels: Sequence[str] = field(default_factory=tuple)
    outcome_name:    str = "n_on_time"


def from_panel(panel, *, category_labels: Sequence[str] = (),
               outcome: str = "n_on_time") -> BinomialDiDData:
    """Build BinomialDiDData from the DiD panel produced by features.panel_for_did.

    Expects columns: n_orders, <outcome>, eligible, post,
                     category_en_code, seller_volume_tier_code, customer_state_code
    """
    return BinomialDiDData(
        n_trials=panel["n_orders"].to_numpy(dtype=np.int64),
        n_successes=panel[outcome].to_numpy(dtype=np.int64),
        eligible=panel["eligible"].to_numpy(dtype=np.int64),
        post=panel["post"].to_numpy(dtype=np.int64),
        category_idx=panel["category_en_code"].to_numpy(dtype=np.int64),
        seller_tier_idx=panel["seller_volume_tier_code"].to_numpy(dtype=np.int64),
        state_idx=panel["customer_state_code"].to_numpy(dtype=np.int64),
        n_categories=int(panel["category_en_code"].max()) + 1,
        n_seller_tiers=int(panel["seller_volume_tier_code"].max()) + 1,
        n_states=int(panel["customer_state_code"].max()) + 1,
        category_labels=tuple(category_labels),
        outcome_name=outcome,
    )


def build_did_binomial(d: BinomialDiDData) -> pm.Model:
    """Construct the hierarchical Bayesian DiD model."""
    coords = {
        "obs":          np.arange(len(d.n_trials)),
        "category":     np.asarray(d.category_labels) if d.category_labels
                         else np.arange(d.n_categories),
        "seller_tier":  np.arange(d.n_seller_tiers),
        "state":        np.arange(d.n_states),
    }

    with pm.Model(coords=coords) as model:
        n_trials  = pm.Data("n_trials",  d.n_trials,        dims="obs")
        eligible  = pm.Data("eligible",  d.eligible,        dims="obs")
        post      = pm.Data("post",      d.post,            dims="obs")
        cat_idx   = pm.Data("cat_idx",   d.category_idx,    dims="obs")
        tier_idx  = pm.Data("tier_idx",  d.seller_tier_idx, dims="obs")
        state_idx = pm.Data("state_idx", d.state_idx,       dims="obs")

        # ---- Hyperpriors ----------------------------------------------
        alpha_bar = pm.Normal("alpha_bar", 0.0, 1.5)
        delta_bar = pm.Normal("delta_bar", 0.0, 1.5)        # global policy effect

        sigma_alpha = pm.Exponential("sigma_alpha", 1.0)
        sigma_delta = pm.Exponential("sigma_delta", 1.0)
        sigma_gamma_s = pm.Exponential("sigma_gamma_s", 1.0)
        sigma_gamma_g = pm.Exponential("sigma_gamma_g", 1.0)

        # ---- Single-shot main effects (not per-category) ---------------
        # eligible captures the basket-size structural effect (T->B->D->Y
        # mediator path that we don't want attributed to the policy).
        # post captures the common time trend.
        beta_eligible = pm.Normal("beta_eligible", 0.0, 1.0)
        beta_post     = pm.Normal("beta_post",     0.0, 1.0)

        # ---- Hierarchical category-level params (non-centered) ---------
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="category")
        z_delta = pm.Normal("z_delta", 0.0, 1.0, dims="category")
        z_gamma_s = pm.Normal("z_gamma_s", 0.0, 1.0, dims="seller_tier")
        z_gamma_g = pm.Normal("z_gamma_g", 0.0, 1.0, dims="state")

        alpha_C = pm.Deterministic("alpha_C",
                                   alpha_bar + z_alpha * sigma_alpha,
                                   dims="category")
        delta_C = pm.Deterministic("delta_C",
                                   delta_bar + z_delta * sigma_delta,
                                   dims="category")
        gamma_S = pm.Deterministic("gamma_S", z_gamma_s * sigma_gamma_s,
                                   dims="seller_tier")
        gamma_G = pm.Deterministic("gamma_G", z_gamma_g * sigma_gamma_g,
                                   dims="state")

        # ---- Linear predictor on logit scale --------------------------
        # The DiD interaction (eligible * post * delta_C) is the policy effect.
        # Everything else absorbs basket-size, time trend, and adjustment set.
        logit_p = (
            alpha_C[cat_idx]
            + beta_eligible * eligible
            + beta_post * post
            + delta_C[cat_idx] * eligible * post     # the policy effect
            + gamma_S[tier_idx]
            + gamma_G[state_idx]
        )
        p = pm.Deterministic("p", pm.math.invlogit(logit_p), dims="obs")

        pm.Binomial(d.outcome_name,
                    n=n_trials, p=p,
                    observed=d.n_successes, dims="obs")

    return model
