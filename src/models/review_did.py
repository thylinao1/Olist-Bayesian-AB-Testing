"""Difference-in-differences ordered-logit review model.

Same DiD correction the binomial and revenue models apply. Replaces the
combined treatment indicator with `eligible`, `post`, and `eligible * post`
so the policy effect is identified separately from the basket-size
structural effect (large baskets → marginally lower review scores) and a
common marketplace-wide time trend.

The original review model had a `month` grouping factor; in the DiD design
that's collinear with `post` (post is just a binarised version of month)
so we drop it. Seller-tier and customer-state are kept as adjustment-set
covariates (forks per the DAG, not affected by treatment).

    review_score_i ~ OrderedLogit(eta = phi_i, cutpoints = kappa)

    phi_i = beta_C[c]
          + beta_eligible * eligible_i
          + beta_post     * post_i
          + delta_C[c]    * (eligible_i * post_i)         <- policy effect
          + gamma_S[s]                                    <- seller-tier adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


@dataclass(frozen=True)
class ReviewDiDData:
    review_score:    np.ndarray   # (N,) values in {0..K-1}
    eligible:        np.ndarray   # (N,) {0,1}
    post:            np.ndarray   # (N,) {0,1}
    category_idx:    np.ndarray   # (N,)
    seller_tier_idx: np.ndarray   # (N,)
    n_categories:    int
    n_seller_tiers:  int
    K:               int = 5
    category_labels: Sequence[str] = field(default_factory=tuple)


def build_model_data_did(
    df: pd.DataFrame,
    *,
    cutover_week: pd.Timestamp,
    subtotal_threshold_brl: float = 150.0,
) -> ReviewDiDData:
    """Build DiD inputs from the dataframe returned by `load_review_panel`."""
    df = df.copy()
    df["eligible"] = (df["items_subtotal"].fillna(0) >= subtotal_threshold_brl).astype(int)
    df["post"]     = (pd.to_datetime(df["purchase_week"]) >= cutover_week).astype(int)

    cats, cat_uniques = pd.factorize(df["category_en"], sort=True)
    tiers, _ = pd.factorize(df["seller_volume_tier"], sort=True)
    return ReviewDiDData(
        review_score=df["review_score"].astype(int).to_numpy() - 1,
        eligible=df["eligible"].to_numpy(dtype=np.int64),
        post=df["post"].to_numpy(dtype=np.int64),
        category_idx=cats.astype(np.int64),
        seller_tier_idx=tiers.astype(np.int64),
        n_categories=int(cats.max()) + 1,
        n_seller_tiers=int(tiers.max()) + 1,
        category_labels=tuple(cat_uniques),
    )


def build_did_ordered_logit(d: ReviewDiDData) -> pm.Model:
    coords = {
        "obs":           np.arange(len(d.review_score)),
        "category":      np.asarray(d.category_labels) if d.category_labels
                          else np.arange(d.n_categories),
        "seller_tier":   np.arange(d.n_seller_tiers),
        "cutpoint":      np.arange(d.K - 1),
        "cutpoint_free": np.arange(d.K - 2),
    }

    with pm.Model(coords=coords) as model:
        cat_idx   = pm.Data("cat_idx",   d.category_idx,    dims="obs")
        tier_idx  = pm.Data("tier_idx",  d.seller_tier_idx, dims="obs")
        eligible  = pm.Data("eligible",  d.eligible,        dims="obs")
        post      = pm.Data("post",      d.post,            dims="obs")

        # Cutpoints with kappa[0] anchored at 0 to break the cumulative-logit
        # location identifiability ridge with beta_bar. See review.py for the
        # full justification. Implementation: kappa = [0, cumsum(positive gaps)]
        # gives strictly-increasing cutpoints by construction.
        kappa_gaps = pm.HalfNormal(
            "kappa_gaps",
            sigma=1.5,
            initval=np.full(d.K - 2, 1.0),
            dims="cutpoint_free",
        )
        kappa = pm.Deterministic(
            "kappa",
            pt.concatenate([pt.zeros(1), pt.cumsum(kappa_gaps)]),
            dims="cutpoint",
        )

        beta_bar  = pm.Normal("beta_bar",     0.0, 1.0)
        delta_bar = pm.Normal("delta_bar",    0.0, 1.0)
        beta_eligible = pm.Normal("beta_eligible", 0.0, 1.0)
        beta_post     = pm.Normal("beta_post",     0.0, 1.0)

        sigma_beta    = pm.Exponential("sigma_beta",    1.0)
        sigma_delta   = pm.Exponential("sigma_delta",   1.0)
        sigma_gamma_s = pm.Exponential("sigma_gamma_s", 1.0)

        z_beta    = pm.Normal("z_beta",    0.0, 1.0, dims="category")
        z_delta   = pm.Normal("z_delta",   0.0, 1.0, dims="category")
        z_gamma_s = pm.Normal("z_gamma_s", 0.0, 1.0, dims="seller_tier")

        beta_C  = pm.Deterministic("beta_C",
                                   beta_bar + z_beta * sigma_beta,
                                   dims="category")
        delta_C = pm.Deterministic("delta_C",
                                   delta_bar + z_delta * sigma_delta,
                                   dims="category")
        gamma_S = pm.Deterministic("gamma_S", z_gamma_s * sigma_gamma_s,
                                   dims="seller_tier")

        phi = (
            beta_C[cat_idx]
            + beta_eligible * eligible
            + beta_post * post
            + delta_C[cat_idx] * eligible * post
            + gamma_S[tier_idx]
        )

        pm.OrderedLogistic(
            "y_review",
            eta=phi,
            cutpoints=kappa,
            observed=d.review_score,
            dims="obs",
        )

    return model


__all__ = [
    "ReviewDiDData",
    "build_model_data_did",
    "build_did_ordered_logit",
]
