"""Hierarchical ordered-logit review-score model — Ch 12.3.

Construction
------------
Olist customers leave a review score of 1-5 (integer, ordered). Treating
this as continuous would imply the gap between 1 and 2 stars is the same
as 4 and 5 stars, which is false in nearly every Likert-style instrument.

Following McElreath §12.3, we use a *cumulative-link* (ordered-logit) model.
For K=5 outcome levels we need K-1=4 cutpoints kappa_k on the logit scale,
and a linear predictor phi_i that shifts the latent distribution.

    review_score_i ~ OrderedLogit(eta = phi_i, cutpoints = kappa)

with linear predictor (treatment + adjustment-set covariates as in Ch 13):

    phi_i = beta_C[c] + tau_C[c] * T_i + gamma_S[s] + delta_M[m]

Hierarchical (partial-pooling) priors on the category effect and category-
varying treatment slope, and unpooled effects for the smaller groupings:

    beta_C[c]  ~ Normal(beta_bar, sigma_beta)        [non-centered]
    tau_C[c]   ~ Normal(tau_bar,  sigma_tau)         [non-centered]
    gamma_S[s] ~ Normal(0,        sigma_gamma)
    delta_M[m] ~ Normal(0,        sigma_delta)

    beta_bar, tau_bar  ~ Normal(0, 1.0)
    sigma_*            ~ Exponential(1)

    kappa_k            ~ Normal(0, 1.5)   subject to  kappa_1 < kappa_2 < ...

The ordering constraint on kappa is enforced via PyMC's `transforms.ordered`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd
import pymc as pm
from pymc.distributions.transforms import ordered

from ..paths import DUCKDB_PATH


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def load_review_panel(
    *,
    cutover_week: pd.Timestamp,
    subtotal_threshold_brl: float = 150.0,
    min_category_size: int = 50,
) -> pd.DataFrame:
    """Pull one row per delivered order with a non-null review score."""
    sql = """
        SELECT
            f.order_id,
            f.customer_unique_id,
            f.review_score,
            f.payment_total,
            f.items_subtotal,
            f.purchase_week,
            f.dominant_category   AS category_en,
            f.is_delivered,
            CAST(EXTRACT(MONTH FROM f.order_purchase_timestamp) AS INTEGER)
                                  AS calendar_month,
            ds.volume_tier        AS seller_volume_tier
        FROM gold.fact_orders     AS f
        LEFT JOIN silver.order_items AS oi
            ON oi.order_id = f.order_id AND oi.order_item_id = 1
        LEFT JOIN gold.dim_seller AS ds
            ON ds.seller_id = oi.seller_id
        WHERE f.review_score IS NOT NULL
          AND f.dominant_category IS NOT NULL
          AND ds.volume_tier IS NOT NULL
          AND f.is_delivered
    """
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        df = con.execute(sql).fetchdf()
    df["purchase_week"] = pd.to_datetime(df["purchase_week"])

    df["treatment"] = (
        (df["purchase_week"] >= cutover_week)
        & (df["items_subtotal"].fillna(0) >= subtotal_threshold_brl)
    ).astype(int)

    cat_counts = df["category_en"].value_counts()
    keep = cat_counts[cat_counts >= min_category_size].index
    df = df[df["category_en"].isin(keep)].copy()
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class ReviewModelData:
    review_score:    np.ndarray   # (N,) values in {1..K}
    treatment:       np.ndarray   # (N,) {0,1}
    category_idx:    np.ndarray   # (N,)
    seller_tier_idx: np.ndarray   # (N,)
    month_idx:       np.ndarray   # (N,)
    n_categories:    int
    n_seller_tiers:  int
    n_months:        int
    K:               int = 5      # number of review-score levels (1..5)
    category_labels: Sequence[str] = field(default_factory=tuple)


def build_model_data(df: pd.DataFrame) -> ReviewModelData:
    cat_codes,  cat_uniques  = pd.factorize(df["category_en"], sort=True)
    tier_codes, _            = pd.factorize(df["seller_volume_tier"], sort=True)
    mon_codes,  _            = pd.factorize(df["calendar_month"], sort=True)

    return ReviewModelData(
        # Subtract 1 so the response is 0..K-1 (PyMC OrderedLogistic expects this)
        review_score    = (df["review_score"].astype(int).to_numpy() - 1),
        treatment       = df["treatment"].astype(int).to_numpy(),
        category_idx    = cat_codes.astype(np.int64),
        seller_tier_idx = tier_codes.astype(np.int64),
        month_idx       = mon_codes.astype(np.int64),
        n_categories    = int(cat_codes.max()) + 1,
        n_seller_tiers  = int(tier_codes.max()) + 1,
        n_months        = int(mon_codes.max()) + 1,
        category_labels = tuple(cat_uniques),
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_ordered_logit(d: ReviewModelData) -> pm.Model:
    coords = {
        "obs":         np.arange(len(d.review_score)),
        "category":    np.asarray(d.category_labels) if d.category_labels
                        else np.arange(d.n_categories),
        "seller_tier": np.arange(d.n_seller_tiers),
        "month":       np.arange(d.n_months),
        "cutpoint":    np.arange(d.K - 1),
    }

    with pm.Model(coords=coords) as model:
        # ---- Data containers --------------------------------------------
        cat_idx     = pm.Data("cat_idx",   d.category_idx,    dims="obs")
        tier_idx    = pm.Data("tier_idx",  d.seller_tier_idx, dims="obs")
        month_idx   = pm.Data("month_idx", d.month_idx,       dims="obs")
        treatment   = pm.Data("treatment", d.treatment,       dims="obs")

        # ---- Cutpoints (kappa_1 < kappa_2 < kappa_3 < kappa_4) ----------
        # Initial values evenly space the cutpoints to start ordered.
        init_cuts = np.linspace(-2.0, 2.0, d.K - 1)
        kappa = pm.Normal(
            "kappa",
            mu=0.0, sigma=1.5,
            transform=ordered,
            initval=init_cuts,
            dims="cutpoint",
        )

        # ---- Hierarchical category & treatment effects ------------------
        beta_bar  = pm.Normal("beta_bar",  0.0, 1.0)
        tau_bar   = pm.Normal("tau_bar",   0.0, 1.0)
        sigma_beta  = pm.Exponential("sigma_beta",  1.0)
        sigma_tau   = pm.Exponential("sigma_tau",   1.0)
        sigma_gamma = pm.Exponential("sigma_gamma", 1.0)
        sigma_delta = pm.Exponential("sigma_delta", 1.0)

        z_beta  = pm.Normal("z_beta",  0.0, 1.0, dims="category")
        z_tau   = pm.Normal("z_tau",   0.0, 1.0, dims="category")
        z_gamma = pm.Normal("z_gamma", 0.0, 1.0, dims="seller_tier")
        z_delta = pm.Normal("z_delta", 0.0, 1.0, dims="month")

        beta_C  = pm.Deterministic("beta_C",  beta_bar + z_beta * sigma_beta,
                                   dims="category")
        tau_C   = pm.Deterministic("tau_C",   tau_bar  + z_tau  * sigma_tau,
                                   dims="category")
        gamma_S = pm.Deterministic("gamma_S", z_gamma  * sigma_gamma,
                                   dims="seller_tier")
        delta_M = pm.Deterministic("delta_M", z_delta  * sigma_delta,
                                   dims="month")

        # ---- Linear predictor on log-cumulative-odds scale --------------
        phi = (
            beta_C[cat_idx]
            + tau_C[cat_idx] * treatment
            + gamma_S[tier_idx]
            + delta_M[month_idx]
        )

        # ---- Likelihood -------------------------------------------------
        pm.OrderedLogistic(
            "y_review",
            eta=phi,
            cutpoints=kappa,
            observed=d.review_score,
            dims="obs",
        )

    return model
