"""Hurdle / zero-inflated repeat-revenue model.

Construction
------------
We model per-customer repeat-purchase revenue in the 180 days following
the customer's first order. Two processes generate the data:

    1.  Did the customer come back at all?  (Bernoulli)
    2.  If they did, how much did they spend?  (LogNormal on repeat revenue)

This is the natural hurdle-style mixture: most customer rows are exact
zeros (no repeat purchase) and the rest have positive continuous spend.

Likelihood
----------
    y_i = 0   with probability  1 - theta_i          # never returned
    y_i ~ LogNormal(mu_i, sigma)  if returns

equivalent to a hurdle distribution:

    p(y_i = 0)        = 1 - theta_i
    p(y_i = y | y>0)  = theta_i * LogNormalPDF(y | mu_i, sigma)

Linear models - hierarchical priors over the first-order product category
. The treatment indicator T enters BOTH stages because the policy
plausibly affects both 'do they return' and 'how much do they spend'.

    logit(theta_i) = alpha_C[c]  + tau_C[c] * T_i
    mu_i           = beta_C[c]   + delta_C[c] * T_i + log_first_subtotal_i

(We use log_first_subtotal as an offset / covariate: bigger initial baskets
predict bigger downstream spend, and not adjusting for it makes the
treatment effect look bigger than it is.)

Adaptive priors (non-centered for stability):

    alpha_C[c] = alpha_bar + z_alpha[c] * sigma_alpha
    tau_C[c]   = tau_bar   + z_tau[c]   * sigma_tau
    beta_C[c]  = beta_bar  + z_beta[c]  * sigma_beta
    delta_C[c] = delta_bar + z_delta[c] * sigma_delta

Hyperpriors:
    alpha_bar, tau_bar, beta_bar, delta_bar ~ Normal(0, 1.0)
    sigma_*                                 ~ Exponential(1)
    sigma                                   ~ Exponential(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd
import pymc as pm

from ..paths import DUCKDB_PATH


# ---------------------------------------------------------------------------
# Data loading & assembly
# ---------------------------------------------------------------------------
def load_repeat_revenue(
    *,
    cutover_week: pd.Timestamp,
    subtotal_threshold_brl: float = 150.0,
) -> pd.DataFrame:
    """Pull `analytics.repeat_revenue` and add the treatment indicator.

    Treatment: first order was placed after the cutover week AND first-order
    subtotal >= threshold (i.e., the customer would have qualified for free
    shipping under the hypothetical policy).
    """
    sql = "SELECT * FROM analytics.repeat_revenue"
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        df = con.execute(sql).fetchdf()
    df["first_order_week"] = pd.to_datetime(df["first_order_week"])
    df = df.dropna(subset=["first_category", "first_subtotal"]).copy()

    df["treatment"] = (
        (df["first_order_week"] >= cutover_week)
        & (df["first_subtotal"] >= subtotal_threshold_brl)
    ).astype(int)

    # Drop categories with very few customers - partial pooling would still
    # work but the report becomes hard to read. Threshold is generous.
    cat_counts = df["first_category"].value_counts()
    keep = cat_counts[cat_counts >= 50].index
    df = df[df["first_category"].isin(keep)].copy()
    return df


@dataclass(frozen=True)
class RevenueModelData:
    has_repeat: np.ndarray            # (N,) {0, 1}
    repeat_revenue: np.ndarray        # (N,) >= 0; >0 only when has_repeat=1
    log_first_subtotal: np.ndarray    # (N,)
    treatment: np.ndarray             # (N,) {0, 1}
    category_idx: np.ndarray          # (N,)
    n_categories: int
    category_labels: Sequence[str] = field(default_factory=tuple)


def build_model_data(df: pd.DataFrame) -> RevenueModelData:
    """Materialise the model inputs.

    Important defensive step: a small number of customers have
    `n_repeat_orders > 0` but `repeat_revenue = 0` (cancelled or
    fully-refunded repeat orders). LogNormal log-prob at y=0 is -inf,
    which crashes NUTS at initialisation. We treat these as
    non-repeats - they were not economic repeats anyway.
    """
    df = df.copy()
    df["has_repeat"] = (
        df["has_repeat"].fillna(False).astype(bool)
        & (df["repeat_revenue"] > 0)
    ).astype(int)

    cats, uniques = pd.factorize(df["first_category"], sort=True)
    return RevenueModelData(
        has_repeat=df["has_repeat"].to_numpy(dtype=np.int64),
        repeat_revenue=df["repeat_revenue"].astype(float).to_numpy(),
        log_first_subtotal=np.log(df["first_subtotal"].clip(lower=1.0)).to_numpy(),
        treatment=df["treatment"].astype(int).to_numpy(),
        category_idx=cats.astype(np.int64),
        n_categories=int(cats.max()) + 1,
        category_labels=tuple(uniques),
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_hurdle_lognormal(d: RevenueModelData) -> pm.Model:
    """Two-stage hurdle model: Bernoulli x LogNormal, both with category effects."""

    coords = {
        "obs":      np.arange(len(d.has_repeat)),
        "category": np.asarray(d.category_labels) if d.category_labels
                     else np.arange(d.n_categories),
    }

    # The repeat-only mask - used to slice the LogNormal observation set
    pos_idx = np.flatnonzero(d.has_repeat == 1)

    with pm.Model(coords=coords) as model:
        # ---- Data containers --------------------------------------------
        cat_idx        = pm.Data("cat_idx",        d.category_idx,         dims="obs")
        treatment      = pm.Data("treatment",      d.treatment,            dims="obs")
        log_first_sub  = pm.Data("log_first_sub",  d.log_first_subtotal,   dims="obs")

        # ---- STAGE 1 -- Bernoulli: did the customer return ----------------
        alpha_bar = pm.Normal("alpha_bar", 0.0, 1.0)
        tau_bar   = pm.Normal("tau_bar",   0.0, 1.0)
        sigma_alpha = pm.Exponential("sigma_alpha", 1.0)
        sigma_tau   = pm.Exponential("sigma_tau",   1.0)

        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="category")
        z_tau   = pm.Normal("z_tau",   0.0, 1.0, dims="category")
        alpha_C = pm.Deterministic("alpha_C", alpha_bar + z_alpha * sigma_alpha,
                                   dims="category")
        tau_C   = pm.Deterministic("tau_C",   tau_bar   + z_tau   * sigma_tau,
                                   dims="category")

        logit_theta = alpha_C[cat_idx] + tau_C[cat_idx] * treatment
        theta = pm.Deterministic("theta", pm.math.invlogit(logit_theta),
                                 dims="obs")

        pm.Bernoulli("y_repeat", p=theta, observed=d.has_repeat, dims="obs")

        # ---- STAGE 2 -- LogNormal: how much they spent if they returned --
        # We slice the model to only the positive-revenue rows.
        pos_cat_idx        = d.category_idx[pos_idx]
        pos_treatment      = d.treatment[pos_idx]
        pos_log_first_sub  = d.log_first_subtotal[pos_idx]
        pos_revenue        = d.repeat_revenue[pos_idx]

        beta_bar  = pm.Normal("beta_bar",  4.0, 1.5)   # log spend ~ R$ 50 baseline
        delta_bar = pm.Normal("delta_bar", 0.0, 1.0)
        gamma_log_first_sub = pm.Normal("gamma_log_first_sub", 0.5, 0.5)
        sigma_beta  = pm.Exponential("sigma_beta",  1.0)
        sigma_delta = pm.Exponential("sigma_delta", 1.0)
        sigma_obs   = pm.Exponential("sigma_obs",   1.0)

        z_beta  = pm.Normal("z_beta",  0.0, 1.0, dims="category")
        z_delta = pm.Normal("z_delta", 0.0, 1.0, dims="category")
        beta_C  = pm.Deterministic("beta_C",  beta_bar  + z_beta  * sigma_beta,
                                   dims="category")
        delta_C = pm.Deterministic("delta_C", delta_bar + z_delta * sigma_delta,
                                   dims="category")

        mu = (
            beta_C[pos_cat_idx]
            + delta_C[pos_cat_idx] * pos_treatment
            + gamma_log_first_sub * pos_log_first_sub
        )
        pm.LogNormal("y_revenue", mu=mu, sigma=sigma_obs,
                     observed=pos_revenue)

    return model
