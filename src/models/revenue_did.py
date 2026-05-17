"""Difference-in-differences hurdle-LogNormal revenue model.

Same correction the binomial DiD applies, lifted into the two-stage hurdle
likelihood. Both stages decompose the combined treatment T = eligible AND post
into three terms:

    Stage 1 (Bernoulli - did the customer come back at all):
        logit(theta_i) = alpha_C[c]
                        + beta_e_b * eligible_i
                        + beta_p_b * post_i
                        + delta_b_C[c] * (eligible_i * post_i)   <- policy effect

    Stage 2 (LogNormal - how much they spent if they did):
        mu_i = beta_C[c]
              + beta_e_l * eligible_i
              + beta_p_l * post_i
              + delta_l_C[c] * (eligible_i * post_i)             <- policy effect
              + gamma * log_first_subtotal_i

Why the second stage needs the same correction
----------------------------------------------
`gamma_log_first_sub` was already adjusting for the *level* of the first basket
on the conditional-spend side. But the original `delta_C[c] * T` was still
absorbing two things at once (size threshold + cutover). Splitting them lets
delta_l_C[c] identify only the policy effect.

Model returns (delta_b_bar, delta_l_bar) as the two policy-effect summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import pymc as pm

from .revenue import load_repeat_revenue


@dataclass(frozen=True)
class RevenueDiDData:
    has_repeat: np.ndarray            # (N,) {0,1}
    repeat_revenue: np.ndarray        # (N,) >= 0
    log_first_subtotal: np.ndarray    # (N,)
    eligible: np.ndarray              # (N,) {0,1} - first_subtotal >= threshold
    post: np.ndarray                  # (N,) {0,1} - first_order_week >= cutover
    category_idx: np.ndarray          # (N,)
    n_categories: int
    category_labels: Sequence[str] = field(default_factory=tuple)


def build_model_data_did(
    df: pd.DataFrame,
    *,
    cutover_week: pd.Timestamp,
    subtotal_threshold_brl: float = 150.0,
) -> RevenueDiDData:
    """Build DiD inputs from the dataframe returned by `load_repeat_revenue`.

    Same defensive treatment of the LogNormal zero-revenue case as in the
    original revenue model: customers with `n_repeat_orders > 0` but
    `repeat_revenue = 0` are recoded as non-repeats (LogNormal logp at y=0
    is -inf and would crash NUTS).
    """
    df = df.copy()
    df["eligible"] = (df["first_subtotal"] >= subtotal_threshold_brl).astype(int)
    df["post"]     = (df["first_order_week"] >= cutover_week).astype(int)
    df["has_repeat"] = (
        df["has_repeat"].fillna(False).astype(bool)
        & (df["repeat_revenue"] > 0)
    ).astype(int)

    cats, uniques = pd.factorize(df["first_category"], sort=True)
    return RevenueDiDData(
        has_repeat=df["has_repeat"].to_numpy(dtype=np.int64),
        repeat_revenue=df["repeat_revenue"].astype(float).to_numpy(),
        log_first_subtotal=np.log(df["first_subtotal"].clip(lower=1.0)).to_numpy(),
        eligible=df["eligible"].to_numpy(dtype=np.int64),
        post=df["post"].to_numpy(dtype=np.int64),
        category_idx=cats.astype(np.int64),
        n_categories=int(cats.max()) + 1,
        category_labels=tuple(uniques),
    )


def build_did_hurdle_lognormal(d: RevenueDiDData) -> pm.Model:
    coords = {
        "obs":      np.arange(len(d.has_repeat)),
        "category": np.asarray(d.category_labels) if d.category_labels
                     else np.arange(d.n_categories),
    }

    pos_idx = np.flatnonzero(d.has_repeat == 1)

    with pm.Model(coords=coords) as model:
        cat_idx       = pm.Data("cat_idx",        d.category_idx,       dims="obs")
        eligible      = pm.Data("eligible",       d.eligible,           dims="obs")
        post          = pm.Data("post",           d.post,               dims="obs")
        log_first_sub = pm.Data("log_first_sub",  d.log_first_subtotal, dims="obs")

        # ---- Stage 1: Bernoulli on has_repeat ---------------------------
        alpha_bar = pm.Normal("alpha_bar",   0.0, 1.0)
        delta_b_bar = pm.Normal("delta_b_bar", 0.0, 1.0)   # policy effect, stage 1
        beta_e_b = pm.Normal("beta_e_b",   0.0, 1.0)        # basket-size, stage 1
        beta_p_b = pm.Normal("beta_p_b",   0.0, 1.0)        # time trend,  stage 1
        sigma_alpha   = pm.Exponential("sigma_alpha",   1.0)
        sigma_delta_b = pm.Exponential("sigma_delta_b", 1.0)

        z_alpha   = pm.Normal("z_alpha",   0.0, 1.0, dims="category")
        z_delta_b = pm.Normal("z_delta_b", 0.0, 1.0, dims="category")
        alpha_C   = pm.Deterministic("alpha_C",
                                     alpha_bar + z_alpha * sigma_alpha,
                                     dims="category")
        delta_b_C = pm.Deterministic("delta_b_C",
                                     delta_b_bar + z_delta_b * sigma_delta_b,
                                     dims="category")

        logit_theta = (
            alpha_C[cat_idx]
            + beta_e_b * eligible
            + beta_p_b * post
            + delta_b_C[cat_idx] * eligible * post
        )
        theta = pm.Deterministic("theta", pm.math.invlogit(logit_theta), dims="obs")
        pm.Bernoulli("y_repeat", p=theta, observed=d.has_repeat, dims="obs")

        # ---- Stage 2: LogNormal on positive-revenue rows ----------------
        pos_cat_idx       = d.category_idx[pos_idx]
        pos_eligible      = d.eligible[pos_idx]
        pos_post          = d.post[pos_idx]
        pos_log_first_sub = d.log_first_subtotal[pos_idx]
        pos_revenue       = d.repeat_revenue[pos_idx]

        beta_bar    = pm.Normal("beta_bar",    4.0, 1.5)
        delta_l_bar = pm.Normal("delta_l_bar", 0.0, 1.0)    # policy effect, stage 2
        beta_e_l    = pm.Normal("beta_e_l",    0.0, 1.0)
        beta_p_l    = pm.Normal("beta_p_l",    0.0, 1.0)
        gamma_log_first_sub = pm.Normal("gamma_log_first_sub", 0.5, 0.5)

        sigma_beta    = pm.Exponential("sigma_beta",    1.0)
        sigma_delta_l = pm.Exponential("sigma_delta_l", 1.0)
        sigma_obs     = pm.Exponential("sigma_obs",     1.0)

        z_beta    = pm.Normal("z_beta",    0.0, 1.0, dims="category")
        z_delta_l = pm.Normal("z_delta_l", 0.0, 1.0, dims="category")
        beta_C    = pm.Deterministic("beta_C",
                                     beta_bar + z_beta * sigma_beta,
                                     dims="category")
        delta_l_C = pm.Deterministic("delta_l_C",
                                     delta_l_bar + z_delta_l * sigma_delta_l,
                                     dims="category")

        mu = (
            beta_C[pos_cat_idx]
            + beta_e_l * pos_eligible
            + beta_p_l * pos_post
            + delta_l_C[pos_cat_idx] * pos_eligible * pos_post
            + gamma_log_first_sub * pos_log_first_sub
        )
        pm.LogNormal("y_revenue", mu=mu, sigma=sigma_obs,
                     observed=pos_revenue)

    return model


__all__ = [
    "RevenueDiDData",
    "build_model_data_did",
    "build_did_hurdle_lognormal",
    "load_repeat_revenue",
]
