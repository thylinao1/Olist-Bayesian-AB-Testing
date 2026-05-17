"""Hierarchical Binomial conversion model.

Model
-----
Per cell i indexed by (category c, seller_tier s, state g, month m, treatment T):

    n_delivered_i ~ Binomial(n_orders_i, p_i)

    logit(p_i) = alpha_C[c] + beta_S[s] + gamma_G[g] + delta_M[m]
                 + tau_C[c] * T_i

Adaptive (hierarchical) priors - partial pooling:

    alpha_C[c] ~ Normal(alpha_bar, sigma_alpha)
    beta_S[s]  ~ Normal(0,         sigma_beta)
    gamma_G[g] ~ Normal(0,         sigma_gamma)
    delta_M[m] ~ Normal(0,         sigma_delta)
    tau_C[c]   ~ Normal(tau_bar,   sigma_tau)

Hyperpriors:

    alpha_bar, tau_bar ~ Normal(0, 1.5)
    sigma_*           ~ Exponential(1)

Notes
-----
* Non-centered parameterisation (z * sigma + mu) is used throughout to avoid
  the divergences that the centered form would produce here.
* `tau_C[c] ~ Normal(tau_bar, sigma_tau)` lets the treatment effect itself
  vary by product category - the headline result of the analysis. A flat A/B
  test gives one number; this gives a posterior over (n_categories) effects
  plus a global mean and across-category variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BinomialModelData:
    """Tensor-shaped inputs to PyMC."""
    n_trials: np.ndarray          # shape (N,) - denominator
    n_successes: np.ndarray       # shape (N,) - numerator (success-coded outcome)
    treatment: np.ndarray         # shape (N,) {0, 1}
    category_idx: np.ndarray      # shape (N,)
    seller_tier_idx: np.ndarray   # shape (N,)
    state_idx: np.ndarray         # shape (N,)
    month_idx: np.ndarray         # shape (N,)

    # Cardinalities
    n_categories: int
    n_seller_tiers: int
    n_states: int
    n_months: int

    # Optional labels for plotting / posterior summaries
    category_labels: Sequence[str] = field(default_factory=tuple)
    outcome_name: str = "y"


def from_panel(
    panel: pd.DataFrame,
    *,
    outcome: str = "n_on_time",
    category_labels: Sequence[str] = (),
) -> BinomialModelData:
    """Convert the aggregated panel from features.panel_for_binomial.

    Parameters
    ----------
    outcome : {"n_on_time", "n_delivered"}
        Which success column to model. Defaults to on-time delivery (~89%
        base rate). `n_delivered` is also available but saturates at ~97%.
    """
    needed = ["n_orders", outcome, "treatment",
              "category_en_code", "seller_volume_tier_code",
              "customer_state_code", "calendar_month_code"]
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing columns: {missing}")

    return BinomialModelData(
        n_trials=panel["n_orders"].to_numpy(dtype=np.int64),
        n_successes=panel[outcome].to_numpy(dtype=np.int64),
        treatment=panel["treatment"].to_numpy(dtype=np.int64),
        category_idx=panel["category_en_code"].to_numpy(dtype=np.int64),
        seller_tier_idx=panel["seller_volume_tier_code"].to_numpy(dtype=np.int64),
        state_idx=panel["customer_state_code"].to_numpy(dtype=np.int64),
        month_idx=panel["calendar_month_code"].to_numpy(dtype=np.int64),
        n_categories=int(panel["category_en_code"].max()) + 1,
        n_seller_tiers=int(panel["seller_volume_tier_code"].max()) + 1,
        n_states=int(panel["customer_state_code"].max()) + 1,
        n_months=int(panel["calendar_month_code"].max()) + 1,
        category_labels=tuple(category_labels),
        outcome_name=outcome,
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_hierarchical_binomial(d: BinomialModelData) -> pm.Model:
    """Construct (but do not yet sample) the PyMC model."""
    coords = {
        "obs":          np.arange(len(d.n_trials)),
        "category":     np.asarray(d.category_labels) if d.category_labels
                         else np.arange(d.n_categories),
        "seller_tier":  np.arange(d.n_seller_tiers),
        "state":        np.arange(d.n_states),
        "month":        np.arange(d.n_months),
    }

    with pm.Model(coords=coords) as model:
        # ---- Data containers (named; show up in ArviZ trace) -------------
        n_trials    = pm.Data("n_trials",    d.n_trials,    dims="obs")
        treatment   = pm.Data("treatment",   d.treatment,   dims="obs")
        cat_idx     = pm.Data("cat_idx",     d.category_idx, dims="obs")
        tier_idx    = pm.Data("tier_idx",    d.seller_tier_idx, dims="obs")
        state_idx   = pm.Data("state_idx",   d.state_idx,   dims="obs")
        month_idx   = pm.Data("month_idx",   d.month_idx,   dims="obs")

        # ---- Hyperpriors -------------------------------------------------
        alpha_bar = pm.Normal("alpha_bar", mu=0.0, sigma=1.5)
        tau_bar   = pm.Normal("tau_bar",   mu=0.0, sigma=1.5)

        sigma_alpha = pm.Exponential("sigma_alpha", 1.0)
        sigma_beta  = pm.Exponential("sigma_beta",  1.0)
        sigma_gamma = pm.Exponential("sigma_gamma", 1.0)
        sigma_delta = pm.Exponential("sigma_delta", 1.0)
        sigma_tau   = pm.Exponential("sigma_tau",   1.0)

        # ---- Group effects, non-centered parameterisation ----------------
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="category")
        z_beta  = pm.Normal("z_beta",  0.0, 1.0, dims="seller_tier")
        z_gamma = pm.Normal("z_gamma", 0.0, 1.0, dims="state")
        z_delta = pm.Normal("z_delta", 0.0, 1.0, dims="month")
        z_tau   = pm.Normal("z_tau",   0.0, 1.0, dims="category")

        alpha_C = pm.Deterministic("alpha_C",
                                   alpha_bar + z_alpha * sigma_alpha,
                                   dims="category")
        beta_S  = pm.Deterministic("beta_S",  z_beta  * sigma_beta,
                                   dims="seller_tier")
        gamma_G = pm.Deterministic("gamma_G", z_gamma * sigma_gamma,
                                   dims="state")
        delta_M = pm.Deterministic("delta_M", z_delta * sigma_delta,
                                   dims="month")
        tau_C   = pm.Deterministic("tau_C",
                                   tau_bar + z_tau * sigma_tau,
                                   dims="category")

        # ---- Linear predictor on logit scale ----------------------------
        logit_p = (
            alpha_C[cat_idx]
            + beta_S[tier_idx]
            + gamma_G[state_idx]
            + delta_M[month_idx]
            + tau_C[cat_idx] * treatment
        )
        p = pm.Deterministic("p", pm.math.invlogit(logit_p), dims="obs")

        # ---- Likelihood -------------------------------------------------
        pm.Binomial(
            d.outcome_name,
            n=n_trials,
            p=p,
            observed=d.n_successes,
            dims="obs",
        )

    return model


# ---------------------------------------------------------------------------
# Prior predictive check helper (always do this BEFORE fitting)
# ---------------------------------------------------------------------------
def prior_predictive_check(
    d: BinomialModelData,
    *,
    samples: int = 500,
    random_seed: int | None = 0,
) -> az.InferenceData:
    """Run a prior predictive simulation. Returns the InferenceData object.

    Caller can then plot `idata.prior_predictive[d.outcome_name]` to confirm
    the priors don't put silly mass on impossible regions (e.g., the prior
    should not strongly imply >95% success rate before we have seen any data).
    """
    model = build_hierarchical_binomial(d)
    with model:
        idata = pm.sample_prior_predictive(
            samples=samples,
            random_seed=random_seed,
        )
    return idata
