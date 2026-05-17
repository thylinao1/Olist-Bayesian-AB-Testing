"""Feature engineering bridge between SQL gold layer and Bayesian models.

This module is the single point where:

    1. The hypothetical free-shipping treatment is defined operationally.
    2. The DAG-justified adjustment set is encoded as integer category codes
       suitable for PyMC index variables in the multilevel grouping.
    3. The panel is filtered to a reasonable analysis window so we do not
       model orders that had insufficient time to be fulfilled.

The treatment definition is concentrated here so it can be sensitivity-tested
later by varying a single parameter rather than rewriting the SQL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import duckdb
import numpy as np
import pandas as pd

from .paths import DUCKDB_PATH


# ---------------------------------------------------------------------------
# Treatment definition - see docs/01_treatment_and_dag.md
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TreatmentSpec:
    """Hypothetical 'free-shipping above subtotal threshold' policy."""

    subtotal_threshold_brl: float = 150.0      # R$ 150 = roughly the median basket
    cutover_week: pd.Timestamp | None = None   # set at runtime to median week
    delivery_grace_weeks: int = 6              # exclude orders too recent to have delivered

    def assign(self, df: pd.DataFrame) -> pd.Series:
        """Return a {0,1} series for the treatment indicator."""
        if self.cutover_week is None:
            raise ValueError("cutover_week must be set before assignment")
        post = pd.to_datetime(df["purchase_week"]) >= self.cutover_week
        eligible = df["items_subtotal"].fillna(0) >= self.subtotal_threshold_brl
        return (post & eligible).astype(int)


# ---------------------------------------------------------------------------
# Build the modelling-ready DataFrame
# ---------------------------------------------------------------------------
def load_orders(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Pull orders + adjustment-set covariates from gold/silver."""
    sql = """
        SELECT
            f.order_id,
            f.customer_unique_id,
            f.purchase_week,
            f.purchase_month,
            CAST(EXTRACT(MONTH FROM f.order_purchase_timestamp) AS INTEGER)
                                                  AS calendar_month,
            f.dominant_category                   AS category_en,
            f.payment_total,
            f.items_subtotal,
            f.items_freight,
            f.n_items,
            f.is_delivered,
            f.is_lost,
            f.is_on_time,
            f.review_score,
            c.first_state                         AS customer_state,
            ds.volume_tier                        AS seller_volume_tier,
            ds.seller_state
        FROM gold.fact_orders                AS f
        LEFT JOIN gold.dim_customer          AS c  USING (customer_unique_id)
        LEFT JOIN silver.order_items         AS oi
            ON oi.order_id = f.order_id AND oi.order_item_id = 1
        LEFT JOIN gold.dim_seller            AS ds ON ds.seller_id = oi.seller_id
        WHERE f.order_purchase_timestamp IS NOT NULL
    """
    own = con is None
    if own:
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = con.execute(sql).fetchdf()
    finally:
        if own:
            con.close()
    df["purchase_week"] = pd.to_datetime(df["purchase_week"])
    return df


def build_modelling_frame(
    spec: TreatmentSpec | None = None,
    *,
    drop_missing_category: bool = True,
) -> tuple[pd.DataFrame, TreatmentSpec, dict[str, dict]]:
    """Return (frame, finalised TreatmentSpec, label encoders).

    The label encoders dict maps each grouping factor (category, seller_tier,
    customer_state, month) to a `{label: integer_code}` mapping; PyMC needs
    integer codes for `pm.Data` index variables.
    """
    spec = spec or TreatmentSpec()

    df = load_orders()

    # Drop pre-modelling junk: orders too recent to be delivered, or missing
    # the adjustment-set variables we need.
    max_week = df["purchase_week"].max()
    cutoff = max_week - pd.Timedelta(weeks=spec.delivery_grace_weeks)
    df = df[df["purchase_week"] <= cutoff].copy()

    if drop_missing_category:
        df = df.dropna(subset=["category_en"])
    df = df.dropna(subset=["seller_volume_tier", "customer_state"]).copy()

    # Cutover week - median purchase week so treated/control are balanced.
    if spec.cutover_week is None:
        cutover_week = pd.to_datetime(df["purchase_week"].median())
        spec = TreatmentSpec(
            subtotal_threshold_brl=spec.subtotal_threshold_brl,
            cutover_week=cutover_week,
            delivery_grace_weeks=spec.delivery_grace_weeks,
        )

    df["treatment"] = spec.assign(df)

    # ---- Encode grouping factors as integer codes ------------------------
    encoders: dict[str, dict] = {}
    for col in ["category_en", "seller_volume_tier", "customer_state",
                "calendar_month"]:
        codes, uniques = pd.factorize(df[col], sort=True)
        df[f"{col}_code"] = codes
        encoders[col] = {label: i for i, label in enumerate(uniques)}

    return df.reset_index(drop=True), spec, encoders


# ---------------------------------------------------------------------------
# Aggregate to the modelling panel grain
# ---------------------------------------------------------------------------
def panel_for_binomial(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (category, seller_tier, state, month, treatment) cells.

    The hierarchical Binomial model uses an aggregated form:
    one row per cell, with `n_trials` and `n_successes` columns. This is
    statistically identical to per-row Bernoulli but ~100x faster to sample.

    Two outcome columns are produced so the modelling code can pick the one
    with usable variance:

        n_delivered  - `is_delivered` (cancelled vs reached customer)
                       Saturated at ~97% on Olist; included for completeness.
        n_on_time    - `is_on_time`   (delivered by estimated date)
                       Base rate ~89%; this is the outcome we actually fit.
    """
    grouped = (
        df.assign(_one=1)
          .groupby(
              ["category_en_code", "seller_volume_tier_code",
               "customer_state_code", "calendar_month_code", "treatment"],
              observed=True,
          )
          .agg(
              n_orders=("_one", "sum"),
              n_delivered=("is_delivered", "sum"),
              n_on_time=("is_on_time", "sum"),
              avg_subtotal=("items_subtotal", "mean"),
          )
          .reset_index()
    )
    grouped["n_delivered"] = grouped["n_delivered"].astype(int)
    grouped["n_on_time"]   = grouped["n_on_time"].astype(int)
    return grouped


def panel_for_did(df: pd.DataFrame, spec: TreatmentSpec) -> pd.DataFrame:
    """Aggregate to (category, seller_tier, state, eligible, post) cells.

    Unlike `panel_for_binomial` this keeps `eligible` and `post` as separate
    indicators so the difference-in-differences model can identify each
    main effect AND the interaction (the policy effect).

    The DiD design avoids the basket-size confound flagged in
    `reports/final_report.md`: the naive `treatment = eligible AND post`
    indicator conflates the policy with the structural basket-size slowness
    AND with any common marketplace-wide time trend. Splitting them lets
    the model attribute each gap to its own term.
    """
    df = df.assign(
        eligible=(df["items_subtotal"].fillna(0) >= spec.subtotal_threshold_brl).astype(int),
        post=(pd.to_datetime(df["purchase_week"]) >= spec.cutover_week).astype(int),
        _one=1,
    )
    grouped = (
        df.groupby(
            ["category_en_code", "seller_volume_tier_code",
             "customer_state_code", "eligible", "post"],
            observed=True,
        )
        .agg(
            n_orders=("_one", "sum"),
            n_delivered=("is_delivered", "sum"),
            n_on_time=("is_on_time", "sum"),
            avg_subtotal=("items_subtotal", "mean"),
        )
        .reset_index()
    )
    grouped["n_delivered"] = grouped["n_delivered"].astype(int)
    grouped["n_on_time"]   = grouped["n_on_time"].astype(int)
    return grouped


__all__ = [
    "TreatmentSpec",
    "load_orders",
    "build_modelling_frame",
    "panel_for_binomial",
    "panel_for_did",
]
