"""Unit tests for the treatment-assignment and feature-building logic.

These tests use small synthetic DataFrames so they run without the full
Olist data, which means they are safe to run in CI without secrets.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import TreatmentSpec


def _toy_orders():
    return pd.DataFrame({
        "purchase_week": pd.to_datetime([
            "2017-06-01", "2017-12-01",     # pre-cutover
            "2018-02-01", "2018-06-01",     # post-cutover
        ] * 2),
        "items_subtotal": [
            50.0, 200.0, 80.0, 300.0,       # mix of below/above threshold
            150.0, 149.99, 0.0, 500.0,
        ],
    })


def test_treatment_assignment_requires_cutover_week():
    spec = TreatmentSpec()
    with pytest.raises(ValueError, match="cutover_week must be set"):
        spec.assign(_toy_orders())


def test_treatment_zero_below_threshold():
    """T must NEVER be 1 for orders below the subtotal threshold."""
    spec = TreatmentSpec(
        subtotal_threshold_brl=150.0,
        cutover_week=pd.Timestamp("2018-01-01"),
    )
    df = _toy_orders()
    t = spec.assign(df)
    below = df["items_subtotal"] < 150.0
    assert (t[below] == 0).all(), (
        f"Treatment leaked below threshold: "
        f"{t[below].tolist()} for subtotals {df.loc[below, 'items_subtotal'].tolist()}"
    )


def test_treatment_zero_pre_cutover():
    """T must NEVER be 1 for orders placed before the cutover week."""
    spec = TreatmentSpec(
        subtotal_threshold_brl=150.0,
        cutover_week=pd.Timestamp("2018-01-01"),
    )
    df = _toy_orders()
    t = spec.assign(df)
    pre = df["purchase_week"] < spec.cutover_week
    assert (t[pre] == 0).all(), (
        f"Treatment leaked pre-cutover: {t[pre].tolist()}"
    )


def test_treatment_one_iff_eligible_and_post():
    """T == 1 exactly when (subtotal >= threshold) AND (week >= cutover)."""
    spec = TreatmentSpec(
        subtotal_threshold_brl=150.0,
        cutover_week=pd.Timestamp("2018-01-01"),
    )
    df = _toy_orders()
    t = spec.assign(df)
    expected = (
        (df["items_subtotal"] >= 150.0)
        & (df["purchase_week"] >= spec.cutover_week)
    ).astype(int)
    pd.testing.assert_series_equal(
        t.reset_index(drop=True), expected.reset_index(drop=True),
        check_names=False,
    )


def test_treatment_handles_nan_subtotal():
    """A NaN subtotal must NEVER be classified as eligible."""
    spec = TreatmentSpec(
        subtotal_threshold_brl=150.0,
        cutover_week=pd.Timestamp("2018-01-01"),
    )
    df = pd.DataFrame({
        "purchase_week": pd.to_datetime(["2018-03-01"]),
        "items_subtotal": [np.nan],
    })
    t = spec.assign(df)
    assert int(t.iloc[0]) == 0, "NaN subtotal should not be treated"
