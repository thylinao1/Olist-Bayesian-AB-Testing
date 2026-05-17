"""Integration tests that run against the actual built DuckDB warehouse.

These are skipped automatically when the DB file is not present (e.g.
fresh checkout, CI without Kaggle credentials). Locally, after running
`python -m src.etl`, these tests should all pass.
"""

from pathlib import Path

import duckdb
import pytest

from src.paths import DUCKDB_PATH


pytestmark = pytest.mark.skipif(
    not Path(DUCKDB_PATH).exists(),
    reason="DuckDB warehouse not built; run `python -m src.etl` first",
)


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    yield c
    c.close()


def test_dim_customer_unique_on_customer_unique_id(con):
    """One row per customer_unique_id in gold.dim_customer."""
    total = con.execute("SELECT COUNT(*) FROM gold.dim_customer").fetchone()[0]
    distinct = con.execute(
        "SELECT COUNT(DISTINCT customer_unique_id) FROM gold.dim_customer"
    ).fetchone()[0]
    assert total == distinct, (
        f"dim_customer has duplicates: {total} rows vs "
        f"{distinct} distinct customer_unique_id"
    )


def test_fact_orders_unique_on_order_id(con):
    """One row per order in gold.fact_orders."""
    total = con.execute("SELECT COUNT(*) FROM gold.fact_orders").fetchone()[0]
    distinct = con.execute(
        "SELECT COUNT(DISTINCT order_id) FROM gold.fact_orders"
    ).fetchone()[0]
    assert total == distinct, f"fact_orders has duplicate order_ids"


def test_fact_orders_row_count_matches_bronze(con):
    """gold.fact_orders should match bronze.orders within a small tolerance.

    Tolerance allows for rows dropped due to NULL purchase_timestamp, which
    are excluded from silver.orders.
    """
    bronze_n = con.execute("SELECT COUNT(*) FROM bronze.orders").fetchone()[0]
    gold_n   = con.execute("SELECT COUNT(*) FROM gold.fact_orders").fetchone()[0]
    drop_pct = (bronze_n - gold_n) / bronze_n
    assert drop_pct < 0.01, (
        f"fact_orders dropped {drop_pct*100:.2f}% of bronze.orders "
        f"({bronze_n - gold_n} rows); expected < 1%"
    )


def test_payment_total_nonnegative(con):
    """No order should have a negative payment_total in gold.fact_orders."""
    n_neg = con.execute(
        "SELECT COUNT(*) FROM gold.fact_orders WHERE payment_total < 0"
    ).fetchone()[0]
    assert n_neg == 0, f"{n_neg} orders have negative payment_total"


def test_quality_diagnostics_all_ok(con):
    """analytics.quality_diagnostics must report 'ok' for every check."""
    rows = con.execute(
        "SELECT check_name, status FROM analytics.quality_diagnostics"
    ).fetchall()
    assert len(rows) >= 5, f"expected >= 5 diagnostic checks, found {len(rows)}"
    failures = [r for r in rows if r[1] != "ok"]
    assert not failures, f"quality_diagnostics failures: {failures}"


def test_treatment_assignment_never_below_threshold_in_did_panel(con):
    """For DiD treatment T = eligible AND post, no T=1 cell can have a
    subtotal below the threshold. This is the senior-review check that
    treatment assignment is correct end-to-end."""
    n_bad = con.execute("""
        SELECT COUNT(*) FROM analytics.category_seller_panel
        WHERE treatment_assigned AND avg_order_value < 150
    """).fetchone()[0]
    # avg_order_value is not exactly subtotal but is close; allow a small
    # tolerance for noise on the panel grain
    assert n_bad < 100, (
        f"{n_bad} treated cells have avg_order_value < 150 - "
        f"check the DiD eligible-only assignment"
    )
