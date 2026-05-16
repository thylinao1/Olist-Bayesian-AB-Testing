-- ============================================================================
-- analytics.quality_diagnostics — single-table audit of pipeline integrity
-- ----------------------------------------------------------------------------
-- Returns one row per check with a pass/fail flag. Anything other than
-- (status='ok') means the model upstream of it shouldn't be trusted. Run
-- this as the last step of every ETL pass.
-- ============================================================================

CREATE OR REPLACE TABLE analytics.quality_diagnostics AS
WITH checks AS (
    -- 1. Fact orders should have exactly one row per source order.
    SELECT
        'fact_orders unique on order_id' AS check_name,
        CASE
            WHEN (SELECT COUNT(*) FROM gold.fact_orders)
               = (SELECT COUNT(DISTINCT order_id) FROM gold.fact_orders)
            THEN 'ok' ELSE 'FAIL'
        END AS status,
        (SELECT COUNT(*) FROM gold.fact_orders)::VARCHAR AS observed
    UNION ALL

    -- 2. dim_customer should have one row per customer_unique_id.
    SELECT
        'dim_customer unique on customer_unique_id',
        CASE
            WHEN (SELECT COUNT(*) FROM gold.dim_customer)
               = (SELECT COUNT(DISTINCT customer_unique_id) FROM gold.dim_customer)
            THEN 'ok' ELSE 'FAIL'
        END,
        (SELECT COUNT(*) FROM gold.dim_customer)::VARCHAR
    UNION ALL

    -- 3. Every order in fact_orders must have a customer dim row.
    SELECT
        'fact_orders -> dim_customer FK',
        CASE
            WHEN (
                SELECT COUNT(*) FROM gold.fact_orders f
                WHERE NOT EXISTS (
                    SELECT 1 FROM gold.dim_customer d
                    WHERE d.customer_unique_id = f.customer_unique_id
                )
            ) = 0
            THEN 'ok' ELSE 'FAIL'
        END,
        (
            SELECT COUNT(*)::VARCHAR FROM gold.fact_orders f
            WHERE NOT EXISTS (
                SELECT 1 FROM gold.dim_customer d
                WHERE d.customer_unique_id = f.customer_unique_id
            )
        )
    UNION ALL

    -- 4. No negative payment totals.
    SELECT
        'fact_orders payment_total >= 0',
        CASE
            WHEN (
                SELECT COUNT(*) FROM gold.fact_orders WHERE payment_total < 0
            ) = 0
            THEN 'ok' ELSE 'FAIL'
        END,
        (SELECT COUNT(*)::VARCHAR FROM gold.fact_orders WHERE payment_total < 0)
    UNION ALL

    -- 5. Cohort retention rates must lie in [0, 1].
    SELECT
        'cohort_retention rate in [0,1]',
        CASE
            WHEN (
                SELECT COUNT(*) FROM analytics.cohort_retention
                WHERE retention_rate < 0 OR retention_rate > 1
            ) = 0
            THEN 'ok' ELSE 'FAIL'
        END,
        (
            SELECT COUNT(*)::VARCHAR FROM analytics.cohort_retention
            WHERE retention_rate < 0 OR retention_rate > 1
        )
)
SELECT * FROM checks;
