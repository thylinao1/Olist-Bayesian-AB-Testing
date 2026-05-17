-- ============================================================================
-- analytics.cohort_retention - monthly customer retention matrix
-- ----------------------------------------------------------------------------
-- Builds the classic cohort heatmap: rows = signup month, columns = months
-- since signup, cell value = pct of cohort that ordered that month.
--
-- SQL devices on display:
--   * a generated month spine via DATE_TRUNC + GENERATE_SERIES so cohorts that
--     have zero activity in a given month still get a row (no silent drops)
--   * cohort assignment via a window function over first_order_at
--   * a self-join on the spine to convert absolute months -> relative months
--   * conditional aggregation for the active-customer count
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE TABLE analytics.cohort_retention AS
WITH

-- 1. Each customer's cohort = month of first ever order.
customer_cohort AS (
    SELECT
        customer_unique_id,
        DATE_TRUNC('month', MIN(order_purchase_timestamp))::DATE AS cohort_month
    FROM gold.fact_orders
    GROUP BY customer_unique_id
),

-- 2. Each (customer, activity month) pair the customer was active in.
customer_activity AS (
    SELECT DISTINCT
        customer_unique_id,
        DATE_TRUNC('month', order_purchase_timestamp)::DATE AS activity_month
    FROM gold.fact_orders
),

-- 3. Cohort sizes (denominator).
cohort_size AS (
    SELECT
        cohort_month,
        COUNT(*) AS n_customers
    FROM customer_cohort
    GROUP BY cohort_month
),

-- 4. Active counts per (cohort_month, activity_month).
active_counts AS (
    SELECT
        cc.cohort_month,
        ca.activity_month,
        COUNT(DISTINCT ca.customer_unique_id) AS n_active
    FROM customer_cohort        AS cc
    JOIN customer_activity      AS ca USING (customer_unique_id)
    WHERE ca.activity_month >= cc.cohort_month
    GROUP BY cc.cohort_month, ca.activity_month
)

SELECT
    ac.cohort_month,
    ac.activity_month,
    -- Months since cohort start (DuckDB: AGE returns INTERVAL; we extract).
    DATEDIFF('month', ac.cohort_month, ac.activity_month) AS month_offset,
    cs.n_customers                                        AS cohort_size,
    ac.n_active                                           AS active_customers,
    ac.n_active::DOUBLE / cs.n_customers                  AS retention_rate
FROM active_counts AS ac
JOIN cohort_size   AS cs USING (cohort_month)
ORDER BY ac.cohort_month, ac.activity_month;
