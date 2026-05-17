-- ============================================================================
-- gold.dim_customer - one row per real customer, with lifetime aggregates
-- ----------------------------------------------------------------------------
-- Lifetime aggregates are computed from gold.fact_orders so we get all the
-- payment and item rollups for free. Tier banding is intentionally
-- deterministic (NTILE quartiles on lifetime spend) so it can be used as a
-- categorical predictor in hierarchical models without leaking the outcome.
-- ============================================================================

CREATE OR REPLACE TABLE gold.dim_customer AS
WITH lifetime AS (
    SELECT
        customer_unique_id,

        -- timing
        MIN(order_purchase_timestamp)            AS first_order_at,
        MAX(order_purchase_timestamp)            AS last_order_at,
        DATE_TRUNC('month', MIN(order_purchase_timestamp))::DATE
                                                  AS cohort_month,

        -- volume
        COUNT(*)                                  AS n_orders,
        SUM(payment_total)                        AS lifetime_revenue,
        AVG(payment_total)                        AS avg_order_value,

        -- experience
        AVG(review_score)                         AS avg_review_score,
        AVG(CASE WHEN is_on_time THEN 1.0 ELSE 0.0 END)
                                                  AS pct_on_time
    FROM gold.fact_orders
    GROUP BY customer_unique_id
),
banded AS (
    SELECT
        l.*,
        -- Quartile spend tier - ordinal 1..4, lowest -> highest spender.
        NTILE(4) OVER (ORDER BY lifetime_revenue) AS spend_quartile
    FROM lifetime AS l
)
SELECT
    b.customer_unique_id,
    c.first_zip_code_prefix,
    c.first_state,
    c.first_city,
    b.first_order_at,
    b.last_order_at,
    b.cohort_month,
    b.n_orders,
    b.lifetime_revenue,
    b.avg_order_value,
    b.avg_review_score,
    b.pct_on_time,
    b.spend_quartile,
    -- Categorical tier label for downstream readability.
    CASE b.spend_quartile
        WHEN 1 THEN 'low'
        WHEN 2 THEN 'mid_low'
        WHEN 3 THEN 'mid_high'
        WHEN 4 THEN 'high'
    END                                          AS spend_tier,
    (b.n_orders >= 2)                            AS is_repeat_customer
FROM banded               AS b
LEFT JOIN silver.customers AS c
    ON c.customer_unique_id = b.customer_unique_id;
