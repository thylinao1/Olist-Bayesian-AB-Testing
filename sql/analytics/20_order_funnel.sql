-- ============================================================================
-- analytics.order_funnel — order lifecycle funnel by cohort
-- ----------------------------------------------------------------------------
-- Olist orders move through a status pipeline:
--   created -> approved -> in_carrier -> delivered  (or canceled / unavailable)
--
-- Each stage has a measurable timestamp on silver.orders. We can therefore
-- treat it like a conversion funnel: of all orders placed in week W, what
-- fraction made it to each stage?
--
-- Why this matters for the project:
--   * The hierarchical Binomial model (Ch 11 + 13) needs a clean conversion
--     definition. This file defines it.
--   * The CV bullet "built funnel analysis with window functions" is real
--     work, not a buzzword.
-- ============================================================================

CREATE OR REPLACE TABLE analytics.order_funnel AS
WITH stage_per_order AS (
    SELECT
        order_id,
        purchase_week,
        purchase_month,
        order_purchase_timestamp,
        order_approved_at,
        order_delivered_carrier_date,
        order_delivered_customer_date,
        order_status,
        is_delivered,
        is_lost,
        is_on_time,

        -- Stage flags. Each stage = "did the order ever reach this point?".
        TRUE                                              AS reached_created,
        (order_approved_at IS NOT NULL)                   AS reached_approved,
        (order_delivered_carrier_date IS NOT NULL)        AS reached_carrier,
        (order_delivered_customer_date IS NOT NULL)       AS reached_delivered,

        -- Time-to-stage deltas in hours (NULL if stage never reached).
        DATEDIFF('hour',
                 order_purchase_timestamp,
                 order_approved_at)                       AS hours_to_approved,
        DATEDIFF('hour',
                 order_purchase_timestamp,
                 order_delivered_carrier_date)            AS hours_to_carrier,
        DATEDIFF('hour',
                 order_purchase_timestamp,
                 order_delivered_customer_date)           AS hours_to_delivered
    FROM silver.orders
),

-- Weekly aggregation. We compute *both* counts and conversion ratios from
-- the previous stage to make funnel drop-off legible.
weekly AS (
    SELECT
        purchase_week,
        COUNT(*)                                          AS n_created,
        SUM(CAST(reached_approved   AS INTEGER))          AS n_approved,
        SUM(CAST(reached_carrier    AS INTEGER))          AS n_carrier,
        SUM(CAST(reached_delivered  AS INTEGER))          AS n_delivered,
        SUM(CAST(is_on_time         AS INTEGER))          AS n_on_time,
        SUM(CAST(is_lost            AS INTEGER))          AS n_lost,
        AVG(hours_to_approved)                            AS avg_hours_to_approved,
        AVG(hours_to_carrier)                             AS avg_hours_to_carrier,
        AVG(hours_to_delivered)                           AS avg_hours_to_delivered
    FROM stage_per_order
    GROUP BY purchase_week
)

SELECT
    purchase_week,
    n_created,
    n_approved,
    n_carrier,
    n_delivered,
    n_on_time,
    n_lost,
    -- Stage-to-stage conversion (always relative to previous stage).
    n_approved::DOUBLE  / NULLIF(n_created,   0)          AS pct_created_to_approved,
    n_carrier::DOUBLE   / NULLIF(n_approved,  0)          AS pct_approved_to_carrier,
    n_delivered::DOUBLE / NULLIF(n_carrier,   0)          AS pct_carrier_to_delivered,
    n_on_time::DOUBLE   / NULLIF(n_delivered, 0)          AS pct_delivered_on_time,
    -- Loss rate (cancelled / unavailable / never delivered).
    n_lost::DOUBLE      / NULLIF(n_created,   0)          AS pct_lost,
    avg_hours_to_approved,
    avg_hours_to_carrier,
    avg_hours_to_delivered
FROM weekly
ORDER BY purchase_week;
