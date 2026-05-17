-- ============================================================================
-- analytics.repeat_revenue - customer-level repeat-purchase revenue panel
-- ----------------------------------------------------------------------------
-- For each customer, given their *first* order:
--   * was the first order in the 'treated' bracket (subtotal >= R$ 150
--     AND placed after the cutover week)?
--   * within 180 days of the first order, what was the total spend on
--     subsequent orders? 0 if no repeat purchase happened.
--
-- This panel feeds the hurdle / zero-inflated revenue model.
-- ============================================================================

CREATE OR REPLACE TABLE analytics.repeat_revenue AS
WITH first_order AS (
    SELECT
        customer_unique_id,
        order_id                     AS first_order_id,
        order_purchase_timestamp     AS first_order_at,
        purchase_week                AS first_order_week,
        items_subtotal               AS first_subtotal,
        dominant_category            AS first_category,
        payment_total                AS first_payment_total
    FROM gold.fact_orders
    WHERE customer_order_rank = 1
),
later_orders AS (
    SELECT
        f.customer_unique_id,
        f.order_id,
        f.order_purchase_timestamp,
        f.payment_total
    FROM gold.fact_orders            AS f
    WHERE f.customer_order_rank >= 2
),
repeat_window AS (
    -- Sum spend on subsequent orders within 180 days of first order
    SELECT
        fo.customer_unique_id,
        SUM(lo.payment_total)        AS repeat_revenue_180d,
        COUNT(*)                     AS n_repeat_orders
    FROM first_order        AS fo
    JOIN later_orders       AS lo
        ON lo.customer_unique_id = fo.customer_unique_id
       AND lo.order_purchase_timestamp >  fo.first_order_at
       AND lo.order_purchase_timestamp <= fo.first_order_at + INTERVAL 180 DAY
    GROUP BY fo.customer_unique_id
)
SELECT
    fo.customer_unique_id,
    fo.first_order_at,
    fo.first_order_week,
    fo.first_subtotal,
    fo.first_category,
    fo.first_payment_total,
    COALESCE(rw.repeat_revenue_180d, 0)::DOUBLE          AS repeat_revenue,
    COALESCE(rw.n_repeat_orders,     0)::INTEGER         AS n_repeat_orders,
    (COALESCE(rw.n_repeat_orders, 0) > 0)                AS has_repeat
FROM first_order        AS fo
LEFT JOIN repeat_window AS rw USING (customer_unique_id);
