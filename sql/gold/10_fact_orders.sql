-- ============================================================================
-- gold.fact_orders — one row per order, denormalised, with derived signals
-- ----------------------------------------------------------------------------
-- This is THE table the Bayesian models read. It bundles together:
--
--   * payment rollup (sum across instalments, count of instalments, dominant
--     payment type via mode)
--   * item rollup  (count, distinct categories/sellers, total revenue, total
--     freight)
--   * review join  (review_score, days from delivery -> review)
--   * sequential context (per-customer order rank, days since previous order,
--     is-first-purchase flag)
--
-- Notable SQL devices used here, deliberately:
--   * CTEs to stage each rollup so types and grains are obvious
--   * MODE() WITHIN GROUP for dominant payment type
--   * COUNT(DISTINCT ...) for cardinality features
--   * ROW_NUMBER() and LAG() over the per-customer order timeline
-- ============================================================================

CREATE OR REPLACE TABLE gold.fact_orders AS
WITH payment_rollup AS (
    SELECT
        order_id,
        SUM(payment_value)                       AS payment_total,
        COUNT(*)                                 AS n_payments,
        MAX(payment_installments)                AS max_installments,
        MODE() WITHIN GROUP (
            ORDER BY payment_type
        )                                        AS dominant_payment_type
    FROM silver.order_payments
    GROUP BY order_id
),
items_rollup AS (
    SELECT
        order_id,
        COUNT(*)                                 AS n_items,
        COUNT(DISTINCT product_id)               AS n_unique_products,
        COUNT(DISTINCT category_en)              AS n_unique_categories,
        COUNT(DISTINCT seller_id)                AS n_unique_sellers,
        SUM(price)                               AS items_subtotal,
        SUM(freight_value)                       AS items_freight,
        SUM(line_revenue)                        AS items_revenue,
        AVG(price)                               AS avg_unit_price,
        MAX(price)                               AS max_unit_price,
        MODE() WITHIN GROUP (
            ORDER BY category_en
        )                                        AS dominant_category
    FROM silver.order_items
    GROUP BY order_id
),
order_with_rollups AS (
    SELECT
        o.*,
        COALESCE(pr.payment_total, 0)            AS payment_total,
        pr.n_payments,
        pr.max_installments,
        pr.dominant_payment_type,
        ir.n_items,
        ir.n_unique_products,
        ir.n_unique_categories,
        ir.n_unique_sellers,
        ir.items_subtotal,
        ir.items_freight,
        ir.items_revenue,
        ir.avg_unit_price,
        ir.max_unit_price,
        ir.dominant_category,
        r.review_score,
        DATEDIFF('day',
                 o.order_delivered_customer_date,
                 r.review_answer_timestamp)      AS review_lag_days
    FROM silver.orders               AS o
    LEFT JOIN payment_rollup         AS pr ON pr.order_id = o.order_id
    LEFT JOIN items_rollup           AS ir ON ir.order_id = o.order_id
    LEFT JOIN silver.order_reviews   AS r  ON r.order_id  = o.order_id
)
SELECT
    *,

    -- ---- per-customer order sequencing -----------------------------------
    -- Rank of this order within the customer's history (1 = first purchase).
    ROW_NUMBER() OVER (
        PARTITION BY customer_unique_id
        ORDER BY order_purchase_timestamp ASC
    )                                            AS customer_order_rank,

    -- Total orders this customer ever places — useful as a fixed feature.
    COUNT(*) OVER (
        PARTITION BY customer_unique_id
    )                                            AS customer_total_orders,

    -- Days since this customer's previous order (NULL on first purchase).
    DATEDIFF(
        'day',
        LAG(order_purchase_timestamp) OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_purchase_timestamp ASC
        ),
        order_purchase_timestamp
    )                                            AS days_since_prior_order,

    -- Convenience flag for first-purchase modelling.
    (ROW_NUMBER() OVER (
        PARTITION BY customer_unique_id
        ORDER BY order_purchase_timestamp ASC
    ) = 1)                                       AS is_first_purchase

FROM order_with_rollups;
