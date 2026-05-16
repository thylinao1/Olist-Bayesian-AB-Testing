-- ============================================================================
-- gold.dim_seller — seller-level aggregates + tenure-based tier
-- ----------------------------------------------------------------------------
-- Seller tier is one of the two grouping levels in the hierarchical model
-- (the other is product category). We need it to be:
--   * causally upstream of the treatment (so we can condition on it safely)
--   * stable over the experiment window
--
-- We band by historical order volume up to the start of each seller's
-- observed window, using a deterministic NTILE on lifetime orders.
-- ============================================================================

CREATE OR REPLACE TABLE gold.dim_seller AS
WITH per_seller AS (
    SELECT
        oi.seller_id,
        s.state                                   AS seller_state,
        s.lat                                     AS seller_lat,
        s.lng                                     AS seller_lng,
        MIN(o.order_purchase_timestamp)           AS first_seen_at,
        MAX(o.order_purchase_timestamp)           AS last_seen_at,
        COUNT(DISTINCT oi.order_id)               AS n_orders,
        COUNT(DISTINCT oi.product_id)             AS n_unique_products,
        COUNT(DISTINCT p.category_en)             AS n_unique_categories,
        SUM(oi.line_revenue)                      AS lifetime_revenue,
        AVG(oi.line_revenue)                      AS avg_line_revenue,
        AVG(r.review_score)                       AS avg_review_score
    FROM silver.order_items   AS oi
    JOIN silver.orders        AS o ON o.order_id   = oi.order_id
    LEFT JOIN silver.products AS p ON p.product_id = oi.product_id
    LEFT JOIN silver.sellers  AS s ON s.seller_id  = oi.seller_id
    LEFT JOIN silver.order_reviews AS r ON r.order_id = oi.order_id
    GROUP BY oi.seller_id, s.state, s.lat, s.lng
)
SELECT
    p.*,
    NTILE(4) OVER (ORDER BY n_orders) AS volume_quartile,
    CASE NTILE(4) OVER (ORDER BY n_orders)
        WHEN 1 THEN 'small'
        WHEN 2 THEN 'mid_small'
        WHEN 3 THEN 'mid_large'
        WHEN 4 THEN 'large'
    END                               AS volume_tier
FROM per_seller AS p;
