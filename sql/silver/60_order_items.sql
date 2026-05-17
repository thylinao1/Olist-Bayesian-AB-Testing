-- ============================================================================
-- silver.order_items - line items joined to product + seller dims
-- ----------------------------------------------------------------------------
-- Plus an item-level revenue split so gold can roll up to order grain quickly.
-- ============================================================================

CREATE OR REPLACE TABLE silver.order_items AS
SELECT
    oi.order_id,
    oi.order_item_id,
    oi.product_id,
    p.category_en,
    oi.seller_id,
    s.state                       AS seller_state,
    oi.shipping_limit_date,
    oi.price,
    oi.freight_value,
    oi.price + oi.freight_value   AS line_revenue
FROM bronze.order_items AS oi
LEFT JOIN silver.products AS p ON p.product_id = oi.product_id
LEFT JOIN silver.sellers  AS s ON s.seller_id  = oi.seller_id;
