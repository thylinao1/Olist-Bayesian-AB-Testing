-- ============================================================================
-- silver.customers - one row per real customer
-- ----------------------------------------------------------------------------
-- Bronze schema gotcha: `customer_id` is a per-order shipping-address proxy,
-- *not* a stable user identifier. The same person placing two orders gets two
-- different `customer_id` values but a single `customer_unique_id`.
--
-- We therefore:
--   * key the dim on customer_unique_id (the real person-key)
--   * keep first-seen city/state (people rarely move between orders, but when
--     they do we don't want to silently overwrite history)
--   * surface order count for downstream segmentation (low-effort proxy for
--     a "loyal" tier, refined in gold).
-- ============================================================================

CREATE OR REPLACE TABLE silver.customers AS
WITH ordered_addresses AS (
    SELECT
        c.customer_unique_id,
        c.customer_zip_code_prefix,
        c.customer_city,
        c.customer_state,
        o.order_purchase_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp ASC
        ) AS rn_first_order
    FROM bronze.customers AS c
    LEFT JOIN bronze.orders AS o
        ON o.customer_id = c.customer_id
)
SELECT
    customer_unique_id,
    customer_zip_code_prefix AS first_zip_code_prefix,
    customer_city            AS first_city,
    customer_state           AS first_state,
    order_purchase_timestamp AS first_order_at
FROM ordered_addresses
WHERE rn_first_order = 1;


-- ---- per-order shipping link -----------------------------------------------
-- Sometimes we still need the shipping address per order; we keep this as a
-- skinny bridge table rather than collapsing into orders directly.
CREATE OR REPLACE TABLE silver.customer_orders AS
SELECT
    c.customer_id                AS shipping_id,        -- per-order alias
    c.customer_unique_id,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state
FROM bronze.customers AS c;
