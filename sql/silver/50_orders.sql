-- ============================================================================
-- silver.orders - orders enriched with customer/shipping + delivery metrics
-- ----------------------------------------------------------------------------
-- The interesting derived fields here:
--   * delivery_days        - purchase -> delivery customer
--   * approval_lag_minutes - purchase -> payment approved
--   * delivery_vs_estimate - actual delivery - estimate (negative => early)
--   * is_delivered_on_time - bool
--
-- We also surface customer_unique_id directly on orders so downstream joins
-- against silver.customers don't need to bounce through customer_orders.
-- ============================================================================

CREATE OR REPLACE TABLE silver.orders AS
WITH joined AS (
    SELECT
        o.order_id,
        o.customer_id,
        co.customer_unique_id,
        o.order_status,
        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_carrier_date,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date
    FROM bronze.orders          AS o
    LEFT JOIN silver.customer_orders AS co
        ON co.shipping_id = o.customer_id
)
SELECT
    j.order_id,
    j.customer_id,
    j.customer_unique_id,
    j.order_status,

    -- date columns
    j.order_purchase_timestamp,
    j.order_approved_at,
    j.order_delivered_carrier_date,
    j.order_delivered_customer_date,
    j.order_estimated_delivery_date,

    -- convenience time grains
    DATE_TRUNC('day',   j.order_purchase_timestamp)::DATE AS purchase_date,
    DATE_TRUNC('week',  j.order_purchase_timestamp)::DATE AS purchase_week,
    DATE_TRUNC('month', j.order_purchase_timestamp)::DATE AS purchase_month,
    EXTRACT(DOW  FROM j.order_purchase_timestamp)         AS purchase_dow,
    EXTRACT(HOUR FROM j.order_purchase_timestamp)         AS purchase_hour,

    -- derived deltas
    DATEDIFF('minute', j.order_purchase_timestamp, j.order_approved_at)
        AS approval_lag_minutes,
    DATEDIFF('day', j.order_purchase_timestamp, j.order_delivered_customer_date)
        AS delivery_days,
    DATEDIFF('day', j.order_estimated_delivery_date, j.order_delivered_customer_date)
        AS delivery_vs_estimate_days,

    -- analytic flags
    (j.order_status = 'delivered')                              AS is_delivered,
    (j.order_status IN ('canceled', 'unavailable'))             AS is_lost,
    (j.order_delivered_customer_date IS NOT NULL
     AND j.order_delivered_customer_date <= j.order_estimated_delivery_date)
                                                                AS is_on_time
FROM joined AS j;
