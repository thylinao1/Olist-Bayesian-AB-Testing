-- ============================================================================
-- analytics.repeat_purchase - gap-and-island session reconstruction
-- ----------------------------------------------------------------------------
-- For each customer we want to identify "purchase sessions" - clusters of
-- orders separated by a long inactive gap. Standard gap-and-island pattern:
--
--   1. order the customer's events by timestamp
--   2. flag each event as the start of a new session if the gap from the
--      previous event exceeds a threshold (we use 30 days here)
--   3. cumulative sum of those flags = the session id
--
-- Output is at the order grain so it joins back to fact_orders cleanly.
-- This is the kind of SQL e-commerce data scientists use daily for
-- sessionisation and is a strong CV signal.
-- ============================================================================

CREATE OR REPLACE TABLE analytics.repeat_purchase AS
WITH
-- 1. Per-order timeline with the previous order's timestamp.
ordered AS (
    SELECT
        order_id,
        customer_unique_id,
        order_purchase_timestamp,
        LAG(order_purchase_timestamp) OVER w_customer
            AS prev_order_at,
        DATEDIFF('day',
                 LAG(order_purchase_timestamp) OVER w_customer,
                 order_purchase_timestamp)
            AS days_since_prior
    FROM gold.fact_orders
    WINDOW w_customer AS (
        PARTITION BY customer_unique_id
        ORDER BY order_purchase_timestamp ASC
    )
),

-- 2. Flag a new session whenever gap > 30 days (or first order).
gap_flagged AS (
    SELECT
        *,
        CASE
            WHEN prev_order_at IS NULL THEN 1   -- first order = new session
            WHEN days_since_prior > 30  THEN 1
            ELSE 0
        END AS is_new_session
    FROM ordered
),

-- 3. Cumulative sum of new-session flags = session id within customer.
sessions AS (
    SELECT
        order_id,
        customer_unique_id,
        order_purchase_timestamp,
        prev_order_at,
        days_since_prior,
        SUM(is_new_session) OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_purchase_timestamp ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_id
    FROM gap_flagged
)

SELECT
    s.order_id,
    s.customer_unique_id,
    s.order_purchase_timestamp,
    s.days_since_prior,
    s.session_id,
    -- Order rank within session (1 = session opener, 2+ = session continuation).
    ROW_NUMBER() OVER (
        PARTITION BY s.customer_unique_id, s.session_id
        ORDER BY s.order_purchase_timestamp ASC
    )                                       AS session_order_rank,
    -- Convenience flag: this order opens a new session.
    (ROW_NUMBER() OVER (
        PARTITION BY s.customer_unique_id, s.session_id
        ORDER BY s.order_purchase_timestamp ASC
    ) = 1)                                  AS is_session_opener
FROM sessions AS s;
