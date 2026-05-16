-- ============================================================================
-- silver.order_reviews — one row per order
-- ----------------------------------------------------------------------------
-- Source has occasional duplicates per order_id (a customer can sometimes edit
-- a review). We keep the most recently answered version using ROW_NUMBER.
-- ============================================================================

CREATE OR REPLACE TABLE silver.order_reviews AS
WITH ranked AS (
    SELECT
        review_id,
        order_id,
        review_score,
        review_creation_date,
        review_answer_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY review_answer_timestamp DESC NULLS LAST
        ) AS rn
    FROM bronze.order_reviews
)
SELECT
    review_id,
    order_id,
    review_score,
    review_creation_date,
    review_answer_timestamp
FROM ranked
WHERE rn = 1;
