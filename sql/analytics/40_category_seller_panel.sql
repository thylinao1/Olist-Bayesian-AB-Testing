-- ============================================================================
-- analytics.category_seller_panel - modelling-grade panel for the Bayesian fit
-- ----------------------------------------------------------------------------
-- The hierarchical Binomial model operates on a panel grouped by:
--   * product category
--   * seller volume tier
--   * purchase week
--
-- Each cell aggregates conversion (delivered / created) and revenue. This is
-- the table PyMC reads.
--
-- Heavy lifting:
--   * we attach a hypothetical-treatment flag here so downstream models see a
--     unified schema. The actual treatment definition lives in src/features.py
--     and is parameterisable; here we just plumb the column through with a
--     placeholder default of FALSE so the table is still queryable before the
--     experiment is defined.
-- ============================================================================

CREATE OR REPLACE TABLE analytics.category_seller_panel AS
WITH base AS (
    SELECT
        f.order_id,
        f.purchase_week,
        f.dominant_category               AS category_en,
        f.is_delivered,
        f.is_lost,
        f.is_on_time,
        f.payment_total,
        f.review_score,
        s.volume_tier                     AS seller_volume_tier,
        s.seller_state
    FROM gold.fact_orders AS f
    LEFT JOIN silver.order_items AS oi
        ON oi.order_id = f.order_id
       AND oi.order_item_id = 1          -- first line item as seller anchor
    LEFT JOIN gold.dim_seller AS s
        ON s.seller_id = oi.seller_id
)

SELECT
    purchase_week,
    category_en,
    seller_volume_tier,
    seller_state,
    COUNT(*)                                                 AS n_orders,
    SUM(CAST(is_delivered AS INTEGER))                       AS n_delivered,
    SUM(CAST(is_lost      AS INTEGER))                       AS n_lost,
    SUM(CAST(is_on_time   AS INTEGER))                       AS n_on_time,
    SUM(payment_total)                                       AS gmv,
    AVG(payment_total)                                       AS avg_order_value,
    AVG(review_score::DOUBLE)                                AS avg_review_score,
    -- Conversion rate (the modelling outcome). NULLIF for safety.
    SUM(CAST(is_delivered AS INTEGER))::DOUBLE / NULLIF(COUNT(*), 0)
                                                             AS conversion_rate,
    -- Placeholder treatment flag - overwritten by feature pipeline.
    FALSE                                                    AS treatment_assigned
FROM base
WHERE category_en IS NOT NULL          -- keep panel tight; missing-category
                                       -- orders go to a separate diagnostics
                                       -- table in 50_quality_diagnostics.sql
GROUP BY purchase_week, category_en, seller_volume_tier, seller_state;
