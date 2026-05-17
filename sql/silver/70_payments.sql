-- ============================================================================
-- silver.order_payments - kept at instalment grain (one row per installment)
-- ----------------------------------------------------------------------------
-- Most analyses want the order-level total. We keep instalment grain in silver
-- because the *number* and *type* of instalments is itself an interesting
-- predictor (multi-instalment buyers tend to be lower-conversion in Brazil).
-- Order-level rollup happens in gold.
-- ============================================================================

CREATE OR REPLACE TABLE silver.order_payments AS
SELECT
    order_id,
    payment_sequential,
    payment_type,
    payment_installments,
    payment_value
FROM bronze.order_payments;
