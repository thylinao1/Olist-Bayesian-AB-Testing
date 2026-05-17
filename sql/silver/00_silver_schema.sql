-- ============================================================================
-- SILVER LAYER - cleaned, typed, deduplicated tables
-- ----------------------------------------------------------------------------
-- Goals:
--   1. Fix the typos in source column names (lenght -> length).
--   2. Resolve the customer-id grain trap: bronze.customers has one row per
--      (customer_unique_id, shipping address, order). Silver gives us a stable
--      one-row-per-person customer dimension keyed on customer_unique_id.
--   3. Dedupe geolocation to one (lat, lng) centroid per zip prefix.
--   4. Translate product categories to English up-front so downstream reports
--      are recruiter-readable.
--   5. Add data-quality flags so downstream models can opt out of broken rows.
-- ============================================================================

DROP SCHEMA IF EXISTS silver CASCADE;
CREATE SCHEMA silver;
