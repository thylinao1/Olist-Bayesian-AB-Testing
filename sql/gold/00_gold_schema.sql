-- ============================================================================
-- GOLD LAYER - analytics-ready facts and dimensions
-- ----------------------------------------------------------------------------
-- The gold layer is the contract between the SQL pipeline and everything
-- downstream (notebooks, models, dashboards). One grain per table, all the
-- joins already resolved, all the window-function-derived attributes baked
-- in so models never have to recompute them.
-- ============================================================================

DROP SCHEMA IF EXISTS gold CASCADE;
CREATE SCHEMA gold;
