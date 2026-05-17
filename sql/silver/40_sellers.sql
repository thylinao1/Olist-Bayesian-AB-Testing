-- ============================================================================
-- silver.sellers - sellers joined to silver.geolocation for lat/lng
-- ----------------------------------------------------------------------------
-- Joining sellers to the deduped geolocation centroid lets us model regional
-- effects (varying intercepts by state; optional Gaussian-process geo).
-- ============================================================================

CREATE OR REPLACE TABLE silver.sellers AS
SELECT
    s.seller_id,
    s.seller_zip_code_prefix         AS zip_prefix,
    s.seller_city                    AS city,
    s.seller_state                   AS state,
    g.lat,
    g.lng
FROM bronze.sellers      AS s
LEFT JOIN silver.geolocation AS g
    ON g.zip_prefix = s.seller_zip_code_prefix;
