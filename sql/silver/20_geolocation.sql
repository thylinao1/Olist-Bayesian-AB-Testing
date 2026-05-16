-- ============================================================================
-- silver.geolocation — one centroid per zip prefix
-- ----------------------------------------------------------------------------
-- Bronze geolocation has 1M+ rows: many (lat, lng) pairs per zip prefix
-- because each order's geocoded address is recorded individually. For
-- analytics we want a single representative coordinate per zip.
--
-- Strategy:
--   * group by zip prefix
--   * use the centroid (mean of lat/lng) as the representative point
--   * pick the modal city/state name (most-frequent city for that prefix) so
--     downstream joins by zip + city don't drift
-- ============================================================================

CREATE OR REPLACE TABLE silver.geolocation AS
WITH city_counts AS (
    SELECT
        geolocation_zip_code_prefix AS zip_prefix,
        geolocation_city            AS city,
        geolocation_state           AS state,
        COUNT(*)                    AS n_obs,
        ROW_NUMBER() OVER (
            PARTITION BY geolocation_zip_code_prefix
            ORDER BY COUNT(*) DESC, geolocation_city ASC
        ) AS rn_modal_city
    FROM bronze.geolocation
    GROUP BY 1, 2, 3
),
centroids AS (
    SELECT
        geolocation_zip_code_prefix AS zip_prefix,
        AVG(geolocation_lat)        AS lat,
        AVG(geolocation_lng)        AS lng,
        COUNT(*)                    AS n_observations
    FROM bronze.geolocation
    GROUP BY 1
)
SELECT
    cen.zip_prefix,
    cen.lat,
    cen.lng,
    cc.city  AS modal_city,
    cc.state AS modal_state,
    cen.n_observations
FROM centroids        AS cen
LEFT JOIN city_counts AS cc
    ON cc.zip_prefix = cen.zip_prefix
   AND cc.rn_modal_city = 1;
