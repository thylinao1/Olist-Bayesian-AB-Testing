-- ============================================================================
-- silver.products — typo fixes + English category names + size proxies
-- ----------------------------------------------------------------------------
-- Olist source has Portuguese-only category labels and three columns named
-- with the misspelling 'lenght'. We rename and translate up-front.
--
-- We also derive a single 'volume_cm3' summary so downstream models don't
-- have to multiply L*W*H every time.
-- ============================================================================

CREATE OR REPLACE TABLE silver.products AS
SELECT
    p.product_id,
    p.product_category_name                          AS category_pt,
    COALESCE(t.product_category_name_english,
             p.product_category_name)                 AS category_en,
    p.product_name_lenght                            AS name_length,
    p.product_description_lenght                     AS description_length,
    p.product_photos_qty                             AS n_photos,
    p.product_weight_g                               AS weight_g,
    p.product_length_cm                              AS length_cm,
    p.product_height_cm                              AS height_cm,
    p.product_width_cm                               AS width_cm,
    -- Derived: volume in cm^3 with NULL handling
    CASE
        WHEN p.product_length_cm IS NULL
          OR p.product_height_cm IS NULL
          OR p.product_width_cm  IS NULL
        THEN NULL
        ELSE p.product_length_cm * p.product_height_cm * p.product_width_cm
    END                                              AS volume_cm3,
    -- Data-quality flag — model can filter on this
    (p.product_category_name IS NULL
     OR p.product_weight_g IS NULL
     OR p.product_length_cm IS NULL)                 AS has_missing_attributes
FROM bronze.products AS p
LEFT JOIN bronze.category_translation AS t
    ON t.product_category_name = p.product_category_name;
