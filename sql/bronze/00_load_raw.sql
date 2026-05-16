-- ============================================================================
-- BRONZE LAYER — raw CSV ingestion
-- ----------------------------------------------------------------------------
-- Goal: get every Olist CSV into DuckDB with as little transformation as
-- possible. Types are explicit so we don't pay the price of bad inference
-- later. We keep the original column names verbatim so that failures here
-- can be diff'd against the source files quickly.
--
-- The {raw_dir} placeholder is filled in by src/etl.py at runtime.
-- ============================================================================

DROP SCHEMA IF EXISTS bronze CASCADE;
CREATE SCHEMA bronze;


-- ---- orders ----------------------------------------------------------------
CREATE TABLE bronze.orders AS
SELECT *
FROM read_csv(
    '{raw_dir}/olist_orders_dataset.csv',
    header = true,
    columns = {{
        'order_id'                     : 'VARCHAR',
        'customer_id'                  : 'VARCHAR',
        'order_status'                 : 'VARCHAR',
        'order_purchase_timestamp'     : 'TIMESTAMP',
        'order_approved_at'            : 'TIMESTAMP',
        'order_delivered_carrier_date' : 'TIMESTAMP',
        'order_delivered_customer_date': 'TIMESTAMP',
        'order_estimated_delivery_date': 'TIMESTAMP'
    }}
);


-- ---- order_items -----------------------------------------------------------
CREATE TABLE bronze.order_items AS
SELECT *
FROM read_csv(
    '{raw_dir}/olist_order_items_dataset.csv',
    header = true,
    columns = {{
        'order_id'           : 'VARCHAR',
        'order_item_id'      : 'INTEGER',
        'product_id'         : 'VARCHAR',
        'seller_id'          : 'VARCHAR',
        'shipping_limit_date': 'TIMESTAMP',
        'price'              : 'DOUBLE',
        'freight_value'      : 'DOUBLE'
    }}
);


-- ---- order_payments --------------------------------------------------------
CREATE TABLE bronze.order_payments AS
SELECT *
FROM read_csv(
    '{raw_dir}/olist_order_payments_dataset.csv',
    header = true,
    columns = {{
        'order_id'            : 'VARCHAR',
        'payment_sequential'  : 'INTEGER',
        'payment_type'        : 'VARCHAR',
        'payment_installments': 'INTEGER',
        'payment_value'       : 'DOUBLE'
    }}
);


-- ---- order_reviews ---------------------------------------------------------
-- review_comment_message has commas + line breaks — DuckDB handles quoted CSV
-- fields, but we pin types just in case.
CREATE TABLE bronze.order_reviews AS
SELECT *
FROM read_csv(
    '{raw_dir}/olist_order_reviews_dataset.csv',
    header = true,
    columns = {{
        'review_id'              : 'VARCHAR',
        'order_id'               : 'VARCHAR',
        'review_score'           : 'INTEGER',
        'review_comment_title'   : 'VARCHAR',
        'review_comment_message' : 'VARCHAR',
        'review_creation_date'   : 'TIMESTAMP',
        'review_answer_timestamp': 'TIMESTAMP'
    }}
);


-- ---- customers -------------------------------------------------------------
CREATE TABLE bronze.customers AS
SELECT *
FROM read_csv(
    '{raw_dir}/olist_customers_dataset.csv',
    header = true,
    columns = {{
        'customer_id'             : 'VARCHAR',
        'customer_unique_id'      : 'VARCHAR',
        'customer_zip_code_prefix': 'VARCHAR',
        'customer_city'           : 'VARCHAR',
        'customer_state'          : 'VARCHAR'
    }}
);


-- ---- products --------------------------------------------------------------
-- Olist column names have stray double underscores; keep them as-is in bronze
-- so the source file is byte-comparable; silver renames them.
CREATE TABLE bronze.products AS
SELECT *
FROM read_csv(
    '{raw_dir}/olist_products_dataset.csv',
    header = true,
    columns = {{
        'product_id'                : 'VARCHAR',
        'product_category_name'     : 'VARCHAR',
        'product_name_lenght'       : 'INTEGER',
        'product_description_lenght': 'INTEGER',
        'product_photos_qty'        : 'INTEGER',
        'product_weight_g'          : 'INTEGER',
        'product_length_cm'         : 'INTEGER',
        'product_height_cm'         : 'INTEGER',
        'product_width_cm'          : 'INTEGER'
    }}
);


-- ---- sellers ---------------------------------------------------------------
CREATE TABLE bronze.sellers AS
SELECT *
FROM read_csv(
    '{raw_dir}/olist_sellers_dataset.csv',
    header = true,
    columns = {{
        'seller_id'             : 'VARCHAR',
        'seller_zip_code_prefix': 'VARCHAR',
        'seller_city'           : 'VARCHAR',
        'seller_state'          : 'VARCHAR'
    }}
);


-- ---- geolocation -----------------------------------------------------------
-- Heavy duplication per zip prefix; silver dedupes to a representative point.
CREATE TABLE bronze.geolocation AS
SELECT *
FROM read_csv(
    '{raw_dir}/olist_geolocation_dataset.csv',
    header = true,
    columns = {{
        'geolocation_zip_code_prefix': 'VARCHAR',
        'geolocation_lat'            : 'DOUBLE',
        'geolocation_lng'            : 'DOUBLE',
        'geolocation_city'           : 'VARCHAR',
        'geolocation_state'          : 'VARCHAR'
    }}
);


-- ---- category translation --------------------------------------------------
CREATE TABLE bronze.category_translation AS
SELECT *
FROM read_csv(
    '{raw_dir}/product_category_name_translation.csv',
    header = true,
    columns = {{
        'product_category_name'        : 'VARCHAR',
        'product_category_name_english': 'VARCHAR'
    }}
);


-- ---- sanity check ----------------------------------------------------------
-- A single SELECT that surfaces row counts on stdout via `python -m src.etl`.
SELECT
    'orders'                 AS table_name, COUNT(*) AS row_count FROM bronze.orders                 UNION ALL
SELECT 'order_items',          COUNT(*)               FROM bronze.order_items          UNION ALL
SELECT 'order_payments',       COUNT(*)               FROM bronze.order_payments       UNION ALL
SELECT 'order_reviews',        COUNT(*)               FROM bronze.order_reviews        UNION ALL
SELECT 'customers',            COUNT(*)               FROM bronze.customers            UNION ALL
SELECT 'products',             COUNT(*)               FROM bronze.products             UNION ALL
SELECT 'sellers',              COUNT(*)               FROM bronze.sellers              UNION ALL
SELECT 'geolocation',          COUNT(*)               FROM bronze.geolocation          UNION ALL
SELECT 'category_translation', COUNT(*)               FROM bronze.category_translation;
