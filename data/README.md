# Data — Olist Brazilian E-commerce

The dataset is **not committed** to the repo. It's ~45 MB across 9 CSV files. Get it once and the ETL pipeline takes care of everything from there.

## Acquire the data

### Option A — Kaggle CLI (preferred, scriptable)

```bash
# 1. Set up Kaggle credentials once: https://www.kaggle.com/docs/api
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)

# 2. Download into data/raw/
cd data/raw/
kaggle datasets download -d olistbr/brazilian-ecommerce --unzip
```

### Option B — Manual download

1. Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Click **Download** (requires a free Kaggle account)
3. Unzip the contents directly into `data/raw/`

## Expected files

After unzipping you should have (filenames are exact):

```
data/raw/
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_orders_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

## Schema overview

| Table | Rows (≈) | Grain | Key joins |
|---|---|---|---|
| `orders` | 99,441 | one per order | `customer_id` → customers; `order_id` → items, payments, reviews |
| `order_items` | 112,650 | one per order × seller × product | `order_id`, `product_id`, `seller_id` |
| `order_payments` | 103,886 | one per payment instalment | `order_id` |
| `order_reviews` | 99,224 | one per review | `order_id` |
| `customers` | 99,441 | one per customer-order shipping address | `customer_unique_id` is the dedup key |
| `products` | 32,951 | one per product | `product_id`, `product_category_name` |
| `sellers` | 3,095 | one per seller | `seller_id`, `seller_zip_code_prefix` |
| `geolocation` | 1,000,163 | one per zip prefix × lat/lng | many duplicates per zip — needs dedup |
| `category_translation` | 71 | category name PT → EN | `product_category_name` |

A note on grain: `customers` has one row per (customer × shipping address × order), so `customer_id` is *not* a stable user ID across orders — `customer_unique_id` is. This is the kind of subtlety the silver layer cleans up.

## License

Olist Brazilian E-commerce Public Dataset © Olist Store, released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Non-commercial portfolio use is fine.
