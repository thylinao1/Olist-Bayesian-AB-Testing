"""End-to-end ETL orchestrator.

Builds the DuckDB warehouse layer-by-layer from the raw Olist CSVs:

    raw CSVs  ->  bronze  ->  silver  ->  gold  ->  analytics

Each layer is defined by a directory of .sql files in `sql/<layer>/`. Files are
executed in alphabetical order. Bronze additionally needs to know where the
raw CSVs live, so the loader templates {raw_dir} into the SQL.

Usage
-----
    python -m src.etl                    # runs all layers
    python -m src.etl bronze             # runs just bronze
    python -m src.etl bronze silver      # runs bronze then silver
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import duckdb

from .paths import (
    ANALYTICS_SQL_DIR,
    BRONZE_SQL_DIR,
    DUCKDB_PATH,
    GOLD_SQL_DIR,
    RAW_DIR,
    SILVER_SQL_DIR,
)

LAYER_DIRS: dict[str, Path] = {
    "bronze": BRONZE_SQL_DIR,
    "silver": SILVER_SQL_DIR,
    "gold": GOLD_SQL_DIR,
    "analytics": ANALYTICS_SQL_DIR,
}


def _check_raw_files() -> None:
    """Fail loudly and helpfully if the user hasn't downloaded the dataset."""
    expected = [
        "olist_customers_dataset.csv",
        "olist_geolocation_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
        "product_category_name_translation.csv",
    ]
    missing = [name for name in expected if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw CSVs in "
            f"{RAW_DIR}:\n  - "
            + "\n  - ".join(missing)
            + "\n\nSee data/README.md for download instructions."
        )


def run_layer(con: duckdb.DuckDBPyConnection, layer: str) -> None:
    """Execute every .sql file in the layer's directory, alphabetically."""
    layer_dir = LAYER_DIRS[layer]
    sql_files = sorted(layer_dir.glob("*.sql"))
    if not sql_files:
        print(f"[{layer}] no .sql files found in {layer_dir}, skipping.")
        return

    print(f"\n[{layer}] running {len(sql_files)} file(s) from {layer_dir}")
    for path in sql_files:
        sql = path.read_text()
        # Bronze needs RAW_DIR templated in so the loader can find CSVs.
        if layer == "bronze":
            sql = sql.format(raw_dir=RAW_DIR.as_posix())
        t0 = time.perf_counter()
        try:
            con.execute(sql)
        except duckdb.Error as exc:
            print(f"  ! {path.name} FAILED: {exc}", file=sys.stderr)
            raise
        dt = time.perf_counter() - t0
        print(f"  + {path.name}  ({dt:.2f}s)")


def main(layers: Sequence[str]) -> None:
    layers = layers or list(LAYER_DIRS.keys())
    invalid = [layer for layer in layers if layer not in LAYER_DIRS]
    if invalid:
        raise ValueError(
            f"Unknown layer(s): {invalid}. Valid: {list(LAYER_DIRS.keys())}"
        )

    if "bronze" in layers:
        _check_raw_files()

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to {DUCKDB_PATH}")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        for layer in layers:
            run_layer(con, layer)

    print("\nDone.")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "layers",
        nargs="*",
        choices=list(LAYER_DIRS.keys()),
        help="One or more layer names to build, in order. Default: all.",
    )
    args = parser.parse_args()
    main(args.layers)


if __name__ == "__main__":
    cli()
