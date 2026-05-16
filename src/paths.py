"""Centralised path constants for the project.

All paths are derived from the repository root so the ETL pipeline works
identically regardless of where it's invoked from.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
DUCKDB_DIR: Path = DATA_DIR / "duckdb"
DUCKDB_PATH: Path = DUCKDB_DIR / "olist.duckdb"

SQL_DIR: Path = PROJECT_ROOT / "sql"
BRONZE_SQL_DIR: Path = SQL_DIR / "bronze"
SILVER_SQL_DIR: Path = SQL_DIR / "silver"
GOLD_SQL_DIR: Path = SQL_DIR / "gold"
ANALYTICS_SQL_DIR: Path = SQL_DIR / "analytics"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
