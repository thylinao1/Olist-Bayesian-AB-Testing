"""Regenerate the per-category forest plots from saved traces with proper sizing.

The fit scripts produce forest plots at default ArviZ figure size, which
crams 58-73 category labels into a fixed-height figure and produces
unreadably overlapping text. This script reloads each saved trace and
re-renders the forest plot with a figure height proportional to the number
of categories, so every label has enough vertical space.

Usage:
    python scripts/regenerate_forest_plots.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arviz as az
import matplotlib.pyplot as plt

from src.paths import DUCKDB_DIR, FIGURES_DIR

warnings.filterwarnings("ignore")

# Width: 11 inches. Height: 0.30 inches per category plus 2.5-inch base for
# title and margins. With 73 categories this gives a 24-inch tall figure -
# tall, but every label is readable. At 140 dpi the PNG is ~3300 px tall
# which renders cleanly in GitHub previews and in the report.
WIDTH = 11.0
HEIGHT_PER_CAT = 0.30
BASE_HEIGHT = 2.5


def _figsize(n_cats: int) -> tuple[float, float]:
    return (WIDTH, BASE_HEIGHT + HEIGHT_PER_CAT * n_cats)


def _regen(trace_path: Path, var_name: str, title: str, out_name: str) -> None:
    if not trace_path.exists():
        print(f"  skip: {trace_path.name} not found")
        return
    idata = az.from_netcdf(str(trace_path))
    try:
        n_cats = idata.posterior[var_name].sizes["category"]
    except KeyError:
        print(f"  skip {out_name}: variable {var_name} not in {trace_path.name}")
        return
    fig, ax = plt.subplots(figsize=_figsize(n_cats))
    az.plot_forest(
        idata, var_names=[var_name], combined=True, hdi_prob=0.94, ax=ax,
    )
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_name}  ({n_cats} categories, "
          f"figsize={_figsize(n_cats)})")


def main() -> None:
    print("Regenerating forest plots with proper per-row spacing...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plots = [
        # (trace_path, var_name, title, output filename)
        (DUCKDB_DIR / "binomial_idata.nc", "tau_C",
         "Per-category treatment effect on on-time delivery (naive, logit)",
         "binomial_tau_C_forest.png"),
        (DUCKDB_DIR / "binomial_did_idata.nc", "delta_C",
         "Per-category POLICY effect on on-time delivery (DiD, logit)",
         "binomial_did_delta_C_forest.png"),
        (DUCKDB_DIR / "revenue_idata.nc", "tau_C",
         "Per-category P(repeat) treatment effect (naive, logit)",
         "revenue_tau_C_forest.png"),
        (DUCKDB_DIR / "revenue_idata.nc", "delta_C",
         "Per-category conditional log-spend treatment effect (naive)",
         "revenue_delta_C_forest.png"),
        (DUCKDB_DIR / "revenue_did_idata.nc", "delta_b_C",
         "Per-category POLICY effect on P(repeat) -- Stage 1 (DiD)",
         "revenue_did_delta_b_C_forest.png"),
        (DUCKDB_DIR / "revenue_did_idata.nc", "delta_l_C",
         "Per-category POLICY effect on log(spend|repeat) -- Stage 2 (DiD)",
         "revenue_did_delta_l_C_forest.png"),
        (DUCKDB_DIR / "review_idata.nc", "tau_C",
         "Per-category treatment effect on review score (naive, cum-logit)",
         "review_tau_C_forest.png"),
        (DUCKDB_DIR / "review_did_idata.nc", "delta_C",
         "Per-category POLICY effect on review score (DiD, cum-logit)",
         "review_did_delta_C_forest.png"),
    ]

    for trace_path, var_name, title, out_name in plots:
        try:
            _regen(trace_path, var_name, title, out_name)
        except Exception as exc:
            print(f"  FAIL {out_name}: {exc!r}")

    print(f"\nAll figures regenerated in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
