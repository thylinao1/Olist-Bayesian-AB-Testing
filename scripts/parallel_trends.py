"""Parallel-trends visualisation and formal pre-trend test for the DiD.

The DiD design in `src/models/binomial_did.py` identifies the policy effect
under the assumption that the pre-cutover on-time delivery trends are
parallel between the eligible (subtotal >= R$ 150) and ineligible cohorts.
This script makes that assumption testable, two ways:

1.  **Visual check** — plot weekly on-time rates for both cohorts across
    the full panel, with a vertical line at the cutover week. If the two
    lines move roughly in parallel before the cutover, the assumption
    holds visually.

2.  **Formal pre-trend regression** — restrict to the pre-cutover window
    and fit:

        logit(on_time) = a + b*time + c*eligible + d*(time x eligible)

    The interaction coefficient `d` is the slope difference between the
    two cohorts in the pre-period. If `d` is statistically
    indistinguishable from zero, the parallel-trends assumption is
    consistent with the data. A significant `d` would invalidate the
    naive DiD identification and require a more flexible specification
    (e.g., differential pre-trends adjusted out, or interrupted-time-
    series with seasonal controls).

Output:
    reports/figures/parallel_trends.png
    reports/parallel_trends.md (writeup with the regression numbers)

Usage:
    python scripts/parallel_trends.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.features import build_modelling_frame
from src.paths import DUCKDB_PATH, FIGURES_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")


def load_weekly_on_time(threshold: float) -> pd.DataFrame:
    """Pull weekly on-time rates split by eligibility from gold.fact_orders."""
    sql = f"""
        SELECT
            purchase_week,
            (items_subtotal >= {threshold})        AS eligible,
            COUNT(*)                               AS n,
            SUM(CAST(is_on_time AS INTEGER))       AS on_time
        FROM gold.fact_orders
        WHERE items_subtotal IS NOT NULL
          AND items_freight  IS NOT NULL
          AND purchase_week  IS NOT NULL
          AND is_delivered                          -- on-time only defined on delivered
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        df = con.execute(sql).fetchdf()
    df["purchase_week"] = pd.to_datetime(df["purchase_week"])
    df["rate"] = df["on_time"] / df["n"]
    return df


def fit_pre_trend(weekly: pd.DataFrame, cutover_week: pd.Timestamp) -> dict:
    """Fit logit(on_time) ~ time + eligible + time:eligible on the pre-period.

    Returns dict with coefficient estimates, SE, p-values, and the
    headline `d` (interaction) coefficient that tests parallel trends.
    """
    pre = weekly[weekly["purchase_week"] < cutover_week].copy()
    # Convert purchase_week to a numeric "weeks since first observation" so
    # the regression coefficient is interpretable as logit-points-per-week.
    pre["t"] = (
        (pre["purchase_week"] - pre["purchase_week"].min()).dt.days / 7.0
    )
    pre["eligible"] = pre["eligible"].astype(float)
    pre["t_x_eligible"] = pre["t"] * pre["eligible"]

    X = sm.add_constant(
        pre[["t", "eligible", "t_x_eligible"]].astype(float).values
    )
    # We have aggregated cell-level data, so use Binomial-family GLM with
    # successes / trials encoding rather than per-observation logistic.
    mod = sm.GLM(
        endog=np.column_stack([pre["on_time"], pre["n"] - pre["on_time"]]),
        exog=X,
        family=sm.families.Binomial(),
    ).fit()

    return {
        "n_weeks":       int(pre["purchase_week"].nunique()),
        "n_orders_pre":  int(pre["n"].sum()),
        "const":         {"coef": float(mod.params[0]),
                          "se":   float(mod.bse[0]),
                          "p":    float(mod.pvalues[0])},
        "time":          {"coef": float(mod.params[1]),
                          "se":   float(mod.bse[1]),
                          "p":    float(mod.pvalues[1])},
        "eligible":      {"coef": float(mod.params[2]),
                          "se":   float(mod.bse[2]),
                          "p":    float(mod.pvalues[2])},
        "time_x_elig":   {"coef": float(mod.params[3]),
                          "se":   float(mod.bse[3]),
                          "p":    float(mod.pvalues[3])},
        "aic":           float(mod.aic),
        "df_resid":      int(mod.df_resid),
    }


def plot_trends(weekly: pd.DataFrame, cutover_week: pd.Timestamp,
                out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for elig, group in weekly.groupby("eligible"):
        label = ("eligible (subtotal >= R$ 150)" if elig
                 else "ineligible (subtotal < R$ 150)")
        ax.plot(group["purchase_week"], group["rate"] * 100,
                marker="o", markersize=4, linewidth=1.2, label=label)
    ax.axvline(cutover_week, color="grey", linestyle="--", linewidth=1.2,
               label=f"cutover ({cutover_week.date()})")
    ax.set_ylabel("On-time delivery rate (%)")
    ax.set_xlabel("Purchase week")
    ax.set_title("Weekly on-time delivery rate by basket eligibility")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_ylim(70, 100)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading weekly on-time rates by eligibility...")
    _, spec, _ = build_modelling_frame()
    weekly = load_weekly_on_time(spec.subtotal_threshold_brl)
    print(f"  {len(weekly):,} weekly cells across "
          f"{weekly['purchase_week'].nunique()} weeks")

    print(f"\nFitting pre-trend regression on the pre-cutover window "
          f"(before {spec.cutover_week.date()})...")
    res = fit_pre_trend(weekly, spec.cutover_week)
    print(f"  n_weeks_pre   = {res['n_weeks']}")
    print(f"  n_orders_pre  = {res['n_orders_pre']:,}")
    print(f"  time coef     = {res['time']['coef']:+.5f}  "
          f"(SE {res['time']['se']:.5f}, p={res['time']['p']:.4f})")
    print(f"  eligible coef = {res['eligible']['coef']:+.5f}  "
          f"(SE {res['eligible']['se']:.5f}, p={res['eligible']['p']:.4f})")
    print(f"  >> time x eligible (parallel-trends test): "
          f"{res['time_x_elig']['coef']:+.5f}  "
          f"(SE {res['time_x_elig']['se']:.5f}, "
          f"p={res['time_x_elig']['p']:.4f})")

    verdict = (
        "PARALLEL-TRENDS ASSUMPTION CONSISTENT WITH DATA"
        if res["time_x_elig"]["p"] >= 0.05
        else "PARALLEL-TRENDS ASSUMPTION VIOLATED (interaction p < 0.05)"
    )
    print(f"\n  Verdict: {verdict}")

    print("\nPlotting weekly trend lines...")
    plot_path = FIGURES_DIR / "parallel_trends.png"
    plot_trends(weekly, spec.cutover_week, plot_path)
    print(f"  saved {plot_path}")

    # ---- Markdown writeup -----------------------------------------------
    p = res["time_x_elig"]["p"]
    passes = p >= 0.05
    lines: list[str] = []
    lines.append("# Parallel-trends test for the DiD identification")
    lines.append("")
    lines.append(
        "The hierarchical Bayesian DiD model in §4.1 identifies the policy "
        "effect under the assumption that the pre-cutover on-time-delivery "
        "trends were parallel between the eligible (subtotal >= R$ 150) "
        "and ineligible cohorts. This document tests that assumption two "
        "ways: visually and via a formal pre-trend regression."
    )
    lines.append("")
    lines.append("## Visual check")
    lines.append("")
    lines.append("![Weekly on-time rate by eligibility](figures/parallel_trends.png)")
    lines.append("")
    lines.append(
        "Both cohorts move broadly in parallel across the panel. The "
        "vertical dashed line marks the cutover week. Both cohorts also "
        "drop together at the right edge - that's the delivery-grace "
        "censoring (orders too recent to have a delivery outcome) and is "
        "filtered out of the modelling panel via the 6-week grace window "
        "in `src/features.py`."
    )
    lines.append("")
    lines.append("## Formal pre-trend regression")
    lines.append("")
    lines.append("Fitted on the pre-cutover window only:")
    lines.append("")
    lines.append("```")
    lines.append("logit(on_time_rate) = a + b*time + c*eligible + d*(time x eligible)")
    lines.append("```")
    lines.append("")
    lines.append("| Coefficient | Estimate | SE | p-value | Interpretation |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| `const`             | {res['const']['coef']:+.4f} "
                 f"| {res['const']['se']:.4f} | {res['const']['p']:.4f} "
                 f"| baseline logit |")
    lines.append(f"| `time`              | {res['time']['coef']:+.4f} "
                 f"| {res['time']['se']:.4f} | {res['time']['p']:.4f} "
                 f"| common weekly trend |")
    lines.append(f"| `eligible`          | {res['eligible']['coef']:+.4f} "
                 f"| {res['eligible']['se']:.4f} | {res['eligible']['p']:.4f} "
                 f"| eligible vs ineligible level |")
    lines.append(f"| **`time x eligible`** | **{res['time_x_elig']['coef']:+.4f}** "
                 f"| {res['time_x_elig']['se']:.4f} | **{res['time_x_elig']['p']:.4f}** "
                 f"| **slope difference - parallel-trends test** |")
    lines.append("")
    lines.append(f"Pre-period sample size: {res['n_orders_pre']:,} orders "
                 f"across {res['n_weeks']} weeks.")
    lines.append("")
    if passes:
        lines.append(
            f"**Result: PARALLEL TRENDS CONSISTENT WITH DATA.** The slope "
            f"difference `time x eligible` is {res['time_x_elig']['coef']:+.4f} "
            f"logit-points per week with p = {p:.3f}, which is not "
            f"statistically distinguishable from zero at the 5% level. The "
            f"DiD identification assumption is supported. Any residual "
            f"non-parallelism is small enough that the +1.5 pp policy "
            f"effect in §4.1 is unlikely to be a pre-trend artefact."
        )
    else:
        lines.append(
            f"**Result: PARALLEL-TRENDS ASSUMPTION VIOLATED.** The slope "
            f"difference `time x eligible` is {res['time_x_elig']['coef']:+.4f} "
            f"logit-points per week with p = {p:.4f}, which IS "
            f"statistically distinguishable from zero. The naive DiD "
            f"identification is contaminated by differential pre-trends. "
            f"The reported +1.5 pp policy effect should be interpreted "
            f"with caution - some of it may be the continuation of an "
            f"already-diverging pre-trend rather than a true policy "
            f"effect. A more flexible specification (e.g., adding "
            f"category-week fixed effects, or restricting the panel to "
            f"a narrower symmetric window around the cutover) would be "
            f"the next step."
        )
    lines.append("")
    out_path = REPORTS_DIR / "parallel_trends.md"
    out_path.write_text("\n".join(lines))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
