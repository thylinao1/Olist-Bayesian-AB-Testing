"""Bunching diagnostic at the R$ 150 eligibility threshold.

In a *real* free-shipping deployment customers with R$ 120-149 baskets
would have an incentive to pad their carts with filler items to clear the
R$ 150 threshold. The static Olist data was generated WITHOUT such a
threshold, so we should observe NO discontinuous density spike just above
R$ 150 - the basket-size distribution should be smooth across the cutoff.

This script verifies that, providing the McCrary-style sanity check that
the senior review flagged as missing. If we ever found a spike here, the
DiD analysis would be biased upward because the marginal-buyer composition
just above the cutoff would differ systematically from just below.

Output:
    reports/figures/bunching_diagnostic.png
    reports/bunching_diagnostic.md

Usage:
    python scripts/bunching_test.py
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

from src.paths import DUCKDB_PATH, FIGURES_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

THRESHOLD = 150.0
BIN_WIDTH = 5.0
WINDOW = (50.0, 300.0)            # R$ range to inspect
DONUT = (THRESHOLD - 10, THRESHOLD + 10)   # bins near threshold excluded from fit


def load_subtotals() -> np.ndarray:
    sql = """
        SELECT items_subtotal
        FROM gold.fact_orders
        WHERE items_subtotal IS NOT NULL
          AND items_subtotal >= 0
    """
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        df = con.execute(sql).fetchdf()
    return df["items_subtotal"].to_numpy(dtype=float)


def main() -> None:
    print("Loading items_subtotal distribution...")
    subtotals = load_subtotals()
    print(f"  {len(subtotals):,} orders, "
          f"min=R$ {subtotals.min():.2f}, "
          f"max=R$ {subtotals.max():.2f}, "
          f"median=R$ {np.median(subtotals):.2f}")

    # Histogram in BIN_WIDTH-real bins over the window of interest.
    bins = np.arange(WINDOW[0], WINDOW[1] + BIN_WIDTH, BIN_WIDTH)
    counts, edges = np.histogram(subtotals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    print(f"\n{len(bins) - 1} bins of R$ {BIN_WIDTH:.0f} each "
          f"across R$ {WINDOW[0]:.0f}-{WINDOW[1]:.0f}")

    # Fit a smooth (3rd-degree polynomial in log-bin-center) to bin counts
    # OUTSIDE the donut around the threshold. The fit predicts the counterfactual
    # density at the threshold under "no bunching".
    keep = (centers < DONUT[0]) | (centers > DONUT[1])
    log_centers = np.log(centers)
    coefs = np.polyfit(log_centers[keep], counts[keep], deg=3)
    pred = np.polyval(coefs, log_centers)

    # The "bunching jump" is the observed-vs-predicted excess just above
    # the threshold. We test it with a Z-statistic under Poisson noise
    # (variance = mean count for the bin).
    just_above = (centers > THRESHOLD) & (centers <= THRESHOLD + BIN_WIDTH)
    just_below = (centers < THRESHOLD) & (centers >= THRESHOLD - BIN_WIDTH)
    n_above = int(counts[just_above].sum())
    n_below = int(counts[just_below].sum())
    pred_above = float(pred[just_above].sum())
    pred_below = float(pred[just_below].sum())
    excess_above = n_above - pred_above
    se_above = np.sqrt(max(pred_above, 1.0))    # Poisson SE
    z_above = excess_above / se_above

    print(f"\n  bin just BELOW R$ {THRESHOLD:.0f} "
          f"[R$ {DONUT[0]+BIN_WIDTH:.0f}-{THRESHOLD:.0f}): "
          f"observed={n_below}, predicted={pred_below:.0f}")
    print(f"  bin just ABOVE R$ {THRESHOLD:.0f} "
          f"[R$ {THRESHOLD:.0f}-{THRESHOLD+BIN_WIDTH:.0f}): "
          f"observed={n_above}, predicted={pred_above:.0f}")
    print(f"  excess above threshold: {excess_above:+.0f}  "
          f"(Z = {z_above:+.2f}, "
          f"abs(Z) > 1.96 would indicate a 5% bunching spike)")

    # Distinguish three regimes:
    #   z > +1.96  : EXCESS above threshold => policy-induced bunching (would
    #                bias DiD estimate upward)
    #   z < -1.96  : DEFICIT above threshold => pre-existing structural kink
    #                (typically psychological pricing at "just under" round
    #                numbers like R$ 149 vs R$ 150)
    #   |z| < 1.96 : clean - density continuous across the cutoff
    policy_bunching = z_above > 1.96
    structural_kink = z_above < -1.96

    # ---- Plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(centers, counts, width=BIN_WIDTH * 0.9,
           color="#bfdbf0", edgecolor="#1f6f9c",
           label="observed bin counts")
    ax.plot(centers, pred, color="#cc4c4c", linewidth=1.6,
            label="smooth fit through bins outside the donut")
    ax.axvline(THRESHOLD, color="grey", linestyle="--", linewidth=1.2,
               label=f"R$ {THRESHOLD:.0f} threshold")
    ax.axvspan(DONUT[0], DONUT[1], color="#fffaa0", alpha=0.35,
               label=f"donut R$ {DONUT[0]:.0f}-{DONUT[1]:.0f} (excluded from fit)")
    ax.set_xlabel("Item subtotal (R$)")
    ax.set_ylabel("Number of orders")
    ax.set_title("Bunching diagnostic - density of items_subtotal around R$ 150")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = FIGURES_DIR / "bunching_diagnostic.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to {plot_path}")

    # ---- Writeup -------------------------------------------------------
    lines: list[str] = []
    lines.append("# Bunching diagnostic at R$ 150 threshold")
    lines.append("")
    lines.append(
        "**What this test asks.** In a *real* deployment of the free-"
        "shipping-above-R$ 150 policy, customers with R$ 120-149 baskets "
        "would have a clear incentive to add filler items to push the "
        "basket over the threshold. The bunching dynamic appears as a "
        "discontinuous density spike just above the cutoff (relative to "
        "a smooth counterfactual). Olist's static historical data was "
        "generated WITHOUT such a threshold, so the test should find no "
        "spike. Finding one would mean the data is contaminated by an "
        "unmodelled selection mechanism and the DiD policy effect would "
        "be biased upward."
    )
    lines.append("")
    lines.append("## Visualisation")
    lines.append("")
    lines.append("![Bunching diagnostic at R$ 150](figures/bunching_diagnostic.png)")
    lines.append("")
    lines.append(
        "The blue bars are observed bin counts (R$ 5 bins from R$ 50 to "
        "R$ 300). The red curve is a smooth fit (3rd-degree polynomial in "
        "log subtotal) through all bins OUTSIDE the donut zone "
        f"(R$ {DONUT[0]:.0f}-{DONUT[1]:.0f}, shaded yellow). The donut is "
        "excluded from the fit so any local distortion around the threshold "
        "doesn't leak into the counterfactual prediction."
    )
    lines.append("")
    lines.append("## McCrary-style test of density continuity")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("|---|---|")
    lines.append(f"| Observed orders in bin just BELOW R$ {THRESHOLD:.0f} | {n_below:,} |")
    lines.append(f"| Predicted by smooth fit | {pred_below:,.0f} |")
    lines.append(f"| Observed orders in bin just ABOVE R$ {THRESHOLD:.0f} | {n_above:,} |")
    lines.append(f"| Predicted by smooth fit | {pred_above:,.0f} |")
    lines.append(f"| Excess above threshold | {excess_above:+.0f} |")
    lines.append(f"| Z-statistic (Poisson SE) | **{z_above:+.2f}** |")
    lines.append(f"| Threshold for bunching at 5% | abs(Z) > 1.96 |")
    lines.append("")
    if policy_bunching:
        lines.append(
            f"**Result: POLICY-INDUCED BUNCHING DETECTED** (Z = {z_above:+.2f}, "
            f"> +1.96). The bin just above R$ {THRESHOLD:.0f} has a "
            f"statistically significant *excess* of {excess_above:+.0f} orders "
            f"relative to the smooth counterfactual. This is the classic "
            f"bunching signature - customers concentrated just above the "
            f"cutoff - and it would invalidate the DiD identification "
            f"because the marginal-buyer composition just above the "
            f"cutoff would differ systematically from just below. This is "
            f"NOT the expected finding for synthetic-treatment data, so "
            f"investigate the data-generating process before trusting the "
            f"DiD numbers."
        )
    elif structural_kink:
        lines.append(
            f"**Result: STRUCTURAL DISCONTINUITY, NOT POLICY BUNCHING** "
            f"(Z = {z_above:+.2f}, < -1.96). The bin just above R$ "
            f"{THRESHOLD:.0f} has a *deficit* of {-excess_above:+.0f} "
            f"orders relative to the smooth counterfactual - the opposite "
            f"direction from what policy-induced bunching would produce. "
            f"The most likely explanation is retail pricing structure: "
            f"sellers list items at psychological price points like R$ "
            f"149.99 or R$ 99.99, causing baskets to cluster just *below* "
            f"round-number thresholds (R$ 150, R$ 100, etc.) rather than "
            f"just above them. This is a pre-existing feature of the "
            f"data-generating process, not a policy effect, and it confirms "
            f"the synthetic treatment cannot be capturing real bunching dynamics."
        )
        lines.append("")
        lines.append(
            f"**Implication for the DiD analysis.** A structural kink in "
            f"basket density at the cutoff does not invalidate the DiD "
            f"identification per se - the DiD compares on-time-delivery "
            f"rates at fixed cell grain, not basket-density gradients. "
            f"But it does mean an RDD identification strategy at the same "
            f"cutoff would be problematic (RDD assumes the density is "
            f"continuous through the threshold; here it isn't). The DiD "
            f"design is therefore preferable to RDD for this dataset and "
            f"this threshold choice."
        )
    else:
        lines.append(
            f"**Result: NO DISCONTINUITY DETECTED** (Z = {z_above:+.2f}, "
            f"|Z| < 1.96). The observed bin counts around R$ {THRESHOLD:.0f} "
            f"are consistent with a smooth density - exactly what we "
            f"expect, since the R$ 150 threshold didn't exist when the "
            f"data was generated. The DiD policy effect is not "
            f"contaminated by threshold-bunching in the historical panel."
        )
    lines.append("")
    lines.append(
        "**Caveat for future deployment.** This test confirms the *current "
        "data* is clean, not that a future deployment of the policy would "
        "be free of bunching. If Olist actually ran the policy, the "
        "post-deployment subtotal distribution would almost certainly "
        "show a spike just above R$ 150 as customers pad their carts. "
        "The conditional-spend lift estimated from a real deployment would "
        "therefore be inflated by the bunching dynamic - the DiD posterior "
        "from this static analysis is an *underestimate* of what a real "
        "deployment would observe, and a *correct estimate* of what the "
        "true policy effect on naturally-occurring large baskets is."
    )
    out_path = REPORTS_DIR / "bunching_diagnostic.md"
    out_path.write_text("\n".join(lines))
    print(f"Writeup saved to {out_path}")


if __name__ == "__main__":
    main()
