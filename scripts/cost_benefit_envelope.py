"""Back-of-envelope cost-benefit for the hypothetical free-shipping policy.

Combines the Bayesian posterior means from the DiD trace files with a SQL
query for average freight cost per eligible order. Produces a markdown
table summarising the trade-off:

    Estimated subsidy cost   = N_eligible_orders * avg_freight_eligible
    Estimated incremental    = N_eligible_orders * P(repeat) * conditional_spend_lift
      revenue                * estimated_margin
    Net envelope             = incremental_margin - subsidy_cost

This is a *rough envelope* not a forecast. Margin assumptions are explicit
and conservative.

Output: reports/cost_benefit_envelope.md  (and stdout).

Usage:
    python scripts/cost_benefit_envelope.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arviz as az
import duckdb
import numpy as np

from src.features import build_modelling_frame
from src.paths import DUCKDB_DIR, DUCKDB_PATH, REPORTS_DIR

warnings.filterwarnings("ignore")

# Conservative back-of-envelope margin assumption.
# Marketplace contribution margins on incremental GMV vary widely; 20% is
# representative of e-commerce platforms net of payment processing,
# customer service, and basic logistics overhead.
ASSUMED_INCREMENTAL_MARGIN = 0.20


def query_freight_by_eligibility(threshold: float, cutover_week) -> dict:
    """Pull avg freight + counts split by (eligible, post-cutover).

    The bug we're fixing: subsidy cost in a real deployment only applies
    to orders that are BOTH eligible AND placed after the policy goes
    live. Earlier versions of this script multiplied avg freight by the
    full eligible count (pre + post), inflating the subsidy ~2x.
    """
    sql = f"""
        SELECT
            (items_subtotal >= {threshold})                 AS eligible,
            (purchase_week >= DATE '{cutover_week.date()}') AS post,
            COUNT(*)                                        AS n_orders,
            AVG(items_freight)                              AS avg_freight,
            AVG(items_subtotal)                             AS avg_subtotal,
            AVG(payment_total)                              AS avg_payment
        FROM gold.fact_orders
        WHERE items_subtotal IS NOT NULL
          AND items_freight  IS NOT NULL
          AND purchase_week  IS NOT NULL
        GROUP BY 1, 2
    """
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        rows = con.execute(sql).fetchall()
    return {(bool(r[0]), bool(r[1])): {
        "n_orders":     int(r[2]),
        "avg_freight":  float(r[3]),
        "avg_subtotal": float(r[4]),
        "avg_payment":  float(r[5]),
    } for r in rows}


def main() -> None:
    print("Loading freight stats from DuckDB...")
    _, spec, _ = build_modelling_frame()
    cells = query_freight_by_eligibility(
        spec.subtotal_threshold_brl, spec.cutover_week,
    )
    # The treated cell (eligible AND post-cutover) is the only one that would
    # actually receive the policy subsidy if Olist switched it on at the
    # cutover week.
    treated = cells[(True,  True)]
    pre_elig = cells[(True,  False)]
    post_inel = cells[(False, True)]
    pre_inel = cells[(False, False)]
    print(f"  treated (post AND eligible):       n={treated['n_orders']:>6,}, "
          f"avg_freight=R$ {treated['avg_freight']:.2f}, "
          f"avg_payment=R$ {treated['avg_payment']:.2f}")
    print(f"  pre-cutover  eligible (NOT subs.): n={pre_elig['n_orders']:>6,}, "
          f"avg_freight=R$ {pre_elig['avg_freight']:.2f}")
    print(f"  post-cutover ineligible:           n={post_inel['n_orders']:>6,}")
    print(f"  pre-cutover  ineligible:           n={pre_inel['n_orders']:>6,}")
    # Backward-compatible variable names for the calc below.
    elig = treated

    print("Loading posterior means from DiD traces...")
    rev = az.from_netcdf(str(DUCKDB_DIR / "revenue_did_idata.nc"))
    delta_b_bar = float(rev.posterior["delta_b_bar"].mean())   # logit Stage 1
    delta_l_bar = float(rev.posterior["delta_l_bar"].mean())   # log   Stage 2
    alpha_bar   = float(rev.posterior["alpha_bar"].mean())     # baseline logit

    p_repeat_control = 1 / (1 + np.exp(-alpha_bar))
    p_repeat_treated = 1 / (1 + np.exp(-(alpha_bar + delta_b_bar)))
    spend_mult       = float(np.exp(delta_l_bar))
    print(f"  P(repeat | control) = {p_repeat_control:.4f}")
    print(f"  P(repeat | treated) = {p_repeat_treated:.4f}")
    print(f"  conditional spend multiplier (treated / control) = {spend_mult:.4f}")

    # ---- Envelope computation -----------------------------------------
    # N_elig is now the POST-CUTOVER eligible count (the only orders that
    # would actually receive the subsidy if the policy went live at the
    # cutover week). Earlier versions of this script accidentally used the
    # pre+post eligible total, inflating subsidy cost ~2x.
    N_elig = elig["n_orders"]
    subsidy_per_eligible = elig["avg_freight"]
    subsidy_cost_total = N_elig * subsidy_per_eligible

    # Incremental revenue:
    #   Treated returners spend conditional_spend_lift more than control returners.
    #   Repeat probability shift (delta_b_bar) is near-null in our DiD - we use
    #   the model's posterior mean either way to be honest.
    avg_repeat_spend_control = elig["avg_payment"]   # rough proxy for typical second-order spend
    incremental_per_eligible = (
        p_repeat_treated * avg_repeat_spend_control * (spend_mult - 1)
        + (p_repeat_treated - p_repeat_control) * avg_repeat_spend_control * spend_mult
    )
    incremental_gmv_total = N_elig * incremental_per_eligible
    incremental_margin_total = incremental_gmv_total * ASSUMED_INCREMENTAL_MARGIN
    net_envelope = incremental_margin_total - subsidy_cost_total

    print()
    print(f"  subsidy cost (total):       R$ {subsidy_cost_total:>14,.0f}")
    print(f"  incremental GMV (total):    R$ {incremental_gmv_total:>14,.0f}")
    print(f"  incremental margin @20%:    R$ {incremental_margin_total:>14,.0f}")
    print(f"  net envelope:               R$ {net_envelope:>14,.0f}")

    # ---- Markdown writeup ----------------------------------------------
    lines: list[str] = []
    lines.append("# Cost-benefit envelope (rough)")
    lines.append("")
    lines.append("**This is a back-of-envelope sanity check, not a forecast.** It uses "
                 "the posterior means from the DiD-corrected revenue model and the "
                 "actual freight statistics on Olist to size the policy's incremental "
                 "revenue vs subsidy cost. Margin assumptions are explicit and "
                 "conservative.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append("| Quantity | Value | Source |")
    lines.append("|---|---|---|")
    lines.append(f"| Subtotal-eligibility threshold | R$ {spec.subtotal_threshold_brl:.0f} | treatment definition |")
    lines.append(f"| N eligible orders (panel total) | {N_elig:,} | DuckDB `gold.fact_orders` |")
    lines.append(f"| Avg freight on an eligible order | R$ {subsidy_per_eligible:.2f} | DuckDB query |")
    lines.append(f"| Avg payment on an eligible order | R$ {elig['avg_payment']:.2f} | DuckDB query |")
    lines.append(f"| Baseline P(repeat within 180d) | {p_repeat_control:.4f} | DiD posterior mean (alpha_bar) |")
    lines.append(f"| Treated P(repeat within 180d) | {p_repeat_treated:.4f} | DiD posterior mean (alpha_bar + delta_b_bar) |")
    lines.append(f"| Conditional spend lift (treated / control) | x{spend_mult:.3f} | DiD posterior mean (exp delta_l_bar) |")
    lines.append(f"| Assumed contribution margin on incremental GMV | {ASSUMED_INCREMENTAL_MARGIN*100:.0f}% | conservative platform assumption |")
    lines.append("")
    lines.append("## Envelope")
    lines.append("")
    lines.append("| Line | R$ |")
    lines.append("|---|---|")
    lines.append(f"| Subsidy cost (N_eligible x avg_freight_eligible) | {subsidy_cost_total:>14,.0f} |")
    lines.append(f"| Incremental GMV from policy lift              | {incremental_gmv_total:>14,.0f} |")
    lines.append(f"| Incremental contribution margin @ {ASSUMED_INCREMENTAL_MARGIN*100:.0f}%        | {incremental_margin_total:>14,.0f} |")
    lines.append(f"| **Net envelope (margin - subsidy)**           | **{net_envelope:>14,.0f}** |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if net_envelope > 0:
        verdict = ("a *positive net envelope*: at the assumed 20% contribution "
                   "margin, the incremental GMV from lifted retention and lifted "
                   "conditional spend outweighs the freight subsidy cost.")
    else:
        verdict = ("a *negative net envelope*: at the assumed 20% contribution "
                   "margin, the freight subsidy cost is larger than the incremental "
                   "margin from lifted retention and conditional spend.")
    lines.append(f"The envelope produces {verdict}")
    lines.append("")
    lines.append("**Sensitivity.** Holding everything else fixed, the break-even contribution "
                 f"margin is approximately `subsidy_cost / incremental_GMV = "
                 f"{(subsidy_cost_total/incremental_gmv_total)*100:.1f}%`. If actual "
                 "margins on the incremental categories are above this threshold the policy "
                 "pays for itself; if below it does not. The DiD spend multiplier carries "
                 "wide uncertainty (94% HDI on the log scale crosses zero), so this envelope "
                 "should be treated as a midpoint estimate, not a prediction.")
    lines.append("")
    lines.append("**What the envelope omits.** Order-acquisition cost savings from retained "
                 "customers (not modelled), lifetime-value impact beyond the 180-day window, "
                 "review-score-driven brand effects (review drops ~0.13 stars), seller-side "
                 "price responses (would inflate observed lift), and the bunching-induced "
                 "basket-padding dynamic that the static data cannot capture. All are listed "
                 "in §7 Limitations of the main report.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "cost_benefit_envelope.md"
    out_path.write_text("\n".join(lines))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
