"""Run the classical A/B baselines on the same data the Bayesian models use.

This script writes a side-by-side comparison to reports/baselines.md so the
final report can quote the gap between flat classical analysis and the
hierarchical Bayesian story.

Usage:
    python scripts/run_baselines.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `src` importable when this script is invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.baselines import (
    chi_square_review,
    mann_whitney,
    two_proportion_z,
    welch_t_test,
)
from src.features import build_modelling_frame, panel_for_binomial
from src.models.review import load_review_panel
from src.models.revenue import load_repeat_revenue
from src.paths import REPORTS_DIR


def _section(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n{title}\n{bar}\n"


def main() -> None:
    t0 = time.perf_counter()
    out_lines: list[str] = []

    df, spec, _ = build_modelling_frame()
    print(f"cutover week: {spec.cutover_week.date()}")
    out_lines.append("# Classical baseline results")
    out_lines.append("")
    out_lines.append(f"Treatment: free-shipping eligibility "
                     f"(post-cutover & subtotal >= R$ {spec.subtotal_threshold_brl:.0f})")
    out_lines.append(f"Cutover week: **{spec.cutover_week.date()}**")
    out_lines.append("")

    # ---- 1.  ON-TIME DELIVERY (binomial / two-proportion z) -----------
    print(_section("Two-proportion z - on-time delivery"))
    out_lines.append("## Two-proportion z-test on on-time delivery\n")

    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]
    succ_t, n_t = int(treated["is_on_time"].sum()), len(treated)
    succ_c, n_c = int(control["is_on_time"].sum()), len(control)
    res = two_proportion_z(succ_t, n_t, succ_c, n_c)
    line = (f"  treated  : {succ_t:>6,}/{n_t:>6,}  ({res.rate_treated*100:5.2f}%)\n"
            f"  control  : {succ_c:>6,}/{n_c:>6,}  ({res.rate_control*100:5.2f}%)\n"
            f"  diff     : {res.diff*100:+.3f} pp,  95% CI "
            f"[{res.ci_low*100:+.3f}, {res.ci_high*100:+.3f}] pp\n"
            f"  z        : {res.z:+.3f}\n"
            f"  p-value  : {res.p_value:.4f}")
    print(line)
    out_lines.append("```\n" + line + "\n```\n")

    # ---- 2.  REVENUE (Welch t-test on customer-level repeat revenue) -----
    print(_section("Welch t-test - customer-level repeat revenue (180 days)"))
    out_lines.append("## Welch t-test on per-customer repeat revenue (180-day window)\n")

    rev = load_repeat_revenue(cutover_week=spec.cutover_week)
    welch = welch_t_test(rev[rev["treatment"] == 1]["repeat_revenue"].to_numpy(),
                         rev[rev["treatment"] == 0]["repeat_revenue"].to_numpy())
    line = (f"  mean(treated): R$ {welch.mean_treated:>8,.2f}\n"
            f"  mean(control): R$ {welch.mean_control:>8,.2f}\n"
            f"  diff         : R$ {welch.diff:+.2f}, 95% CI "
            f"[{welch.ci_low:+.2f}, {welch.ci_high:+.2f}]\n"
            f"  t            : {welch.t:+.3f},  df={welch.df:.1f}\n"
            f"  p-value      : {welch.p_value:.4f}")
    print(line)
    out_lines.append("```\n" + line + "\n```\n")

    # ---- 3.  REVENUE (Mann-Whitney as a non-parametric stress test) ------
    print(_section("Mann-Whitney U - same outcome, non-parametric"))
    out_lines.append("## Mann-Whitney U on per-customer repeat revenue\n")
    mw = mann_whitney(rev[rev["treatment"] == 1]["repeat_revenue"].to_numpy(),
                      rev[rev["treatment"] == 0]["repeat_revenue"].to_numpy())
    line = (f"  median(treated): R$ {mw.median_treated:>8,.2f}\n"
            f"  median(control): R$ {mw.median_control:>8,.2f}\n"
            f"  U statistic    : {mw.u_statistic:,.0f}\n"
            f"  p-value        : {mw.p_value:.4f}")
    print(line)
    out_lines.append("```\n" + line + "\n```\n")

    # ---- 4.  REVIEW SCORE (chi-square + Mann-Whitney) --------------------
    print(_section("Chi-square - review score distribution by treatment"))
    out_lines.append("## Chi-square test of independence: review score x treatment\n")
    rev_df = load_review_panel(cutover_week=spec.cutover_week)
    chi = chi_square_review(rev_df)
    print(chi.contingency)
    print(f"  chi^2  : {chi.chi2:.3f}, df={chi.df}")
    print(f"  p-value: {chi.p_value:.4f}")
    out_lines.append("```")
    out_lines.append(chi.contingency.to_string())
    out_lines.append(f"chi^2  : {chi.chi2:.3f}, df={chi.df}")
    out_lines.append(f"p-value: {chi.p_value:.4f}")
    out_lines.append("```\n")

    # Mann-Whitney on review score (treats the ordinal as ranks - defensible)
    mw_rev = mann_whitney(
        rev_df[rev_df["treatment"] == 1]["review_score"].to_numpy(),
        rev_df[rev_df["treatment"] == 0]["review_score"].to_numpy(),
    )
    out_lines.append(f"Mann-Whitney U on review score: U={mw_rev.u_statistic:,.0f}, "
                     f"p={mw_rev.p_value:.4f}\n")
    print(f"Mann-Whitney U on review score: U={mw_rev.u_statistic:,.0f}, "
          f"p={mw_rev.p_value:.4f}")

    # ---- Save -----------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "baselines.md"
    out_path.write_text("\n".join(out_lines))
    print(f"\nWritten to {out_path}")
    print(f"Total time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
