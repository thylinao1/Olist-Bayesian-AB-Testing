"""Rank product categories by per-category policy effect across the three DiD models.

Loads the saved ArviZ traces from `data/duckdb/*_did_idata.nc`, extracts the
per-category posterior on the policy parameter (`delta_C` for binomial and
review, `delta_b_C` and `delta_l_C` for revenue), and writes a markdown
table summarising:

    * top 5 categories by mean policy effect on on-time delivery
    * top 5 categories by mean policy effect on P(repeat)
    * top 5 categories by mean policy effect on conditional spend
    * top 5 categories by mean policy effect on review score
      (and bottom 5, since the global effect is negative)

Output: reports/category_recommendations.md  (and stdout).

Usage:
    python scripts/category_recommendations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arviz as az
import numpy as np
import pandas as pd

from src.paths import DUCKDB_DIR, REPORTS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def per_category_summary(idata: az.InferenceData, var_name: str) -> pd.DataFrame:
    """Return one row per category: mean / HDI / P(>0) for var_name[category]."""
    arr = idata.posterior[var_name]                        # (chain, draw, category)
    flat = arr.stack(sample=("chain", "draw")).values      # (category, sample)
    cats = arr.coords["category"].values

    rows = []
    for i, cat in enumerate(cats):
        s = flat[i]
        rows.append({
            "category":   str(cat),
            "mean":       float(s.mean()),
            "hdi_3":      float(np.percentile(s, 3)),
            "hdi_97":     float(np.percentile(s, 97)),
            "P_pos":      float((s > 0).mean()),
            "n_samples":  int(s.size),
        })
    return pd.DataFrame(rows)


def fmt_logit_pp(x: float, base_p: float = 0.89) -> str:
    """Convert a small logit shift to a probability-point shift at base_p."""
    logit_base = np.log(base_p / (1 - base_p))
    new_p = 1 / (1 + np.exp(-(logit_base + x)))
    return f"{(new_p - base_p) * 100:+.2f} pp"


def render_section(title: str, df: pd.DataFrame, *,
                   ascending: bool = False, n: int = 5,
                   pp_base: float | None = None) -> list[str]:
    """Return markdown lines for one ranking."""
    df = df.sort_values("mean", ascending=ascending).head(n)
    out = [f"### {title}\n"]
    if pp_base is not None:
        out.append(
            "| Rank | Category | δ̅ (logit) | 94% HDI | P(δ>0) | ≈ Δ at base rate |"
        )
        out.append("|---|---|---|---|---|---|")
        for rank, row in enumerate(df.itertuples(), 1):
            pp = fmt_logit_pp(row.mean, base_p=pp_base)
            out.append(
                f"| {rank} | {row.category} | "
                f"{row.mean:+.3f} | "
                f"({row.hdi_3:+.3f}, {row.hdi_97:+.3f}) | "
                f"{row.P_pos*100:.1f}% | {pp} |"
            )
    else:
        out.append("| Rank | Category | δ̅ | 94% HDI | P(δ>0) |")
        out.append("|---|---|---|---|---|")
        for rank, row in enumerate(df.itertuples(), 1):
            out.append(
                f"| {rank} | {row.category} | "
                f"{row.mean:+.3f} | "
                f"({row.hdi_3:+.3f}, {row.hdi_97:+.3f}) | "
                f"{row.P_pos*100:.1f}% |"
            )
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    traces = {
        "binomial_did": DUCKDB_DIR / "binomial_did_idata.nc",
        "revenue_did":  DUCKDB_DIR / "revenue_did_idata.nc",
        "review_did":   DUCKDB_DIR / "review_did_idata.nc",
    }
    missing = [name for name, p in traces.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trace files (run the matching scripts/fit_*_did.py first):\n  - "
            + "\n  - ".join(f"{n}: {traces[n]}" for n in missing)
        )

    print("Loading DiD traces...")
    idatas = {name: az.from_netcdf(str(p)) for name, p in traces.items()}

    # ---- Extract per-category posteriors --------------------------------
    on_time = per_category_summary(idatas["binomial_did"], "delta_C")
    repeat  = per_category_summary(idatas["revenue_did"],  "delta_b_C")
    spend   = per_category_summary(idatas["revenue_did"],  "delta_l_C")
    review  = per_category_summary(idatas["review_did"],   "delta_C")

    print(f"  on-time     categories : {len(on_time)}")
    print(f"  repeat      categories : {len(repeat)}")
    print(f"  spend       categories : {len(spend)}")
    print(f"  review      categories : {len(review)}")

    # ---- Build markdown report -----------------------------------------
    lines: list[str] = []
    lines.append("# Where the policy is most worth running")
    lines.append("")
    lines.append(
        "Per-category posterior summaries from the three DiD-corrected "
        "Bayesian models. Hierarchical partial pooling over the 73 product "
        "categories means each estimate borrows strength from the rest, so "
        "the rankings are robust to small-cell noise."
    )
    lines.append("")
    lines.append(
        "**How to read each table.** δ̅ is the policy effect on the "
        "model's link scale (logit for binomial / Bernoulli, log for "
        "spend, cumulative-logit for review). The HDI is the 94% credible "
        "interval. `P(δ>0)` is the posterior probability the policy "
        "*helps* the outcome in that category."
    )
    lines.append("")

    # On-time delivery - base rate ~89%
    lines.extend(render_section(
        "On-time delivery - top 5 categories where the policy lifts most",
        on_time, ascending=False, n=5, pp_base=0.89,
    ))

    # Stage 1 P(repeat) - base rate ~2.3%
    lines.extend(render_section(
        "P(repeat purchase) - top 5 categories where the policy lifts retention most",
        repeat, ascending=False, n=5, pp_base=0.023,
    ))

    # Stage 2 conditional spend - log scale, no base-rate conversion
    lines.append("### Conditional spend if customer returns - top 5 categories\n")
    sub = spend.sort_values("mean", ascending=False).head(5)
    lines.append("| Rank | Category | δ̅ (log) | × multiplier | 94% HDI | P(δ>0) |")
    lines.append("|---|---|---|---|---|---|")
    for rank, row in enumerate(sub.itertuples(), 1):
        lines.append(
            f"| {rank} | {row.category} | {row.mean:+.3f} | "
            f"× {np.exp(row.mean):.3f} | "
            f"({row.hdi_3:+.3f}, {row.hdi_97:+.3f}) | "
            f"{row.P_pos*100:.1f}% |"
        )
    lines.append("")

    # Review score - global effect is negative, so show TWO views
    lines.extend(render_section(
        "Review score - top 5 categories where the policy hurts LEAST",
        review, ascending=False, n=5,
    ))
    lines.extend(render_section(
        "Review score - bottom 5 categories where the policy hurts MOST",
        review, ascending=True, n=5,
    ))

    # ---- Recommendation summary -----------------------------------------
    lines.append("## Recommendation summary")
    lines.append("")
    lines.append("If the platform were to roll the free-shipping policy out "
                 "to a *subset* of categories rather than universally, the "
                 "ones where the on-time lift, the conditional-spend lift, "
                 "and the review-score impact are jointly most favourable "
                 "are the ones to target. A simple aggregate score is the "
                 "sum of standardised per-category mean policy effects "
                 "across the four outcomes (negating the review effect "
                 "since lower is worse).")
    lines.append("")

    # Aggregate score: standardise each, sum signed
    def z(s):
        return (s - s.mean()) / s.std()

    agg = (
        on_time[["category", "mean"]].rename(columns={"mean": "on_time"})
        .merge(repeat[["category", "mean"]].rename(columns={"mean": "repeat"}),  on="category", how="outer")
        .merge(spend [["category", "mean"]].rename(columns={"mean": "spend"}),   on="category", how="outer")
        .merge(review[["category", "mean"]].rename(columns={"mean": "review"}),  on="category", how="outer")
    ).dropna()
    agg["score"] = (
        z(agg["on_time"]) + z(agg["repeat"]) + z(agg["spend"]) - z(agg["review"])
    )
    agg = agg.sort_values("score", ascending=False).head(8)
    lines.append("| Rank | Category | Aggregate z-score | δ on-time | δ repeat | δ spend | δ review |")
    lines.append("|---|---|---|---|---|---|---|")
    for rank, row in enumerate(agg.itertuples(), 1):
        lines.append(
            f"| {rank} | {row.category} | {row.score:+.2f} | "
            f"{row.on_time:+.3f} | {row.repeat:+.3f} | "
            f"{row.spend:+.3f} | {row.review:+.3f} |"
        )
    lines.append("")
    lines.append(
        "Categories at the top of this list are where every channel of the "
        "policy is most favourable simultaneously. They are the natural "
        "first wave for a phased rollout."
    )

    # ---- Write -----------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "category_recommendations.md"
    out_path.write_text("\n".join(lines))
    print(f"\nWritten: {out_path}\n")

    # Echo a short preview to stdout
    for line in lines[:60]:
        print(line)


if __name__ == "__main__":
    main()
