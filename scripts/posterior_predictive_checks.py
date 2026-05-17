"""Posterior predictive checks for the three Bayesian models.

For each fitted model, reconstructs the model, draws posterior-predictive
samples from the saved trace, and produces a PPC plot. The PPC compares the
distribution of model-simulated outcomes to the observed outcome. If they
overlap well, the model captures the data-generating process; systematic
gaps suggest mis-specification.

Saved figures (one per model):
    reports/figures/ppc_binomial_did.png
    reports/figures/ppc_revenue_did_stage1.png
    reports/figures/ppc_revenue_did_stage2.png
    reports/figures/ppc_review_did.png

Usage:
    python scripts/posterior_predictive_checks.py
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
import numpy as np
import pymc as pm

from src.features import build_modelling_frame, panel_for_did
from src.models.binomial_did import build_did_binomial, from_panel as binom_from_panel
from src.models.revenue_did import (
    build_did_hurdle_lognormal,
    build_model_data_did as revenue_build_data,
    load_repeat_revenue,
)
from src.models.review_did import (
    build_did_ordered_logit,
    build_model_data_did as review_build_data,
)
from src.models.review import load_review_panel
from src.paths import DUCKDB_DIR, FIGURES_DIR

warnings.filterwarnings("ignore")


def _save_ppc(idata: az.InferenceData, var_name: str,
              title: str, out_path: Path,
              kind: str = "kde",
              num_pp_samples: int = 100) -> None:
    """Wrap az.plot_ppc with reasonable defaults."""
    fig, ax = plt.subplots(figsize=(9, 5))
    az.plot_ppc(
        idata,
        var_names=[var_name],
        kind=kind,
        num_pp_samples=num_pp_samples,
        ax=ax,
        random_seed=0,
    )
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ppc_binomial() -> None:
    print("\n=== Binomial DiD PPC ===")
    idata = az.from_netcdf(str(DUCKDB_DIR / "binomial_did_idata.nc"))
    df, spec, encoders = build_modelling_frame()
    panel = panel_for_did(df, spec)
    labels = sorted(encoders["category_en"].keys(),
                    key=lambda k: encoders["category_en"][k])
    mdata = binom_from_panel(panel, category_labels=labels, outcome="n_on_time")
    model = build_did_binomial(mdata)

    print("  drawing posterior predictive samples...")
    with model:
        pp = pm.sample_posterior_predictive(
            idata, var_names=["n_on_time"], random_seed=0, progressbar=False,
        )
    idata.extend(pp)

    obs_mean = mdata.n_successes.sum() / mdata.n_trials.sum()
    pp_arr = pp.posterior_predictive["n_on_time"].values
    pp_rate = pp_arr.sum(axis=-1) / mdata.n_trials.sum()
    print(f"  observed on-time rate: {obs_mean:.4f}")
    print(f"  PP rate: mean={pp_rate.mean():.4f}, "
          f"5/95={np.percentile(pp_rate, 5):.4f}/{np.percentile(pp_rate, 95):.4f}")
    print(f"  --> {'GOOD' if abs(pp_rate.mean() - obs_mean) < 0.005 else 'CHECK'} "
          f"(PP covers observation within {abs(pp_rate.mean()-obs_mean)*100:.2f} pp)")

    _save_ppc(idata, "n_on_time",
              "Posterior predictive check - Binomial DiD on-time delivery",
              FIGURES_DIR / "ppc_binomial_did.png")


def ppc_revenue() -> None:
    print("\n=== Revenue DiD PPC (both stages) ===")
    idata = az.from_netcdf(str(DUCKDB_DIR / "revenue_did_idata.nc"))
    _, spec, _ = build_modelling_frame()
    df = load_repeat_revenue(cutover_week=spec.cutover_week)
    mdata = revenue_build_data(
        df, cutover_week=spec.cutover_week,
        subtotal_threshold_brl=spec.subtotal_threshold_brl,
    )
    model = build_did_hurdle_lognormal(mdata)

    print("  drawing posterior predictive samples...")
    with model:
        pp = pm.sample_posterior_predictive(
            idata, var_names=["y_repeat", "y_revenue"],
            random_seed=0, progressbar=False,
        )
    idata.extend(pp)

    obs_repeat = mdata.has_repeat.mean()
    pp_repeat = pp.posterior_predictive["y_repeat"].values.mean(axis=-1)
    print(f"  Stage 1 observed P(repeat): {obs_repeat:.4f}")
    print(f"  Stage 1 PP P(repeat): mean={pp_repeat.mean():.4f}, "
          f"5/95=[{np.percentile(pp_repeat, 5):.4f}, "
          f"{np.percentile(pp_repeat, 95):.4f}]")

    pos_rev = mdata.repeat_revenue[mdata.has_repeat == 1]
    obs_log_med = float(np.log(pos_rev[pos_rev > 0]).mean())
    pp_rev = pp.posterior_predictive["y_revenue"].values
    pp_log_med = float(np.log(np.clip(pp_rev, 1e-6, None)).mean())
    print(f"  Stage 2 observed mean(log spend|repeat): {obs_log_med:.3f}")
    print(f"  Stage 2 PP        mean(log spend|repeat): {pp_log_med:.3f}")

    _save_ppc(idata, "y_repeat",
              "Posterior predictive check - Revenue Stage 1 (Bernoulli: repeat)",
              FIGURES_DIR / "ppc_revenue_did_stage1.png")
    _save_ppc(idata, "y_revenue",
              "Posterior predictive check - Revenue Stage 2 (LogNormal: conditional spend)",
              FIGURES_DIR / "ppc_revenue_did_stage2.png")


def ppc_review() -> None:
    print("\n=== Review DiD PPC ===")
    idata = az.from_netcdf(str(DUCKDB_DIR / "review_did_idata.nc"))
    _, spec, _ = build_modelling_frame()
    df = load_review_panel(cutover_week=spec.cutover_week)
    # Match the subsample the fit script used (30k rows, stratified)
    max_rows = 30000
    if len(df) > max_rows:
        df = (
            df.groupby("category_en", group_keys=False)
              .apply(lambda g: g.sample(
                  n=max(1, int(max_rows * len(g) / len(df))),
                  random_state=0,
              ))
              .reset_index(drop=True)
        )
    mdata = review_build_data(
        df, cutover_week=spec.cutover_week,
        subtotal_threshold_brl=spec.subtotal_threshold_brl,
    )
    model = build_did_ordered_logit(mdata)

    print("  drawing posterior predictive samples...")
    with model:
        pp = pm.sample_posterior_predictive(
            idata, var_names=["y_review"], random_seed=0, progressbar=False,
        )
    idata.extend(pp)

    # Compare the empirical distribution to the PP distribution per score
    obs_counts = np.bincount(mdata.review_score, minlength=mdata.K) / len(mdata.review_score)
    pp_arr = pp.posterior_predictive["y_review"].values
    pp_props = np.stack(
        [np.bincount(pp_arr[c, d], minlength=mdata.K) / pp_arr.shape[-1]
         for c in range(pp_arr.shape[0]) for d in range(pp_arr.shape[1])],
        axis=0,
    )
    print("  observed vs PP mean (proportion per score 1..5):")
    for k in range(mdata.K):
        print(f"    score={k+1}: observed={obs_counts[k]:.3f}, "
              f"PP mean={pp_props[:, k].mean():.3f}")

    _save_ppc(idata, "y_review",
              "Posterior predictive check - Review DiD (ordered logit)",
              FIGURES_DIR / "ppc_review_did.png",
              kind="kde", num_pp_samples=50)


def main() -> None:
    print("Posterior predictive checks for the three DiD-corrected Bayesian models.")
    ppc_binomial()
    ppc_revenue()
    ppc_review()
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
