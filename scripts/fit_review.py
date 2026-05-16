"""Fit the hierarchical ordered-logit review-score model.

Usage:
    python scripts/fit_review.py
    python scripts/fit_review.py --use-nutpie
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `src` importable when this script is invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from src.features import build_modelling_frame
from src.models.review import (
    build_model_data,
    build_ordered_logit,
    load_review_panel,
)
from src.paths import DUCKDB_DIR, FIGURES_DIR


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    # Reuse cutover week from the binomial pipeline so all three models
    # share an identical treatment definition.
    _, spec, _ = build_modelling_frame()
    print(f"cutover week: {spec.cutover_week.date()}")

    df = load_review_panel(cutover_week=spec.cutover_week)
    print(f"orders with reviews     : {len(df):,}")
    print(f"treated (T=1)           : {int(df['treatment'].sum()):,} "
          f"({df['treatment'].mean()*100:.2f}%)")
    print(f"categories (>=50 orders): {df['category_en'].nunique()}")

    if args.max_rows and len(df) > args.max_rows:
        df = (
            df.groupby("treatment", group_keys=False)
              .apply(lambda g: g.sample(
                  n=int(args.max_rows * len(g) / len(df)),
                  random_state=0,
              ))
              .reset_index(drop=True)
        )
        print(f"  --> stratified subsample to {len(df):,} rows for memory safety")

    print(f"\nReview score distribution (overall):")
    print(df["review_score"].value_counts(normalize=True).sort_index()
          .round(3).to_string())

    print("\nReview score distribution by treatment (mean only):")
    print(df.groupby("treatment")["review_score"].mean().round(3).to_string())

    mdata = build_model_data(df)
    model = build_ordered_logit(mdata)

    print(f"\nSampling: {args.chains} chains x {args.draws} draws "
          f"({args.tune} tune)...")
    t1 = time.perf_counter()
    with model:
        if args.use_nutpie:
            import nutpie
            idata = nutpie.sample(
                nutpie.compile_pymc_model(model),
                draws=args.draws, tune=args.tune,
                chains=args.chains, seed=0,
            )
        else:
            idata = pm.sample(
                draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.cores,
                target_accept=0.9, random_seed=0, progressbar=True,
            )
    print(f"  sampled in {time.perf_counter()-t1:.1f}s")

    # ---- Diagnostics ------------------------------------------------------
    print("\n--- Hyperparameter posterior summary ---")
    s = az.summary(idata,
                   var_names=["kappa","beta_bar","tau_bar",
                              "sigma_beta","sigma_tau",
                              "sigma_gamma","sigma_delta"],
                   hdi_prob=0.94)
    print(s[["mean","sd","hdi_3%","hdi_97%","ess_bulk","r_hat"]].round(3).to_string())

    post = idata.posterior
    tau_bar = post["tau_bar"].values.flatten()
    print(f"\nGlobal treatment effect on review score (logit-cumulative):")
    print(f"  mean = {tau_bar.mean():+.3f}, "
          f"94% HDI = ({np.percentile(tau_bar,3):+.3f}, "
          f"{np.percentile(tau_bar,97):+.3f})")
    print(f"  P(tau_bar > 0) = {(tau_bar > 0).mean()*100:.1f}%")

    # ---- Save -------------------------------------------------------------
    nc_path = DUCKDB_DIR / "review_idata.nc"
    idata.to_netcdf(nc_path)
    print(f"\nTrace written to {nc_path}")

    az.plot_forest(idata, var_names=["tau_C"], combined=True, hdi_prob=0.94)
    plt.title("Per-category treatment effect on review score (logit-cumulative)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "review_tau_C_forest.png", dpi=150)
    plt.close()

    az.plot_posterior(idata, var_names=["kappa"], hdi_prob=0.94)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "review_cutpoints.png", dpi=150)
    plt.close()

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Memory-safer defaults: 2 chains x 1000 draws, run sequentially.
    # The 94k-row OrderedLogistic gradient is the OOM risk; halving chains
    # in parallel halves peak RAM. Use --chains 4 --cores 4 on a beefier box.
    p.add_argument("--draws",  type=int, default=1000)
    p.add_argument("--tune",   type=int, default=800)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--cores",  type=int, default=1)
    p.add_argument("--max-rows", type=int, default=30000,
                   help="Stratified subsample to this many rows (default 30k). "
                        "Pass 0 to disable.")
    p.add_argument("--use-nutpie", action="store_true")
    run(p.parse_args())
