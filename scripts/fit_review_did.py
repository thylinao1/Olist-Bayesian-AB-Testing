"""Fit the DiD-corrected ordered-logit review-score model.

Decomposes the combined treatment into:
    beta_eligible - basket-size structural effect on review scores
    beta_post     - common time trend
    delta_C[c]    - POLICY EFFECT, varying by category

Usage:
    python scripts/fit_review_did.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from src.features import build_modelling_frame
from src.models.review_did import (
    build_did_ordered_logit,
    build_model_data_did,
)
from src.models.review import load_review_panel
from src.paths import DUCKDB_DIR, FIGURES_DIR


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    _, spec, _ = build_modelling_frame()
    print(f"cutover week: {spec.cutover_week.date()}")

    df = load_review_panel(cutover_week=spec.cutover_week)
    print(f"orders with reviews     : {len(df):,}")

    if args.max_rows and len(df) > args.max_rows:
        df = (
            df.groupby("category_en", group_keys=False)
              .apply(lambda g: g.sample(
                  n=max(1, int(args.max_rows * len(g) / len(df))),
                  random_state=0,
              ))
              .reset_index(drop=True)
        )
        print(f"  --> stratified subsample to {len(df):,} rows for memory safety")

    mdata = build_model_data_did(
        df,
        cutover_week=spec.cutover_week,
        subtotal_threshold_brl=spec.subtotal_threshold_brl,
    )
    print(f"\nDiD 2x2 cell sizes:")
    cell = pd.DataFrame({
        "eligible": mdata.eligible,
        "post":     mdata.post,
        "score":    mdata.review_score + 1,   # back to 1..5
    })
    print(cell.groupby(["eligible","post"]).agg(
        n=("score","size"),
        avg_score=("score","mean"),
    ).round(3).to_string())

    model = build_did_ordered_logit(mdata)

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
                target_accept=0.95, random_seed=0,
                init="adapt_diag", progressbar=True,
            )
    print(f"  sampled in {time.perf_counter()-t1:.1f}s")

    print("\n--- Hyperparameter posterior summary ---")
    s = az.summary(idata, var_names=[
        "kappa","beta_bar","delta_bar","beta_eligible","beta_post",
        "sigma_beta","sigma_delta","sigma_gamma_s",
    ], hdi_prob=0.94)
    print(s[["mean","sd","hdi_3%","hdi_97%","ess_bulk","r_hat"]]
          .round(3).to_string())

    post = idata.posterior
    delta_bar = post["delta_bar"].values.flatten()
    beta_e    = post["beta_eligible"].values.flatten()
    beta_p    = post["beta_post"].values.flatten()

    print("\n--- DECOMPOSITION (cumulative-logit scale) ---")
    print(f"  basket-size (beta_eligible): mean={beta_e.mean():+.3f}, "
          f"94% HDI=({np.percentile(beta_e,3):+.3f}, "
          f"{np.percentile(beta_e,97):+.3f})")
    print(f"  time trend  (beta_post):     mean={beta_p.mean():+.3f}, "
          f"94% HDI=({np.percentile(beta_p,3):+.3f}, "
          f"{np.percentile(beta_p,97):+.3f})")
    print(f"  >>> POLICY (delta_bar): mean={delta_bar.mean():+.3f}, "
          f"94% HDI=({np.percentile(delta_bar,3):+.3f}, "
          f"{np.percentile(delta_bar,97):+.3f})")
    print(f"  P(delta_bar > 0) = {(delta_bar > 0).mean()*100:.1f}%")

    nc_path = DUCKDB_DIR / "review_did_idata.nc"
    idata.to_netcdf(nc_path)
    print(f"\nTrace written to {nc_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    az.plot_forest(idata, var_names=["delta_C"], combined=True, hdi_prob=0.94)
    plt.title("Per-category POLICY effect on review score (DiD, cum-logit)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "review_did_delta_C_forest.png", dpi=150)
    plt.close()

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draws",  type=int, default=1000)
    p.add_argument("--tune",   type=int, default=800)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--cores",  type=int, default=1)
    p.add_argument("--max-rows", type=int, default=30000,
                   help="Stratified subsample (default 30k). Pass 0 to disable.")
    p.add_argument("--use-nutpie", action="store_true")
    run(p.parse_args())
