"""Fit the hurdle-LogNormal repeat-revenue model.

Usage:
    python scripts/fit_revenue.py
    python scripts/fit_revenue.py --use-nutpie
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
import pandas as pd
import pymc as pm

from src.features import build_modelling_frame
from src.models.revenue import (
    build_hurdle_lognormal,
    build_model_data,
    load_repeat_revenue,
)
from src.paths import DUCKDB_DIR, FIGURES_DIR


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    # We need the cutover week from the binomial features pipeline so the
    # treatment definition is consistent across the three models.
    df_full, spec, _ = build_modelling_frame()
    print(f"cutover week (consistent with binomial model): {spec.cutover_week.date()}")

    df = load_repeat_revenue(cutover_week=spec.cutover_week)
    print(f"customers in panel       : {len(df):,}")
    print(f"customers with repeat    : {int(df['has_repeat'].sum()):,} "
          f"({df['has_repeat'].mean()*100:.2f}%)")
    print(f"treated (T=1)            : {int(df['treatment'].sum()):,}")
    print(f"categories (>= 50 custs) : {df['first_category'].nunique()}")
    print(f"avg repeat spend if any  : R$ "
          f"{df.loc[df['has_repeat']==1, 'repeat_revenue'].mean():,.2f}")

    mdata = build_model_data(df)
    model = build_hurdle_lognormal(mdata)

    print(f"\nSampling: {args.chains} chains x {args.draws} draws "
          f"({args.tune} tune)...")
    t1 = time.perf_counter()
    with model:
        if args.use_nutpie:
            # Note: nutpie's jitter init can fail on this model because
            # initial draws of beta_bar / sigma_obs sometimes produce
            # LogNormal mu values that evaluate to -inf on the observed
            # repeat_revenue range. The fallback below catches that.
            try:
                import nutpie
                idata = nutpie.sample(
                    nutpie.compile_pymc_model(model),
                    draws=args.draws, tune=args.tune,
                    chains=args.chains, seed=0,
                )
            except RuntimeError as exc:
                print(f"  nutpie failed ({exc.__class__.__name__}); "
                      f"falling back to pm.sample with ADVI init")
                idata = pm.sample(
                    draws=args.draws, tune=args.tune,
                    chains=args.chains, cores=args.cores,
                    target_accept=0.9, random_seed=0, progressbar=True,
                    init="adapt_diag",
                )
        else:
            idata = pm.sample(
                draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.cores,
                target_accept=0.9, random_seed=0, progressbar=True,
                init="adapt_diag",
            )
    print(f"  sampled in {time.perf_counter()-t1:.1f}s")

    # ---- Diagnostics ------------------------------------------------------
    print("\n--- Hyperparameter posterior summary ---")
    s = az.summary(idata,
                   var_names=["alpha_bar","tau_bar","sigma_alpha","sigma_tau",
                              "beta_bar","delta_bar","sigma_beta","sigma_delta",
                              "sigma_obs","gamma_log_first_sub"],
                   hdi_prob=0.94)
    print(s[["mean","sd","hdi_3%","hdi_97%","ess_bulk","r_hat"]].round(3).to_string())

    # Headline result: P(repeat) and conditional spend, treatment vs control
    post = idata.posterior
    tau_bar    = post["tau_bar"].values.flatten()
    delta_bar  = post["delta_bar"].values.flatten()
    print(f"\nGlobal stage-1 (P(repeat)) treatment effect (logit):")
    print(f"  mean = {tau_bar.mean():+.3f}, "
          f"94% HDI = ({np.percentile(tau_bar, 3):+.3f}, "
          f"{np.percentile(tau_bar, 97):+.3f})")
    print(f"  P(tau_bar > 0) = {(tau_bar > 0).mean()*100:.1f}%")
    print(f"\nGlobal stage-2 (log spend) treatment effect:")
    print(f"  mean = {delta_bar.mean():+.3f} (multiplicative R$ "
          f"factor {np.exp(delta_bar.mean()):.3f})")
    print(f"  94% HDI = ({np.percentile(delta_bar, 3):+.3f}, "
          f"{np.percentile(delta_bar, 97):+.3f})")
    print(f"  P(delta_bar > 0) = {(delta_bar > 0).mean()*100:.1f}%")

    # ---- Save -------------------------------------------------------------
    nc_path = DUCKDB_DIR / "revenue_idata.nc"
    idata.to_netcdf(nc_path)
    print(f"\nTrace written to {nc_path}")

    az.plot_forest(idata, var_names=["tau_C"], combined=True, hdi_prob=0.94)
    plt.title("Per-category effect on P(repeat purchase)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "revenue_tau_C_forest.png", dpi=150)
    plt.close()

    az.plot_forest(idata, var_names=["delta_C"], combined=True, hdi_prob=0.94)
    plt.title("Per-category effect on log(repeat revenue) | repeat happened")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "revenue_delta_C_forest.png", dpi=150)
    plt.close()

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draws",  type=int, default=1500)
    p.add_argument("--tune",   type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores",  type=int, default=4)
    p.add_argument("--use-nutpie", action="store_true")
    run(p.parse_args())
