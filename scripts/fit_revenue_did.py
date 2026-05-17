"""Fit the DiD-corrected hurdle-LogNormal repeat-revenue model.

This decomposes both stages of the hurdle into:
    eligible (basket-size structural effect),
    post (common time trend),
    eligible * post (the policy effect itself).

Usage:
    python scripts/fit_revenue_did.py
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
import pymc as pm

from src.features import build_modelling_frame
from src.models.revenue_did import (
    build_did_hurdle_lognormal,
    build_model_data_did,
    load_repeat_revenue,
)
from src.paths import DUCKDB_DIR, FIGURES_DIR


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    _, spec, _ = build_modelling_frame()
    print(f"cutover week: {spec.cutover_week.date()}")

    df = load_repeat_revenue(cutover_week=spec.cutover_week)
    mdata = build_model_data_did(
        df,
        cutover_week=spec.cutover_week,
        subtotal_threshold_brl=spec.subtotal_threshold_brl,
    )

    print(f"customers in panel : {len(mdata.has_repeat):,}")
    print(f"with repeat        : {int(mdata.has_repeat.sum()):,} "
          f"({mdata.has_repeat.mean()*100:.2f}%)")
    print(f"eligible (>= R$ {spec.subtotal_threshold_brl:.0f}): "
          f"{int(mdata.eligible.sum()):,}")
    print(f"post (>= cutover)  : {int(mdata.post.sum()):,}")
    print(f"both (E AND P)     : {int((mdata.eligible & mdata.post).sum()):,}")

    model = build_did_hurdle_lognormal(mdata)

    print(f"\nSampling: {args.chains} chains x {args.draws} draws "
          f"({args.tune} tune)...")
    t1 = time.perf_counter()
    with model:
        if args.use_nutpie:
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
                    target_accept=0.9, random_seed=0,
                    init="adapt_diag", progressbar=True,
                )
        else:
            idata = pm.sample(
                draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.cores,
                target_accept=0.9, random_seed=0,
                init="adapt_diag", progressbar=True,
            )
    print(f"  sampled in {time.perf_counter()-t1:.1f}s")

    print("\n--- Hyperparameter posterior summary ---")
    s = az.summary(idata, var_names=[
        "alpha_bar","delta_b_bar","beta_e_b","beta_p_b",
        "sigma_alpha","sigma_delta_b",
        "beta_bar","delta_l_bar","beta_e_l","beta_p_l",
        "gamma_log_first_sub",
        "sigma_beta","sigma_delta_l","sigma_obs",
    ], hdi_prob=0.94)
    print(s[["mean","sd","hdi_3%","hdi_97%","ess_bulk","r_hat"]]
          .round(3).to_string())

    post = idata.posterior
    delta_b = post["delta_b_bar"].values.flatten()
    delta_l = post["delta_l_bar"].values.flatten()
    beta_e_b = post["beta_e_b"].values.flatten()
    beta_p_b = post["beta_p_b"].values.flatten()
    beta_e_l = post["beta_e_l"].values.flatten()
    beta_p_l = post["beta_p_l"].values.flatten()

    print("\n--- DECOMPOSITION (Stage 1: P(repeat)) ---")
    print(f"  basket-size (beta_e_b): mean={beta_e_b.mean():+.3f}, "
          f"94% HDI=({np.percentile(beta_e_b,3):+.3f}, "
          f"{np.percentile(beta_e_b,97):+.3f})")
    print(f"  time trend  (beta_p_b): mean={beta_p_b.mean():+.3f}, "
          f"94% HDI=({np.percentile(beta_p_b,3):+.3f}, "
          f"{np.percentile(beta_p_b,97):+.3f})")
    print(f"  >>> POLICY (delta_b_bar): mean={delta_b.mean():+.3f}, "
          f"94% HDI=({np.percentile(delta_b,3):+.3f}, "
          f"{np.percentile(delta_b,97):+.3f})")
    print(f"  P(delta_b_bar > 0) = {(delta_b > 0).mean()*100:.1f}%")

    print("\n--- DECOMPOSITION (Stage 2: log conditional spend) ---")
    print(f"  basket-size (beta_e_l): mean={beta_e_l.mean():+.3f}, "
          f"94% HDI=({np.percentile(beta_e_l,3):+.3f}, "
          f"{np.percentile(beta_e_l,97):+.3f})")
    print(f"  time trend  (beta_p_l): mean={beta_p_l.mean():+.3f}, "
          f"94% HDI=({np.percentile(beta_p_l,3):+.3f}, "
          f"{np.percentile(beta_p_l,97):+.3f})")
    print(f"  >>> POLICY (delta_l_bar): mean={delta_l.mean():+.3f} "
          f"(multiplier x{np.exp(delta_l.mean()):.3f})")
    print(f"  94% HDI=({np.percentile(delta_l,3):+.3f}, "
          f"{np.percentile(delta_l,97):+.3f})")
    print(f"  P(delta_l_bar > 0) = {(delta_l > 0).mean()*100:.1f}%")

    nc_path = DUCKDB_DIR / "revenue_did_idata.nc"
    idata.to_netcdf(nc_path)
    print(f"\nTrace written to {nc_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    az.plot_forest(idata, var_names=["delta_b_C"], combined=True, hdi_prob=0.94)
    plt.title("Per-category POLICY effect on P(repeat) - Stage 1 (DiD)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "revenue_did_delta_b_C_forest.png", dpi=150)
    plt.close()

    az.plot_forest(idata, var_names=["delta_l_C"], combined=True, hdi_prob=0.94)
    plt.title("Per-category POLICY effect on log(spend|repeat) - Stage 2 (DiD)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "revenue_did_delta_l_C_forest.png", dpi=150)
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
