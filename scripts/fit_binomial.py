"""Fit the hierarchical Binomial on-time-delivery model on the full Olist panel.

Usage:
    python scripts/fit_binomial.py                          # default: 4x1500 NUTS
    python scripts/fit_binomial.py --draws 2000 --chains 4  # heavier run
    python scripts/fit_binomial.py --use-nutpie             # 3-5x faster sampler

Output:
    reports/figures/binomial_*.png        — posterior plots
    data/duckdb/binomial_idata.nc         — full ArviZ InferenceData (NetCDF)
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

from src.features import build_modelling_frame, panel_for_binomial
from src.models.binomial import build_hierarchical_binomial, from_panel
from src.paths import DUCKDB_DIR, FIGURES_DIR


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    print("Loading data...")
    df, spec, encoders = build_modelling_frame()
    panel = panel_for_binomial(df)
    labels = sorted(encoders["category_en"].keys(),
                    key=lambda k: encoders["category_en"][k])
    mdata = from_panel(panel, outcome=args.outcome, category_labels=labels)
    print(f"  outcome     : {mdata.outcome_name}")
    print(f"  panel cells : {len(mdata.n_trials):,}")
    print(f"  successes   : {mdata.n_successes.sum():,} of {mdata.n_trials.sum():,} "
          f"({mdata.n_successes.sum()/mdata.n_trials.sum()*100:.2f}%)")
    print(f"  cutover wk  : {spec.cutover_week.date()}")

    model = build_hierarchical_binomial(mdata)

    print("\nPrior predictive check...")
    with model:
        prior = pm.sample_prior_predictive(samples=500, random_seed=0)
    prior_p = prior.prior["p"].values.flatten()
    print(f"  prior P(success) median = {np.median(prior_p)*100:.1f}%, "
          f"[5%, 95%] = [{np.percentile(prior_p,5)*100:.1f}%, "
          f"{np.percentile(prior_p,95)*100:.1f}%]")

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
                target_accept=0.9, random_seed=0,
                progressbar=True,
            )
    print(f"  sampled in {time.perf_counter()-t1:.1f}s")

    # ---- Diagnostics ------------------------------------------------------
    print("\n--- Hyperparameter posterior summary ---")
    s = az.summary(idata, var_names=["alpha_bar", "tau_bar",
                                      "sigma_alpha", "sigma_beta",
                                      "sigma_gamma", "sigma_delta",
                                      "sigma_tau"], hdi_prob=0.94)
    print(s[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]]
          .round(3).to_string())

    post_tau_bar = idata.posterior["tau_bar"].values.flatten()
    p_pos = (post_tau_bar > 0).mean()
    print(f"\nGlobal treatment effect (logit scale):")
    print(f"  mean   = {post_tau_bar.mean():+.3f}")
    print(f"  94% HDI= ({np.percentile(post_tau_bar,3):+.3f}, "
          f"{np.percentile(post_tau_bar,97):+.3f})")
    print(f"  P(tau_bar > 0) = {p_pos*100:.1f}%")

    # ---- Save artefacts ---------------------------------------------------
    nc_path = DUCKDB_DIR / "binomial_idata.nc"
    idata.to_netcdf(nc_path)
    print(f"\nTrace written to {nc_path}")

    print("Plotting...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    az.plot_forest(idata, var_names=["tau_C"], combined=True, hdi_prob=0.94)
    plt.title("Per-category treatment effect on on-time delivery (logit)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "binomial_tau_C_forest.png", dpi=150)
    plt.close()

    az.plot_posterior(idata, var_names=["tau_bar"], hdi_prob=0.94, ref_val=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "binomial_tau_bar_posterior.png", dpi=150)
    plt.close()

    print(f"Figures saved to {FIGURES_DIR}")
    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outcome", default="n_on_time",
                   choices=["n_on_time", "n_delivered"])
    p.add_argument("--draws",   type=int, default=1500)
    p.add_argument("--tune",    type=int, default=1000)
    p.add_argument("--chains",  type=int, default=4)
    p.add_argument("--cores",   type=int, default=4)
    p.add_argument("--use-nutpie", action="store_true",
                   help="Use the Rust-backed nutpie sampler (3-5x faster).")
    run(p.parse_args())
