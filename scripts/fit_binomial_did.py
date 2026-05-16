"""Fit the hierarchical Bayesian DiD model on Olist on-time delivery.

Difference-in-differences gives us a clean policy effect estimate by
removing two confounds the naive `treatment = eligible AND post`
specification was lumping in:

    * basket-size structural effect (eligible captures it)
    * common time trend          (post captures it)

The interaction `eligible * post * delta_C[c]` is the policy effect,
allowed to vary by product category via partial pooling.

Usage:
    python scripts/fit_binomial_did.py
    python scripts/fit_binomial_did.py --use-nutpie
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

from src.features import build_modelling_frame, panel_for_did
from src.models.binomial_did import build_did_binomial, from_panel
from src.paths import DUCKDB_DIR, FIGURES_DIR


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    print("Loading data...")
    df, spec, encoders = build_modelling_frame()
    panel = panel_for_did(df, spec)
    labels = sorted(encoders["category_en"].keys(),
                    key=lambda k: encoders["category_en"][k])
    mdata = from_panel(panel, category_labels=labels, outcome="n_on_time")

    print(f"  outcome     : {mdata.outcome_name}")
    print(f"  panel cells : {len(mdata.n_trials):,}")
    print(f"  successes   : {mdata.n_successes.sum():,} of {mdata.n_trials.sum():,} "
          f"({mdata.n_successes.sum()/mdata.n_trials.sum()*100:.2f}%)")
    print(f"  cutover wk  : {spec.cutover_week.date()}")
    print(f"\n  cell sizes (eligible, post) -> n_orders, on_time_rate:")
    cell = panel.groupby(["eligible", "post"]).agg(
        n=("n_orders", "sum"),
        on=("n_on_time", "sum"),
    )
    cell["pct_on_time"] = cell["on"] / cell["n"]
    print(cell.to_string())

    model = build_did_binomial(mdata)

    print("\nPrior predictive check...")
    with model:
        prior = pm.sample_prior_predictive(samples=200, random_seed=0)
    prior_p = prior.prior["p"].values.flatten()
    print(f"  prior P(success) median = {np.median(prior_p)*100:.1f}%, "
          f"5/95 = [{np.percentile(prior_p,5)*100:.1f}%, "
          f"{np.percentile(prior_p,95)*100:.1f}%]")

    print(f"\nSampling: {args.chains} x {args.draws} draws ({args.tune} tune)...")
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
        "alpha_bar", "delta_bar",        # global baseline + global policy effect
        "beta_eligible", "beta_post",    # basket-size + time-trend
        "sigma_alpha", "sigma_delta",
        "sigma_gamma_s", "sigma_gamma_g",
    ], hdi_prob=0.94)
    print(s[["mean","sd","hdi_3%","hdi_97%","ess_bulk","r_hat"]].round(3).to_string())

    post = idata.posterior
    delta_bar = post["delta_bar"].values.flatten()
    beta_elig = post["beta_eligible"].values.flatten()
    beta_post = post["beta_post"].values.flatten()

    print("\n--- DECOMPOSITION OF THE NAIVE -2pp HEADLINE ---")
    print(f"  basket-size structural effect (beta_eligible): "
          f"mean={beta_elig.mean():+.3f}, "
          f"94% HDI=({np.percentile(beta_elig,3):+.3f}, "
          f"{np.percentile(beta_elig,97):+.3f})")
    print(f"  common time trend            (beta_post):     "
          f"mean={beta_post.mean():+.3f}, "
          f"94% HDI=({np.percentile(beta_post,3):+.3f}, "
          f"{np.percentile(beta_post,97):+.3f})")
    print(f"\n  >>> POLICY EFFECT (delta_bar): "
          f"mean={delta_bar.mean():+.3f}, "
          f"94% HDI=({np.percentile(delta_bar,3):+.3f}, "
          f"{np.percentile(delta_bar,97):+.3f})")
    print(f"  P(delta_bar > 0) = {(delta_bar > 0).mean()*100:.1f}%")

    nc_path = DUCKDB_DIR / "binomial_did_idata.nc"
    idata.to_netcdf(nc_path)
    print(f"\nTrace written to {nc_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    az.plot_forest(idata, var_names=["delta_C"], combined=True, hdi_prob=0.94)
    plt.title("Per-category POLICY EFFECT on on-time delivery (DiD, logit scale)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "binomial_did_delta_C_forest.png", dpi=150)
    plt.close()

    az.plot_posterior(idata,
                      var_names=["delta_bar", "beta_eligible", "beta_post"],
                      hdi_prob=0.94, ref_val=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "binomial_did_decomposition.png", dpi=150)
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
