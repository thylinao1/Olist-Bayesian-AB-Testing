"""Classical A/B-test baselines - the comparison set for the Bayesian models.

The point of this module is to make it easy to say:

    "On the marginal data, the two-proportion z-test gives p=0.04 and the
     Welch t-test gives p=0.31. The hierarchical Bayesian model finds
     posterior mass on a positive treatment effect for 14 of 73 categories
     even though the pooled comparison is null - Simpson's-paradox-style
     heterogeneity invisible to flat A/B analysis."

That is the headline result of the project. The classical baselines are
not strawmen; they are the methods a marketplace data scientist would
use by default, and the Bayesian story has to land *next to* them.

References
----------
- WAIC / PSIS-LOO for Bayesian model comparison (Vehtari et al.).
- Two-proportion z, Welch's t, Mann-Whitney U: standard introductory
  statistics - included here to demonstrate they are not the right tool
  for hierarchical data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Two-proportion z-test for the binomial (on-time) outcome
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TwoProportionResult:
    rate_treated:  float
    rate_control:  float
    diff:          float
    z:             float
    p_value:       float
    ci_low:        float
    ci_high:       float


def two_proportion_z(
    successes_treated: int, n_treated: int,
    successes_control: int, n_control: int,
    *, alpha: float = 0.05,
) -> TwoProportionResult:
    p_t = successes_treated / n_treated
    p_c = successes_control / n_control
    p_pool = (successes_treated + successes_control) / (n_treated + n_control)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_treated + 1 / n_control))
    z = (p_t - p_c) / se_pool

    # Wald CI for the difference (uses unpooled SE)
    se_unpool = np.sqrt(p_t * (1 - p_t) / n_treated +
                        p_c * (1 - p_c) / n_control)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    return TwoProportionResult(
        rate_treated=p_t,
        rate_control=p_c,
        diff=diff,
        z=z,
        p_value=2 * (1 - stats.norm.cdf(abs(z))),
        ci_low=diff - z_crit * se_unpool,
        ci_high=diff + z_crit * se_unpool,
    )


# ---------------------------------------------------------------------------
# Welch's t-test (revenue) - unequal variance, the "default" continuous test
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WelchTResult:
    mean_treated:  float
    mean_control:  float
    diff:          float
    t:             float
    df:            float
    p_value:       float
    ci_low:        float
    ci_high:       float


def welch_t_test(treated: np.ndarray, control: np.ndarray,
                 *, alpha: float = 0.05) -> WelchTResult:
    t = stats.ttest_ind(treated, control, equal_var=False)
    # CI for the difference of means
    var_t = treated.var(ddof=1)
    var_c = control.var(ddof=1)
    se = np.sqrt(var_t / len(treated) + var_c / len(control))
    df = (var_t / len(treated) + var_c / len(control)) ** 2 / (
        (var_t / len(treated)) ** 2 / (len(treated) - 1) +
        (var_c / len(control)) ** 2 / (len(control) - 1)
    )
    diff = treated.mean() - control.mean()
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    return WelchTResult(
        mean_treated=float(treated.mean()),
        mean_control=float(control.mean()),
        diff=float(diff),
        t=float(t.statistic),
        df=float(df),
        p_value=float(t.pvalue),
        ci_low=float(diff - t_crit * se),
        ci_high=float(diff + t_crit * se),
    )


# ---------------------------------------------------------------------------
# Mann-Whitney U - non-parametric alternative for revenue / scores
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MannWhitneyResult:
    median_treated:  float
    median_control:  float
    u_statistic:     float
    p_value:         float


def mann_whitney(treated: np.ndarray, control: np.ndarray) -> MannWhitneyResult:
    res = stats.mannwhitneyu(treated, control, alternative="two-sided")
    return MannWhitneyResult(
        median_treated=float(np.median(treated)),
        median_control=float(np.median(control)),
        u_statistic=float(res.statistic),
        p_value=float(res.pvalue),
    )


# ---------------------------------------------------------------------------
# Chi-square (review-score distribution test)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChiSquareResult:
    chi2:        float
    df:          int
    p_value:     float
    contingency: pd.DataFrame


def chi_square_review(df: pd.DataFrame,
                      treatment_col: str = "treatment",
                      score_col: str = "review_score") -> ChiSquareResult:
    contingency = pd.crosstab(df[treatment_col], df[score_col])
    chi2, p, dof, _ = stats.chi2_contingency(contingency.values)
    return ChiSquareResult(chi2=float(chi2), df=int(dof),
                           p_value=float(p), contingency=contingency)


# ---------------------------------------------------------------------------
# Convenience: side-by-side report
# ---------------------------------------------------------------------------
def format_result_row(name: str, effect: float, ci: tuple[float, float],
                      p_value: float, *, sig_threshold: float = 0.05) -> str:
    sig = "*" if p_value < sig_threshold else " "
    return (f"{name:<24} effect={effect:+.4f}   "
            f"95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]   "
            f"p={p_value:.4f} {sig}")
