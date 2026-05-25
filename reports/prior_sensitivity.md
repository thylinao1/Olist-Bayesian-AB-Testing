# Prior-sensitivity check on the Binomial DiD

The headline on-time policy effect in section 4.1 uses an `Exponential(1)` prior on the across-category scale `sigma_delta`. This script re-fits the model under two alternative priors and tabulates `delta_bar` and `sigma_delta` from each run. If the headline is being driven by the prior, `delta_bar` will move materially across rows; if the likelihood dominates, it will move by a few thousandths of a logit.

| Hyperprior on `sigma_delta` | `delta_bar` mean | 94% HDI | `sigma_delta` mean | P(delta_bar > 0) | divergences |
|---|---|---|---|---|---|
| `Exponential(1)` | +0.1699 | (+0.0678, +0.3005) | 0.1990 | 99.5% | 20 |
| `Exponential(2)` | +0.1695 | (+0.0620, +0.2950) | 0.1979 | 99.6% | 3 |
| `HalfNormal(1)` | +0.1696 | (+0.0515, +0.2884) | 0.1957 | 99.8% | 5 |

The spread in `delta_bar` across the three runs is +0.0004 logit (~0.00 pp on the probability scale at the ~89% baseline). The headline +1.5 pp on-time policy effect is insensitive to the hyperprior choice - the posterior is dominated by the 97k-order likelihood, not by the prior.
