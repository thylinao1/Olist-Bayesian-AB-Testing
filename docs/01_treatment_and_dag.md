# 01 - Treatment Definition & Causal DAG

This document does two jobs that must precede any model fit:

1. **Defines the "treatment" we're going to estimate the effect of.** Olist is observational data; there is no real RCT in it. To stay honest we frame the analysis as *the inferential machinery that would be used if Olist ran this experiment*. The treatment is synthetic but the exposure variable is constructed from variables that are actually present in the data, and the modelling pipeline is the one we would deploy if the assignment were real. Recruiters reading this should see: "this person knows the difference between observational and experimental data and treats causal claims with appropriate humility."
2. **Encodes the causal assumptions as a DAG**, identifies the backdoor paths, and resolves the minimal adjustment set using the standard four-elemental-confounds framework (the fork, the pipe, the collider, the descendant).

---

## 1. The hypothetical intervention

> **"Olist switches on a free-shipping policy: for orders whose item-subtotal exceeds R$ 150, the freight charge is waived."**

Why this is the right pick:

- It is a policy levers e-commerce platforms actually pull (Shopee, Lazada, Amazon, Mercado Livre have all run versions of it).
- It plausibly moves *all three* of our outcome variables - conversion, basket revenue, and review score - so it exercises a hierarchical Binomial, a hurdle/zero-inflated continuous, and an ordered-categorical model together.
- The ingredients we need to construct exposure (subtotal, freight charge, week-of-purchase) all exist on `gold.fact_orders`.
- It maps cleanly to a "cutover" research design: assume the policy goes live at week W\*, so orders with `purchase_week >= W*` and `items_subtotal >= 150` are *treated* and everything else is *control*.

### Encoding

| Symbol | Definition | Source column |
|---|---|---|
| `T = 1` | order is in the treated stratum (purchase\_week ≥ W\* AND items\_subtotal ≥ 150) | derived in `src/features.py` |
| `T = 0` | otherwise | derived |

`W*` will be chosen at the median purchase week of the panel so the treated and control halves are roughly balanced. The exact threshold (R$ 150) is the median item-subtotal in the data - both choices are documented and held fixed.

---

## 2. Outcomes

Three outcomes, each modelled with its own likelihood:

| Outcome | Glyph | Distribution |
|---|---|---|
| Order delivered (conversion proxy) | `Y_conv` | Bernoulli / Binomial |
| Customer-level revenue (incl. zero spenders) | `Y_rev` | Hurdle / zero-inflated continuous |
| Review score (1-5) | `Y_rev_score` | Cumulative-link ordered logit |

---

## 3. Variables in the DAG

| Symbol | Meaning | Role |
|---|---|---|
| `T` | Free-shipping treatment indicator | Exposure |
| `Y` | One of the three outcomes above | Outcome |
| `C` | Product category | Confounder (fork) - causes both basket-subtotal eligibility and outcome propensity |
| `S` | Seller volume tier | Confounder (fork) - large sellers ship faster, also drive basket size |
| `G` | Geography (customer state) | Confounder (fork) - drives baseline freight cost (so eligibility) and review propensity |
| `M` | Calendar month / season | Confounder (fork) - seasonality moves both basket size and review score |
| `F` | Freight value charged | **Mediator (pipe)** - `T` reduces `F`; `F` lifts conversion |
| `B` | Number of basket items | **Mediator (pipe)** - `T` incentivises larger baskets |
| `D` | Delivery days | **Mediator (pipe)** - basket size and category affect delivery time |
| `R` | Customer leaves a review | **Descendant of Y** - caused by the outcome experience (Y) only. Earlier drafts of this DAG drew a T → R edge as well, making R a collider; we dropped it because the claim (treatment changes whether someone bothers to leave a review) is weak and keeping it would have required handling collider-conditioning when filtering the review panel to non-null scores. |

---

## 4. The DAG

```
              ┌──────── M (month / season) ────────┐
              │                                     │
              │      ┌── G (geography) ──┐          │
              │      │                   │          │
              ▼      ▼                   ▼          ▼
              T  ───────────────────────►  Y
              ▲      ▲                   ▲          ▲
              │      │                   │          │
              │      │                   │          │
              C ─────┘                   └──── S    │
              │                                     │
              └──────── F (freight) ───┐            │
                                       │            │
                                       ▼            │
                            B ──► D ───┘            │
                                                    │
                                       R ◄──────────┘
                                       ▲
                                       │
                                       T
```

(Rendered programmatically by `src/dag.py`; edit there, then re-run.)

---

## 5. Backdoor analysis

Following the standard backdoor-adjustment recipe:

> *(1) List all paths between T and Y. (2) Classify each as open or closed. (3) Classify each as a backdoor (arrow into T). (4) Close any open backdoors; do **not** open closed ones.*

### Backdoor paths from T to Y

| # | Path | Open? | Backdoor? | Action |
|---|---|---|---|---|
| 1 | `T ← C → Y` | open (fork at C) | yes | **condition on C** |
| 2 | `T ← S → Y` | open (fork at S) | yes | **condition on S** |
| 3 | `T ← G → Y` | open (fork at G) | yes | **condition on G** |
| 4 | `T ← M → Y` | open (fork at M) | yes | **condition on M** |
| 5 | `T → F → Y` | open (pipe) | **no** - leaves T forward | **do NOT condition on F** (post-treatment bias) |
| 6 | `T → B → D → Y` | open (pipe) | **no** - leaves T forward | **do NOT condition on B or D** |
| 7 | `T → Y → R` | open (pipe through Y) | **no** - leaves T forward via Y | **do NOT condition on R** (conditioning on a descendant of Y would weakly partial-out Y; we filter `review_score IS NOT NULL` for the review-score analysis, which is mild selection on R, acknowledged in §9 Limitations of the main report) |

### Adjustment set

```
adjust_for = { C  (product category),
               S  (seller volume tier),
               G  (customer state),
               M  (purchase month) }
```

These are the variables that go into the model as covariates / hierarchical grouping factors. Everything *downstream of T* - freight, basket size, delivery time, review - is excluded from the adjustment set, because conditioning on a mediator introduces post-treatment bias.

### Implied conditional independencies

If the DAG is correct, the following should hold in the data:

- `B ⊥ M | T, C, S` - basket size is independent of season once treatment, category, and seller tier are fixed.
- `F ⊥ G | T, C, S` - freight is independent of geography once treatment and structural drivers are fixed.

We test these in `notebooks/02_dag_treatment.ipynb` as a sanity check on the DAG itself (the data can falsify a DAG even though it can never confirm one).

---

## 6. What this means for the model spec

The hierarchical Binomial model becomes:

```
y_{ij}     ~ Binomial(n_{ij}, p_{ij})
logit(p_{ij}) = α_C[c] + β_S[s] + γ_G[g] + δ_M[m] + τ_C[c] · T_i

# adaptive (hierarchical) priors
α_C[c]     ~ Normal(ᾱ, σ_α)        # category-level baseline
β_S[s]     ~ Normal(0,   σ_β)
γ_G[g]     ~ Normal(0,   σ_γ)
δ_M[m]     ~ Normal(0,   σ_δ)

τ_C[c]     ~ Normal(τ̄, σ_τ)        # category-varying treatment effect

# hyperpriors
ᾱ, τ̄      ~ Normal(0, 1.5)
σ_α, σ_β, σ_γ, σ_δ, σ_τ ~ Exponential(1)
```

Two non-trivial bits:

1. **`α_C[c] ~ Normal(ᾱ, σ_α)`** is a partial-pooling structure: categories that are well-observed get their own intercept; sparsely-observed categories are pulled toward the global mean `ᾱ` by `σ_α`.
2. **`τ_C[c] ~ Normal(τ̄, σ_τ)`** lets the *treatment effect itself* vary by category - the headline result of the analysis. A flat A/B test gives one number; this gives a posterior distribution over (number-of-categories) treatment effects, plus a global mean and the variance across categories.

The zero-inflated revenue model and the ordered-categorical review model re-use the same adjustment set with different likelihoods.

---

## 7. SUTVA & interference

The hierarchical Bayesian models in this project assume the **stable-unit treatment value assumption** (SUTVA): the policy's effect on one order is independent of how the policy treats other orders. In a marketplace this assumption is plausible *as a first-pass* but not bulletproof, and an honest analysis needs to call out where it might fail.

Two specific interference channels matter for a real free-shipping deployment:

1. **Seller price adjustment.** Sellers know the platform is subsidising shipping for eligible baskets. To recover the margin they would otherwise have charged in freight, sellers may inflate the list price of items that frequently push a basket above the eligibility threshold. The treatment effect on per-order revenue would then partially reflect *seller pricing response* rather than genuine policy effect. The direction of this bias is upward — observed conditional spend lifts overstate the true policy effect.

2. **Seller logistics prioritisation.** Eligible orders carry no freight margin for the seller (the platform pays it). Sellers may prioritise these orders for faster fulfilment to avoid penalties on delivery-time SLAs, *crowding out* faster handling of ineligible orders within the same warehouse capacity. The ineligible (control) group's on-time delivery rate would then degrade *because of* the policy applied to the eligible (treated) group. This is classical SUTVA violation — a treatment effect on a unit's neighbours. The bias direction here is also upward, because the control gets slower while the treated stays the same.

Both channels would inflate the apparent policy effect relative to the true causal effect. Neither is detectable with the static observational data on hand — they require a real deployment to identify.

**The cleanest fix for a real deployment is a clustered RCT.** Randomise at the *seller* level, not the order level: a randomly chosen subset of sellers participate in the free-shipping policy and the rest do not. This breaks both interference channels because (a) participating sellers cannot raise prices on customers visible to non-participating sellers without losing them, and (b) logistics prioritisation only crowds out other orders within the same seller, not across the marketplace. The cost is statistical power — clustered designs have effectively smaller sample sizes — and the design must be powered accordingly.

For this portfolio analysis, the headline numbers should be read as *upper-bound estimates* of the policy effect under SUTVA-violating channels that the data cannot rule out.

---

## 8. References & inspiration

The Bayesian methodology used here (DAGs and the four elemental confounds, hierarchical / partial-pooling models, non-centered parameterisations, hurdle and ordered-logit likelihoods) is drawn from Richard McElreath's *Statistical Rethinking* book and accompanying YouTube course.
