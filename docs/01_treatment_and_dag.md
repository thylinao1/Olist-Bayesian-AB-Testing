# 01 — Treatment Definition & Causal DAG

This document does two jobs that must precede any model fit:

1. **Defines the "treatment" we're going to estimate the effect of.** Olist is observational data; there is no real RCT in it. To stay honest we frame the analysis as *the inferential machinery that would be used if Olist ran this experiment*. The treatment is synthetic but the exposure variable is constructed from variables that are actually present in the data, and the modelling pipeline is the one we would deploy if the assignment were real. Recruiters reading this should see: "this person knows the difference between observational and experimental data and treats causal claims with appropriate humility."
2. **Encodes the causal assumptions as a DAG**, identifies the backdoor paths, and resolves the minimal adjustment set — following the four-elemental-confounds framework in Statistical Rethinking Ch 6 (the fork, the pipe, the collider, the descendant).

---

## 1. The hypothetical intervention

> **"Olist switches on a free-shipping policy: for orders whose item-subtotal exceeds R$ 150, the freight charge is waived."**

Why this is the right pick:

- It is a policy levers e-commerce platforms actually pull (Shopee, Lazada, Amazon, Mercado Livre have all run versions of it).
- It plausibly moves *all three* of our outcome variables — conversion, basket revenue, and review score — so it exercises the full Ch 11 / Ch 12 / Ch 13 stack.
- The ingredients we need to construct exposure (subtotal, freight charge, week-of-purchase) all exist on `gold.fact_orders`.
- It maps cleanly to a "cutover" research design: assume the policy goes live at week W\*, so orders with `purchase_week >= W*` and `items_subtotal >= 150` are *treated* and everything else is *control*.

### Encoding

| Symbol | Definition | Source column |
|---|---|---|
| `T = 1` | order is in the treated stratum (purchase\_week ≥ W\* AND items\_subtotal ≥ 150) | derived in `src/features.py` |
| `T = 0` | otherwise | derived |

`W*` will be chosen at the median purchase week of the panel so the treated and control halves are roughly balanced. The exact threshold (R$ 150) is the median item-subtotal in the data — both choices are documented and held fixed.

---

## 2. Outcomes

Three outcomes, each modelled in its own chapter of the book:

| Outcome | Glyph | Distribution | Chapter |
|---|---|---|---|
| Order delivered (conversion proxy) | `Y_conv` | Bernoulli / Binomial | 11.1 |
| Customer-level revenue (incl. zero spenders) | `Y_rev` | Hurdle / zero-inflated continuous | 12.2 |
| Review score (1-5) | `Y_rev_score` | Cumulative-link ordered logit | 12.3 |

---

## 3. Variables in the DAG

| Symbol | Meaning | Role |
|---|---|---|
| `T` | Free-shipping treatment indicator | Exposure |
| `Y` | One of the three outcomes above | Outcome |
| `C` | Product category | Confounder (fork) — causes both basket-subtotal eligibility and outcome propensity |
| `S` | Seller volume tier | Confounder (fork) — large sellers ship faster, also drive basket size |
| `G` | Geography (customer state) | Confounder (fork) — drives baseline freight cost (so eligibility) and review propensity |
| `M` | Calendar month / season | Confounder (fork) — seasonality moves both basket size and review score |
| `F` | Freight value charged | **Mediator (pipe)** — `T` reduces `F`; `F` lifts conversion |
| `B` | Number of basket items | **Mediator (pipe)** — `T` incentivises larger baskets |
| `D` | Delivery days | **Mediator (pipe)** — basket size and category affect delivery time |
| `R` | Customer leaves a review | **Collider** — caused by the treatment effect AND the outcome experience |

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

Following the recipe from §6.4 of Statistical Rethinking:

> *(1) List all paths between T and Y. (2) Classify each as open or closed. (3) Classify each as a backdoor (arrow into T). (4) Close any open backdoors; do **not** open closed ones.*

### Backdoor paths from T to Y

| # | Path | Open? | Backdoor? | Action |
|---|---|---|---|---|
| 1 | `T ← C → Y` | open (fork at C) | yes | **condition on C** |
| 2 | `T ← S → Y` | open (fork at S) | yes | **condition on S** |
| 3 | `T ← G → Y` | open (fork at G) | yes | **condition on G** |
| 4 | `T ← M → Y` | open (fork at M) | yes | **condition on M** |
| 5 | `T → F → Y` | open (pipe) | **no** — leaves T forward | **do NOT condition on F** (post-treatment bias) |
| 6 | `T → B → D → Y` | open (pipe) | **no** — leaves T forward | **do NOT condition on B or D** |
| 7 | `T → R ← Y` | closed (collider at R) | no | **do NOT condition on R** (would open the collider) |

### Adjustment set

```
adjust_for = { C  (product category),
               S  (seller volume tier),
               G  (customer state),
               M  (purchase month) }
```

These are the variables that go into the model as covariates / hierarchical grouping factors. Everything *downstream of T* — freight, basket size, delivery time, review — is excluded from the adjustment set, because conditioning on a mediator is exactly the post-treatment bias §6.2 warns about.

### Implied conditional independencies

If the DAG is correct, the following should hold in the data:

- `B ⊥ M | T, C, S` — basket size is independent of season once treatment, category, and seller tier are fixed.
- `F ⊥ G | T, C, S` — freight is independent of geography once treatment and structural drivers are fixed.

We test these in `notebooks/02_dag_treatment.ipynb` as a sanity check on the DAG itself (the data can falsify a DAG even though it can never confirm one — McElreath §6.4.3).

---

## 6. What this means for the model spec

The hierarchical Binomial model (Ch 11 + 13) becomes:

```
y_{ij}     ~ Binomial(n_{ij}, p_{ij})
logit(p_{ij}) = α_C[c] + β_S[s] + γ_G[g] + δ_M[m] + τ_C[c] · T_i

# adaptive (hierarchical) priors — Ch 13
α_C[c]     ~ Normal(ᾱ, σ_α)        # category-level baseline
β_S[s]     ~ Normal(0,   σ_β)
γ_G[g]     ~ Normal(0,   σ_γ)
δ_M[m]     ~ Normal(0,   σ_δ)

τ_C[c]     ~ Normal(τ̄, σ_τ)        # category-varying treatment effect

# hyperpriors
ᾱ, τ̄      ~ Normal(0, 1.5)
σ_α, σ_β, σ_γ, σ_δ, σ_τ ~ Exponential(1)
```

Two non-trivial bits, drawn directly from the book:

1. **`α_C[c] ~ Normal(ᾱ, σ_α)`** is the partial-pooling structure from §13.1 (the Reedfrogs tadpoles model). Categories that are well-observed get their own intercept; sparsely-observed categories are pulled toward the global mean `ᾱ` by `σ_α`.
2. **`τ_C[c] ~ Normal(τ̄, σ_τ)`** lets the *treatment effect itself* vary by category — the headline result of the analysis. A flat A/B test gives one number; this gives a posterior distribution over (number-of-categories) treatment effects, plus a global mean and the variance across categories.

The zero-inflated revenue model (Ch 12.2) and the ordered-categorical review model (Ch 12.3) re-use the same adjustment set with different likelihoods. We'll define those in `02_revenue_model.md` and `03_review_model.md`.

---

## 7. References

- McElreath, *Statistical Rethinking* (2nd ed.), Ch 6 §§6.2–6.4 (post-treatment bias, collider bias, the four elemental confounds, backdoor adjustment).
- McElreath, Ch 11 §11.1 (binomial regression, logit link, relative vs. absolute effects).
- McElreath, Ch 13 §§13.1–13.3 (multilevel tadpoles model, varying intercepts, hyperpriors, partial pooling).
