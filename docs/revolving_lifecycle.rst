Four revolving account lifecycles
================================

Run examples/revolving_lifecycle.py or source examples/revolving_lifecycle.R
and call run_revolving_lifecycle(). Both run all four products. The optional
product_ids filter supports a short representative check. return_artifacts=True
(or TRUE in R) retains the original sealed models, reports and replay inputs.
All data are independent synthetic examples, not estimated bank calibrations.

Credit card, charge card, overdraft and home-equity line each have three distinct
actual accounts: a fully repaid account, a default/cure/redefault account and an
early voluntary exit. Person, application, household, facility and claim IDs stay
linked throughout origination, performing monitoring, delinquency, restructuring,
default workout and portfolio review. Local account and release CSVs pass the
same source controls as external files. A limit revision effective February15
but only known February20 cannot enter the February15 exposure forecast.

Each product has its own native fitted application GLM using 2018 frequency
observations and an independently fitted 2020 logistic calibration. Their
availability precedes every 2024 decision. Funded application targets measure
first default within365 days of first draw, conditional on funding; terminal
exit without full follow-up is unknown, never an invented nondefault. These
models do not establish transportability to rejected applications. Prior2021
behavioral, EAD and LGD fits and prior2022 pool/transition data are also separate
from actual held-out outcomes. The behavioral intercept fit is0.25; original
predictions remain available while monitoring uses an explicitly declared
8-decimal reporting view. Original sealed artifacts are never rounded.

The card February15 exposure is500+0.5*(1000-500)=750. The overdraft is
400+0.75*(1000-400)=850; its borrowing receipts are excluded from income.
The home-equity exposure is50000+0.25*(100000-50000)=62500. Charge-card
exposure uses a fitted direct lognormal EAD model: an authorization ceiling is
not an undrawn contractual commitment and no CCF is inferred. The charge
balance is fully paid February1 and new purchases occur February10. Optional
future purchase deferral never silently changes the existing March1 due amount.

Home-equity draw rights end March1. The repaid zero account exits on that date;
it cannot remain an active first-default risk with neither debt nor draw rights.
The indebted account enters repayment. Joint-household income and existing
mortgage obligations are each counted once despite joint legal recourse.
Committed combined LTV is(120000+100000)/275000=0.8. The February29 household
stress and actual March payment observation are distinct: a scenario income
reduction is not claimed to have occurred. Raw affordability_inputs and the
sealed affordability validation bind the same account and existing mortgage.

The main account first defaults April15 with600 times the product scale.
Full payment May10, ten covered clean days and a validated May20 cure precede
a separately approved May31 amendment and June1 new draw of200 times scale.
June20 is a new default episode, not an extension of the first open claim.
July recoveries, costs and principal write-offs are separate ledger entries;
write-off is never recovered cash. First-workout LGD is0. Second-workout LGDs
are0.65,0.525,0.75 and0.55 for card, charge, overdraft and home equity.
First-default behavioral risk sets continue to exclude the historical defaulter
following cure. Historical approved amendments do not imply that a March or
April offered restructuring was accepted.

The April30 collections review values the same still-open April15 claim before
its future May payment is known. Five separate prior2022 histories support real
cause-specific, Fine-Gray and multistate fits. First-cure competing-risk curves
are alternatives, not additive probabilities. Supported landmark transition
predictions assume time-inhomogeneous Markov behavior in the declared augmented
states. Three mutually exclusive monetary histories each have probability1/3;
conditional gross recovery is7/12 of the remaining claim. Uptake0.25 applies
once, accepted cost5 times scale, declined recovery0.2 of the claim and offer
cost2 times scale imply expected net recovery174.25 times scale. The offered
plan remains unaccepted and does not change the historical contract or default
state. No causal treatment effect is identified. Missing probation coverage
produces unknown cure targets. The separate residual-recovery model is explicitly
a common frozen synthetic assumption transported to the residual stage, not an
empirically identified product-specific recovery forecast.

Vintage hazards use separate prior calendar-bucket observations. Cumulative
first-default probability is1-(1-p)^j, with p0.1 except overdraft0.2. Monthly
endpoint roll transitions allow cure/redefault; a separate absorbing-default
matrix produces first-hit projections. Lifetime loss uses marginal first-event
masses, prior mean LGD0.4 and the explicitly assumed original authorization
ceiling as conditional exposure. An independent nine-path enumeration has the
same expected loss. It never multiplies each cumulative PD by the full loss.
Grade calibration derives0/5 and2/5 default frequencies from the same prior
pool rows actually used in the example; current grades follow observed utilization.

The six required diagnostic families assess actual retail cohorts, vintage,
roll rates, collections, retail pools and replay. Five overlapping monthly
windows belong to repeated customers, so no independent-window confidence
interval or large-sample acceptance is claimed. Small pools explicitly retain
insufficient-support diagnostics. The same held-out pool has count-weighted
observed default rate0.5 and exposure-weighted rate1. The first-event survival
check treats observed terminal closure as a permanent end to the expressly
defined facility default-before-exit event, not issuer nondefault or a hypothetical
continued contract. Application follow-up and monitoring instead retain their
stated censoring contracts. These are different estimands, not interchangeable
labels. A predeclared illustrative validation-report criterion is bound to the
actual result and manifests; it is not bank model approval.

Frozen model/calibrator/policy replay is separate from fresh refitting. Event
replay deduplicates the original opening event and respects the later-known
revision, recovering limits1000 then600 times scale. Full raw reports retain
local integrity seals. Cross-language fresh financial summaries are compared
with absolute tolerance1e-10 and relative tolerance1e-12, not identical fit hashes.

Default tests exercise a representative real fitted card chain and negative
controls. CREDITRATINGSYSTEM_EXTENDED_TESTS=true additionally runs all four;
the installed example call always covers all four. Tests reject altered ledgers,
false vintage outcomes, premature amendments and model/product mismatches.
Numerical backends are limited to one thread in the verification harness.

Method references: Friedman, Hastie and Tibshirani(2010), Regularization Paths
for Generalized Linear Models via Coordinate Descent, Journal of Statistical
Software33(1), https://doi.org/10.18637/jss.v033.i01; R stats generalized linear
model documentation, https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html.
See the accompanying survival and retail_collections_review documentation for
competing-risk and multistate estimation contracts and their primary references.

The home-equity branch additionally validates both actually closed LGD claims
against the original prior severity0.4. Count-weighted bias is0.125 and
EAD-weighted bias is0.2625 for losses0 and0.55 on EAD60000 and20000.
Both episodes retain cluster heloc-main-person-household; uncertainty is none,
so they are not treated as independent households. Joint application, household
budget, cohort split and workout validation retain that same household identity.
The funding ledger links heloc-main-application to account heloc-main; the
pre-funding household graph describes the existing senior mortgage separately.
