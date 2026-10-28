# Local offer economics

These functions calculate explicit local scenarios. They do not issue offers,
change limits, execute treatments, or estimate causal price or treatment effects.
Risk, affordability, eligibility and operational rules remain separate.

All three methods accept a versioned profile (`profile_id`, `version`,
`known_at`, `currency`), `as_of` and optional calculation `context`.
Profile and assumption knowledge cannot follow `as_of`; currency and physical
measure must agree with context. Outputs are CalcResults containing a canonical
SHA-256-bound value. Every candidate retains its full input fingerprint. Only
finite numeric inputs are accepted; missing evidence is never a favorable value.

## Limit frontier

`compute_limit_frontier(offer_candidates, constraint_profile, *, as_of, context=None)`

Each row has `offer_id`, `version`, `known_at`, `currency`, `subject_id`, `amount`,
integer `tenor_days`, `assumption_reference`, `target`, `pd`, `lgd`, `ead`,
`risk_conditioning='CONDITIONAL_ON_THIS_OFFER'`, `risk_reference`,
`affordability_status`, explicit nullable `affordability_maximum_amount`, and
`eligibility_status`. Offer IDs are distinct; all offers concern the same subject.
Affordability states are AFFORDABLE, UNAFFORDABLE or UNKNOWN; eligibility states
are ELIGIBLE, INELIGIBLE or UNKNOWN. An unknown cap cannot establish feasibility.

The profile supplies the exact common `target`, `maximum_amount`, `maximum_ead`,
`maximum_expected_loss`, `maximum_concentration`, positive
`concentration_denominator`, `existing_group_exposure` and nonnegative numerical
`tolerance`. The default target has `event='default'`, positive integer `horizon`,
`time_unit='days'`, `probability_measure='PHYSICAL'` and explicit `conditioning`.
It is the same comparison horizon for all limit candidates; tenor may differ.

Expected loss is PD times conditional LGD times EAD. Economic LGD may be signed;
no regulatory cap or floor is silently added. Concentration is existing group
exposure plus candidate EAD divided by the supplied fixed **post-decision**
denominator. The denominator must already reflect the bank's intended budget
convention. No portfolio revaluation or inferred limit-to-EAD conversion occurs.
Each comparison reports slack, violations and numerical binding flags. The
scalar tolerance applies in the units of each reported comparison; choose it
small enough for all fields. Affordability and eligibility are independent gates.

All discrete offers are evaluated, including nonmonotone PD/amount relationships.
The frontier contains feasible offers not dominated by a same-tenor offer with
at least as much amount and no more EL, with one strict improvement. Different
tenors are not assigned a hidden utility ordering. `maximum_feasible_amount`
is null when none qualify. The result does not interpolate between candidates.

## Risk-based price grid

`compute_risk_based_price_grid(quote_candidates, pricing_profile, *, as_of, context=None)`

The profile declares `default_timing='BEFORE_DUE'`,
`prepayment_timing='AFTER_COUPON'`, `funding_treatment='DISCOUNT_CURVE_ONLY'`, and
`selection_assumption='QUOTE_CONDITIONED_INPUTS_NOT_CAUSAL'`.
Each quote has the common offer identification fields above, nonnegative
`annual_rate`, `target`,
`risk_conditioning='CONDITIONAL_ON_THIS_ACCEPTED_QUOTE'`, `risk_reference`,
`takeup_probability`, `takeup_reference`, `upfront_fee`, `origination_cost`,
`offer_cost`, and chronological `periods`.

The target horizon equals the quote's entire `tenor_days`, measured from `as_of`
to the final payment. The supplied conditional hazards apply to accepted accounts
at this quote, not to a generic borrower baseline. Each period requires:

- `payment_date`, `scheduled_principal`;
- `conditional_default_probability`, `conditional_prepayment_probability`;
- economic `lgd`, `recovery_date`;
- positive `payment_discount_factor`, `recovery_discount_factor`;
- nonnegative `servicing_cost`, `prepayment_fee`.

Default and prepayment are mutually exclusive conditional on performing at the
start of each period. Their sum cannot exceed one. Starting with survival S=1,
period masses are S*q_default and S*q_prepayment; closing survival is
S*(1-q_default-q_prepayment). Principal follows the supplied scheduled amortization
only for continuing accounts. Surviving scheduled principal must repay the entire
loan; there is no unpriced residual value. Coupons use fixed simple ACT/365F on
opening principal and the quoted annual rate.

Default occurs before due-date payments, receives no current coupon, and exposes
opening principal. Net recovery is that principal times (1-LGD), paid at the
explicit recovery date and discounted to `as_of`. LGD therefore describes the
undiscounted net recovery at that date; supplying an already discounted LGD would
double discount. Negative recovery can represent net workout costs. Recovery
cannot precede the default endpoint. Prepayment occurs after the coupon, repays
all opening principal and the specified fee. Continuing accounts pay the coupon
and scheduled principal. Servicing cost applies to every opening survivor,
including accounts that default during that interval. No accrued default interest
or partial-period default timing is invented.

Book NPV equals minus the original amount, plus upfront fee, minus origination
cost, plus all discounted expected cashflows. Funding costs enter **only** through
the supplied funding discount factors; do not add another funding-interest cost.
The same date must have the same discount factor within a quote. No monotonic
curve restriction is imposed because negative funding rates are possible.
Per-offer NPV is takeup probability times book NPV minus offer cost. Offer cost is
paid regardless of takeup; origination cost and upfront fee are conditional on
booking. Discounted EL is reported separately and is already reflected through
recovery in the cashflow NPV; it must not be subtracted a second time.

For amount100, one-year coupon10 and funding discount1/1.1, no risk or costs give
zero NPV. With conditional default.2, prepayment.3, LGD.4 and prepayment fee2,
expected year-end cash is100.6: interest8, principal80, fee.6 and recovery12.
Its NPV is100.6/1.1-100. A higher price may lower NPV when its supplied accepted
population has higher risk or lower uptake. Break-even flags are per candidate;
there is no claim of a unique rate root or estimated causal demand elasticity.

## Treatment candidates

`evaluate_treatment_candidates(treatment_candidates, treatment_profile, *, as_of, context=None)`

The profile adds `subject_id`, `baseline_id`, `baseline_present_value`,
`identification='CONDITIONAL_SCENARIO_NOT_CAUSAL'`, and `allowed_actions`.
Each candidate supplies `treatment_id`, `version`, `known_at`, `subject_id`,
`currency`, `action`, `prediction_reference`, `support_status` (SUPPORTED or
UNSUPPORTED), `affordability_status` (AFFORDABLE, UNAFFORDABLE, NOT_REQUIRED,
UNKNOWN), `eligibility_status`, `uptake_probability`, `offer_cost`,
`accepted_upfront_cost`, `rejected_present_value` and `cashflows`.

Cashflow records have distinct `cashflow_id`, future or same-day `date`,
`expected_recovery`, `expected_cost`, and positive `discount_factor`, anchored
at `as_of`. Recovery and cost are nonnegative, already probability-weighted
**within acceptance**. No second cure/default probability is applied.
Accepted value sums discounted net cashflows minus accepted upfront cost.
Expected value mixes acceptance and the explicit rejection value, then subtracts
the offer cost once. The baseline is a separate comparator, not an implicit
rejection assumption. Scenario difference from baseline is not a causal effect.

Unsupported predictions yield null monetary values and cannot enter the feasible
ranking. Unsupported, prohibited, unaffordable and ineligible treatments retain
separate reasons. Supported but unaffordable treatments retain their monetary
scenario result while remaining infeasible. Feasible treatments rank by expected
value, ties by ID. Equal nominal recovery can have different present value because
of timing and costs. Historical action selection and missing counterfactuals
remain unresolved by this calculation.
