# Affordability arithmetic

These functions quantify payment capacity independently of PD and credit policy.
They accept scalars or one-dimensional vectors, with explicit length-one expansion.
All monetary values must be in the same declared currency. No legal or universal
cutoffs are built in. Examples use arbitrary synthetic amounts and thresholds.

`debt_to_income(debt_stock, income, *, income_frequency, currency, ratio_period,
income_basis="net", context=None)` divides a debt stock by income. With
`ratio_period="annual"`, the supplied income is annualized at 52 weekly, 26
biweekly, 13 four-weekly, 12 monthly, 4 quarterly or 1 annual payments. With
`ratio_period="income_period"`, no annualization is performed. A stock of 60,000
and monthly income of 2,500 yields annual DTI 2, not monthly DTI 24.

`debt_service_to_income(debt_service, income, *, period, currency,
income_basis="net", context=None)` divides same-period service by income. Service
of 900 and monthly income of 3,000 yields DSTI 0.3. Stock debt and debt service
are different numerators. Nonpositive income produces a null ratio, explicit
`NONPOSITIVE_INCOME` and `applicable=False`, never a safe zero.

`residual_income(net_income, recurring_expenses, existing_debt_service,
new_debt_service, *, period, currency, context=None)` subtracts the three disjoint
outflow components. The caller must reconcile obligations and household transfers
before aggregation. The function preserves negative income and negative surplus.
Expenses and debt service must be nonnegative. Output includes `residual`,
`negative_buffer`, `minimum` and a component trace.

`assess_affordability(metrics, threshold_profile, evidence, context=None)` consumes
scalar named metrics and explicit availability flags. A versioned profile contains
`schema_version="1.0"`, `policy_id`, `version`, `required_evidence` and `rules`.
Rules have unique `rule_id`, `metric`, `operator` and finite `threshold`. Operators
are `<`, `<=`, `>` and `>=`; comparisons use unrounded values. Positive headroom
points in the passing direction. At equality a strict rule fails with zero headroom.
Missing metrics or evidence produce `INSUFFICIENT_EVIDENCE`, with any known breaches
still retained. Otherwise the result is `AFFORDABLE` or `NOT_AFFORDABLE`.
No result constitutes credit approval or changes the risk estimate.

`normalize_income_period(components, *, target_period, target_basis, currency,
payments_per_year, variable_income_factor, context=None)` works on economic income
sources. Every component has `economic_id`, `amount`, `frequency`, `basis` (`net`
or `gross`), `currency`, and `kind` (`recurring`, `variable`, `internal_transfer`).
An amount may be null when unknown. `payments_per_year` is an explicit mapping,
for example `{"weekly":52,"monthly":12,"annual":1,"thirteenth_salary":1}`.
The conversion is source amount times source annual count divided by target
annual count. Enter a thirteenth salary as a separate annual component, not as
another monthly amount. This calendar is an averaging convention, not a dated
cashflow forecast.

Crossing net/gross basis requires an explicit nonnegative `tax_amount` in the
same payment period: gross minus supplied deductions gives net, and net plus
supplied deductions gives gross. No tax rate or jurisdiction is guessed.
Identical `economic_id` repeats count once; conflicting economic amounts or
bases fail. Internal transfers are excluded. `variable_income_factor` in [0,1]
reduces positive variable earnings. Negative earnings remain fully included.
Unknown included amounts or tax bridges make `total_income` null; `known_income`
and missing components remain available. Missing is not a zero-income assumption.

`estimate_recurring_expenses(expenses, *, target_period, currency,
payments_per_year, household_profile, benchmarks=None, context=None)` reconciles
already categorized periodic amounts. Rows have `economic_id`, `category`,
`amount`, `frequency`, `currency` and `kind` (`essential`, `discretionary`,
`debt_service`, `internal_transfer`). For repeated evidence of the same economic
obligation it takes the **maximum normalized known amount**, so a bank statement
and contract do not add twice. It is not a detector for recurring transactions;
the caller identifies economic obligations and their periodic amount first.
Transfers are excluded and debt service is a separate output, never part of
living expenses. Unknown debt service stays null with `known_debt_service`
reported separately.

`household_profile` supplies `profile_id` and an array `required_categories`.
An optional local benchmark table has `profile_id`, `category`, `amount`,
`frequency`, `currency`, `kind`; only exact profile matches apply. Household
size and composition must therefore be encoded by the chosen profile. Each
benchmark is a category floor after reconciling all observed obligations in that
category. It is not added on top of observed spending. Missing categories or
unobserved obligations keep total recurring expenses unknown. The output gives
essential/discretionary totals, reconciled evidence counts and remaining gaps.

`combined_loan_to_value(claims, collateral, allocations, *, currency,
context=None)` calculates static secured-debt ratios and rank coverage. Claims
have `claim_id`, `amount`, `currency`; collateral has `collateral_id`, `value`,
`currency`, `valuation_date`, `freshness` (`CURRENT` or `STALE`, supplied by the
caller). Allocation rows give `claim_id`, `collateral_id`, `amount`, integer
`priority`. Amount allocations must partition each claim once, within a relative
or absolute 1e-12 reconciliation tolerance. This explicit partition is not a
legal model of unlimited cross-collateralization. Each collateral value is
counted once. Lower ranks consume value first; equal ranks share remaining value
proportionally. The output separates allocated loan amounts, cumulative rank
LTV and covered/uncovered amounts. A zero value returns a null LTV, not infinity.
Claims of 80 and 20 over value 125 give CLTV 0.8; over value 90, coverage is 80
and 10. Realization costs and legal enforceability are outside this static ratio.

`stress_affordability(baseline, scenario, *, context=None)` projects a supplied
joint monthly path. It does not draw independent shocks. The deterministic
scenario may be generated externally with shared innovations. Every period
explicitly maps all income, expense, collateral and contract IDs, preserving
the dependence in that supplied path. This supports sensitivity to income,
payment obligations and collateral; these factors are relevant to creditworthiness
assessment in the [EBA loan origination and monitoring guidance](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring?version=2020).
No regulatory thresholds or compliance conclusion are supplied.

The baseline is a named object with:

- `as_of`, reporting `currency`, nonnegative `liquid_buffer`.
- `incomes`: records with unique `economic_id`, monthly net `monthly_amount`,
  `basis="net"` and currency; `expenses`: unique economic ID, monthly amount and
  currency. Expenses must exclude all contract debt service.
- `contracts`: economic ID, currency, nonnegative `principal`, integer
  `remaining_months`, nonnegative nominal `annual_rate`, `repayment` (`annuity`
  or `interest_only`), `balloon_amount`, `rate_type` (`fixed` or `reset`), array
  `reset_dates`, nonnegative `rate_floor` and `rate_cap`. The initial coupon must
  lie within the bounds. An interest-only contract requires balloon = principal.
- `collateral`: unique economic ID, value and currency. `cltv_scope` explicitly
  equals `all_contracts_secured_by_pool` or `not_applicable`. Only a declared common
  collateral pool supports aggregate CLTV; otherwise use the separate allocation
  function for appropriate secured subsets.
- `thresholds`: `minimum_residual`, `maximum_dsti`, `maximum_cltv`, all explicit.

The scenario has `schema_version="1.0"`, `scenario_id`, and a `periods` array.
Every period contains `date`, `income_factors`, `expense_factors`,
`collateral_factors`, `rate_shifts` and `fx_to_reporting`. Factor and shift maps
cover their corresponding baseline IDs exactly. FX maps supply positive reporting
currency units per source currency unit, with reporting currency FX = 1. Dates
must be consecutive monthly anniversaries of `as_of`, clamped to month-end where
necessary. Factors scale original baseline amounts. Rate shifts are absolute
annual-rate changes relative to each contract's original coupon, applied only
at an explicit reset and bounded by its floor/cap. Fixed coupons ignore market
shifts. A reset within the projected path must coincide with a payment date;
intra-month accrual requires a different contract.

Monthly interest is the opening balance times nominal annual coupon / 12.
An annuity solves `B = P*(1-(1+r)^(-n))/r + balloon*(1+r)^(-n)`; at zero rate,
`P=(B-balloon)/n`. At a reset, the payment is recomputed over remaining maturity.
The maturity payment clears actual principal plus interest, including the
balloon; no refinancing proceeds are invented. This distinguishes regular payment
capacity from the terminal payment risk described by the
[CFPB balloon-payment explanation](https://www.consumerfinance.gov/ask-cfpb/what-is-a-balloon-payment-when-is-one-allowed-en-104/).

Outputs retain each contract's coupon, interest, payment, repaid principal and
closing balance. Household outputs include residual income, debt service, DSTI,
opening debt / annualized current income, closing CLTV, breaches and cumulative
liquidity. `first_liquidity_exhaustion` is the first date the buffer becomes
strictly negative. The path continues after exhaustion to display the funding
gap; it does not simulate default, capitalization or collection. Nonpositive
income gives null ratios. Zero collateral in an applicable pool with outstanding
debt produces `ZERO_COLLATERAL_VALUE`. All user-supplied values and paths remain
unchanged; baseline and scenario hashes support replay.

Native R uses the same signatures with named arguments, `NULL` and ordinary
named lists for objects. Array fields must be unnamed lists even when empty or
of length one. These four methods return structured records rather than the
scalar/vector interfaces of the initial ratios.
