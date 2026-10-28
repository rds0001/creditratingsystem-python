Debt service and liquidity allocation
=====================================

``build_debt_schedule(loan_terms, calendar, base_rate_fixings=None)`` produces
contractual payment legs with opening/closing principal, interest, principal,
positive debt service and any lender refund. Terms declare facility_id, currency,
principal, rate_type (fixed/floating), rate_convention
(nominal_simple/effective_annual), day_count (ACT/365F or ACT/360), amortization
(bullet/equal_principal/annuity/custom), maturity_date, and annual_rate or spread.

Calendar records carry start_date, end_date, payment_date, and floating fixing_date
or custom principal_due where applicable. Periods are contiguous and payment
dates strictly increase. Final payment equals maturity; all principal must be
repaid. Floating fixings explicitly contain fixing_date and base_rate. No unused
fixings, fees, capitalization, prepayment options or assumed refinancing enter.

Simple growth is 1 + rate * year_fraction; effective growth is
(1 + rate) ** year_fraction. Negative rates are retained if growth stays positive.
A fixed-rate constant annuity equals initial principal divided by the sum of
cumulative discount factors, including irregular periods. Floating annuity resets
require explicit principal schedules instead of implicit perfect foresight.

``apply_cash_waterfall(cash_periods, waterfall_policy, currency=...)`` processes
each scenario chronologically. Records contain scenario_id, date, currency,
cash_available, reserve_target and amounts_due keyed by every declared claim ID.
Amounts due are NEW obligations and explicitly include zeros.

The policy declares policy_id/version, initial_reserve, reserve_priority,
excess_reserve_policy (retain/release_before_claims), arrears_policy
(carry_without_interest), and claims (claim_id, unique priority, reserve_eligible).
Reserve top-up and claims execute by increasing priority. Eligible claims use
free cash first, then restricted reserve. Previous unpaid obligations carry once.
Remaining cash is paid to equity; no implicit retention occurs. Opening cash plus
opening reserve equals claim payments plus closing reserve plus equity cash.

Reserve releases are liquidity movements, not revenue. Negative operating flows
are represented by cost claims and actual available liquidity. The default-event
function separately assesses DSCR/LTV, available debt-service liquidity, UTP and
material arrears. Uncommitted refinancing and hypothetical collateral proceeds
are not available liquidity. The core calculations do not infer materiality rules
or arrears ages from aggregate unpaid monetary balances.
