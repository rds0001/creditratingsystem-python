# Dated behavioral credit risk

The behavioral functions connect local account evidence, historical payment
allocation, future event labels and state dynamics. A live account with a zero
balance stays in its account risk set. Closure, prepayment, sale, default and
operational cure remain explicit states. Loan default probability, observed
delinquency and balance movement are distinct quantities.

Use ISO `YYYY-MM-DD` strings, finite numeric amounts and explicit currency codes.
Tables are lists of named rows; the Python data adapter also accepts data frames.
Native R accepts data frames or lists of named rows. Objects use named lists and
array fields use unnamed lists in R, including empty and singleton arrays.
Calculation results expose `.value` in Python and `$value` in R. These methods
make local calculations and do not change accounts or communicate with customers.

## Payment allocation and state assignment

`compute_days_past_due(obligations, payments, *, as_of, known_at,
allocation_policy="fifo", currency, context=None)` operates on one account.
Obligations require unique `obligation_id`, `account_id`, `due_at`, `known_at`,
nonnegative `amount` and currency. Payments require unique `payment_id`, account,
`paid_at`, `known_at`, nonnegative amount, currency and `kind` (`payment` or
`reversal`). A reversal supplies `reverses_payment_id` and must refer to a visible
earlier payment. Sum of reversals cannot exceed that payment.

Only obligations known by the cutoff and due by the observation date are used.
Only payments/reversals economically effective by the observation date and known
by the cutoff are included. Under FIFO the net payment pool covers due
obligations in due-date/ID order. This is an explicit current-pool reallocation
contract; it does not model a bank's historical servicing waterfall or separate
principal/interest/fee priorities. Under `explicit`, payment records additionally
supply `obligation_id` and cannot fund a different obligation. Unused credit is
reported separately; future obligations are not treated as already due.

DPD is the largest positive age among unpaid due obligations. An unpaid amount
due today has DPD zero and is included in `unpaid_due_amount`; it is not yet in
`past_due_amount`. Example: obligations 100 on January 1 and February 1, payment
150 on January 15, reversal 100 on February 10, known February 12. At February
29 in a leap year the net 50 covers part of January's obligation, leaving
arrears 150 and DPD 59. A January snapshot retains its original payment evidence.
No grace or materiality threshold is hidden in this calculation.

`assign_delinquency_state(dpd, past_due_amount, exposure, events, state_profile,
context=None)` maps the resulting facts under an explicit profile. Profile fields:

- `schema_version="1.0"`, `profile_id`, `immaterial_state`.
- `materiality`: nonnegative `absolute`, `relative` thresholds and `logic`
  (`and` or `or`). Threshold equality qualifies; arrears must still be positive.
  At zero exposure, positive arrears satisfy the relative criterion.
- `buckets`: ordered `{state, lower, upper}` integer DPD intervals, lower
  inclusive/upper exclusive, gap-free from zero. Last `upper` is null.
- `event_priority`: ordered `{event, state, at_risk, absorbing}` records.
  Event flags are supplied separately as booleans. Every supplied flag must
  have a declared priority. For example, UTP can trigger default at DPD zero.
- Optional `dpd_default_rule`: `{minimum_dpd, state, material_required}`.

Event rules take precedence, then an explicit DPD default rule, then material
delinquency buckets or the immaterial state. A bucket alone never creates a
regulatory default claim. `absorbing` describes this risk target; operational
cure and first-default history are separate concepts.

## Frozen account snapshots

`build_account_month_panel(accounts, statements, account_events, *,
observation_dates, reporting_lag_days=0, obligations=None, payments=None,
context=None)` builds calendar month ends without treating a statement cycle as
the month itself. Accounts require unique `account_id`, `customer_id`,
`product_id`, `opened_at`, `known_at`, currency. Statements require unique
`statement_id`, account ID, `observed_at`, `known_at`, balance, credit limit,
integer DPD, past-due amount and currency. The statement observation date is its
economic reporting date, for example its cycle close; month-end snapshots may
therefore use an older cycle and report `source_age_days`.

Lifecycle events require unique `event_id`, account ID, `event_type`,
`occurred_at`, `known_at`. Allowed lifecycle types are `ACTIVE`, `DEFAULT`,
`CHARGEOFF`, `CURED`, `CLOSED`, `PREPAID`, `SOLD`. Simultaneous unresolved events
or statement revisions are rejected. A legal terminal state cannot be silently
reopened. Default and charge-off remain in `ever_default` after operational cure.
Closed records remain present for historical/recovery context.

At every requested month end, the information cutoff is **month end minus
reporting_lag_days**. Only facts effective by the month end and available by this
cutoff can be used. This parameter is a conservative evidence delay; it does not
permit look-ahead into the next month. The latest economic statement and latest
available revision are selected. No statement yields missing features, not zero
balance. Optional obligations and payments replace statement DPD/arrears with
the dated FIFO calculation. These tables must be complete for that intended
schedule scope; the function cannot discover missing source obligations.

The result contains `accounts` and `customers`. Account features are balance,
credit limit, DPD, arrears and utilization (null for zero limit). Each snapshot
has a canonical hash, product, currency, lifecycle state, calendar-month age,
first-default risk-set flag and source availability. Active undrawn credit is
retained. Customer aggregates separate currencies and report active balances
and account counts; aggregation is not a customer PD.

## Future labels and cohort performance

`build_behavioral_targets(panel, events, *, horizon_months, target_event,
competing_events, exit_treatment, evaluation_cutoff, coverage, context=None)`
constructs **first-event** labels from the unmodified snapshot date to its
monthly horizon anniversary: `(as_of, horizon_end]`. Use the same event-table
columns described above, with caller-defined target/exit event names. A known
prior target, prior default or terminated snapshot is outside the first-event
risk set; cure does not erase first-default history. This contract does not
estimate recurrent delinquency or re-default episodes.

`coverage` records have account ID, `observed_through`, `known_at` and explicitly
declare completeness of outcome observation. Absence of a reported target alone
does not justify zero. A positive event can be identified before full calendar
maturity. A complete mature event-free window gets zero. Other windows remain
`IMMATURE` or `OUTCOME_PENDING`, with null labels. If coverage and an observed
event are both unavailable, follow-up duration is null. Same-day competing
target/exit events require explicit upstream ordering and are rejected here.
`exit_treatment="competing"` assigns zero to a first competing exit;
`"censor"` leaves its target label null. The former is a loan-lifetime target,
not an assertion about hypothetical risk after closure. Customers and overlapping
windows are dependent: use grouped temporal splits and appropriate uncertainty.

`build_vintage_performance(accounts, panel, events, *, months_on_book,
evaluation_cutoff, coverage, default_event, delinquency_states, exit_events,
context=None)` forms origination-month/currency cohorts. Accounts additionally
supply nonnegative `original_balance`. MOB zero ends at origination month's
last day; MOB one ends at the next month's last day. Panel states must be observed
at those month ends for a complete stock measure.

Denominators are fixed cohort account counts; original balance is the loss-rate
denominator. First default and first competing exit are retained separately.
For every cell, the output reports cohort count, observed stock count, complete
outcome count, first defaults, period incidence and exits. Cumulative/default
incidence rates stay null until the calendar window is mature and every cohort
member has known complete outcomes or an observed absorbing first event. Stock
delinquency stays null until all member snapshots are available. Operational
cure can lower current delinquency without reducing ever-default rates.
Optional `LOSS` and `RECOVERY` event rows carry nonnegative amount and cohort
currency; observed net losses subtract recovery. These are observed cashflow
totals, not ultimate recovery estimates. Young vintages remain missing. Age,
cohort and calendar period are linearly related; these descriptive triangles
do not separately identify their causal effects.

## Estimated transitions and projections

`estimate_retail_roll_rates(panel, *, states, from_date, to_date, group_by,
weighting, missing_successor, context=None)` matches each origin account to its
explicit destination at the two dates. State rows require account ID, `as_of`,
`state`, grouping columns and opening `balance` (or `features.balance`) for
balance weighting. Groups are defined at the origin date; product and MOB may
be included. `weighting="count"` estimates account transitions;
`"opening_balance"` estimates opening-balance-weighted flows. Zero-balance
accounts retain origin counts in both modes. Closures and multi-bucket jumps
remain explicit destinations.

`missing_successor="error"` requires complete successors. `"censor"` returns
their missing count and mass separately and estimates probabilities conditional
on observed successors. This is not automatically unbiased under informative
dropout. Every origin count/mass equals observed destination total plus censored
count/mass. Empty observed origin rows return null probabilities, not identity
rows. The output includes full counts, masses, probability matrices and gaps.

`project_retail_roll_rates(initial_distribution, transition_matrices, *, states,
default_states, closure_states, exposure_by_period, context=None)` accepts one
row-stochastic transition matrix per projected period. Date/MOB/macro-dependent
matrices must be constructed under the caller's declared scenario. Null estimated
rows cannot be projected without an explicit modelling assumption. First-default
and closure states must be individually absorbing, and closed-state exposures
must be zero. The probability distribution multiplies successive matrices; flows
are `origin_probability * transition_probability`.

`exposure_by_period` supplies a separate nonnegative period-by-state amount matrix.
These conditional exposures encode drawdown, amortization, maturity and other
balance evolution. Expected destination balances are state probability times
state exposure; `balance_flows` multiply probability flows by destination
exposure. They are expected conditional balances, not cash transfers or a
self-financing balance process. The output separates marginal/new cumulative
default from any initial default mass. Operational cure models must use a
separate state process when first-default absorption is required.

This distinction between account transitions and exposure dynamics is consistent
with the separation of credit-card PD and EAD processes in the
[Federal Reserve's supervisory model description](https://www.federalreserve.gov/publications/2025-june-supervisory-stress-test-methodology-descriptions-supervisory-models.htm).
No supervisory coefficients, cutoffs or calibrations are copied.

## Frozen behavioral prediction

`predict_behavioral_risk(snapshot, model, target, calibrator=None, offset=None,
context=None)` validates snapshot integrity and a frozen model's
`behavioral_contract`: `risk_unit="account"`, `event_policy="first_default"`,
arrays `product_ids`/`allowed_states` and `maximum_source_age_days`. It checks
active first-default eligibility, source age, exact physical target and model
feature order before invoking the shared model predictor. It currently supports
the models available through `predict_pd`; unsupported fitted model families
fail explicitly. Optional calibration is applied under the same target and base
model version; isotonic calibration must declare `metadata.input_scale="probability"`.
Model training and calibration knowledge dates must not exceed the snapshot's
information cutoff; later-fitted models cannot be presented as historical scores.
The result reports raw/calibrated PD, IDs, dates and the snapshot hash. Account
actions, early-warning cutoffs, rating grades and EAD are separate calculations.
