Frozen shared-scenario transaction paths
=======================================

``simulate_transactions(transaction_recipes, scenario_cube, entity_innovations,
parameter_snapshot)`` composes the standalone debt, operating, asset, counterparty,
liquidity, default, recovery and risk calculations. Each recipe represents one
complete obligor cash pool with explicitly declared full legal cross-collateralization.
Separate obligors may share scenario factors and innovation channels; they cannot
reuse monetary operating contracts, facilities, reserves or collateral. Contractual
compartments must not be silently combined into such a pool.

Inputs and frozen execution
---------------------------

Each recipe identifies its transaction, subject and parameter snapshot entry, with
content hashes of that entry and the common ScenarioCube. The calendar maps every
cube time to a contiguous contractual period ending on its payment date. Monthly,
quarterly and annual frequencies are checked from the origin, preserving end of
month. Numeric times are in years. This prevents a shorter observed path being
reported as a longer PD horizon.

A physical ScenarioCube carries scenario/time/factor axes, identifiers, units,
normalized weights and its forecast, stationary or cycle-mixture sampling contract.
PIT/TTC/stress conditioning is preserved, never inferred from the average of paths.
The caller is responsible for the validity of that scenario law. Factor bindings
name their factor and unit, with explicit level or exponential transformation.
All entity innovations are identified by scenario, date and channel; missing and
unused innovations are rejected. There is no hidden random-number generation.

Parameter entries contain loan terms and liquidity ranks; operating cash inputs;
assets with market-factor bindings and innovation channels; reserve policy;
materiality thresholds; independently assessed outside-obligation evidence;
default and recovery policies; and the declared sampling design. Hashes bind
contents, not the truth or legal validity of supplied evidence.

Chronology and accounting
-------------------------

Counterparty events occur before receipts. Baseline contract receipts must already
be included once in operating CFADS: only the simulated cash change is added.
The same counterparty must have identical default probabilities and innovations
across transactions. Its event is not added to borrower PD.

The pool pays operating deficits, debt claims in declared rank, and finally reserve
topups. One reserve can fund several facilities only through the cash-conserving
waterfall. Contractually scheduled interest is paid before principal; unpaid
interest and principal are retained separately, with no additional arrears interest.
Actual EAD uses unpaid principal and due interest, not the contractual future
closing balance. Refinancing is never assumed.

Borrower default is assessed **after due payments** on each grid date. The economic
DSCR/LTV joint test is distinct from independent Unlikely-to-pay evidence and material
past-due signals. Available reserves may remedy the economic shortfall but never
cancel those independent signals. Own continuous material-arrears counters use
explicit absolute and relative thresholds and the declared absence of unrecorded
between-grid payments. Outside-obligation evidence is required on the entire grid,
including explicit assessed-no-indicator rows.

At first default the method freezes actual exposure, collateral and remaining cash.
Going-concern output stops there. Recovery uses the same event identity, allocates
collateral once and includes the remaining reserve once. The current recovery
recipe uses supplied fixed realization fractions, costs and delays, discounting
from default with an explicit effective annual rate. More general realization
scenarios remain available through the standalone recovery waterfall.

Scope and inference
-------------------

This recipe supports nonnegative scheduled coupons and no going-concern asset
sales; it rejects unsupported lender-refund or asset-removal state transitions.
Negative operating cashflows are supported. Explicit contract structures must be
modeled, not approximated by duplicated liquidity or assets. Recovery completeness,
legal entitlements, materiality applicability and outside evidence remain caller
responsibilities; the method does not certify regulatory approval.

PD and expected loss are calculated from unique shared scenario trials. Initial
defaults are excluded from initially-performing PD estimates. IID unit trials,
weighted scenarios and dependent/QMC designs retain their separate uncertainty
rules. Loss is averaged jointly with default and actual EAD/recovery, never obtained
by multiplying unrelated marginal averages.

Small tests cover reserve support without assumed refinancing, partial principal
repayment, independent arrears despite good collateral, initial default, two loans
sharing one reserve, counterparty interruption, complete repayment, frozen hashes,
calendar alignment, duplicate resources and transaction order invariance. The
baseline hand case has EAD100 and recovery80 at maturity, producing loss20.
