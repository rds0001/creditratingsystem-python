# Dated asset pool paths

`simulate_asset_pool(asset_pool, scenario_contract, obligor_paths, asset_paths,
shared_innovations, context=None)` consumes a bound `CreditAssetPool` from
`build_credit_asset_pool`. It returns `AssetPoolPaths` with asset events, shared
default-group events, dated cashflows, purchases and unresolved ending states.
The same arguments are available natively in R. Dates are ISO calendar dates;
optional result context must match the actual horizon in days and conditioning.

The versioned scenario contract supplies `contract_id`, `version`, unique
`scenario_ids`, normalized nonnegative `weights`, increasing `dates` after the
pool snapshot, `currency`, `probability_measure='PHYSICAL'`, `conditioning`, and
`sampling_design` (`IID_UNWEIGHTED`, `FIXED_WEIGHTED_SCENARIOS`, or
`DEPENDENT_OR_QMC`). IID requires uniform weights. It also supplies
`default_timing='BEFORE_DUE'`, `prepayment_timing='AFTER_COUPON'`,
`dilution_treatment='PROPORTIONAL_REMAINING_SCHEDULE'`,
`reinvestment_funding='PRINCIPAL_CASH_ONLY'`, and nonnegative
`initial_reinvestment_cash` (zero for static pools).

Each scenario/default-group/date row contains `conditional_pd` and
`default_channel`. Each scenario/asset/date row contains
`conditional_prepayment_probability`, `prepayment_channel`, `dilution_fraction`,
`recovery_fraction`, integer `recovery_delay_days`, `recovery_cost`,
`servicing_cost`, `interest_multiplier`, integer `overdue_days`, and Boolean
`disputed`. Probabilities, dilution and recovery fractions lie in [0,1]; costs
and multipliers are nonnegative. The complete Cartesian grids are required.
Shared innovations contain `scenario_id`, `date`, `channel_id`, and `uniform`
in [0,1); unused channels and missing referenced channels are rejected. The
caller supplies the dependence model and innovations; this function makes no
independence or model-estimation claim.

All assets start performing. A default group has one absorbing event, applied
to every remaining claim in the group. On a risk date, dilution first reduces
the remaining contractual principal and coupons proportionately. Default then
precedes coupon and principal payments. Conditional on no default, coupon is
paid before prepayment or contractual principal. A prepaid or matured claim
cannot subsequently default. Servicing cost applies to opening performing
owned claims. Dilution is reported separately from credit loss.

Gross default recovery and workout cost are booked once at default date plus
the declared delay. Recovery dates between risk dates create cash-only ledger
rows, without additional hazard draws. For two claims of 100 on one debtor,
40% recovery and cost 5 per claim imply gross recovery 80, cost 10 and modeled
credit loss 130. No coupon is paid on that default date. Recovery beyond the
calendar remains in `pending_recoveries`; it is never silently treated as zero
or received cash. `modeled_credit_loss` anticipates the supplied recovery;
`recovery_gross` and `workout_cost` report actual dated scenario cash.

Revolving pools retain principal until the exclusive revolving endpoint and
purchase only predeclared, already observed warehouse claims after the
purchase-date payments. Warehouse claims evolve before purchase; earlier
coupons and principal belong to their previous owner. Purchases require a
performing positive remaining balance, current aging/dispute eligibility and
sufficient retained principal cash. Priority then assignment ID breaks ties.
A failed dated purchase is not retried. Recoveries and interest do not fund
purchases under this explicit funding convention. Remaining principal cash is
released when the revolving period ends. The ledger exposes both asset
principal and reinvestment cash conservation residuals.

Every contractual payment within the horizon and every purchase must appear
on the risk grid. Intermediate assignment transfer-out is rejected because
it requires an explicit sale-price contract. Future originations and initial
workout states are outside this engine. Floating-rate behavior is a supplied
coupon multiplier path, not an inferred rate model. Unpaid principal, retained
cash and pending recovery prevent `terminal_complete`; downstream ultimate
loss metrics must respect this distinction. These are economic cash ledgers,
not legal eligibility or regulatory certification.
