# Counterparty exposure paths

The three methods consume local user-valued scenarios and explicit contractual
assumptions. They do not price trades, infer legal opinions, simulate default
probabilities, or calculate regulatory capital. Shared path identifiers preserve
the ability to join exposure, default and recovery scenarios for dependent losses.

`aggregate_netting_set_exposure(trade_value_paths, netting_agreements, fx_paths,
as_of=..., reporting_currency=..., valuation_convention='EX_DIVIDEND_BEFORE_MARGIN')`
requires a complete common path/date/trade grid. Trade rows contain `path_id`,
`date`, `trade_id`, `currency`, and signed `value`. FX rows contain `path_id`,
`date`, `currency`, and positive `rate` in reporting units per trade-currency unit;
the reporting currency explicitly has rate one. No interpolation is performed.
Trade currency remains constant across the grid.

Each agreement contains `netting_set_id`, `counterparty_id`, `trade_ids`, Boolean
`legally_enforceable`, `evidence_reference`, `known_at`, `effective_from`, and
`effective_until` (date or null). Membership is unique, complete, known at as_of,
and effective at every valuation date under [from,until). Multiple trades require
an enforceable agreement; a non-nettable trade uses its own singleton set with
`legally_enforceable=False`. Signed converted values are summed within the set
before flooring at zero. Separate sets never offset each other.

`simulate_collateral_margin(netting_paths, margin_policies, call_calendar,
collateral_market_paths)` returns a bound collateral scenario artifact. One policy
per netting set specifies:

- `known_at`, `collateral_currency`, signed `initial_units` (positive held, negative posted).
- Nonnegative `threshold_received`, `threshold_posted`, `minimum_transfer_amount`.
- `ownership`: TITLE_TRANSFER or SECURITY_INTEREST; `received_enforceable=True`.
- `posted_returnability`: UNSECURED_CLAIM or BANKRUPTCY_REMOTE. Title transfer cannot be treated as bankruptcy remote.
- Boolean `posted_claim_netting_enforceable`, and `reuse`: ALLOWED or PROHIBITED.

A single homogeneous collateral asset is represented by units. At each path/date/set,
market rows declare `currency`, positive `unit_price`, positive `fx_rate`, and
`received_haircut`/`posted_haircut` in [0,1). Calls require unique `call_id`,
`path_id`, `netting_set_id`, `valuation_date`, `call_date`, `settlement_date`, and
nonnegative integer `dispute_delay_days`. Valuation cannot follow the call;
settlement cannot precede it. Effective settlement adds the declared calendar-day
dispute delay and must be on the grid or beyond its end. One call per date/set/path
avoids ambiguous intraday order.

Existing settlements occur before the new call; immediate settlement occurs after
that call. Desired signed effective collateral is V−received threshold for
positive exposure, V+posted threshold for negative exposure, and zero inside the
threshold band. Convert it to target units using valuation-date price, FX and the
applicable haircut. Subtract settled plus pending units to obtain the new transfer.
MTA is measured using gross reporting-currency market value at that valuation date.
No call is issued below MTA or for zero transfer. Pending transfers beyond the
horizon remain visible. Initial units are the snapshot balance before the first
grid date's events; the caller supplies any pre-snapshot pending obligations by
extending the explicit scenario, not by embedding them in initial units.

Recognized received collateral is positive nominal market value after its haircut.
Posted collateral is either bankruptcy remote or a full nominal unsecured return
claim; its haircut only determines required posting. With legally enforceable
posted-claim offset, exposure is max(V−recognized received+posted claim,0).
Otherwise it is max(V−recognized received,0)+posted claim. Reuse permission is
recorded, but the library adds no reinvestment or funding benefit.

`closeout_exposure_at_default(netting_paths, collateral_paths, default_events,
intermediate_cashflows, freeze_timing=..., cashflow_convention=...)` verifies that
both artifacts belong to the same netting scenario. Each event identifies a path
and netting set plus `default_date`, `closeout_date`, `margin_observation_date`.
Observation is no later than default and closeout no earlier than default. Default
and closeout must be exact grid dates. The declared observation anchor is caller
policy evidence; it is not inferred from default date. Outputs distinguish
`margin_period_days` (observation to closeout) from `default_to_closeout_days`.

BEFORE_DATE_EVENTS freezes opening units before all settlements/calls that day;
AFTER_DATE_EVENTS freezes closing units after them. Frozen units are revalued using
closeout price, FX and haircut. Later margin never reduces closeout exposure.
Cashflow rows have unique `cashflow_id`, path/set, `date`, signed `amount`, reporting
`currency`, and `included_in_closeout_value=False`. Under the required convention
UNSETTLED_EXCLUDED_FROM_CLOSEOUT_VALUE they are outstanding contractual claims in
(default,closeout], added exactly once. Intraday default-day unpaid cashflows are
outside this date-based contract. Insufficient path extension fails explicitly.

The vocabulary follows [BCBS CRE50.15–19 and 50.23](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/50/inforce/2019-12-15/published/2024-07-05).
Regulatory MPOR refers to the relevant margin-call market observation, rather than
simply the default-to-closeout delay. These APIs implement declared economic
contracts and apply no regulatory floor or approval assumption.

Hand example: +100 and −60 in one set yield40; in separate sets, total exposure is100.
If40 collateral units settle at default and their closeout price falls to0.5,
a closeout value90 plus10 unpaid claims gives exposure80. A later margin call
raising collateral to180 units does not alter that frozen-default calculation.
