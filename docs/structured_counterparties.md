# Pool counterparty cash stress

`stress_pool_counterparties(baseline_cashflows, exposure_contracts,
netting_sets, exposure_paths, collateral_paths, settlement_terms,
scenario_contract, party_mapping, party_default_paths, shared_innovations,
as_of=..., context=None)` applies caller-supplied physical shared defaults to
existing monetary claims and supplied replacement marks. It returns a bound
`PoolCounterpartyStress` with adjusted cash, removed original monetary IDs,
netting/counterparty attribution and cash reconciliation. The same native R
arguments are available, with `as_of` as an ordinary named argument.

The scenario and party contracts match the invoice engine: versioned scenario
IDs, normalized weights, increasing dates, currency, physical measure, sampling
design and conditioning; unique legal-party/default-group mapping; complete
conditional hazard grid and exactly used channel uniforms in [0,1). Optional
context matches the actual horizon in calendar days. Counterparty roles never
obtain independent defaults for the same legal identity.

Baseline rows contain scenario, unique `cashflow_id`, `date` and nonnegative
`amount`; each monetary ID exists in every scenario. They represent existing
incoming claims held from origin until remittance, not unearned future income.
Exposure contracts contain unique `exposure_id`, `counterparty_id`,
`netting_set_id`, `role`, `basis` and disjoint `cashflow_ids`. Supported pairs are
SERVICER/ACCOUNT_BANK with CASH_HELD, GUARANTOR with GUARANTEE_RECEIVABLE, and
SWAP with REPLACEMENT. A guarantee receivable is an existing baseline claim;
this method does not create an extra guarantee benefit over the same exposure.
Replacement contracts have no linked baseline cash. No monetary ID may be used
by two exposures.

Each netting set declares its counterparty, Boolean `enforceable`, unique
`collateral_contract_id`, `collateral_convention='RECEIVED_TRANSFERABLE'`, and
`default_timing='AFTER_COLLATERAL_SNAPSHOT_BEFORE_CASHFLOWS'`. Without enforceable
netting a set must contain exactly one exposure. Complete exposure snapshots
contain scenario, exposure, date and finite `signed_replacement_mark`. Cash-linked
exposures require zero added mark because their value derives from baseline
claims. Complete collateral snapshots contain scenario, netting set, date,
`known_at`, nonnegative `collateral_value` and `haircut` in [0,1]. Knowledge cannot
postdate the declared default snapshot. Posted-collateral claims are outside
this scoped stress method and belong in the counterparty exposure engine.

At default, unpaid linked baseline cash scheduled on or after that date is
removed once. Already remitted cash remains untouched. Add supplied signed
replacement marks to those linked claims before flooring the net exposure.
Received haircut-adjusted collateral covers the positive net claim first.
Settlement terms, once per scenario/counterparty, supply
`estate_recovery_fraction`, `estate_capacity`, `estate_delay_days`,
`collateral_delay_days`, `replacement_cost` and `replacement_delay_days`.
Estate capacity applies once across all unsecured netting-set claims and is
allocated pro rata. Additional replacement cost is charged once per counterparty.
All resulting settlement dates must fit the explicit horizon; otherwise the
method requires an extended scenario rather than silently omitting a tail.

Signed replacement marks create a replacement cash outlay when positive and an
inflow when negative. A negative net closeout is paid once. This accounting is
necessary when cash claims and swap liabilities legally offset. For held cash
100 and a replacement mark -60, net closeout is 40. Haircut-adjusted collateral
10 and 50% estate recovery imply credit loss 15. With additional replacement
cost 5, adjusted net cash is 80. If the replacement mark is -150, replacement
inflow 150 and net closeout payment 50 cancel the missing cash claim 100;
credit loss is zero and only the additional cost reduces cash to 95. For a
standalone replacement mark 100, the expense is booked once and subsequent
collateral/recovery offsets it; a separate second loss charge is never added.

Outputs explicitly reconcile baseline cash minus adjusted cash to credit loss
plus additional replacement cost. Positive-claim role attribution allocates a
netting set's resulting loss proportionately to its positive gross claims;
this is an allocation convention, not a second calculation of loss. Source
contracts and hashes remain in the artifact. Legal enforceability, derivative
prices and collateral title are supplied assumptions. Unaffected baseline
cash is preserved and source-separated recovery dates expose liquidity delays.
