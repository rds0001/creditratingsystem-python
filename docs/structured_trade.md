# Invoice payment, recourse and servicing paths

`simulate_trade_receivables(invoice_pool, scenario_contract, party_mapping,
party_default_paths, invoice_scenarios, seller_terms, servicer_terms,
shared_innovations, context=None)` evaluates an observed static invoice pool.
Each owned invoice has one zero-interest bullet obligation and a continuously
valid pool assignment. Both Python and native R return a hashed
`TradeReceivablePaths` with dated investor cash, shared default events and
separate credit, dilution, seller and operational-risk attribution.

The scenario contract supplies identity/version, unique `scenario_ids`,
normalized nonnegative `weights`, increasing risk `dates` after snapshot,
`currency`, `probability_measure='PHYSICAL'`, `sampling_design` and
`conditioning`. Supported sampling designs are `IID_UNWEIGHTED` (equal weights),
`FIXED_WEIGHTED_SCENARIOS` and `DEPENDENT_OR_QMC`. The caller supplies the
innovations; the method draws no random numbers. Optional context must match
physical currency/conditioning and the final risk-grid horizon in days.

`party_mapping` maps every participating `party_id` to one `default_group_id`,
covering debtors, sellers and servicers. The debtor mapping must match the frozen
pool. A legal identity cannot have independent defaults in different roles.
The complete `party_default_paths` grid has scenario, group, risk date,
`conditional_pd` and `default_channel`. Innovations have scenario, date,
`channel_id` and `uniform` in [0,1). Groups share an absorbing default event;
missing or unused innovation channels are rejected.

Each scenario/invoice row supplies `asset_id`, `payment_date`, `dilution_date`,
`dilution_fraction`, `debtor_recovery_fraction`, `debtor_recovery_date` and
`debtor_recovery_cost`. Payment cannot precede the original due date; early
settlement needs a separately modeled discount contract. An actual delayed
payment remains exposed to debtor default through the payment date. Default
occurs before same-day payment. Commercial dilution must precede that payment
or default and proportionately reduces the claim. Default recovery is applied
to the resulting claim. Full dilution creates no credit event, collection or
workout cost. All supplied cash dates must fit the declared horizon; omitted
future cash or unresolved claims are rejected rather than assigned a recovery.

Seller terms contain scenario, `seller_id`, `settlement_date`,
`performing_capacity`, `default_estate_capacity`, `default_recovery_fraction`,
and `recourse_scope` (`CREDIT_ONLY` or `CREDIT_AND_DILUTION`). Settlement follows
all affected debtor recoveries/payments. Nonrecourse invoices receive zero.
Residual eligible claims are aggregated by seller, multiplied by recovery
fraction if the shared seller default has occurred, and capped once by the
relevant capacity. Payment is allocated pro rata across claims and split between
credit and commercial dilution components. Seller payments go directly to the
investor, outside the servicer account; they do not insure servicing losses.

For two invoices of 100 with debtor default and 40% recovery, residual credit
claims total 120. Seller capacity 50 pays 25 per invoice, leaving credit loss
70 and investor cash 130. If the seller also defaults, its 50% recovery is
further capped by estate capacity 20, leaving loss 100. With-recourse therefore
does not remove debtor or joint seller risk.

Servicer terms contain scenario, `servicer_id`, `remittance_date`,
`default_commingling_recovery_fraction`, `default_commingling_capacity` and
`replacement_cost`. Remittance follows affected debtor cash collection. Before
failure, collections are held until remittance. If the servicer defaults before
or on remittance, this held cash forms one aggregate capped commingling claim.
Collections on or after failure redirect directly to the investor. Replacement
cost is charged once on failure, not per invoice. Two held collections totaling
200, recovery 50% and capacity 30 produce remittance 30 and commingling loss 170;
replacement cost 5 leaves net investor cash 25. Failure before collection instead
redirects the full 200 and leaves only the declared replacement cost.

Every scenario reconciles owned principal minus net credit loss, net commercial
dilution, commingling loss and workout/replacement costs to actual investor
cash. The method models initially performing existing invoices and explicit
in-horizon events. It does not infer legal recourse enforceability, future
origination, revolving acquisitions, early-payment discounts or a derivatives
pricing model.
