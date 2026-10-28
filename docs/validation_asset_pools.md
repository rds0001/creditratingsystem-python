# Observed asset-pool validation

`asset_pool_backtest` reconciles a frozen `CreditAssetPool` and `AssetPoolPaths`
against independent actual servicing, ownership allocations, pool accounts and
tranche cash. Supply their original builder/simulator keyword arguments as
`pool_inputs` and `forecast_inputs`; both artifacts are replayed and reconciled.
Forecast scenario recoveries are never used as observed workout recoveries.

The common validation `design` binds model/version, data fingerprint, holdout
population, physical default target and evaluation cutoff. Additionally supply
`currency` and `forecast_known_at` no later than the pool origin. The forecast
calendar must end at the exact target horizon and use its PIT/TTC conditioning.
This validates supplied frozen scenario probabilities, not their estimation.

Each `servicing` row has unique `event_id`, `asset_id`, `date`, `known_at`, integer
intraday `sequence`, `currency`, `event_type`, `opening_principal`,
`closing_principal`, nullable `default_event_id`, and these nonnegative amounts:
`scheduled_principal`, `prepayment_principal`, `default_ead`, `dilution`,
`interest`, `recovery_gross`, `workout_cost`, `servicing_cost`.
Types are PAYMENT, PREPAYMENT, DEFAULT, RECOVERY, COST and DILUTION. Principal
is checked sequentially against the original whole-asset tape:

```
opening - scheduled principal - prepaid principal - default EAD - dilution = closing
```

Default is absorbing; full prepayment is a separate competing exit. Recoveries
and workout costs reference the prior default event. Their ownership follows
the claim at default, not a subsequently changed performing-asset allocation.
Post-default claim sales require a separate claim ledger and are outside this
interface. Late-known or post-horizon events are explicitly excluded.

`allocations` rows have `allocation_id`, `event_id`, `pool_id`, `share`,
`known_at` and a complete `amounts` map with the eight fields above. Amounts must
match whole-event amount times ownership share. Total ownership/allocation
cannot exceed one; duplicate claims and missing allocations are reported.
`ownership_history` optionally supplies dated assignments with the original
builder's assignment schema. Effective intervals are start-inclusive,
end-exclusive. Ownership is supplied evidence, not a legal enforceability opinion.

`cash_contract` supplies origin `as_of`, `known_at`, `currency`, `opening_cash`.
Each dated `pool_accounts` row contains `date`, `known_at`, `currency`,
`opening_cash`, `closing_cash`, `purchase_cash`, `sale_cash`, `external_injection`,
`external_distribution`, `fees_paid`, `tranche_interest_paid`,
`tranche_principal_paid`, `equity_paid`. Reconciliation checks opening cash plus
owned collections, sales and injections less owned costs and all disbursements.
Every owned servicing date needs an account row, including zero-cash defaults.
Purchase/sale consideration is supplied account evidence; transaction pricing
and legal transfer validity are not inferred from cash conservation.

`tranche_contracts` bind `tranche_id`, origin-known `known_at`, `currency`,
`notional`. `tranche_payments` have `payment_id`, `tranche_id`, `date`, `known_at`,
`currency`, `principal`, `interest`. Principal cannot exceed notional and daily
payment totals must match the pool account. This does **not** validate contractual
waterfall priority, coupons, deficiency ledgers or legal payment entitlement;
`waterfall_priority_validation` explicitly says so.

Each asset's `coverage` records `asset_id`, Boolean `complete`, `complete_through`,
`known_at`, `currency` and the exact five-field validation `target`. Receipt
cannot precede the declared coverage end. Whole horizons must mature before
binary metrics, including for early known defaults. Missing or incomplete
coverage is unknown, never a non-default. The caller's completeness declaration
is not independently authenticated. Cash/probability forecast comparisons also
require unchanged, static original holdings and valid ledgers. Dynamic or changed
holdings retain cash audits but return non-applicable forecast scores.

Metrics separate default and prepayment Brier/log-loss/calibration/discrimination,
actual owned cash, expected-minus-observed cash errors and tranche payments.
One observed pool path and weighted forecast scenarios do not supply empirical
confidence intervals. No-violation output is `NOT_ASSESSED`, not release approval.

An independent example defaults a whole claim of 100. Actual recovery is 30 and
workout cost 5, funding tranche principal 25. The forecast instead has default
probability .5 and modeled conditional recovery 40: expected recovery 20,
recovery bias -10 and default Brier .25. A separate prepayment of 100 plus
interest 5 funds principal 80, interest 5 and equity 20 without a default.

The dated [BCBS CRE40.93 transparency criterion](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/40/inforce/2022-01-01/published/2019-12-15)
motivates distinct principal, interest, prepayment, default and disbursement
records. The implementation uses independently derived accounting identities;
it does not certify compliance with that criterion or current regulation.
