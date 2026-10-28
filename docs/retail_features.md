# Point-in-time retail data and economic relationships

The retail feature APIs consume local user-supplied records. Economic observation time and information availability are separate. All dates are ISO calendar dates; `information_cutoff <= as_of`. Results contain dated contexts, explicit missing values and source or release lineage. No function fetches external customer data or infers identity from a name or address.

## Relational validation

```python
validate_retail_dataset(tables, data_contract, *, as_of,
                        information_cutoff, context=None)
```

`tables` is a mapping from table name to row records (or data frames). The returned `CalcResult.value` contains `valid`, `applicability`, row-level `violations`, `row_counts`, zero-based `eligible_row_indices` and `source_coverage`. Invalid records are reported without being silently repaired. An invalid contract raises a controlled error.

The versioned `data_contract` has `schema_version="1.0"`, nonempty `contract_id` and `version`, `table_specs`, `prohibited_feature_fields` and `source_coverage`. Every table specification contains:

| Field | Contract |
|---|---|
| `table` | Explicit table name |
| `role` | `IDENTITY`, `FEATURE`, `OUTCOME` or `DECISION` |
| `primary_key` | Nonempty array of field names; an application ID should identify an application, not its customer |
| `fields` | Records with `name`, `type` and boolean `nullable`; types are `string`, `date`, `number`, `integer`, `boolean` |
| `foreign_keys` | Records with child `field`, parent `table`, and `target_field` |
| `date_order` | Pairs `[earlier_field,later_field]` of declared date fields |
| `event_field`, `known_field` | Date fields determining economic and information eligibility |
| `feature_fields` | Explicit model-input field names; permitted only on a `FEATURE` table |
| `allow_extra_fields` | Explicit boolean permitting or rejecting undeclared columns |

Each source coverage record has a unique `source_id`, `status` (`COMPLETE`, `PARTIAL`, `UNKNOWN`) and `known_at`. A future coverage assessment appears as `UNKNOWN` at the current cutoff. Missing income and unknown bureau coverage remain missing; neither becomes zero. Common direct outcome names (`target`, `outcome`, `label`, `default_in_horizon`, `future_default`, `lgd_target`, `recovery_target`) and the user-supplied prohibited fields cannot be designated as features. This is an explicit semantic contract, not an automated proof that arbitrary derived variables are free of leakage or appropriate for lending.

Foreign keys are checked against supplied records and, for an eligible child, against parents already eligible at the cutoff. Repeated applications for the same customer are valid when application IDs differ. A reversal can declare a self-referencing payment foreign key, preserving traceability to the original payment. Economic payment allocation and reversal amounts are handled by the payment-history API and `behavioral.compute_days_past_due`; structural validation does not create default labels.

## Dated customer and household graph

```python
build_customer_household_graph(entities, relationships, *,
                               as_of, information_cutoff, context=None)
```

An entity record contains unique `entity_id`, `entity_type` and `known_at`. Types are `CUSTOMER`, `HOUSEHOLD`, `APPLICATION`, `FACILITY`, `BUSINESS`. IDs are explicit and shared across the supplied records; no personal identifying text is required.

Relationship records contain `relationship_id`, positive integer `release_seq`, `source_id`, `target_id`, `role`, `effective_from`, nullable `effective_until`, `known_at`, nullable `liability_fraction`, nullable `budget_share`, and nonempty `evidence_reference`. The supported roles run from customer to household (`MEMBER_OF`), application (`COAPPLICANT_ON`), facility (`BORROWER_ON`, `GUARANTOR_ON`), or business (`OWNER_OF`). Ownership here is a typed link, not an automatic debt guarantee or copied owner PD.

For each logical relationship ID, the latest cutoff-known `(known_at,release_seq)` is selected first. The selected release must then be effective in `[effective_from,effective_until)`. Later corrections cannot rewrite earlier information states. Duplicate active customer/target/role edges and multiple active economic households for the same customer are rejected. This graph deliberately represents one selected economic household per customer at a snapshot; separate analyses can use different explicit household definitions.

Debt fractions are used only for borrower/guarantor roles. Each fraction is in `[0,1]`, or null when unknown. Internal borrower budget shares for one facility cannot sum above one. Legal liability shares can each be one for jointly and severally liable borrowers. The graph does not infer joint liability, enforceability, cross-default or legal default-unit identity.

The content-bound `RETAIL_RELATIONSHIP_GRAPH` contains visible entities, selected edges, household memberships and customer/facility liability paths, together with its cutoff and SHA-256 binding. Excluded and superseded releases remain in diagnostics. Downstream aggregation rejects a changed graph whose binding no longer matches.

## Household obligations

```python
aggregate_household_obligations(
    obligations, household_graph, coverage, *, currency, period,
    guarantee_budget_policy, context=None,
)
```

Obligation snapshots contain `facility_id`, positive integer `release_seq`, `observed_at`, `known_at`, `source_id`, `debt_stock`, `scheduled_debt_service`, `currency`, `period`, and `status`. Amounts are nonnegative or explicitly null. Status is `ACTIVE`, `CLOSED` or `UNKNOWN`; closed facilities require explicit zero residual amounts. A snapshot must be observed before it is known. Latest-known releases are selected per facility. The graph supplies the independent customer-to-facility links, so the same shared facility does not need to be repeated for each borrower. Distinct creditors/facilities retain distinct IDs.

`coverage` contains `status`, named `sources` and `known_at`. Complete coverage requires at least one source, and all supplied obligation sources must be declared. Currency and debt-service period must match exactly; there is no implicit FX conversion or monthly/annual scaling. Snapshot age and selected release remain visible.

`guarantee_budget_policy` must be `exclude` or `include_declared_share`. The latter includes guarantees only using the explicitly supplied budget shares; it does not infer that a contingent guarantee has become a current repayment obligation. Person-level legal liability is always reported separately.

For two household members borrowing the same 100-unit facility with monthly debt service 10, legal fractions one each and budget shares one half each, the household linked debt is 100, its allocated budget debt is 100, its allocated service is 10, and each person's legal debt is 100. Those are alternative views, **not additive exposures**. If only one half-budget borrower belongs to the household, linked debt remains 100 while its allocated budget debt is 50. A guarantee or ownership link never turns collateral sufficiency into a nondefault classification.

Outputs distinguish known linked and budget subtotals from totals. `PARTIAL` or `UNKNOWN` coverage, missing snapshots, unknown status or missing amounts leave totals null. Missing budget shares separately prevent a total budget estimate. A known subtotal of zero with unknown coverage does not mean a debt-free customer. No risk score, approval or empirical validation follows from this arithmetic.

## Minimal relationship example

```python
from creditratingsystem.retail_features import build_customer_household_graph

entities = [dict(entity_id=k, entity_type=t, known_at="2024-01-01")
            for k, t in [("person", "CUSTOMER"), ("house", "HOUSEHOLD")]]
relationships = [dict(relationship_id="membership", release_seq=1,
    source_id="person", target_id="house", role="MEMBER_OF",
    effective_from="2024-01-01", effective_until=None, known_at="2024-01-01",
    liability_fraction=None, budget_share=None,
    evidence_reference="locally verified household membership")]
graph = build_customer_household_graph(entities, relationships,
    as_of="2024-01-31", information_cutoff="2024-01-31")
print(graph.value["memberships"])
```

These data contracts are library conventions. The [Federal Reserve interagency statement](https://www.federalreserve.gov/newsevents/pressreleases/bcreg20191203b.htm) identifies account cashflow data as an alternative data source; it does not prescribe these formulas or certify their suitability. Historical evidence, feature eligibility, data coverage and eventual predictive validation remain distinct.

## Bureau, thin-file and income evidence

`build_bureau_features(snapshots, tradelines, inquiries, provider_mapping, *,
customer_id, provider_id, as_of, information_cutoff, lookback_days,
max_age_days, currency, context=None)` describes one supplied provider file.
The versioned mapping declares `mapping_id`, `version`, `known_at` and three
canonical-to-source field maps named `snapshots`, `tradelines`, `inquiries`.
An empty map means identity. Every canonical field must exist; missing numeric
values must be explicit `None` (`NULL` in R).

* Snapshot fields: `snapshot_id`, `release_seq`, `customer_id`, `provider_id`,
  `observed_at`, `reported_at`, `known_at`, `coverage_status`, `currency`.
  Coverage is `COMPLETE`, `PARTIAL` or `NO_HIT`. Dates follow observation,
  reporting, then local receipt. Choose the latest receipt-visible revision per
  snapshot, then the newest observed snapshot.
* Tradeline fields: `snapshot_id`, **`snapshot_release_seq`**, `tradeline_id`,
  `opened_at`, nullable `closed_at`, `status` (`OPEN`, `CLOSED`, `UNKNOWN`),
  nullable nonnegative `balance`, `limit`, integer `max_dpd`. Lines belong to the
  exact snapshot revision; a later correction cannot overwrite an earlier file.
  Closure is an actual dated event, not a missing open record.
* Inquiry fields: `inquiry_id`, `release_seq`, `customer_id`, `provider_id`,
  `occurred_at`, `known_at`, `kind` (`HARD`, `SOFT`, `UNKNOWN`). Counts cover
  `(as_of-lookback_days, as_of]` from the supplied local history; they do not
  independently establish that all inquiries were supplied.

The result distinguishes `ABSENT`, `NO_HIT`, `STALE`, `CURRENT`, age and coverage.
Open balances and limits, reported closed balances, known subtotals, nullable
complete-file totals and utilization remain separate. A zero limit gives
undefined utilization, not zero utilization. No-hit is not zero household debt.
Maximum reported DPD reflects the supplied tradeline history and is not recast
as a lookback-window maximum. The method does not reproduce a provider score.

`build_thin_file_features(bureau_features, evidence, eligibility_profile, *,
customer_id, as_of, information_cutoff, context=None)` requires the same bureau
customer and date snapshot. Its dated profile declares `profile_id`, `version`,
`known_at`, `minimum_tradelines`, `minimum_history_days`,
`maximum_evidence_age_days`, `allow_partial_evidence`, and a
`feature_requirements` mapping from feature names to lists of required evidence
kinds. Other evidence rows have `evidence_id`, `release_seq`, `known_at`,
`customer_id`, `evidence_type`, `observed_from`, `observed_until`, `coverage_status`
(`COMPLETE`, `PARTIAL`, `UNKNOWN`). The latest known release and newest evidence
interval per kind determine availability. `BUREAU` is reserved for the supplied
bureau result. Output strata describe absent, no-hit, stale, thin or established
files; supported and unsupported features report missing evidence explicitly.
Thresholds are user policy. None of these strata is itself a credit-quality label.
Direct outcome names cannot be designated supported features.

`reconcile_income_evidence(declarations, documentary_evidence,
transaction_evidence, inclusion_policy, *, customer_id, as_of,
information_cutoff, currency, period, basis, context=None)` selects each stable
income component once. Each source row declares `evidence_id`, `release_seq`,
`known_at`, `customer_id`, `component_id`, `component_type`, `observed_from`,
`observed_until`, nonnegative `amount`, `basis` (`NET`/`GROSS`), `period`,
`currency`, `evidence_reference`. Evidence IDs are globally unique across these
three supplied tables; their revisions retain identity. Salary and bonus need
separate component IDs. The policy declares `policy_id`, `version`, `known_at`,
`source_priority` (each of `DECLARED`, `DOCUMENT`, `TRANSACTION` once),
`included_component_types`, `expected_component_ids`, `coverage_status`,
`maximum_age_days`, `allow_stale`, and `relative_discrepancy_threshold`.
Supported income types are `SALARY`, `BONUS`, `PENSION`, `RENT_NET`,
`NET_BUSINESS_INCOME`, `OTHER_NET_INCOME`.

The newest eligible evidence per source and component is reconciled by the
supplied priority. Document support is an asserted source, not an authenticity
verification. Source amounts and differences from the declaration remain visible;
a zero declaration has no relative discrepancy denominator. Gross/net, period
and currency mismatches are explicitly excluded, with no guessed conversion.
Business sales, borrowing, owner transfers and refunds are not disposable income.
Known selected income, current document-supported income, other selected income
and missing components remain separate. Only complete declared coverage without
missing expected components permits a total income; known subtotals remain
available otherwise.

These are explicit local evidence contracts rather than universal thresholds.
The [CFPB correction and update](https://www.consumerfinance.gov/data-research/research-reports/technical-correction-and-update-to-the-cfpbs-credit-invisibles-estimate/)
distinguishes absent records from unscored records; no population estimates or
provider scoring rules are imported into these functions (reviewed 2026-10-13).

## Payment histories and revolving accounts

`build_payment_history_features(obligations, payments, coverage, *, account_id,
as_of, information_cutoff, lookback_days, allocation_policy, currency,
grace_days, context=None)` replays a single account's contractual ledger daily
on `(as_of-lookback_days, as_of]`. Obligation and payment/reversal fields and
FIFO/explicit allocation are those of `compute_days_past_due`. Supply the complete
historical ledger, including pre-window unpaid obligations and prepayments;
truncating input records at the feature window gives a different and generally
incorrect ledger. At each date the replay uses the earlier of that date and the
feature information cutoff. Thus later-known reversals never rewrite earlier
point-in-time states. No materiality threshold or default classification is added.

Coverage declares `status` (`COMPLETE`, `PARTIAL`, `UNKNOWN`), `known_at`,
`observed_from`, `complete_through`. It is an explicit assertion about supplied
history. Complete features require coverage across the whole requested window;
otherwise only `known_history_features` are available and `features` are null.
Daily DPD and unpaid amounts, arrears-day counts, mean/max DPD and transitions
between descriptive `CURRENT`/`ARREARS` states are returned. An unpaid amount on
its due date has zero DPD and becomes arrears on the following day.

Each obligation due inside the window is assessed at `due_at + grace_days` using
knowledge then available. Zero, partial and full allocations remain distinct.
The assessment payment ratio is total allocated amount divided by assessed
obligation amount; zero denominators are undefined. Not-yet-elapsed grace and
obligations not contemporaneously known are reported as unassessable, not missed
payments. Grace changes only assessment timing, never the contractual due date
or DPD. A subsequent reversal can change later daily arrears while preserving the
original point-in-time assessment.

`build_revolving_features(balance_history, statements, coverage, *, account_id,
as_of, information_cutoff, lookback_days, currency, context=None)` uses explicit
daily balance snapshots and statement repayment evidence. Balance rows declare
`snapshot_id`, `release_seq`, `account_id`, `observed_at`, `known_at`, nonnegative
`balance`, `credit_limit`, `activity_status` (`ACTIVE`, `INACTIVE`, `CLOSED`) and
`currency`. After latest-known revision selection there must be one resolved row
per day. A closed facility must have zero balance and limit. Inactivity is never
inferred from an undrawn active facility. Missing days are not filled forward;
complete overlimit-day totals require every calendar date and complete coverage.

Statements declare `statement_id`, `release_seq`, `account_id`, `issued_at`,
`due_at`, `observed_at`, `known_at`, `statement_balance`, `minimum_due`,
`paid_by_due`, `currency`. `paid_by_due` means payments allocated by the contractual
due date, not all payments eventually received. The observation must extend to
the due date to assess repayment. Below-minimum, minimum-without-full, full, and
zero-amount-due states remain separate. Repayment ratios have explicit nullable
zero denominators. These are observed statement counts, not claims that unseen
statements were supplied.

For consecutive observed balances, utilization change is decomposed as
`(new_balance-old_balance)/new_limit` plus
`old_utilization*(old_limit-new_limit)/new_limit`. The first describes net balance
movement; it does not identify gross draws. Thus a limit cut can increase
utilization without new borrowing. Either zero limit makes the decomposition
undefined. Trend is the endpoint utilization change per elapsed calendar day;
the observed utilization mean does not pretend missing days were observed.

## Booked transactions and liquidity observations

`build_transaction_features(transactions, daily_balances, category_mapping,
coverage, *, customer_id, own_account_ids, as_of, information_cutoff,
lookback_days, currency, context=None)` uses the supplied own-account set as an
explicit ownership assertion. Coverage has the dated historical fields used by
payment features. The local category mapping declares `mapping_id`, `version`,
`known_at`, `categories` (local code to canonical category).

Transactions declare `transaction_id`, `release_seq`, `account_id`, `posted_at`,
`value_at`, `known_at`, signed `amount`, `currency`, `kind` (`ORIGINAL` or
`REVERSAL`), `category_code`. Positive amounts are credits. Reversals additionally
identify `reverses_transaction_id`; the original must be an earlier visible
same-account posting of the opposite sign. Partial reversals may not cumulatively
exceed the original. Supply original records outside the observation window when
needed for reversal lineage. Reversals inherit the original classification and
retain their own posting date. Thus a purchase and reversal cancel within a common
window; a reversal of a pre-window purchase remains an actual current credit,
not a fabricated current purchase. Value dates remain visible, while the feature
window consistently uses posting dates.

Canonical categories are `SALARY`, `OTHER_INCOME`, `BUSINESS_REVENUE`,
`CONSUMPTION`, `DEBT_SERVICE`, `BORROWING`, `REFUND`, `OWN_TRANSFER`, `OWNER_DRAW`,
`TAX`, `UNKNOWN`. Missing mappings remain `UNKNOWN`; their absolute volume and
fraction are reported. Category totals separate credits, debits and signed net
amounts. Classified household income includes only salary and other income;
classified expense comprises consumption, debt service and tax. Those are
classified subtotals, not assertions that unknown cashflows are not income or
expense. Sales, borrowing, refunds and owner draws are not household income.

Own transfers require `transfer_id` and `counterparty_own_account_id`. Two supplied
legs must be opposite, same-currency movements between the explicitly owned
accounts. A single visible leg is reported as unmatched, not reclassified as
income. Own transfers and owner draws are excluded from external-flow statistics.
Daily external net cashflow and its population standard deviation include zero
posting days; complete statistics require asserted complete history. Mean and
variance calculations are scaled to avoid unnecessary squaring overflow.

Daily balances declare `balance_id`, `release_seq`, `account_id`, `observed_at`,
`known_at`, signed `closing_balance`, `currency`. One resolved observation per
account-day is required. Overdraft account-days count negative accounts separately;
a positive balance on another account does not erase them. Household buffer
observations require all declared accounts on that date. Missing balance days
are not carried forward and do not become zero overdraft days. Complete overdraft
totals require both the full account-day grid and complete source coverage.

## Purchase plans and BNPL obligations

`build_bnpl_obligation_features(plans, installments, events, provider_coverage, *,
customer_id, as_of, information_cutoff, lookback_days, near_term_days, currency,
context=None)` aggregates explicit schedules and allocated financial events.
Provider coverage declares `status` (`COMPLETE`, `PARTIAL`, `UNKNOWN`), `sources`
(provider IDs), `known_at`; unseen foreign-provider plans are not inferred absent.

Plans declare `plan_id`, `release_seq`, `customer_id`, `provider_id`,
`purchased_at`, `known_at`, `currency`, nullable actual `closed_at`, nullable
`refinances_plan_id`, and boolean `schedule_complete`. Installments declare
`installment_id`, `plan_id`, `release_seq`, `known_at`, `due_at`, nonnegative
`amount`, `currency`. Schedule completeness is explicit, not inferred from one
visible installment. A plan with no supplied installments cannot establish zero
debt. Explicit refinancing must reference an earlier known plan; repeated
purchases alone are not refinancing.

Events declare `event_id`, `release_seq`, `known_at`, `occurred_at`, `plan_id`,
nullable `installment_id`, `kind`, nonnegative `amount`, `currency`, nullable
`related_event_id`. Financial kinds are `PAYMENT`, `MERCHANT_REFUND`,
`PAYMENT_REVERSAL`, `REFUND_REVERSAL`. They must identify the affected installment;
there is no guessed FIFO allocation. Reversals identify an earlier same-installment
original and cannot exceed it cumulatively. Net payments plus merchant refunds
cannot exceed contractual installment amounts. Excess cash returned to a customer
belongs in separate cashflows. Financial events after actual plan closure require
a corrected closure or a separate cashflow record.

`DISPUTE_OPEN` and `DISPUTE_CLOSE` have zero financial amount; closure identifies
one earlier opening on the same plan. A dispute does not automatically cancel or
suspend a contractual obligation and does not imply default. Refunds reduce only
their explicitly allocated installment. Closed plans must have no residual known
unpaid installments.

The result distinguishes overdue, due-today and next-`near_term_days` obligations,
with known future cashflow dates; the near-term window is `(as_of, as_of+days]`.
Plan and provider concurrency uses actual half-open purchase/closure intervals
visible at the feature cutoff. This is a retrospective known-history description,
not the contemporaneous-knowledge replay used for payment DPD. Actual closure is
not guessed from scheduled maturity or a zero observed balance. Active contracts
and plans with positive outstanding amounts therefore have separate counts.
Total debt requires complete provider coverage and complete supplied schedules;
known subtotals remain available otherwise.

The [CFPB January 2025 research report, section V](https://files.consumerfinance.gov/f/documents/cfpb_BNPL_Report_2025_01.pdf)
motivates explicit cross-provider overlap and coverage reporting (reviewed
2026-10-13). The implementation uses actual supplied dates rather than the
report's research proxy for overlap, and imports no population risk estimates.

## Small-business and household interactions

`build_retail_sme_features(business_cashflows, owner_drawings,
household_obligations, liability_links, coverage, *, business_id,
owner_customer_ids, business_debt_facility_ids, household_debt_service_window,
as_of, information_cutoff, lookback_days, currency, context=None)` preserves
business and household cash identities. `coverage` is the dated history contract
used by transaction features. `household_obligations` must be the full result of
`aggregate_household_obligations` (a CalcResult or its complete dictionary), with
matching dates and currency. The explicit service-window declaration has `from`,
`through`, `period`: dates must equal the cashflow window and period must match
the household result. No monthly/annual conversion is guessed.

Business cashflows declare `transaction_id`, `release_seq`, `business_id`,
`posted_at`, `known_at`, signed `amount`, `category`, `currency`. Categories are
`SALES`, `OPERATING_COST`, `TAX`, `CAPEX`, `DEBT_SERVICE`, `FINANCING`,
`OWNER_DRAW`, `OWNER_CONTRIBUTION`. Sales additionally identify `counterparty_id`;
debt service identifies a declared business `facility_id`. Owner draws are debits
and contributions credits. Other categories use signed net cash amounts, allowing
an explicitly classified expense reimbursement or sales return.

Operating cash is sales plus signed operating costs and tax. Capex produces cash
before business debt; debt service produces cash after business debt. Financing,
owner drawings and contributions are excluded from this cash-capacity sequence.
Business sales are not household income, and an owner draw is not additional
consolidated income.

Drawing rows declare `drawing_id`, `release_seq`, `known_at`,
`business_transaction_id`, `household_transaction_id`, `owner_customer_id`,
`business_posted_at`, `household_posted_at`, `business_amount`, `household_amount`,
`currency`. Business evidence must match an actual business owner-draw record;
legs must have opposite amounts and are matched once. Dates may differ but neither
leg may anticipate receipt. Household legs are supplied local evidence, not
independently obtained bank records. Unmatched business drawings remain visible
and excluded from operating cash.

Owner households are joined by explicit customer IDs and counted once per
household. A shared business/household facility ID suppresses combined debt
subtraction to prevent duplicate service. Missing households or uncertain totals
also leave the combined figure undefined. Cash after both business and household
debt excludes household living costs and is **not** an affordability or
distributable-income measure.

Liability rows declare `liability_id`, `release_seq`, `known_at`, `business_id`,
`customer_id`, `liability_type` (`LIMITED_OWNERSHIP`, `PERSONAL_GUARANTEE`,
`UNLIMITED_LIABILITY`), `effective_from`, nullable `effective_until`, nullable
`liability_fraction` in `[0,1]`, nullable nonnegative `maximum_amount`, and
`evidence_reference`. Active half-open intervals are retained as separate legal
evidence. Ownership never automatically becomes a personal guarantee or an owner
PD copied to the business.

Net sales by counterparty produce a Herfindahl share-squared concentration index
only for positive total sales and nonnegative customer net amounts. Calendar-month
net sales (including zero-sales months within the window) and peak-month share
are descriptive observations, not an estimated seasonal forecast. Negative net
customer/month amounts make the corresponding share statistic undefined.

## Frozen retail pools and local parameter estimation

`fit_retail_risk_pools(observations, pooling_spec, *, target, training, model_id,
model_version, context=None)` returns a `FitResult` with a content-bound
`ModelArtifact`. The algorithm is deterministic greedy adjacent merging. It is
not a clustering black box and makes no global optimality or feasibility claim.
Its complete merge path and unresolved constraints are retained.

The pooling specification declares `spec_id`, `version`, `known_at`,
`stratum_fields`, `score_field`, `segments`, `minimum_cases`, `minimum_defaults`,
`minimum_lgd_count`, `minimum_ead_count`, and `maximum_bin_mean_spread` with
separate nullable `pd`, `lgd`, `ead` thresholds. Stratum fields must explicitly
include `product`, `borrower_type`, `facility_type`, `delinquency_state` and may
include other declared non-outcome fields. Each segment has `segment_id`, exact
`values` for every stratum, finite `score_min`/`score_max`, strictly increasing
interior `cutpoints`, and separate `monotonicity` directions (`none`, `increasing`,
`decreasing`) for PD, LGD and EAD. A zero minimum permits no evidence for that
conditional parameter, whose diagnostics then remain undefined. A null spread
threshold explicitly disables that parameter's merge restriction.

Training metadata declares `role='TRAIN'`, `known_at`, `split_id`,
`data_fingerprint`, and optional `data_origin`. Training rows declare:

* unique `observation_id` and `default_unit_id`, `role='TRAIN'`, `index_date`,
  `feature_observed_at`, `feature_known_at`, score and all stratum values;
* `label_window_end`, `label_known_at`, boolean `defaulted`, nullable
  `default_date`, `lgd_status` (`CLOSED`, `OPEN`, `NOT_DEFAULTED`);
* explicitly nullable `lgd`, `ead`, `lgd_known_at`, `ead_known_at`.

Features must be observed and known by the index. The whole label horizon must
mature and be known by the training cutoff. Nondefault rows cannot carry LGD/EAD
outcomes. LGD is an observed closed-workout outcome; open-workout final LGD remains
null. Conditional outcomes have their own knowledge dates, between actual default
and the training cutoff. A missing numeric outcome has a null knowledge date.
The target declares `event='default'`, positive integer `horizon`, `time_unit='days'`,
`probability_measure='PHYSICAL'`, `conditioning='not_defaulted_at_index'`,
`default_unit` (`facility` or `obligor`), `default_definition`, and `currency`.
Default dates lie in `(index_date, label_window_end]`.

Within each declared segment, original score bins use left-closed/right-open
intervals with the final domain endpoint included. Merging first addresses
minimum support. Among admissible neighboring pairs touching an undersupported
block, it chooses the smallest sum of squared normalized differences in available
PD/LGD/EAD means, breaking ties by the leftmost bin. Once no support merge is
available, it considers neighboring monotonicity violations by the same rule.
A candidate is admissible only if every enabled spread constraint holds across
its original nonempty-bin parameter means. The algorithm repeats until no
admissible required merge remains. This local rule can coarsen more than a globally
optimized partition; the result does not prove global impossibility if constraints
remain unmet.

PD case/default counts, closed-LGD and EAD counts, separate means, within-outcome
population standard deviations, and original-bin mean spreads remain visible.
The spread restriction is a descriptive between-bin constraint, not a statistical
proof of homogeneity. Any remaining support or monotonicity violation leaves the
whole affected segment `INSUFFICIENT_SUPPORT` and unassignable. There is no global
monotone LTV assumption or automatic pooling across products/delinquency states.
Pool IDs contain the model ID/version, ruleset fingerprint prefix, segment and
merged original-bin interval. The artifact also binds the actual training inputs.

`assign_retail_risk_pool(observations, fitted_model, *, as_of,
information_cutoff, expected_target, uncovered_policy, context=None)` uses the
frozen model without refitting. Snapshot rows declare `observation_id`,
`index_date`, `feature_observed_at`, `feature_known_at`, strata and score. Model
availability, feature chronology and exact target agreement are checked. The
policy is `unassigned` or `error`; missing inputs, unseen strata, out-of-domain
scores and unsupported segments never become a good pool by default. Output
includes matched strata, original score bin, merged interval and bound
`RETAIL_POOL_ASSIGNMENTS` artifact. Delinquency changes can legitimately change
an assignment. Python and R replay the same frozen artifact exactly.

`estimate_pool_risk_parameters(pool_assignments, cohort_outcomes, workouts,
estimation_profile, *, evaluation_cutoff, calibration_id, calibration_version,
context=None)` returns a bound `CalibrationArtifact` with type
`RETAIL_POOL_RISK_PARAMETERS`, its base model identity, pool risk table and
aggregate mature-unit/default counts. This is a parameter table, not a score
transformation to pass to `apply_calibrator`.

Cohort outcomes declare unique `observation_id`, `default_unit_id`, `index_date`,
`label_window_end`, `label_known_at`, `outcome_status` (`MATURE`, `CENSORED`,
`OPEN_HORIZON`), nullable `defaulted`/`default_date`, nonnegative
`reference_exposure`, and `reference_known_at`. Exposure weights must be frozen by
the index. Only full-horizon mature, cutoff-known, covered outcomes contribute;
missing, censored and unassigned observations are reported separately rather than
imputed nondefault. One row per declared default unit is required. Any borrower
with multiple facilities must first follow the user's declared default-unit rule.

Workout rows declare `episode_id`, `observation_id`, `default_date`,
`default_known_at`, `ead_at_default`, `episode_status` (`OPEN`/`CLOSED`),
`valuation_known_at`, `reference_date` (the cohort index),
`reference_known_at`, `reference_drawn`, `reference_limit`, `recovery_cashflows`,
`discount_curve`, `fx_contract`. The last three use the explicit contracts of
`discount_recoveries`. Recoveries are valued at the supplied workout valuation
knowledge date, which cannot exceed evaluation cutoff. `workout_lgd` preserves
raw economic LGD and open-workout incompleteness; `realized_ccf` preserves raw
CCF and undefined zero-undrawn denominators. Default conditioning and dates must
match the cohort. Each observation contributes at most one workout episode.

The dated estimation profile declares `profile_id`, `version`, `known_at`,
`pd_weighting` (`default_unit`/`reference_exposure`), `lgd_weighting`
(`episode`/`ead`), `ead_weighting` (`episode`/`reference_exposure`), `ccf_weighting`
(`episode`/`undrawn`), `confidence`, `minimum_pool_cases`, and
`independence_assumption='independent_default_units'`.

Table status describes only mature-cohort support; separate parameter statuses
identify absent/zero-weight PD, closed LGD, EAD and CCF samples.
Every weighted estimate reports its numerator, denominator, positive-weight count,
observed range and an independent-episode weighted-mean standard error (undefined
for fewer than two contributing episodes). Those errors do not resolve selection,
censoring or dependence. Closed-workout LGD, open-workout counts, zero-EAD
closed workouts, missing default workouts and undefined CCF are separate. EAD and
CCF condition on observed defaults with available records. Sparse pools retain
descriptive estimates under an explicit insufficient-support status.

Unweighted PD uses `calibrate_grade_pd` without monotonic recalibration or priors
for exact binomial intervals, plus `low_default_bound` for a one-sided independent
binomial upper bound. These intervals concern unit counts, not exposure-weighted
PD. They are conditional on the selected pools and do not account for pool-fit
uncertainty. Local observed PD is not automatically TTC or long-run PD; the
separate long-run calibration routines and representativeness assessments remain
necessary for that target.

[BCBS CRE36.16, .17 and .24](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/36/inforce/2023-01-01/published/2022-12-08)
provide the conceptual reference for separate borrower/transaction/delinquency
characteristics and pool-level PD/LGD/EAD assessment (reviewed 2026-10-13).
These explicit statistical procedures do not establish regulatory approval or
complete compliance for a bank's data and rating system.
