# Application evidence and conditional risk

The application family separates information available before a decision,
contract terms, selection into funded loans, default risk and economic household
contributions. Every function returns `CalcResult`; Python uses `.value` and
native R uses `$value`. Tables follow the data family's row-list/data.frame
adapters. All IDs are strings; dates are ISO `YYYY-MM-DD` and observation horizons
are calendar days. These APIs produce local evidence and calculations, not loan
approval or external lending actions.

## Frozen application information

`build_application_snapshot(applications, feature_records, term_events=None,
required_features=None)` returns one snapshot per application. Applications have
unique `application_id`, `customer_id`, `product_id`, `population_id`,
`application_at`, `information_cutoff`, and a `requested_terms` mapping. Repeated
applications by the same customer remain distinct. Application date cannot follow
information cutoff.

Features use a long table: `application_id`, `feature`, explicit `value` (possibly
missing), `observed_at`, `known_at`, `source_id`. Known time cannot precede
observation. For every feature, restrict both dates to the information cutoff,
then select latest observation and latest known revision. Equal-date ties raise
an ambiguity error, even when values happen to agree. Foreign application IDs
are rejected. The snapshot retains each selected source ID and both dates.

Term events contain `application_id`, `stage` (`offered`, `accepted`, `booked`),
`terms` mapping, `effective_at`, `known_at`. The same availability/revision rule
applies separately to each stage. Original requested terms never become later
booked terms. Stages without known evidence remain missing; no later limit or
price is inserted into the initial customer features. Optional `required_features`
returns a list of missing feature names but does not impute them.

Output contains identity and date fields, `features`, `feature_sources`, separate
`terms` and `term_sources`, `missing_features`, and `snapshot_hash`. The hash uses
the shared `cr-canonical-v1` contract and matches across Python and R. Outputs are
detached copies. They remain mutable in memory, but prediction and offer comparison
verify the signature and reject changes. Rebuilding a snapshot with later data is
an explicit new snapshot, not a revision of the old prediction evidence.

## Joint applications

`assemble_joint_application(application_id, parties, income_items,
obligation_items, currency)` accepts:

| Table | Required columns | Meaning |
|---|---|---|
| Parties | `party_id, role, liability_share` | Role is borrower, co_borrower or guarantor; optional household_id |
| Income items | `economic_id, party_id, amount_monthly, currency` | Recognized monthly income supplied by caller |
| Obligation items | `economic_id, party_id, amount_monthly, currency` | Monthly debt-service amount supplied by caller |

At least one borrower/co-borrower is required. Liability share lies in `[0,1]`;
shares are retained and are **not** assumed to sum to one because several parties
can be jointly liable for the entire debt. Amounts are finite and nonnegative;
all currencies match the declared currency. Income normalization, evidence
eligibility and debt repayment schedules must be calculated before this call.

Items with the same economic ID within each table describe the **total** shared
amount, not each party's share. Conflicting duplicates fail; consistent shared
income or debt is counted once. IDs are not deduplicated across the two distinct
income/debt tables. Guarantor-only resources and obligations are excluded from the
borrower budget and explicitly listed. Provided households and shared economic
items create dependency flags; identities are never inferred from addresses.

Output includes monthly income and debt service, the original legal parties,
unique economic contributions with their party IDs, excluded rows and dependency
flags. It does not average PDs or apply guarantee coverage to the applicant PD.

## Selection-aware application targets

`build_application_targets(applications, decisions, bookings, default_events,
exits, horizon_days, evaluation_cutoff, performance_origin="first_draw_at")`
uses the application table above and these event tables:

- Decisions: unique `decision_id`, `application_id`, `status`
  (`approved`, `rejected`, `pending`, `withdrawn`), `decided_at`, `known_at`;
  optional `accepted_at` requires `accepted_known_at`.
- Bookings: unique `application_id` and `facility_id`, `booked_at`, `known_at`;
  optional `first_draw_at` requires `first_draw_known_at`; optional `followup_end`
  and `outcome_available_at` pass to the data observation-window contract.
- Defaults: unique `event_id`, `facility_id`, `default_at`, `known_at`.
- Exits: `facility_id`, `exit_at`, `known_at`.

The latest decision effective/knowledge pair observable by evaluation cutoff is
selected; ties fail. A known booking requires known approval and acceptance. Acceptance attached to a
nonapproved decision is inconsistent and raises a controlled error.
Decisions cannot precede applications; booking cannot precede either occurrence
or knowledge of acceptance; first draw cannot precede booking. Applications with information cutoffs after the evaluation cutoff
are excluded by a controlled cohort-cutoff error, not silently admitted to the
historical population. This contract allows one booked facility per application;
multi-facility financing must use separate application-facility records upstream.

Selection statuses remain distinct: `pending_decision`, `pending`, `withdrawn`,
`rejected`, `approved_not_taken_up`, `accepted_not_booked`, `booked_not_drawn`,
`funded`. No known default of another facility substitutes for an unfunded loan.
Performance origin is explicitly either `first_draw_at` or `booked_at`; in the
latter contract, the booked facility is treated as the funded risk set even before
its first draw. This choice defines the target and must match prediction metadata.

Labels use `(origin, origin + horizon_days]`. The prediction information cutoff
stays separate from funding origin. First default by the inclusive horizon end
is 1. Complete follow-up without default is 0. Rejects, unfunded stages, incomplete
follow-up and pending ascertainment remain missing. Early repayment/exit is
censoring; same-day default wins. Prior-default facilities are excluded from the
first-default population. Default ascertainment must be complete through supplied
`followup_end`; a future `outcome_available_at` prevents premature zero labels.

Every row contains selection-stage booleans, facility ID, target, label_observed,
eligible, censored, calendar-mature flag, label_status, event ID, origin and
observation-window dates. Calendar maturity does not itself mean label observed.

## Applying an explicit model

`predict_application_risk(snapshot, model, target, calibrator=None, offset=None)`
uses an existing supported frozen classification artifact; it does not select a
model or fit features. Currently the shared classifier applies binary GLM
artifacts with identity preprocessing (already transformed numeric features).
Other artifacts receive the classifier's controlled unsupported-model error.

The model must additionally carry:

```python
model["application_contract"] = {
    "product_ids": ["consumer"],
    "population_ids": ["new_customer"],
    "performance_origin": "first_draw_at",
    "term_stage": "offered",
    "term_features": {"amount": "offer_amount"}
}
```

`term_features` maps term field to fitted feature name; names are unique and must
exist in fitted feature order. Use an explicit empty mapping when terms are not
model features. These feature names must be absent from customer features to
avoid overwriting evidence. The chosen term stage must be available. Required
customer features and term values must be finite numeric inputs; missing values
raise an error that identifies missing names. Imputation and encoding belong to
the frozen preprocessing contract upstream.

`target` must equal the complete fitted target and include the same
`performance_origin`; its probability measure must be `PHYSICAL`. Product and
population applicability are checked against explicit lists. Classification
validates event, horizon, feature order, link, offset presence and parameters.
An optional calibrator is checked against target and base model version. Isotonic
calibrators, which can otherwise consume arbitrary scores, require
`metadata.input_scale="probability"` to consume the model's PD here. Training and
calibration provenance do not establish future empirical performance.

Result fields include `pd`, `unadjusted_pd`, `calibrated`, snapshot/model IDs,
complete target, product/population, term stage and numerical feature coverage.
Coverage is feature completeness, not representativeness or approval eligibility.

## Offered-contract scenarios

`compare_offer_risk(snapshot, offers, model, target, support_bounds,
calibrator=None, lgd=None, offset=None)` requires an offered-term conditional model
with a nonempty term-feature map. Offers have unique `offer_id` and `terms` mapping.
Every varied term must be an actual model feature; no unmodelled amount, tenor or
price is silently ignored. Snapshot customer data remain fixed and are not
reloaded. Identical offers produce identical PDs.

`support_bounds` maps every term to `[lower, upper]`. Extrapolation is flagged by
term; predictions outside support remain visible. These marginal ranges are
user-supplied and do not establish multivariate support or causal identification.
No proportional PD adjustment is used: each offer goes through the actual fitted
conditional model and optional calibration.

Optional `lgd` is a fixed scalar in `[0,1]`; EL then equals
`PD * LGD * offered amount`. This requires an explicitly modelled `amount` and
`requested_terms.currency` in the snapshot, which is returned as EL currency.
Without LGD, expected_loss and currency remain missing. Dynamic EAD, term-specific
LGD, take-up and counterfactual causal effects require their separate models.
Offers are scenarios, not decisions or claims that larger loans cause a given PD
change.

## Controlled errors

Application-specific codes are `APPLICATION_SCHEMA`, `APPLICATION_VALUE`,
`APPLICATION_DATE_ORDER`, `APPLICATION_FOREIGN_KEY`, `APPLICATION_CURRENCY`,
`APPLICATION_AMBIGUOUS_FEATURE`, `APPLICATION_AMBIGUOUS_TERMS`,
`APPLICATION_AMBIGUOUS_ECONOMIC_ITEM`, `APPLICATION_AMBIGUOUS_DECISION`,
`APPLICATION_COHORT_CUTOFF`, `APPLICATION_SELECTION_CONFLICT`,
`APPLICATION_SNAPSHOT_INTEGRITY`, `APPLICATION_MODEL_CONTRACT`,
`APPLICATION_APPLICABILITY`, `APPLICATION_TARGET`, `APPLICATION_MISSING_TERMS`,
`APPLICATION_FEATURE_COLLISION`, `APPLICATION_MISSING_FEATURES`,
`APPLICATION_CALIBRATION_SCALE`, `APPLICATION_OFFER_MODEL`, `APPLICATION_SUPPORT`.
Data, core, classification and calibration failures preserve their own controlled
codes. Inputs are in-memory small/medium tables, with explicit scans rather than
an indexed event store.
