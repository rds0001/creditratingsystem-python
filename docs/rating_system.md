# Local rating objects, model versions and replay

These eleven functions operate on supplied local data and return `CalcResult`.
Python uses `.value`, native R `$value`. A registry is a returned object, not a
server or a mutable global store. Nothing approves loans, contacts a service,
changes accounts or executes code from a stored artifact.

Objects use schema version `1.0`. Dates are ISO `YYYY-MM-DD`, with economic
`as_of <= known_at`. Effective intervals are **[effective_from, effective_to)**;
missing `effective_to` means no declared expiry. Knowledge of a future-effective
rule is allowed; use before its effective date is not. Date equality is inclusive
for knowledge. These are calendar-date contracts, not intraday ordering.

All returned content is copied. Registries, assessments, batches and event states
carry the shared canonical SHA-256 hash. Changing a returned object invalidates
that hash; functions consuming it check integrity. A hash binds content and does
not authenticate its author. Python objects are dictionaries/lists; native R uses
named lists for objects, unnamed lists for arrays (including singletons), and
`structure(list(), names=character())` for an empty object. R data.frame row tables
are also accepted. Do not serialize executable objects or callbacks.

## Subjects and relationships

`build_rating_subjects(tables, links, as_of, known_at)` accepts tables keyed by
`customer`, `obligor`, `facility`, `issue`, `transaction`, or `portfolio`. Each row
has `subject_id`, `version`, `attributes` (object), `effective_from`, `effective_to`
and `known_at`. Subject type comes from the table name and must not be duplicated
in the row. IDs are unique within type and version. More than one effective
version for an ID produces `OVERLAPPING_VERSIONS` and excludes that ambiguous
subject from the resolved graph; a later stored version does not silently win.

Links have unique `link_id`, `relation`, `from_type`, `from_id`, `to_type`, `to_id`
and the same effective/knowledge dates. Allowed relations are `legal_obligor`,
`joint_liability`, `economic_dependency`, `default_correlation`, `data_identity`,
`contains`, and `pd_reference`. Endpoints must resolve to selected typed subjects;
unresolved links are returned separately. This graph reports relationships. It
never merges parties, copies a grade, or infers shared PD from a customer ID.

## Routing and data applicability

`route_rating_profile(snapshot, rules, as_of, known_at)` uses a subject snapshot:

```python
snapshot = {
    "subject_id": "O1", "subject_type": "obligor",
    "as_of": "2024-01-01", "known_at": "2024-01-02",
    "attributes": {
        "income": {"value": 2000, "observed_at": "2023-12-31",
                   "known_at": "2024-01-01"}
    }
}
```

The snapshot economic date must equal the requested economic date, and its
knowledge date cannot exceed the request. Every attribute has an explicit value
(possibly null), observation date and knowledge date. Future observations or
knowledge after the snapshot cutoff are excluded and reported as `UNAVAILABLE`.
`subject_id` and `subject_type` are reserved identity fields usable in conditions.

Rules have `rule_id`, `profile_id`, `profile_version`, finite numeric `priority`,
`conditions`, and effective/knowledge dates. Each condition is
`{field, operator, value}`. Operators: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `in`,
`not_in`, `present`, `absent`. The last two need no value; membership requires an
array; ordered comparisons require finite numbers. Missing values fail every
comparison except `absent`. Conditions are combined with AND; an empty list is
an explicit unconditional rule. There is no expression evaluation.

The smallest matching priority wins. Equal best priority yields `AMBIGUOUS`,
including when rules name the same profile. No match yields `NO_ROUTE`. A unique
match yields `ROUTED` with profile and rule identity. The full deterministic trace
contains eligibility and individual condition results; there is no fallback to
an unrequested method.

`assess_rating_applicability(snapshot, contract, as_of, known_at)` uses the same
snapshot. Contract fields are `contract_id`, `version`, `required_features` (array
of distinct names), `allowed_values` (field to nonempty array), `support_bounds`
(field to inclusive numeric [lower, upper]), and `maximum_age_days` (field to
nonnegative integer). Empty maps are explicit. A fact can also declare
`conflict=true`. Missing required/selection information, unavailable facts, stale
facts, source conflicts and values outside empirical support yield `REFER` with
findings. A known disallowed product/purpose/population field yields
`NOT_APPLICABLE`. Otherwise the result is `APPLICABLE`. These statuses assess
model use; they are neither default labels nor credit decisions.

## Registering and selecting model versions

`register_rating_model_version(registry, bundle, registration)` starts with
`registry=None` / `NULL`. A bundle has `model` (shared frozen ModelArtifact),
`dependencies` and `configuration` objects. Put dependency versions/hashes and
configuration evidence in these explicit maps; they are included in the bundle
hash. Registration has `model_id`, `model_version`, `registered_at`, and optional
additional data-only metadata. IDs must match the model. Registration cannot
precede the model's training knowledge date.

Output is a new registry with sorted `entries` and `registry_hash`. Each entry
contains the full bundle, registration, IDs, registration date and `model_hash`
(the hash of the **whole bundle**). Registering identical content and registration
again is idempotent. Reusing an ID/version for different content or registration
metadata raises `RATING_VERSION_CONFLICT`. Original registries remain unchanged.

`select_effective_model(registry, assignments, profile_id, purpose, as_of,
known_at)` uses explicit assignment rows: `assignment_id`, `profile_id`, `purpose`,
`model_id`, `model_version`, `model_hash`, `effective_from`, `effective_to`,
`known_at`. Every effective assignment must pin the exact registered bundle hash.
The selected entry must have been registered by the historical knowledge cutoff.
Overlapping effective assignments return `AMBIGUOUS_MODEL`; absent/unavailable
artifacts return `MISSING_MODEL`. Known assignments entirely in the future return
`NOT_YET_EFFECTIVE`; other known gaps/expired intervals return `EXPIRED_MODEL`.
A successful `SELECTED` result contains the full assignment and immutable entry.
A later version or champion does not replace a historical explicit assignment.

Alternatively, `assignments` may be the sealed `ModelAssignmentHistory` returned
by `assign_model_role` or `rollback_model_assignment`. Selection projects only
champion events visible at the requested knowledge date, and derives each
effective endpoint from the next visible assignment for that profile/purpose.
Later-known or backdated events cannot rewrite an earlier knowledge view.

## Composition, freshness and lifecycle

`compose_risk_assessment(assessment_id, subject, components, binding, as_of,
known_at, calculated_at, effects=None)` preserves independently computed
quantities. `subject` contains `subject_id` and `subject_type`. `binding` contains
nonempty strings `profile_id`, `profile_version`, `model_id`, `model_version`,
`model_hash`, `input_hash`. Calculation date cannot precede knowledge cutoff.

Each component has `component_id`, `quantity`, `value`, `subject_id`,
`subject_type`, `variant`, and a complete `definition` with `event`, positive
`horizon`, `time_unit` (`days`, `months`, `years`), `probability_measure`
(`PHYSICAL`, `RISK_NEUTRAL`) and `conditioning`. Supported quantities:

| Quantity | Contract |
|---|---|
| PD | Finite probability in [0,1] |
| GRADE | String plus `scale_id`, `scale_version`, `graded_quantity` (PD/EL/SCORE) |
| LGD | Finite supplied loss ratio; negative/net gains and ratios above one remain explicit |
| EAD | Finite nonnegative exposure and explicit `currency` |
| EL | Finite supplied loss amount and explicit `currency` |
| SCORE | Finite supplied score, never silently treated as PD |

A component at another subject level requires `relationship_id`; otherwise
composition reports `UNLINKED_SUBJECT_LEVEL`. Different complete definitions
within one subject/variant report `TARGET_DEFINITION_CONFLICT`. Duplicate
quantities within that group report `DUPLICATE_QUANTITY_VARIANT`. Mixed currencies
are reported, never converted or aggregated. Variants (e.g. standalone,
supported, PIT, TTC) remain separate. Composition does not calculate EL or infer
LGD, grade mappings, support effects, guarantee coverage or cross-level PDs.

Effects have `effect_id`, `kind` (`support`, `transfer`, `override`, `cashflow`),
nonnegative integer `order`, `level`, `input_component_id`, `output_component_id`,
and nonempty `reason`. Orders are unique, references exist and differ, level
matches the output, output components are not overwritten twice, and downstream
effects follow the order of their inputs' producers. Cycles fail. The caller
identifies economic effects uniquely; the library cannot infer hidden duplicate
contracts from supplied names.

Output is a hashed assessment with sorted components/effects, `conflicts`,
`composition_status` (`CONSISTENT`/`CONFLICT`), initial `state=DRAFT`, and empty
`history`. Component conflicts remain visible; a conflicted assessment cannot be
locally finalized.

`check_assessment_freshness(assessment, policy, events, as_of, known_at)` accepts
policy `maximum_age_days`, `maximum_information_age_days`, `trigger_types`, and
optional `effective_to`. Ages equal to limits are still current. Event rows have
`event_id`, `subject_id`, `type`, `occurred_at`, `known_at`, optionally
`supersedes_assessment_id`. Known new material information can stale an assessment
even when it corrects an older economic fact. Only events effective by `as_of`
and known by cutoff apply. Future assessment/lifecycle state is rejected. Result:
`CURRENT`, `STALE`, or `SUPERSEDED`, plus explicit age/expiry/event findings.

`transition_assessment_state(assessment, event, transitions)` returns a new
assessment. Event fields: `event_id`, `type`, `occurred_at`, `known_at`, `reason`,
`gates` (object). Transition rows: `from_state`, `to_state`, `event_type`,
`required_gates` (array). Allowed state names are DRAFT, SUBMITTED, REVIEWED, FINAL,
SUPERSEDED, but the caller must declare each permitted transition. Exactly one
transition must match; every required gate must be explicitly true. Dates and
history are monotone. Duplicate events fail. FINAL can only move to SUPERSEDED;
SUPERSEDED is terminal. History records the prior assessment hash, old/new state
and full event, while the input remains intact. FINAL is local processing status,
not institutional approval.

## Stable batch inference and historical replay

`rate_subject_batch(requests, registry, batch_id, known_at)` sorts unique string
`row_id`s and binds all execution to the batch knowledge cutoff. Each request has
`row_id`, input `known_at`, pinned `model_id`, `model_version`, `model_hash`, `recipe`
and `input`. Model registration and request knowledge must not exceed cutoff.
Recipes are data objects with `method`, `version="1.0"`, `parameters`:

| Method | Input | Parameters |
|---|---|---|
| `application_risk` | `{snapshot: frozen application snapshot}` | `target`, optional `calibrator`, `offset` |
| `behavioral_risk` | `{snapshot: frozen account snapshot}` | `target`, optional `calibrator`, `offset` |
| `binary_glm` | `{X: row matrix, feature_order: array}` | `expected_target`, optional `offset` |
| `pd_model` | `{X: row matrix, feature_order: array}` | `expected_target`, optional `offset` where supported by `predict_pd` |

Application input cutoff and behavioral snapshot knowledge cannot exceed request knowledge. Arbitrary code, dynamic
imports from user values and persisted callbacks are unsupported. Version 1 remains available for these compact inference requests. Version 2 below
adds typed multi-method assessment execution. Model registration must be available
by each request cutoff, even when the batch itself is run much later.

Rows contain `status=OK` and actual computed `value`, or `status=ERROR`, missing
value and a controlled `error_code`. They also bind `input_hash`, `recipe_hash`,
`model_hash`, and `result_hash` (hash of the result value). The batch contains a
sorted request hash, registry hash and its own hash. Reordering requests does not
change a batch. Simulation steps consume explicitly supplied, content-bound innovations; request
order never allocates random streams or changes a seed.
Numerical output hashes are exact; a runtime-dependent numerical difference is
therefore visible during replay.

`apply_assessment_event(state, event, dependencies, known_at)` starts with missing
state and returns `{status, state, affected_assessment_ids}`. Event fields are
`event_id`, string `revision`, `occurred_at`, `known_at`, `dependencies` (array of
`{type,id}`), `payload` (data), and optional `corrects_revision`. Dependency rows
have `assessment_id`, `dependency_type`, `dependency_id`. Matching pairs project
the event to sorted unique assessment IDs; no risk/account mutation occurs.
Identical ID/revision/content is `DUPLICATE` with unchanged state. Different
content under the same revision fails. Corrections require a distinct revision
and explicit reference to the latest processed revision; all older content stays
in history. Future events are `DEFERRED`. A state containing history after the
requested cutoff is rejected, so callers use the appropriate historical state.

`replay_assessment_history(requests, registry, expected, known_at,
replay_id="replay")` reruns the pinned recipes and compares expectations containing
`row_id`, `input_hash`, `recipe_hash`, `model_hash`, `result_hash`. Successful prior
batch rows can be passed directly as expectations. Missing historical bundles,
unknown references/requests or changed input/recipe/model/result hashes produce
`NOT_REPRODUCIBLE` and row findings. Otherwise status is `REPRODUCED`. No current
champion or external source is consulted. Replay proves reproducibility of those
inputs and methods; it does not establish correctness or future model performance.


## Content-bound assessment pipelines, version 2

Public function signatures remain unchanged. Set the request recipe to
`{"method":"assessment_pipeline","version":"2.0","parameters":...}`.
The identical recipe must be stored in the registered bundle's
`configuration.assessment_recipe`; its content is therefore covered by `model_hash`.
The primary model remains a valid `ModelArtifact`, including when a recipe also uses
structural, cashflow, market or portfolio calculations. Additional parameter sets,
models, policies, scenarios and innovations live in `bundle.dependencies.artifacts`.
This is a local calculation contract, not evidence of empirical validation or
institutional model approval.

Every dependency is keyed by an immutable local artifact ID and contains
`{kind, content, sha256, known_at}`. `sha256` is the library's canonical content hash
of `content` (the same canonical format used for model serialization), not a file
byte hash. Use kind `innovations` for explicit random inputs. All dependencies are
verified before execution; none may postdate the assessment information cutoff.
Models/transformations fitted in development are stored as their frozen
`ModelArtifact` or structured result with its original `method_id`. No empirical
estimator is fitted on a subject's inference rows.

The request `input` contains:

- `snapshot`: the dated subject snapshot accepted by routing/applicability;
- `datasets`: ID-keyed `{value, as_of, known_at}` envelopes for supplied data;
- `assessment_id`, `assessment_known_at`, `calculated_at`;
- `lifecycle_events`: ordered review/finalization/superseding event objects;
- `events`: ordered dependency event revisions, including optional repeated or
  deferred events.

Snapshot and dataset information timestamps, primary model availability and frozen
dependencies must be no later than `assessment_known_at`. Dataset economic dates
cannot follow the subject's economic `as_of`. The calculation date lies between
assessment knowledge and the request's historical cutoff. Lifecycle events can be
later than the original assessment but cannot exceed that request cutoff. These
separate dates let replay rebuild an earlier assessment and its later local review
history without exposing the original inference to later information. Future
forecast/scenario dates are legitimate within a supplied conditional simulation;
they are not asserted to be observed historical data.

All eight parameter fields are explicit: `routing_rules`, `applicability_contract`,
`steps`, `components`, `effects`, `transitions`, `event_dependencies`, `decision`.
Use arrays for collections and `null`/`NULL` for an absent decision. Routing must
select exactly one profile and applicability must return `APPLICABLE` before any
risk calculation runs. Components/effects and lifecycle transitions have the same
contracts as the granular composition and transition APIs.

A reference is exactly
`{"$ref":{"source":"step","id":"pd","path":["value",0]}}`.
Sources are `input` (a dataset envelope), `artifact` (dependency content), `model`
(the primary model under ID `primary`) and `step` (prior structured output).
Paths contain object keys or zero-based array indices in both languages. References
have no expression language, executable objects or user-controlled imports.
Non-reference data are literal immutable recipe arguments.

Each step contains `step_id`, `method_id`, `arguments`, `argument_types` and
`output_contract`. Distinct step IDs form a DAG through references. Ready steps run
in lexicographic ID order; missing predecessors and cycles fail explicitly. Method
IDs resolve through a static native operation catalogue. Supported roles are
calculation, numerical market pricing/calibration, explicit simulation and
diagnostics. Empirical fit operations have role `PRE_FITTED`: their only argument
is `artifact`, referencing the frozen output of that same original method. Registry
mutation, batch/replay recursion and state transitions are handled through their
dedicated orchestration stages. A numerical bond/hazard calibration remains a
pricing operation; it is never represented as empirical physical-PD fitting.

`argument_types` maps argument names to `numeric_matrix`, `numeric_vector`,
`character_vector` or `logical_vector` when native argument conversion is needed.
Unspecified arguments preserve their record/list structure. Conversion is strict:
numeric matrices must be finite rectangular arrays, and categorical strings are
never coerced into numeric values. This makes native R's matrix/vector requirements
explicit while retaining sequences, categories and grouped Bayesian inputs.

The output contract requires `shape` (`scalar`, `vector`, `matrix`, `object` or
`records`), `unit`, and explicit nullable `target`. Length-one arrays/scalars are
normalized only according to the declared shape. Actual units and available
model/context/diagnostic targets are checked. Mixed quantity objects may additionally
specify `fields`, each `{path, unit, target, target_path}`: `path` selects a value
within the result and `target_path` optionally selects its actual stored target.
When present the actual nested target must match exactly, including its sampling
law. Component references must match the producing field's quantity/currency and
full target. PD, LGD, EAD, EL, score and grade cannot be relabelled by exchanging
field names. Physical and risk-neutral measures, horizons, conditional loss targets
and declared PIT/TTC sampling semantics remain separate; this engine performs no
implicit conversion between them.

For `explainability.estimate_credit_shapley_values`, use a data-only predictor
`{kind:"FROZEN_PD", model:<reference>, feature_order:[...], expected_target:...}`.
The engine constructs the native generic-PD call internally and verifies the
explanation's model identity and probability scale. It does not deserialize a
caller callback. Other non-probability explanation workflows remain available
through their granular APIs with explicit local callable contracts.

After computation, the engine composes the assessment and replays declared lifecycle
transitions. Exact repeated lifecycle IDs are idempotent; changed content under the
same ID fails. Dependency event revisions use `apply_assessment_event` semantics,
retaining corrections and deferred events. Projection identifies affected
assessments; it does not silently modify a previously computed PD. Recalculation
requires a new request with its own dated input content and immutable binding.

An optional `decision` object contains arguments for `apply_credit_decision_policy`.
Its PD and target must equal an explicitly composed PD component. Its frozen policy,
affordability, eligibility and operational inputs remain separately traceable; the
decision cannot replace or alter the calculated risk component.

Successful version-2 values contain the final `assessment`, `routing`,
`applicability`, ordered step trace, artifact manifest, event state/trace and optional
decision. Each step stores argument and value hashes with its output contract and
actual value. Rows additionally bind `assessment_hash`, `event_state_hash`,
`artifact_manifest_hash`, `innovation_hash` (the manifest subset tagged
`innovations`) and `decision_hash` (nullable). Replay requires all these extra fields
in a version-2 expected row, recomputes every calculation and local history, and
compares them. Missing or corrupt artifacts yield `NOT_REPRODUCIBLE`, never a silent
fallback to another model. Exact numerical hashes intentionally expose last-bit
runtime differences; numerical parity evidence is reported separately from exact
historical hash reproduction. Content hashes establish integrity, not a signature
or proof that a supplied original record was truthful.
