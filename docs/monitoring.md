# Local model roles and monitoring findings

`assign_model_role(history, registry, assignment)` appends a versioned local
role event. The registry is produced by `register_rating_model_version`.
The first history is `None`; later calls accept the returned CalcResult or its
value. Assignment fields are `event_id`, `profile_id`, `purpose`, `role`,
`slot_id`, `model_id`, `model_version`, `model_hash`, `effective_from`,
`known_at`, `evaluation_reference`, `reason`, and `required_feature_order`.
All IDs and references are nonempty strings. Features are an ordered array
and must match the pinned model exactly. A model must already be registered
and known when assigned. Evaluation references are assertions supplied by the
caller, not independent evidence verification.

Roles are `champion` or `challenger`. Each profile/purpose has one champion slot,
named `champion`. Separate named challenger slots permit concurrent challengers.
Events append in nondecreasing knowledge date order. The same event ID and
content is idempotent; conflicting content fails. Two events with the same
scope, role slot and effective date fail even if their model is the same.

The resulting `ModelAssignmentHistory` contains every original event and a
canonical content hash. `select_effective_model` accepts this history in place
of a flat assignment table. It selects champion events known at the requested
cutoff and derives start-inclusive/end-exclusive validity intervals. A later
event closes the previous interval only in knowledge states that include the
later event. Backdated information therefore changes a revised assessment but
cannot alter a replay using the original knowledge cutoff. Histories do not
expire without another assignment; callers must provide explicit validity
restrictions elsewhere when a model has a fixed end of use.

`rollback_model_assignment(history, registry, target_event_id, rollback_event)`
adds another event to the target historical model and scope. The rollback record
declares new `event_id`, `effective_from`, `known_at`, `reason`,
`evaluation_reference` and `required_feature_order`. It cannot change the model
or role pinned by the target event. Current feature compatibility is checked
against the caller's declared required schema. Rollback neither deletes the
intervening model nor cancels independently scheduled future events.

`classify_monitoring_findings(metric_panel, policy)` applies predeclared
thresholds and persistence without recalibration, deployment or messaging.
Policy schema `1.0` contains `policy_id`, `version`, ordered distinct `periods`,
the full expected `series_ids`, and rules. Each rule declares `rule_id`, `metric`,
`operator` (`gt`, `ge`, `lt`, `le`), finite `threshold`, positive integer
`consecutive_periods`, `severity` (`WARNING` or `CRITICAL`) and `action_code`.

Panel rows contain `series_id`, `metric`, `period`, `status`, and `value` (possibly
null). Supported input statuses are `OK`, `MISSING_OUTCOME`,
`INSUFFICIENT_INFORMATION`, `TECHNICAL_ERROR`, and `NOT_APPLICABLE`. Missing rows,
including an entire required series, generate insufficient information. Unavailable
values break the consecutive breach count. Exact threshold equality follows the
chosen strict or non-strict operator. A breach before the required persistence
is `WATCH`; an observed nonbreach is `WITHIN_POLICY`.

Every period and rule remains in the output trace. Overall status summarizes the
last declared period, ordered by technical error, critical, warning, missing
outcome, insufficient information, not applicable, watch, within policy. It is
not a statement that every historical period was complete or satisfactory.
Action codes arise only from persisted warning/critical rule breaches. The
policy content hash is preserved. These are user-defined monitoring rules,
not universal statistical significance levels or a compliance certificate.

The other cohort, drift and comparative monitoring methods are developed under
separate contracts; these three functions do not substitute for those analyses.
