# Overrides and separately observed support/transfer stages

`override_monitoring(pre_override, post_override, override_ledger, *, policy,
design, metrics, uncertainty, cluster_resamples=None, confidence_level=.95,
context=None)` accepts two model entries in the `benchmark_predictions` format.
Distinct model identifiers distinguish pre/post snapshots. The paired benchmark
requires the same case identities, labels, dates and weights and uses only common
mature observations for performance. Override rates separately use all paired
forecast cases, including forecasts whose outcomes are still immature.

The policy supplies `reference`, `approved_authorities` and an explicit
`concentration_threshold` in (0,1]. Ledger records require unique `override_id`,
`case_id`, `reason`, `authority`, `applied_at`, `known_at`, `expires_at`, `from_pd`
and `to_pd`. Dates are day-resolution ISO dates. An action is active when
`applied_at <= forecast_at < expires_at` and `known_at <= forecast_at`. Knowledge
cannot precede the action; expiry must follow it. Future-known and expired
records remain separately reported. Active actions require approved authority.
Equal action/knowledge dates are ambiguous and must be resolved upstream.

For each case, active actions ordered by action/knowledge date must form an exact
chain from its pre-PD to its post-PD (tolerance 1e-12). No active action requires
unchanged predictions. This contract needs self-contained active chains; it does
not silently resurrect an expired intermediary. Outputs include rates by segment
and cohort, PD-increase/decrease directions, reasons, authorities, multiple-action
cases and full chain traces. Concentration flags compare active-action shares to
the supplied threshold. Changes in paired Brier, log loss or discrimination are
descriptive: selective intervention is not random assignment or a causal effect.
Uncertainty uses actual shared cluster resamples through the paired benchmark.

`support_transfer_backtest(records, *, stage_contract, union_contract, design,
uncertainty, cluster_resamples=None, confidence_level=.95, context=None)` validates
separately observed stage outcomes. Rows carry the common dated validation
identity/model/horizon fields plus optional nonnegative `weight`, complete
`stage_predictions`, and `stage_outcomes` keyed by stage name. The named stage
contract provides `kind` (binary/amount), `observable` Boolean, `event_definition`
and `reference`; amount stages also declare the same currency as `design`.
Predictions and amount outcomes are nonnegative; binary predictions are in [0,1].
An unobservable stage must not carry purported observed labels. This explicitly
allows standalone counterfactual stages to remain NOT_APPLICABLE.

Each observed stage provides `value`, `known_at`, `complete_through`, `event_ids`
and, for binary stages, aligned `event_times`. Positive binary outcomes require
source events; negative outcomes require empty event arrays. Events must lie
inside the forecast horizon and be known by the stage knowledge date. The same
source ID must carry the same event date across stages. All stages require mature
horizons and complete follow-up through horizon end; future-known and incomplete
stage outcomes are excluded separately. This complete-case estimand does not
correct informative missingness or censoring. For final cash amounts, callers
must declare a stage definition consistent with the measured period and currency;
settlement delays require complete, known amounts rather than imputed payments.

`union_contract` identifies four distinct binary stage names using `component_a`,
`component_b`, `joint`, `union`. Predicted intersection must obey Frechet bounds;
union PD must equal A+B−joint. When all four labels are independently available,
intersection and union labels must reconcile logically, and the union's source
IDs must equal the deduplicated union of component IDs. Missing component labels
are never reconstructed from final defaults. A shared event counts once.
Observed guarantee cash-flow errors are distinct amount-stage metrics; no
counterfactual support benefit is inferred from realized cash payments.

Outputs provide stage-specific coverage, weighted bias (prediction minus outcome),
MAE/RMSE, binary Brier scores and actual cluster percentile intervals. Degenerate
or wholly unobservable resamples yield explicit non-estimability. The independent
oracle with entity PD .2, transfer .1 and joint .05 has final PD .25; outcomes 1/0
give final Brier .3125. Predicted guarantee amount 30 versus observations 20/0 has
bias 20. A common source event yields one union event, not two.
