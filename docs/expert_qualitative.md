# Qualitative criteria and dated evidence

`compile_qualitative_rubric` freezes structured criteria, evidence requirements and explicit scoring rules. `evaluate_qualitative_evidence` evaluates that contract against a declared observation and knowledge cutoff. Neither function infers facts from narrative text. Points remain distinct from PD, grade assignment and credit decisions. Both return `CalcResult`.

## Rubric contract

Supply `rubric_contract` with `rubric_id`, `version`, `population_id`, `reference`, `known_at`, `score_direction='higher_is_worse'`, `ordered_grades` (possibly empty), and a default target: event, horizon, time unit, probability measure and conditioning. The context must match the target. Grades are ordered from lower to higher risk and are used only for explicit rating limits.

Each criterion has a unique `criterion_id`, nonnegative `weight`, nonempty `required_evidence_types`, positive `minimum_independent_sources`, `missing_policy` (`refer` or `worst`), `not_applicable_policy` (`refer`, `worst`, `fixed`), and distinct scored `levels`. Total weight must be positive; normalization is fixed at compilation and never changes because observations disappear.

Categorical criteria declare `kind='categorical'`, the complete `categories`, and levels `{'level_id':..., 'values':[...], 'points':...}`. The level value sets must partition categories exactly. Numeric criteria declare `kind='numeric'`, `risk_order` (`increasing`, `decreasing`, `none`), and levels `{'level_id':..., 'lower':..., 'upper':..., 'points':...}`. Numeric intervals must be finite, nonempty, contiguous and disjoint. Each interval is [lower,upper), with the last upper endpoint included. Points must obey any declared increasing/decreasing direction. Unknown categorical values and numeric observations outside the compiled domain refer at evaluation.

A missing criterion either has unknown points with `REFER`, or receives its maximum declared points with `ASSUMED_WORST`. There is no available-case denominator and no missing-is-average or missing-is-good rule. Evidenced non-applicability is separate: a `fixed` non-applicable score must be explicitly supplied in `not_applicable_points` and lie within the criterion's declared point range. It is usable only when the observation has sufficient valid evidence.

`score_policy` contains finite `score_floor` and `score_cap`, with floor<cap, `minimum_evidence_coverage` in [0,1], and `unresolved_interaction_policy` (`refer` or `apply_adverse`). Weighted criterion contributions use the frozen normalized weights. Explicit interaction deltas are then added, and floor/cap applied. If any contribution is unknown, a score interval is provided and the point estimate remains null. Insufficient evidence coverage produces `REFER` even if a conservative numerical score is calculable.

## Interactions and rating limits

Optional interactions have `interaction_id`, `conditions`, nonnegative `points_delta`, and a `reference`. Conditions map criterion IDs to allowed level-ID lists and combine by AND. A condition that is known false makes the interaction false even if another criterion is unresolved. Missing, assumed-worst and evidenced non-applicable criteria have no invented level; their conditions remain unknown. An unknown interaction either refers or conservatively applies its adverse delta under the declared policy. Negative favorable interactions are rejected by this contract.

Optional rating limits have `limit_id`, conditions in the same format, `best_allowed_grade`, `worst_allowed_grade`, and reference. Active limits intersect the declared grade intervals. Conflicting active bounds or unresolved conditions refer. The output reports permitted grades without mapping qualitative points to a grade or PD. No implicit rule priority resolves inconsistent bounds.

## Observations, evidence and conflicts

Evaluation requires `as_of` and `known_at` ISO dates. Every observation contains `observation_id`, `criterion_id`, `value`, an explicit Boolean `not_applicable`, `observed_at`, `known_at`, `valid_from`, optional `valid_until`, and referenced `evidence_ids`. Non-applicable observations have null value. Observations cannot be known before they occur. Future, unknown and expired records are excluded with reasons; validity is [valid_from,valid_until). The rubric itself must have been available at the knowledge cutoff.

Every evidence record contains `evidence_id`, `evidence_type`, `source_id`, `independence_group`, `content_fingerprint`, `status` (`VERIFIED`, `UNVERIFIED`, `REJECTED`), and the same knowledge/validity fields. Only verified known/valid evidence can support a criterion. At least all required evidence types and the declared number of distinct independence groups must be present among its eligible linked records. Different document IDs or source IDs alone do not establish independence; groups are explicit caller assertions. The function cannot independently verify authenticity or independence.

Multiple active observations of one criterion are allowed when their structured values and non-applicable flags agree. Eligible evidence can then be combined. Differing active facts always produce a conflict and `REFER`, even when missing policy is `worst`; the function never chooses the more favorable observation or silently takes the latest record.

Optional dated conflict records have `conflict_id`, at least two `observation_ids` belonging to one criterion, `status` (`OPEN` or `RESOLVED`), a reference and validity/knowledge fields. A resolved conflict must choose one linked `resolution_observation_id`; other linked observations are excluded from that criterion's fact assessment. Conflicting resolutions, unavailable selected records, unresolved explicit conflicts and remaining contradictory observations all refer. A future resolution cannot resolve a historical decision.

## Output and an arithmetic example

A three-criterion rubric with weights 1,2,1 and observed points 100,70,0 yields base points (100+2*70)/4=60. An applicable adverse interaction of 15 gives 75. Removing the favorable third observation and using its worst score100 changes base points to85; a cap80 gives80, so missing evidence creates no score bonus. The fixed denominator and nonnegative adverse interaction rules preserve that property.

Results include score/interval, `READY`, `READY_WITH_ASSUMPTIONS` or `REFER`, weighted evidence coverage, each criterion's level/points/contribution, supporting IDs, independent groups, conflicts/reasons, interaction trace, rating limits and exclusions. SHA-256 content fingerprints of the rubric and sorted observation/evidence/conflict records identify the exact inputs used. They are content checks, not authentication. A numerical score accompanied by `REFER` does not authorize approval or remove unresolved judgments.

These are explicit implementation contracts for qualitative modeling. [SHELF](https://shelf.sites.sheffield.ac.uk/) provides background on structured elicitation; these functions do not claim that a SHELF workshop was conducted. Default tests use small hand-calculated cases, date boundaries, evidence independence, conflicting facts and missingness checks; no expensive stochastic fitting is involved.
