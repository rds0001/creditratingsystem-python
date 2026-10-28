# Retail cohort and monthly transition audits

`validate_retail_cohort(application_targets, account_panel, behavioral_targets, *,
target_profile, split_memberships, source_lineage, design, context=None)` audits
existing operational outputs. It accepts application-target rows from
`build_application_targets`, account rows (or the full account-panel result) from
`build_account_month_panel`, and targets from `build_behavioral_targets`.
It does not construct competing features or substitute default labels.

The target profile supplies `profile_id`, `version`, `reference`, `known_at`,
`default_definition`, `default_group_rule`, `default_unit` (facility/obligor),
`first_default`, `exit_treatment` (competing/censor), distinct `default_event`,
`chargeoff_event`, `cure_event`, `competing_events`, `application_horizon_days`,
`behavioral_horizon_months`, `split_isolation_keys`, and `risk_set_by_product`.
Each product rule supplies `active_states` and Boolean `include_zero_balance`.
Obligor profiles require supplied default_group_id assignments rather than
interpreting household membership as default contagion. Profile knowledge must
precede original prediction cutoffs.

Population flow preserves application, approval, acceptance, booking and funding.
Accepted-but-unfunded cases remain distinct; a never-funded own contract cannot
have an observed repayment/default label. NO_HIT, THIN_FILE, PSEUDO and HYPOTHETICAL
label origins are not factual outcomes. Default labels require explicit event
provenance. Negative labels require mature, complete follow-up, except explicitly
defined first-competing-exit behavioral targets. Early observed defaults remain
visible but are excluded from mature-performance coverage.

Account-Month dates, calendar MOB, account opening, knowledge and product risk
membership are checked. A product-qualified active revolving zero balance remains
at risk. First-default profiles exclude previously defaulted accounts; cure,
charge-off and default concepts remain distinct. Account snapshots and behavioral
targets must match one-to-one by account/date, customer, target event and horizon.
The profile specifies interpretation; this audit does not authenticate raw event
histories or group-assignment decisions.

Split memberships require `customer_id`, `role` and every profile isolation key
(e.g. customer/household IDs). An isolated group cannot cross roles. All observed
customers need split evidence. Source-lineage rows supply `source_id`,
`prediction_key`, `source_hash`, `kind`, `observed_at`, `available_at`. Keys are
`application:<application_id>` or `account:<account_id>@<as_of>`. Every snapshot
requires lineage; source observation/availability cannot follow its information
cutoff. TARGET, OUTCOME and POST_DECISION feature sources are reported as leakage.

Outputs are structured violations, flow and maturity/risk counts, fingerprints
and a scope-qualified PASS/FAIL/NOT_APPLICABLE contract decision. PASS here means
only the supplied tables passed this declared audit; it is not evidence of
empirical predictive performance, regulatory approval or authentic source records.
Grouped isolation follows the rationale in the
[official cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

`roll_rate_backtest(forecasts, account_panel, *, states, state_contract, design,
weighting, uncertainty, cluster_resamples=None, confidence_level=.95,
context=None)` validates frozen monthly transition forecasts. The design target
conditions on `origin_state` and uses a positive whole number of months. Forecast
rows have common validation identity/model/date fields, `account_id`,
`origin_state`, `risk_episode_id`, optional `default_episode_id`,
`transition_matrix`, and nonnegative `opening_balance` for monetary weighting.

Observed state rows require account ID, `as_of`, `known_at`, `state`,
`risk_episode_id`, and a nonempty `default_episode_id` in default states.
Here knowledge cannot precede actual state observation; these are observed state
endpoints, not lagged feature estimates relabelled as known outcomes. Exact
historically known origin snapshots must agree with forecast state/episode IDs.
For later dates, the latest revision known by evaluation cutoff is used. The
endpoint must be the exact horizon; only a declared absorbing state may carry
forward from an earlier observation. A missing next month is never compressed
into a shorter horizon or made a self-transition.

The state contract is the migration contract (identified/versioned frozen scale,
reference/known_at, default/cure/exit/absorbing state lists), extended with
`chargeoff_states`, a separately designated subset of default states. Default
need not be absorbing if cure and re-default are modeled. A transition from
default to cure requires a new risk episode. A later default requires a new
default episode; changing default IDs while continuously defaulted or leaving an
absorbing state is a reported episode violation and excluded from endpoint
scoring. Observed intermediate transitions are audited, not unobserved paths
inferred between snapshots.

`weighting` is count or opening_balance. Zero-balance origins retain separate
account counts even when their monetary weight is zero. All multiclass Brier,
log loss, expected/observed transition tables, coverage and actual cluster
bootstrap intervals reuse `rating_migration_backtest`. Extra diagnostics expose
missing exact successors, episode violations, cure/re-default counts and
charge-off destinations. Calibration is conditional on valid observed successors;
informative missingness is not silently repaired.
