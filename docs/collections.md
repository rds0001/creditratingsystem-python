# Collections evidence and repayment economics

These functions analyze local supplied evidence and scenarios. They do not
contact customers or change contracts, treatment assignments or default status.
The [OCC Retail Lending handbook](https://occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/retail-lending/pub-ch-retail-lending.pdf)
provides background on tracking collections actions and subsequent performance.
The operational state rules below are explicit user inputs; no jurisdiction's
legal or regulatory cure definition is built in.

## Episodes and sustained cure

`build_collections_episodes(states, actions, coverage, episode_rule, *,
evaluation_cutoff, context=None)` consumes complete state-change histories.
State rows have `state_event_id`, `account_id`, `state`, `occurred_at`, `known_at`.
Knowledge cannot precede occurrence. Both dates must be within the evaluation
cutoff. Simultaneous changes require explicit resolution outside this daily
contract. A state persists between changes only under the independent assertion
of complete observation in `coverage`: one row per account with `observed_from`,
`observed_through`, `known_at`. An initial state at or before coverage start is
required. Missing changes or baseline cannot establish a clean period.

The rule contains `rule_id`, `version`, disjoint `problem_states`, `clean_states`,
`terminal_states`, `default_states` (a subset of problem states), `probation_days`
and `separation_days`. A problem state starts an episode. A continuously clean
interval of `max(probation_days, separation_days)` closes it as `CURED` only
when complete coverage confirms that interval. A problem at exactly the boundary
interrupts it. Terminal states close as `TERMINATED`; the data endpoint never
closes an open episode. Episode IDs derive from the account and opening event.
Later episodes link to the preceding one. Historical default remains recorded.

Action rows contain `action_id`, `account_id`, `action_type`, `stage`,
`occurred_at`, `known_at`. Stages are ASSIGNED, OFFERED, ACCEPTED, PERFORMED,
CANCELLED. Dated actions link to the corresponding episode; unmatched actions
remain visible. The sealed output includes **all** known state changes, including
events after a cured episode has closed. This matters when later termination
censors post-cure re-default observation. Both episode and complete history hashes
are checked by downstream functions; a hash binds content, not author identity.

`build_cure_targets(episodes, *, horizon_days, redefault_horizon_days,
evaluation_cutoff, context=None)` uses the complete sealed output at the same
cutoff. Sustained cure occurs after `probation_days`, independently of the longer
episode-separation rule. A relapse during probation prevents cure. A positive
label requires observed sustained cure within the horizon from episode start.
A zero requires complete follow-up; termination censors. After observed cure,
a separate first re-default window begins. Termination during that window
censors its negative label, even if the original episode was already closed.
Cure never deletes the first-default history. These are operational targets,
not automated regulatory status changes.

## Action histories without hindsight features

`build_collections_action_history(actions, features, outcomes, coverage, *,
index_stage, required_features, horizon_days, maximum_feature_age_days,
evaluation_cutoff, context=None)` links action stages using `treatment_id`.
Each stage has its own `action_id`; repeated offers require separate treatment
IDs. Account and action type remain fixed across a treatment. `index_stage`
must be ASSIGNED or OFFERED. Acceptance and actual performance are output fields,
never the basis for selecting the index population.

Feature rows have `feature_record_id`, `account_id`, `feature`, `value`,
`observed_at`, `known_at`. Both information dates must strictly precede the action
occurrence date, even when the action was logged later. Same-day records are
excluded because date resolution cannot establish pre-action availability. The latest eligible
revision is selected; unresolved timestamp ties fail. Missing and stale features
are listed. Later acceptance, payment and willingness cannot become pre-action
covariates. Outcome rows have `outcome_id`, `account_id`, `event_type`,
`occurred_at`, `known_at`. The prospective window is (index, horizon end].
Same-day outcomes are flagged as prior-or-same-day because daily timestamps do
not establish within-day order. Complete coverage is separate from observed
events. Invalid stage chronology is visible and excludes descriptive eligibility.
Groups describe factual treatment selection; no causal effect is identified.

## Repayment schedules and values

`simulate_repayment_arrangement(contract, baseline, candidate,
affordability_paths, discount_factors, *, period_end_dates, currency,
context=None)` compares two explicit period schedules. Contract fields:
`currency`, `principal`, `principal_arrears`, `accrued_interest`, `fees`, `as_of`.
Arrears principal is already included in principal and must not be added again.
Dates are increasing future period ends. Discount factors are supplied values.

Each plan declares `plan_id`, vectors `payments`, `period_interest_rates`,
`new_fees`, `servicing_costs`, `receipt_fractions`, and Boolean
`capitalize_unpaid_interest`. Rates are nonnegative **effective period rates**
on opening principal, not annual rates. Receipts are capped scheduled obligations
times the supplied receipt fraction in [0,1]. Cash settles fees, interest, then
principal. Unpaid interest is capitalized only when explicitly requested.
Balances, paid components, capitalized interest, costs and unused payment
capacity are reconciled at each date. The final unpaid claim is reported; it has
no assumed terminal recovery value in NPV. Receipt fractions are deterministic
scenario assumptions, not estimated probabilities or automatic cure evidence.

Affordability rows have `path_id`, `currency`, `available_cash` per period after
other household outgoings. Residuals subtract scheduled obligations, not reduced
receipts. Negative residuals and their dates remain visible. Output includes
baseline/candidate schedules, NPVs and their difference. A no-change proposal
has zero NPV difference. Deferral moves cashflow dates without duplicating debt.

`expected_collections_value(actions, discount_factors, *, ead, currency,
period_end_dates, context=None)` combines uptake and conditional recovery
scenarios. Each action has `action_id`, `currency`, `uptake_probability`,
`accepted_scenarios`, `declined_scenarios`, `offer_cost`, `support_status`,
`assumption_reference`. Both branches explicitly contain conditional scenarios
whose probabilities sum to one. Scenario fields are `scenario_id`, `probability`,
`cashflows`; cashflow fields are `cashflow_id`, `kind` (RECOVERY or COST),
`period_end`, `amount`. Amounts are nonnegative, and unique economic cashflow IDs
prevent duplicated records within a scenario. Alternative branches are exclusive.
Offer cost occurs once at time zero and must not be repeated as a scenario cost.

The method integrates discounted recovery less costs and uptake probability.
EAD supplies a denominator, not a cap on economic recovery. With zero EAD the
ratio is absent. Reported min/max values are scenario ranges, not confidence
intervals. An explicitly supplied no-action scenario serves as baseline. The
method neither optimizes actions nor turns a conditional association into a
causal treatment effect.

## Cure and subsequent re-default curves

`predict_cure_and_redefault(current_episode, model, action_conditions,
horizon_grid, state_recoveries, *, target, currency, extrapolation="error",
context=None)` reuses a frozen Aalen–Johansen artifact from
`estimate_multistate_transitions`. The episode supplies `episode_id`, `state`,
`model_time`, `known_at`, `ever_default`. Model training must have been available
at that cutoff. The model's `collections_contract` declares `cure_states`,
`default_states`, `supported_initial_states`, `supported_actions`, and
`action_interpretation="observational_conditioning"`. Cure states must already
represent the bank's specified sustained-cure rule in the training history.
The function does not convert a single clean payment into sustained cure.

`action_conditions` contains `action_id` and `assumption_reference`. Unsupported
actions fail by default. Explicit flat extrapolation returns the unchanged
reference-model projection with a support warning; it does not estimate how the
new action would change behavior. Forecast horizons are elapsed model-clock
units from `model_time`; they follow the frozen target's time convention.

Current state occupancy, first sustained cure, and **cure followed by a subsequent
default** are separate curves. An augmented state tracks whether cure has already
occurred, then whether the first subsequent default has occurred. The last flag
is absorbing even when the operational account cures again. For an initially
defaulted account, existing default occupancy is not a re-default event. The
forecast inherits the multistate estimator's Markov and observation assumptions.

`state_recoveries` is a horizon-by-state matrix, in the model's state order,
of conditional **period cashflow** expectations given the end-of-period state.
The function reports undiscounted expected cashflows and their total. Inputs
must not be cumulative recovery totals or repeated liquidation values. The
two-account test has cure probability .5 at time one, subsequent re-default
probability .5 at time two, and default occupancy returning to one; these
quantities are deliberately different.
