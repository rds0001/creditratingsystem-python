# Policy evaluation, selection and robustness

These APIs reuse the frozen rule evaluator and preserve separate risk,
affordability, eligibility and operations inputs. They never grant credit,
change probability estimates to fit an approval goal, or infer outcomes for
rejected applicants. Local policy recommendations are not executed decisions.

## Held-out decision grid

`evaluate_decision_grid(policy_candidates, cases, candidate_inputs,
observed_outcomes, evaluation_profile, *, evaluation_cutoff, context=None)`

The profile declares `profile_id`, `version`, `known_at`, `currency`,
`registered_at`, `outcome_access_date`, `evaluation_role` (VALIDATION,
OOF_VALIDATION or TEST), `prediction_design` (HELD_OUT or OUT_OF_FOLD), `split_id`,
`data_fingerprint`, `target`, `candidate_hashes`, `objective`, `constraints`,
`tie_break`, `tolerance`, `loss_bounds`, and `profit_bounds`.

The physical default target has an integer horizon in days and explicit
conditioning. Currency, measure, horizon, time unit and conditioning must agree
with any supplied context. Registration must precede declared outcome access,
which cannot follow the evaluation cutoff. The role OOF_VALIDATION requires
OUT_OF_FOLD predictions. Exclusion and registration dates are explicit caller
assertions, not a claim that the library independently audited the bank's data.
The actual input content fingerprint is returned separately from the supplied
data reference.

Each policy candidate has `candidate_id`, raw frozen `policy` value from
`compile_credit_decision_policy`, and `offer_rule_reference`. The exact candidate
object's canonical fingerprint must equal its predeclared `candidate_hashes`
entry. No extra or missing candidate is accepted. All policies use the profile
target. They remain complete in the evaluation artifact for replay and selection.

Each case supplies `case_id`, `decision_date`, nonnegative `weight`, `split_id`,
`prediction_reference`, `excluded_from_model_training=True` and
`excluded_from_calibration=True`. OOF exclusion applies to the model and calibrator
that produced that case's prediction, not merely to an aggregate sample label.
Total weight must be positive. Candidate inputs form an exact Cartesian grid of
`candidate_id` and `case_id`, with four separate dated source objects (`risk`,
`affordability`, `eligibility`, `operations`), explicit nullable `offer`, and
`nonapproval_costs` mapping REFER, DECLINE and BLOCKED to nonnegative amounts.
Different offers may require different risk and affordability inputs; the API
does not copy a baseline PD into a changed offer. The existing policy evaluator
checks source freshness, target, date and approval gates. Its knowledge cutoff
is exactly the historical decision date.

An APPROVE recommendation requires a versioned `offer` with `offer_id`, `version`,
`known_at`, `currency`, `target`,
`risk_conditioning='CONDITIONAL_ON_THIS_ACCEPTED_OFFER'`, `assumption_reference`,
`support_status` and the explicitly nullable economic fields
`takeup_probability`, `conditional_booked_expected_loss`,
`conditional_booked_npv`, `ead`, `offer_cost`.
SUPPORTED offers need finite economic inputs; UNSUPPORTED offers remain
unidentified for any positive-weight approved case. These are supplied conditional
forecasts/assumptions, never calibrated from the observed outcome table by this
function. Book NPV already includes loss, funding and origination economics;
EL is a separate metric and is not subtracted again. Per-applicant contributions
are takeup times conditional book NPV minus offer cost, takeup times EL and
takeup times EAD. Nonapproval actions incur their explicit costs.

Recommendation rates use all evaluation weight. Supported assumption economics
are divided by that same weight. Thus capacity, value and loss constraints refer
to a common applicant population. An unsupported approved offer makes prospective
monetary metrics null, while action rates and rule traces remain available.

### Observed outcomes are a separate evidence table

Each row has `case_id`, Boolean `funded`, Boolean `complete` and the explicit
nullable fields `booked_offer_hash`, `funded_at`, `funding_known_at`, `outcome_end`,
`known_at`, `defaulted`, `economic_loss`, `net_profit`. Unfunded cases must have
no credit-outcome values and `complete=False`. Funding knowledge cannot precede
funding. The observation horizon runs from actual funding for exactly the target
number of days. Complete observations need Boolean default and finite loss/profit
inside the declared monetary bounds, with outcome knowledge no earlier than the
horizon end. Incomplete labels, loss and profit remain null.

Observed statistics include only cases recommended APPROVE whose actual booked
offer hash exactly matches the hypothetical offer object and whose funding was
known by cutoff. Outcome summaries additionally require complete mature labels
known by cutoff. Hashing the full offer is deliberately strict: changed versions
or assumptions are not silently considered equivalent booked terms. Late-known
funding, different booked offers, unfunded applicants, missing evidence and
incomplete labels have distinct status codes. No rejected or unfunded applicant
is counted as a nondefault.

`observed_pd` describes only the mature matched-funded cohort. Monetary bounds
apply the supplied lower/upper amount to unmatched **outcomes within the known
matched-funded cohort**; they do not bound the effects of unknown funding or
policy-driven acceptance selection. With no matched funding the bounds are null.
`full_candidate_causal_value` is always null: this evaluator does not identify it.

## Constrained selection

`select_constrained_policy(evaluated_grid, *, context=None)`

Selection verifies the grid's canonical fingerprint, each complete candidate
specification binding and its frozen policy. It accepts only VALIDATION or
OOF_VALIDATION, never TEST. The predeclared objective is maximization of
`expected_npv_per_application`, with deterministic `candidate_id` ties.
Constraints have distinct `constraint_id`, `metric`, `operator` (le or ge) and
finite `threshold`; allowed metrics are approval/refer/decline/blocked rates and
expected loss, expected NPV and expected funded EAD per application.

Constraint slacks and numerical binding flags use the predeclared tolerance.
Unknown economic metrics never qualify. Constraints are recomputed rather than
trusting a stored feasible flag. The Pareto report uses the utility objective
and all constraint directions. Dominated candidates are excluded before the
objective/ID tie rule, so an equal-value higher-loss candidate cannot win merely
because its ID sorts first. No feasible identified candidate returns INFEASIBLE
and null selected policy. The result embeds the selected immutable policy and
its candidate/grid/profile bindings. It does not refit that policy.

Evaluate the final choice on a separate untouched test period. Probability
estimation and policy thresholds are distinct tasks; tuning both on the same
training observations can overfit. This separation follows the official
[scikit-learn threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html)
(reviewed 2026-10-13). The local contracts and economics are library-specific;
they are not asserted to be a scikit-learn algorithm or regulatory approval.

## Finite-scenario robustness

`assess_policy_robustness(policy, baseline_cases, perturbation_scenarios,
robustness_profile, *, as_of, context=None)`

The versioned common profile adds `target`, `scenario_design` (BOUNDED_GRID,
BOOTSTRAP_REPLAY or STRESS_SET), `generation_reference`,
`interpretation='PERTURBATION_NOT_CAUSAL'` and numerical `tolerance`.
Each baseline case has `case_id`, `decision_date`, `weight`, the four source
objects, explicit nullable `offer_terms`, `utility_reference`,
`utility_by_action` and `feasible_by_action`. Both action maps contain APPROVE,
REFER, DECLINE and BLOCKED. Utilities are finite currency amounts or null;
feasibility is Boolean or null. Approved cases require offer terms (`offer_id`,
`version`, `amount`, `annual_rate`). These terms are supplied scenarios: the
rule evaluator itself does not price or optimize them.

Each perturbation has `scenario_id`, normalized nonnegative `weight`,
`perturbation_reference` and an exact complete `cases` table. Every scenario
retains the baseline case IDs, weights and dates. Thus shared macro/model/data
perturbations can move the complete application cohort together, without an
invented independent resampling of each case. Source dates still obey historical
knowledge restrictions.

The same policy is replayed for every case. Outputs report action flips,
changed input sources, amount/rate changes where both actions are APPROVE,
separate selected-action feasibility, and scenario utility regret. A changed
source is a recorded association with the flip, not causal feature attribution.
Term-change rates use all case weight; the both-approved condition is explicit.
Known infeasible action rates do not treat unknown feasibility as success;
unknown dimensions are reported separately.

Regret compares the selected feasible action with the best explicitly feasible
alternative only when all relevant action feasibility and utility entries are
known. Any positive-weight unidentified case makes that scenario's mean regret
null; any positive-weight unidentified scenario makes worst-case mean regret null.
Zero-weight scenarios cannot establish a worst-case guarantee. The reported worst
case is the maximum scenario mean over supplied positive-weight scenarios, not
a bound over unmodeled uncertainty. Bootstrap validity and dependence assumptions
remain the caller's responsibility; no confidence interval or causal effect is
invented. No policy is tuned inside this robustness calculation.

The optional evaluation-profile `target_contract_version="1.1"` requires the complete six-field target, including `performance_origin="first_draw_at"`, and admits `default` or `first_default` without renaming either event. The validation design must explicitly repeat `policy_target_contract_version="1.1"` and the same performance origin; its ordinary five-field target is reconstructed and compared with the complete profile/policy/offer contract. Missing version or `"1.0"` preserves the existing default-only behavior; unknown versions and other origins/events are rejected. This extension does not widen quote/pricing target gates. Incomplete realized net-profit/loss observations remain unknown even when actual approval, uptake and funding stages are observed.
