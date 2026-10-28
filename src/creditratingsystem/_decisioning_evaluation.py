"""Held-out policy comparisons, predeclared selection and finite-scenario robustness."""
from copy import deepcopy
from datetime import timedelta
from .core import CalcResult, _fail, _result, _fingerprint
from .data import _rows, _required, _date
from .decisioning import _checked, apply_credit_decision_policy
from ._decisioning_economics import _num, _text, _sum, _context, _bound, _target

_ACTIONS = ('APPROVE', 'REFER', 'DECLINE', 'BLOCKED')
_METRICS = ('approval_rate', 'refer_rate', 'decline_rate', 'blocked_rate', 'expected_loss_per_application', 'expected_npv_per_application', 'expected_funded_ead_per_application')



def _evaluation_target(profile):
    version=profile.get('target_contract_version','1.0');target=profile['target']
    if version=='1.0':
        _target(target)
    elif version=='1.1':
        if not isinstance(target,dict) or set(target)!={'event','horizon','time_unit','probability_measure','conditioning','performance_origin'} or target.get('performance_origin')!='first_draw_at':
            _fail('DECISION_TARGET_CONTRACT','Version1.1 requires the complete first-draw target contract')
        _target(target,allowed_events=('default','first_default'))
    else:
        _fail('DECISION_TARGET_CONTRACT','Unknown target contract version')


def _constraints(profile):
    constraints = _rows(profile['constraints'], 'constraints')
    names = set()
    for c in constraints:
        _required(c, ['constraint_id', 'metric', 'operator', 'threshold'])
        _text(c['constraint_id'], 'constraint_id')
        if c['constraint_id'] in names or c['metric'] not in _METRICS or c['operator'] not in ('le', 'ge'):
            _fail('DECISION_SELECTION_CONTRACT', 'Distinct constraints on declared metrics with le/ge required')
        names.add(c['constraint_id'])
        _num(c['threshold'], 'threshold')
    return constraints


def _feasibility(metrics, constraints, tolerance):
    out = []
    for c in constraints:
        value = metrics[c['metric']]
        slack = None if value is None else _sum([c['threshold'], -value]) if c['operator'] == 'le' else _sum([value, -c['threshold']])
        out.append(dict(constraint_id=c['constraint_id'], metric=c['metric'], value=value, slack=slack, status='UNIDENTIFIED' if slack is None else 'VIOLATED' if slack < -tolerance else 'SATISFIED', binding=slack is not None and abs(slack) <= tolerance))
    return out


def _sources(row, policy, as_of):
    _required(row, ['risk', 'affordability', 'eligibility', 'operations'])
    return apply_credit_decision_policy(policy, row['risk'], row['affordability'], row['eligibility'], row['operations'], as_of=as_of, known_at=as_of).value


def _nullable(row, keys):
    if any(k not in row for k in keys):
        _fail('DATA_SCHEMA', 'Explicit nullable fields required: ' + ','.join(keys))


def evaluate_decision_grid(policy_candidates, cases, candidate_inputs, observed_outcomes, evaluation_profile, *, evaluation_cutoff, context=None):
    """Keep prospective assumption economics separate from matched-funded evidence."""
    profile = deepcopy(evaluation_profile)
    ctx = _context(profile, evaluation_cutoff, context)
    _required(profile, ['registered_at', 'outcome_access_date', 'evaluation_role', 'prediction_design', 'split_id', 'data_fingerprint', 'target', 'candidate_hashes', 'objective', 'constraints', 'tie_break', 'tolerance', 'loss_bounds', 'profit_bounds'])
    _evaluation_target(profile)
    if not _date(profile['registered_at'], 'registered_at') <= _date(profile['outcome_access_date'], 'outcome_access_date') <= _date(evaluation_cutoff, 'evaluation_cutoff'):
        _fail('DECISION_SELECTION_CONTRACT', 'Predeclare candidates/objective/constraints before outcome access')
    if profile['evaluation_role'] not in ('VALIDATION', 'OOF_VALIDATION', 'TEST') or profile['prediction_design'] not in ('HELD_OUT', 'OUT_OF_FOLD') or (profile['evaluation_role'] == 'OOF_VALIDATION' and profile['prediction_design'] != 'OUT_OF_FOLD'):
        _fail('DECISION_SPLIT', 'Explicit held-out or out-of-fold validation/test design required')
    if profile['objective'] != 'expected_npv_per_application' or profile['tie_break'] != 'candidate_id':
        _fail('DECISION_SELECTION_CONTRACT', 'Predeclared expected-NPV objective and ID tie break required')
    for k in ('split_id', 'data_fingerprint'):
        _text(profile[k], k)
    tolerance = _num(profile['tolerance'], 'tolerance', 0)
    constraints = _constraints(profile)
    for key in ('loss_bounds', 'profit_bounds'):
        bound = profile[key]
        if not isinstance(bound, list) or len(bound) != 2 or _num(bound[0], key) > _num(bound[1], key):
            _fail('DECISION_OUTCOME_BOUNDS', 'Explicit finite lower/upper monetary bounds required')
    candidates, cmap = _rows(policy_candidates, 'policy_candidates'), {}
    for c in candidates:
        _required(c, ['candidate_id', 'policy', 'offer_rule_reference'])
        _text(c['candidate_id'], 'candidate_id')
        _text(c['offer_rule_reference'], 'offer_rule_reference')
        policy = _checked(c['policy'], 'DecisionPolicy')
        if c['candidate_id'] in cmap or policy['specification']['target'] != profile['target']:
            _fail('DECISION_SELECTION_CONTRACT', 'Distinct policies on the declared target required')
        cmap[c['candidate_id']] = c
    if not cmap or not isinstance(profile['candidate_hashes'], dict) or profile['candidate_hashes'] != {k: _fingerprint(v) for k, v in cmap.items()}:
        _fail('DECISION_SELECTION_CONTRACT', 'Candidate set must equal its predeclared content hashes')
    case_map, total_weight = {}, []
    for c in _rows(cases, 'cases'):
        _required(c, ['case_id', 'decision_date', 'weight', 'split_id', 'prediction_reference', 'excluded_from_model_training', 'excluded_from_calibration'])
        _text(c['case_id'], 'case_id')
        _text(c['prediction_reference'], 'prediction_reference')
        if c['case_id'] in case_map or c['split_id'] != profile['split_id'] or c['excluded_from_model_training'] is not True or c['excluded_from_calibration'] is not True:
            _fail('DECISION_SPLIT', 'Unique held-out/OOF cases excluded from training and calibration required')
        if _date(c['decision_date'], 'decision_date') > _date(evaluation_cutoff, 'evaluation_cutoff'):
            _fail('DECISION_FUTURE_INPUT', 'Decision date cannot follow evaluation')
        total_weight.append(_num(c['weight'], 'weight', 0))
        case_map[c['case_id']] = c
    weight = _sum(total_weight)
    if weight <= 0:
        _fail('DECISION_SPLIT', 'Positive evaluation weight required')
    inputs = {}
    for row in _rows(candidate_inputs, 'candidate_inputs'):
        _required(row, ['candidate_id', 'case_id', 'nonapproval_costs'])
        _nullable(row, ['offer'])
        key = (row['candidate_id'], row['case_id'])
        if key in inputs or key[0] not in cmap or key[1] not in case_map:
            _fail('DECISION_GRID', 'Distinct candidate/case input rows required')
        for a in ('REFER', 'DECLINE', 'BLOCKED'):
            _num(row['nonapproval_costs'].get(a), a + '_cost', 0)
        inputs[key] = row
    if set(inputs) != {(p, c) for p in cmap for c in case_map}:
        _fail('DECISION_GRID', 'Complete candidate by case input grid required')
    outcomes = {}
    for o in _rows(observed_outcomes, 'observed_outcomes'):
        _required(o, ['case_id', 'funded', 'complete'])
        _nullable(o, ['booked_offer_hash', 'funded_at', 'funding_known_at', 'outcome_end', 'known_at', 'defaulted', 'economic_loss', 'net_profit'])
        if o['case_id'] not in case_map or o['case_id'] in outcomes or type(o['funded']) is not bool or type(o['complete']) is not bool:
            _fail('DECISION_OUTCOME', 'Distinct explicit funded/outcome records required')
        c = case_map[o['case_id']]
        if not o['funded']:
            if any(o[k] is not None for k in ('booked_offer_hash', 'funded_at', 'funding_known_at', 'outcome_end', 'known_at', 'defaulted', 'economic_loss', 'net_profit')) or o['complete']:
                _fail('DECISION_OUTCOME', 'Unfunded cases cannot supply fabricated credit outcomes')
        else:
            _text(o['booked_offer_hash'], 'booked_offer_hash')
            funded = _date(o['funded_at'], 'funded_at')
            funding_known = _date(o['funding_known_at'], 'funding_known_at')
            if funding_known < funded:
                _fail('DECISION_OUTCOME', 'Funding knowledge cannot precede funding')
            end = _date(o['outcome_end'], 'outcome_end')
            if funded < _date(c['decision_date'], 'decision_date') or (end - funded).days != profile['target']['horizon']:
                _fail('DECISION_OUTCOME', 'Outcome horizon begins at actual funding')
            if o['known_at'] is not None and _date(o['known_at'], 'known_at') < funded:
                _fail('DECISION_OUTCOME', 'Outcome knowledge cannot precede funding')
            if o['complete']:
                if o['known_at'] is None or _date(o['known_at'], 'known_at') < end or type(o['defaulted']) is not bool:
                    _fail('DECISION_OUTCOME', 'Complete mature funded outcomes require dated labels')
                for field, bounds in (('economic_loss', profile['loss_bounds']), ('net_profit', profile['profit_bounds'])):
                    _num(o[field], field, bounds[0], bounds[1])
            elif any(o[k] is not None for k in ('defaulted', 'economic_loss', 'net_profit')):
                _fail('DECISION_OUTCOME', 'Incomplete outcomes remain explicitly unknown')
        outcomes[o['case_id']] = o
    table = []
    for cid in sorted(cmap):
        c = cmap[cid]
        counts = {a: [] for a in _ACTIONS}
        loss_terms, profit_terms, ead_terms, traces = [], [], [], []
        unidentified, matched_weight, mature_weight = False, [], []
        obs_loss, obs_profit, obs_defaults, excluded = [], [], [], []
        for case_id in sorted(case_map):
            case, row = case_map[case_id], inputs[(cid, case_id)]
            w = case['weight']
            decision = _sources(row, c['policy'], case['decision_date'])
            action = decision['action']
            counts[action].append(w)
            offer = row['offer']
            offer_hash = None
            supported = True
            if action == 'APPROVE':
                if not isinstance(offer, dict):
                    _fail('DECISION_GRID', 'Approved cases need explicit versioned offer assumptions')
                _required(offer, ['offer_id', 'version', 'known_at', 'currency', 'target', 'risk_conditioning', 'assumption_reference', 'support_status'])
                _nullable(offer, ['takeup_probability', 'conditional_booked_expected_loss', 'conditional_booked_npv', 'ead', 'offer_cost'])
                for k in ('offer_id', 'version', 'assumption_reference'):
                    _text(offer[k], k)
                if offer['currency'] != profile['currency'] or offer['target'] != profile['target'] or offer['risk_conditioning'] != 'CONDITIONAL_ON_THIS_ACCEPTED_OFFER':
                    _fail('TARGET_MISMATCH', 'Matched offer-conditioned economics target required')
                if _date(offer['known_at'], 'known_at') > _date(case['decision_date'], 'decision_date'):
                    _fail('DECISION_FUTURE_INPUT', 'Offer assumptions must precede historical decision')
                if offer['support_status'] not in ('SUPPORTED', 'UNSUPPORTED'):
                    _fail('DECISION_GRID', 'Explicit economic assumption support required')
                offer_hash = _fingerprint(offer)
                supported = offer['support_status'] == 'SUPPORTED'
                if supported:
                    uptake = _num(offer['takeup_probability'], 'takeup_probability', 0, 1)
                    loss = _num(offer['conditional_booked_expected_loss'], 'conditional_booked_expected_loss')
                    profit = _num(offer['conditional_booked_npv'], 'conditional_booked_npv')
                    exposure = _num(offer['ead'], 'ead', 0)
                    cost = _num(offer['offer_cost'], 'offer_cost', 0)
                    loss_terms.append(w * uptake * loss)
                    profit_terms.append(w * (uptake * profit - cost))
                    ead_terms.append(w * uptake * exposure)
                elif w > 0:
                    unidentified = True
            else:
                profit_terms.append(-w * row['nonapproval_costs'][action])
            outcome = outcomes.get(case_id)
            observed_status = 'NOT_APPROVED' if action != 'APPROVE' else 'NO_FUNDING_EVIDENCE'
            if action == 'APPROVE' and outcome is not None:
                if not outcome['funded']:
                    observed_status = 'NOT_FUNDED'
                elif outcome['booked_offer_hash'] != offer_hash:
                    observed_status = 'DIFFERENT_BOOKED_OFFER'
                elif _date(outcome['funded_at'], 'funded_at') > _date(evaluation_cutoff, 'evaluation_cutoff'):
                    observed_status = 'FUTURE_FUNDING'
                elif _date(outcome['funding_known_at'], 'funding_known_at') > _date(evaluation_cutoff, 'evaluation_cutoff'):
                    observed_status = 'FUTURE_FUNDING_EVIDENCE'
                else:
                    matched_weight.append(w)
                    if outcome['complete'] and _date(outcome['known_at'], 'known_at') <= _date(evaluation_cutoff, 'evaluation_cutoff') and _date(outcome['outcome_end'], 'outcome_end') <= _date(evaluation_cutoff, 'evaluation_cutoff'):
                        observed_status = 'MATURE_MATCHED_FUNDED'
                        mature_weight.append(w)
                        obs_loss.append(w * outcome['economic_loss'])
                        obs_profit.append(w * outcome['net_profit'])
                        obs_defaults.append(w * outcome['defaulted'])
                    else:
                        observed_status = 'MATCHED_FUNDED_OUTCOME_UNKNOWN'
            excluded.append(dict(case_id=case_id, status=observed_status))
            traces.append(dict(case_id=case_id, weight=w, action=action, decision=decision, offer_hash=offer_hash, economic_support=supported))
        metrics = {key: _sum(counts[a]) / weight for a, key in zip(_ACTIONS, ('approval_rate', 'refer_rate', 'decline_rate', 'blocked_rate'))}
        metrics.update(expected_loss_per_application=None if unidentified else _sum(loss_terms) / weight, expected_npv_per_application=None if unidentified else _sum(profit_terms) / weight, expected_funded_ead_per_application=None if unidentified else _sum(ead_terms) / weight)
        checks = _feasibility(metrics, constraints, tolerance)
        mw, ow = _sum(matched_weight), _sum(mature_weight)
        unknown = max(0., mw - ow)
        observed = dict(scope='OBSERVED_MATCHED_FUNDED_ONLY', matched_funded_weight=mw, mature_matched_weight=ow, missing_outcome_weight=unknown, observed_pd=None if ow == 0 else _sum(obs_defaults) / ow, observed_loss_sum=_sum(obs_loss), observed_profit_sum=_sum(obs_profit), matched_funded_mean_loss_bounds=None if mw == 0 else [_sum([_sum(obs_loss), unknown * b]) / mw for b in profile['loss_bounds']], matched_funded_mean_profit_bounds=None if mw == 0 else [_sum([_sum(obs_profit), unknown * b]) / mw for b in profile['profit_bounds']], full_candidate_causal_value=None, case_status=excluded)
        table.append(dict(candidate_id=cid, candidate_specification=deepcopy(c), candidate_hash=profile['candidate_hashes'][cid], policy=deepcopy(c['policy'].value if isinstance(c['policy'], CalcResult) else c['policy']), metrics=metrics, constraints=checks, status='UNIDENTIFIED' if unidentified else 'FEASIBLE' if all(x['status'] == 'SATISFIED' for x in checks) else 'INFEASIBLE', observed_outcomes=observed, decisions=traces))
    value = _bound(dict(schema_version='1.0', kind='DecisionGrid', evaluation_cutoff=evaluation_cutoff, profile=profile, input_hash=_fingerprint(dict(cases=cases, candidate_inputs=candidate_inputs, observed_outcomes=observed_outcomes)), total_weight=weight, candidates=table))
    return _result(value, 'decisioning.evaluate_decision_grid', 'decision_grid', ctx, limitations=('Training/calibration exclusion and pre-registration are caller assertions checked for explicit consistency, not independently audited. Candidate economics are supplied offer-conditioned assumptions, separate from observed mature funded outcomes at exactly matching booked terms. Missing and rejected outcomes are never nondefaults. Monetary bounds describe only the observed matched-funded cohort under supplied bounds, not causal policy value or missing funding selection. TEST grids can be evaluated but cannot be used by select_constrained_policy.',))


def select_constrained_policy(evaluated_grid, *, context=None):
    """Select only from predeclared feasible validation candidates; never TEST."""
    grid = _checked(evaluated_grid, 'DecisionGrid')
    _required(grid, ['profile', 'candidates', 'evaluation_cutoff'])
    profile = grid['profile']
    ctx = _context(profile, grid['evaluation_cutoff'], context)
    if profile['evaluation_role'] not in ('VALIDATION', 'OOF_VALIDATION'):
        _fail('DECISION_SPLIT', 'Policy selection requires validation; TEST is untouched for final assessment')
    if profile['objective'] != 'expected_npv_per_application' or profile['tie_break'] != 'candidate_id':
        _fail('DECISION_SELECTION_CONTRACT', 'Frozen declared objective and tie break required')
    _evaluation_target(profile)
    _num(profile['tolerance'], 'tolerance', 0)
    constraints = _constraints(profile)
    candidates = _rows(grid['candidates'], 'candidates')
    if set(profile['candidate_hashes']) != {c['candidate_id'] for c in candidates} or len(candidates) != len(profile['candidate_hashes']):
        _fail('DECISION_SELECTION_CONTRACT', 'Frozen candidate set mismatch')
    feasible, reports = [], []
    for c in candidates:
        if c['candidate_hash'] != profile['candidate_hashes'][c['candidate_id']]:
            _fail('DECISION_SELECTION_CONTRACT', 'Candidate hash mismatch')
        _required(c, ['candidate_specification', 'policy', 'metrics'])
        if _fingerprint(c['candidate_specification']) != c['candidate_hash'] or c['candidate_specification']['candidate_id'] != c['candidate_id'] or c['candidate_specification']['policy'] != c['policy']:
            _fail('DECISION_SELECTION_CONTRACT', 'Candidate specification binding mismatch')
        _checked(c['policy'], 'DecisionPolicy')
        _nullable(c['metrics'], _METRICS)
        for key in _METRICS:
            if c['metrics'][key] is not None:
                _num(c['metrics'][key], key)
        checks = _feasibility(c['metrics'], constraints, profile['tolerance'])
        identified = all(c['metrics'][key] is not None for key in _METRICS)
        ok = identified and all(x['status'] == 'SATISFIED' for x in checks)
        if ok:
            feasible.append(c)
        reports.append(dict(candidate_id=c['candidate_id'], feasible=ok, constraints=checks))
    # Dominance considers the objective and every constraint metric in its preferred direction.
    directions = [(profile['objective'], 1)] + [(c['metric'], -1 if c['operator'] == 'le' else 1) for c in constraints]
    dominated = []
    for c in feasible:
        for q in feasible:
            diffs = [(q['metrics'][k] - c['metrics'][k]) * d for k, d in directions]
            if all(x >= 0 for x in diffs) and any(x > 0 for x in diffs):
                dominated.append(dict(candidate_id=c['candidate_id'], dominated_by=q['candidate_id']))
    dominated_ids = {d['candidate_id'] for d in dominated}
    ordered = sorted((c for c in feasible if c['candidate_id'] not in dominated_ids), key=lambda c: (-c['metrics'][profile['objective']], c['candidate_id']))
    selected = ordered[0] if ordered else None
    value = _bound(dict(schema_version='1.0', kind='PolicySelection', grid_hash=grid['sha256'], selection_profile_hash=_fingerprint(profile), status='SELECTED' if selected else 'INFEASIBLE', selected_candidate_id=None if selected is None else selected['candidate_id'], selected_candidate_hash=None if selected is None else selected['candidate_hash'], selected_policy=None if selected is None else deepcopy(selected['policy']), objective_value=None if selected is None else selected['metrics'][profile['objective']], feasibility=reports, dominance=dominated, tie_break='candidate_id'))
    return _result(value, 'decisioning.select_constrained_policy', 'policy_selection', ctx, limitations=('Discrete predeclared validation-candidate selection under supplied economics only; no causal policy-effect identification. Final performance requires a separate untouched test period. Approval references remain supplied assertions and no financial action is executed.',))


def assess_policy_robustness(policy, baseline_cases, perturbation_scenarios, robustness_profile, *, as_of, context=None):
    """Replay a fixed policy on identified, finite shared perturbation scenarios."""
    policy = _checked(policy, 'DecisionPolicy')
    profile = deepcopy(robustness_profile)
    ctx = _context(profile, as_of, context)
    _required(profile, ['target', 'scenario_design', 'generation_reference', 'interpretation', 'tolerance'])
    if profile['target'] != policy['specification']['target'] or profile['scenario_design'] not in ('BOUNDED_GRID', 'BOOTSTRAP_REPLAY', 'STRESS_SET') or profile['interpretation'] != 'PERTURBATION_NOT_CAUSAL':
        _fail('DECISION_ROBUSTNESS_CONTRACT', 'Compatible target and explicit noncausal finite perturbation design required')
    _text(profile['generation_reference'], 'generation_reference')
    tolerance = _num(profile['tolerance'], 'tolerance', 0)

    def parse_case(row):
        _required(row, ['case_id', 'decision_date', 'weight', 'utility_reference', 'utility_by_action', 'feasible_by_action'])
        _nullable(row, ['offer_terms'])
        _text(row['case_id'], 'case_id')
        _text(row['utility_reference'], 'utility_reference')
        _num(row['weight'], 'weight', 0)
        if _date(row['decision_date'], 'decision_date') > _date(as_of, 'as_of'):
            _fail('DECISION_FUTURE_INPUT', 'Perturbations must replay dates no later than evaluation')
        for a in _ACTIONS:
            _nullable(row['utility_by_action'], [a])
            _nullable(row['feasible_by_action'], [a])
            if row['utility_by_action'][a] is not None:
                _num(row['utility_by_action'][a], a + '_utility')
            if row['feasible_by_action'][a] is not None and type(row['feasible_by_action'][a]) is not bool:
                _fail('DECISION_ROBUSTNESS_CONTRACT', 'Feasibility is explicit Boolean or unknown')
        decision = _sources(row, policy, row['decision_date'])
        terms = row['offer_terms']
        if terms is not None:
            _required(terms, ['offer_id', 'version', 'amount', 'annual_rate'])
            for k in ('offer_id', 'version'):
                _text(terms[k], k)
            for k in ('amount', 'annual_rate'):
                _num(terms[k], k, 0)
        if decision['action'] == 'APPROVE' and terms is None:
            _fail('DECISION_ROBUSTNESS_CONTRACT', 'Approval requires explicit local offer terms')
        return decision

    baseline, decisions = {}, {}
    for row in _rows(baseline_cases, 'baseline_cases'):
        decision = parse_case(row)
        if row['case_id'] in baseline:
            _fail('DECISION_DUPLICATE', 'Unique baseline cases required')
        baseline[row['case_id']], decisions[row['case_id']] = row, decision
    total = _sum(r['weight'] for r in baseline.values())
    if total <= 0:
        _fail('DECISION_ROBUSTNESS_CONTRACT', 'Positive baseline case weight required')
    scenarios, sids, weights = _rows(perturbation_scenarios, 'perturbation_scenarios'), set(), []
    paths, weighted_flips, weighted_terms, weighted_violations = [], [], [], []
    regrets, unidentified = [], set()
    for scenario in sorted(scenarios, key=lambda s: s.get('scenario_id', '')):
        _required(scenario, ['scenario_id', 'weight', 'perturbation_reference', 'cases'])
        _text(scenario['scenario_id'], 'scenario_id')
        _text(scenario['perturbation_reference'], 'perturbation_reference')
        sw = _num(scenario['weight'], 'scenario_weight', 0)
        if scenario['scenario_id'] in sids:
            _fail('DECISION_DUPLICATE', 'Unique perturbation scenarios required')
        sids.add(scenario['scenario_id'])
        weights.append(sw)
        rows = _rows(scenario['cases'], 'scenario_cases')
        if len(rows) != len(baseline) or {r.get('case_id') for r in rows} != set(baseline):
            _fail('DECISION_ROBUSTNESS_CONTRACT', 'Each shared scenario must cover exactly the baseline case set')
        case_results, flips, term_changes, violations, values = [], [], [], [], []
        regret_known = True
        for row in sorted(rows, key=lambda r: r['case_id']):
            base = baseline[row['case_id']]
            if row['decision_date'] != base['decision_date'] or row['weight'] != base['weight']:
                _fail('DECISION_ROBUSTNESS_CONTRACT', 'Shared scenarios preserve case dates and weights')
            decision = parse_case(row)
            action, baseline_action, w = decision['action'], decisions[row['case_id']]['action'], row['weight']
            changed_sources = [k for k in ('risk', 'affordability', 'eligibility', 'operations') if _fingerprint(row[k]) != _fingerprint(base[k])]
            flip = action != baseline_action
            flips.append(w * flip)
            both_approved = action == baseline_action == 'APPROVE'
            amount_change = row['offer_terms']['amount'] - base['offer_terms']['amount'] if both_approved else None
            price_change = row['offer_terms']['annual_rate'] - base['offer_terms']['annual_rate'] if both_approved else None
            terms_changed = both_approved and (abs(amount_change) > tolerance or abs(price_change) > tolerance)
            term_changes.append(w * terms_changed)
            feasible = row['feasible_by_action'][action]
            violations.append(w * (feasible is False))
            if feasible is None and sw * w > 0:
                unidentified.add('selected_action_feasibility')
            utilities = row['utility_by_action']
            all_known = all(row['feasible_by_action'][a] is not None and (not row['feasible_by_action'][a] or utilities[a] is not None) for a in _ACTIONS)
            available = [utilities[a] for a in _ACTIONS if row['feasible_by_action'][a] is True and utilities[a] is not None]
            regret = max(0., _sum([max(available), -utilities[action]])) if all_known and available and feasible is True and utilities[action] is not None else None
            if regret is None and w > 0:
                regret_known = False
            else:
                values.append(w * (regret or 0))
            case_results.append(dict(case_id=row['case_id'], baseline_action=baseline_action, action=action, action_flipped=flip, changed_input_sources=changed_sources, amount_change_if_both_approved=amount_change, annual_rate_change_if_both_approved=price_change, terms_changed_if_both_approved=terms_changed, selected_action_feasible=feasible, scenario_regret=regret, decision=decision))
        scenario_regret = _sum(values) / total if regret_known else None
        if scenario_regret is not None and sw > 0:
            regrets.append(scenario_regret)
        elif sw > 0:
            unidentified.add('utility_regret')
        flip_rate, term_rate, violation_rate = _sum(flips) / total, _sum(term_changes) / total, _sum(violations) / total
        weighted_flips.append(sw * flip_rate)
        weighted_terms.append(sw * term_rate)
        weighted_violations.append(sw * violation_rate)
        paths.append(dict(scenario_id=scenario['scenario_id'], scenario_weight=sw, action_flip_rate=flip_rate, offer_term_change_rate_on_all_cases=term_rate, known_infeasible_action_rate=violation_rate, mean_scenario_regret=scenario_regret, cases=case_results))
    if not scenarios or abs(_sum(weights) - 1) > 1e-12:
        _fail('DECISION_ROBUSTNESS_CONTRACT', 'Normalized nonnegative finite scenario weights required')
    value = _bound(dict(schema_version='1.0', kind='PolicyRobustness', policy_hash=policy['sha256'], profile=profile, as_of=as_of, baseline_hash=_fingerprint(baseline_cases), perturbation_hash=_fingerprint(perturbation_scenarios), weighted_action_flip_rate=_sum(weighted_flips), weighted_offer_term_change_rate=_sum(weighted_terms), weighted_known_infeasible_action_rate=_sum(weighted_violations), worst_case_mean_regret=None if 'utility_regret' in unidentified else max(regrets, default=None), unidentified_dimensions=sorted(unidentified), scenarios=paths))
    return _result(value, 'decisioning.assess_policy_robustness', 'policy_robustness', ctx, limitations=('Finite supplied shared perturbations, not a worst-case guarantee outside this set or an automatically valid confidence interval. Bootstrap validity, dependence and generation design are caller responsibilities. Changed input sources record association with flips, not causal attribution. Amount/rate changes are supplied offer scenarios; the fixed rule policy itself does not price or optimize limits. Regret compares only explicitly feasible actions with fully supplied scenario utilities; unknown utility or feasibility remains unidentified. No policy is refitted or selected on these scenarios.',))
