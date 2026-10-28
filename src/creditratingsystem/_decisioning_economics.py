"""Explicit local offer economics; scenario assumptions are not causal estimates."""
from copy import deepcopy
from datetime import date
import math
from .core import CalcResult, _fail, _result, _fingerprint, _context_value
from .data import _rows, _required, _date


def _num(x, name, minimum=None, maximum=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or (minimum is not None and x < minimum) or (maximum is not None and x > maximum):
        _fail('DECISION_ECONOMICS_INPUT', f'{name} outside finite numeric domain')
    return float(x)


def _text(x, name):
    if not isinstance(x, str) or not x.strip():
        _fail('DECISION_ECONOMICS_INPUT', f'{name} must be nonempty text')
    return x


def _sum(xs):
    try:
        value = math.fsum(xs)
    except (ValueError, OverflowError):
        _fail('NUMERICAL_FAILURE', 'Decision economics outside finite representation')
    if not math.isfinite(value):
        _fail('NUMERICAL_FAILURE', 'Decision economics outside finite representation')
    return value


def _context(profile, as_of, context):
    _date(as_of, 'as_of')
    _required(profile, ['profile_id', 'version', 'known_at', 'currency'])
    for k in ('profile_id', 'version', 'currency'):
        _text(profile[k], k)
    if _date(profile['known_at'], 'known_at') > _date(as_of, 'as_of'):
        _fail('DECISION_FUTURE_INPUT', 'Profile must be known by decision date')
    ctx = _context_value(context)
    if ctx.get('currency') not in (None, profile['currency']) or ctx.get('probability_measure') not in (None, 'PHYSICAL'):
        _fail('TARGET_MISMATCH', 'Physical offer economics and currency must match context')
    if isinstance(profile.get('target'), dict):
        for key in ('horizon', 'time_unit', 'conditioning'):
            if ctx.get(key) is not None and ctx[key] != profile['target'].get(key):
                _fail('TARGET_MISMATCH', 'Context conflicts with declared target')
    return ctx


def _bound(payload):
    value = deepcopy(payload)
    value['sha256'] = _fingerprint(value)
    return value


def _target(target, allowed_events=('default',)):
    _required(target, ['event', 'horizon', 'time_unit', 'probability_measure', 'conditioning'])
    if target['event'] not in allowed_events or target['time_unit'] != 'days' or target['probability_measure'] != 'PHYSICAL':
        _fail('TARGET_MISMATCH', 'Physical default target in days required')
    n = _num(target['horizon'], 'horizon', 1)
    if n != int(n):
        _fail('TARGET_MISMATCH', 'Integer day horizon required')
    _text(target['conditioning'], 'conditioning')


def _offer(row, ids, as_of, currency):
    _required(row, ['offer_id', 'version', 'known_at', 'currency', 'subject_id', 'amount', 'tenor_days', 'assumption_reference'])
    for k in ('offer_id', 'version', 'subject_id', 'assumption_reference'):
        _text(row[k], k)
    if row['offer_id'] in ids:
        _fail('DECISION_DUPLICATE', 'Unique versioned offer IDs required')
    ids.add(row['offer_id'])
    if _date(row['known_at'], 'known_at') > _date(as_of, 'as_of'):
        _fail('DECISION_FUTURE_INPUT', 'Offer assumptions must be known by decision date')
    if row['currency'] != currency:
        _fail('TARGET_MISMATCH', 'Offer currency differs from profile')
    _num(row['amount'], 'amount', 0)
    n = _num(row['tenor_days'], 'tenor_days', 1)
    if n != int(n):
        _fail('DECISION_ECONOMICS_INPUT', 'Integer tenor days required')


def compute_limit_frontier(offer_candidates, constraint_profile, *, as_of, context=None):
    """Enumerate a discrete local limit frontier without assuming monotone risk."""
    profile = deepcopy(constraint_profile)
    ctx = _context(profile, as_of, context)
    _required(profile, ['target', 'maximum_amount', 'maximum_ead', 'maximum_expected_loss', 'maximum_concentration', 'concentration_denominator', 'existing_group_exposure', 'tolerance'])
    _target(profile['target'])
    for k in ('maximum_amount', 'maximum_ead', 'maximum_expected_loss', 'existing_group_exposure', 'tolerance'):
        _num(profile[k], k, 0)
    _num(profile['maximum_concentration'], 'maximum_concentration', 0, 1)
    denominator = _num(profile['concentration_denominator'], 'concentration_denominator', 0)
    if denominator == 0:
        _fail('DECISION_ECONOMICS_INPUT', 'Positive fixed post-decision concentration denominator required')
    rows = _rows(offer_candidates, 'offer_candidates')
    ids, out, subject = set(), [], None
    for row in sorted(rows, key=lambda x: x.get('offer_id', '')):
        _offer(row, ids, as_of, profile['currency'])
        if subject is not None and row['subject_id'] != subject:
            _fail('DECISION_SUBJECT', 'One subject per discrete limit frontier')
        subject = row['subject_id']
        _required(row, ['target', 'pd', 'lgd', 'ead', 'risk_conditioning', 'risk_reference', 'affordability_status', 'eligibility_status'])
        if row['target'] != profile['target'] or row['risk_conditioning'] != 'CONDITIONAL_ON_THIS_OFFER':
            _fail('TARGET_MISMATCH', 'Recalculated or explicitly assumed offer-conditioned risk with shared target required')
        _text(row['risk_reference'], 'risk_reference')
        pd = _num(row['pd'], 'pd', 0, 1)
        lgd, ead = _num(row['lgd'], 'lgd'), _num(row['ead'], 'ead', 0)
        if row['affordability_status'] not in ('AFFORDABLE', 'UNAFFORDABLE', 'UNKNOWN') or row['eligibility_status'] not in ('ELIGIBLE', 'INELIGIBLE', 'UNKNOWN'):
            _fail('DECISION_ECONOMICS_INPUT', 'Explicit affordability and eligibility states required')
        if 'affordability_maximum_amount' not in row:
            _fail('DATA_SCHEMA', 'Explicit nullable affordability cap required')
        cap = row['affordability_maximum_amount']
        if cap is not None:
            _num(cap, 'affordability_maximum_amount', 0)
        el = _sum([pd * lgd * ead])
        concentration = _sum([profile['existing_group_exposure'], ead]) / denominator
        tests = [('AMOUNT', row['amount'], profile['maximum_amount']), ('EAD', ead, profile['maximum_ead']), ('EXPECTED_LOSS', el, profile['maximum_expected_loss']), ('CONCENTRATION', concentration, profile['maximum_concentration'])]
        if cap is not None:
            tests.append(('AFFORDABILITY_AMOUNT', row['amount'], cap))
        slacks = {name: _sum([bound, -value]) for name, value, bound in tests}
        failures = [name for name, slack in slacks.items() if slack < -profile['tolerance']]
        if row['affordability_status'] != 'AFFORDABLE' or cap is None:
            failures.append('AFFORDABILITY_NOT_ESTABLISHED')
        if row['eligibility_status'] != 'ELIGIBLE':
            failures.append('ELIGIBILITY_NOT_ESTABLISHED')
        out.append(dict(offer_id=row['offer_id'], version=row['version'], offer_hash=_fingerprint(row), amount=row['amount'], tenor_days=row['tenor_days'], pd=pd, lgd=lgd, ead=ead, expected_loss=el, concentration=concentration, feasible=not failures, failures=failures, slacks=slacks, binding_constraints=[name for name, slack in slacks.items() if abs(slack) <= profile['tolerance']]))
    feasible = [r for r in out if r['feasible']]
    # A frontier point is not dominated at the SAME tenor; cross-tenor utility is unspecified.
    frontier = [r['offer_id'] for r in feasible if not any(q['tenor_days'] == r['tenor_days'] and q['amount'] >= r['amount'] and q['expected_loss'] <= r['expected_loss'] and (q['amount'] > r['amount'] or q['expected_loss'] < r['expected_loss']) for q in feasible)]
    maximum = max((r['amount'] for r in feasible), default=None)
    value = _bound(dict(schema_version='1.0', kind='LimitFrontier', as_of=as_of, profile=profile, subject_id=subject, candidates=out, frontier_offer_ids=frontier, maximum_feasible_amount=maximum, status='FEASIBLE' if feasible else 'INFEASIBLE'))
    return _result(value, 'decisioning.compute_limit_frontier', 'limit_frontier', ctx, limitations=('Discrete candidate enumeration only; no interpolation, monotonic risk assumption or operative limit change. Concentration uses a supplied fixed post-decision denominator. Offer-conditioned risk is supplied, not inferred from baseline PD. Affordability and eligibility do not modify PD.',))


def compute_risk_based_price_grid(quote_candidates, pricing_profile, *, as_of, context=None):
    """Price complete amortizing quotes under supplied competing event hazards."""
    profile = deepcopy(pricing_profile)
    ctx = _context(profile, as_of, context)
    _required(profile, ['default_timing', 'prepayment_timing', 'funding_treatment', 'selection_assumption'])
    if profile['default_timing'] != 'BEFORE_DUE' or profile['prepayment_timing'] != 'AFTER_COUPON' or profile['funding_treatment'] != 'DISCOUNT_CURVE_ONLY' or profile['selection_assumption'] != 'QUOTE_CONDITIONED_INPUTS_NOT_CAUSAL':
        _fail('DECISION_PRICING_CONTRACT', 'Explicit default/prepayment ordering and funding/selection contract required')
    rows = _rows(quote_candidates, 'quote_candidates')
    ids, out = set(), []
    for row in sorted(rows, key=lambda x: x.get('offer_id', '')):
        _offer(row, ids, as_of, profile['currency'])
        _required(row, ['annual_rate', 'target', 'risk_conditioning', 'risk_reference', 'takeup_probability', 'takeup_reference', 'upfront_fee', 'origination_cost', 'offer_cost', 'periods'])
        _target(row['target'])
        _context(dict(profile, target=row['target']), as_of, ctx)
        if row['target']['horizon'] != row['tenor_days'] or row['risk_conditioning'] != 'CONDITIONAL_ON_THIS_ACCEPTED_QUOTE':
            _fail('TARGET_MISMATCH', 'Quote lifetime target and accepted-quote risk conditioning required')
        for k in ('risk_reference', 'takeup_reference'):
            _text(row[k], k)
        rate = _num(row['annual_rate'], 'annual_rate', 0)
        takeup = _num(row['takeup_probability'], 'takeup_probability', 0, 1)
        for k in ('upfront_fee', 'origination_cost', 'offer_cost'):
            _num(row[k], k, 0)
        periods = _rows(row['periods'], 'periods')
        if not periods:
            _fail('DECISION_PRICING_CONTRACT', 'Complete amortization calendar required')
        outstanding, survival, previous = float(row['amount']), 1., as_of
        cash, contributions, el_terms = [], [-outstanding, row['upfront_fee'], -row['origination_cost']], []
        default_mass = prepayment_mass = 0.
        discount_by_date = {as_of: 1.}
        for p in periods:
            _required(p, ['payment_date', 'scheduled_principal', 'conditional_default_probability', 'conditional_prepayment_probability', 'lgd', 'recovery_date', 'payment_discount_factor', 'recovery_discount_factor', 'servicing_cost', 'prepayment_fee'])
            end, start = _date(p['payment_date'], 'payment_date'), _date(previous, 'previous')
            recovery_date = _date(p['recovery_date'], 'recovery_date')
            if end <= start or recovery_date < end:
                _fail('DECISION_PRICING_CONTRACT', 'Increasing payment dates and recovery no earlier than default endpoint required')
            principal = _num(p['scheduled_principal'], 'scheduled_principal', 0)
            if principal > outstanding + 1e-12 * max(1, row['amount']):
                _fail('DECISION_PRICING_CONTRACT', 'Scheduled principal cannot exceed opening balance')
            principal = min(principal, outstanding)
            qd = _num(p['conditional_default_probability'], 'conditional_default_probability', 0, 1)
            qp = _num(p['conditional_prepayment_probability'], 'conditional_prepayment_probability', 0, 1)
            if qd + qp > 1:
                _fail('DECISION_PRICING_CONTRACT', 'Default and prepayment are mutually exclusive conditional hazards')
            lgd = _num(p['lgd'], 'lgd')
            df, rdf = _num(p['payment_discount_factor'], 'payment_discount_factor', 0), _num(p['recovery_discount_factor'], 'recovery_discount_factor', 0)
            if df == 0 or rdf == 0:
                _fail('DECISION_PRICING_CONTRACT', 'Positive date-specific funding discount factors required')
            for when, factor in ((p['payment_date'], df), (p['recovery_date'], rdf)):
                if when in discount_by_date and discount_by_date[when] != factor:
                    _fail('DECISION_PRICING_CONTRACT', 'Same cash date must use the same funding discount factor')
                discount_by_date[when] = factor
            cost, fee = _num(p['servicing_cost'], 'servicing_cost', 0), _num(p['prepayment_fee'], 'prepayment_fee', 0)
            dm, pm, remaining = survival * qd, survival * qp, survival * (1 - qd - qp)
            coupon = _sum([outstanding * rate * (end - start).days / 365])
            interest = (survival - dm) * coupon
            principal_cash = pm * outstanding + remaining * principal
            prepay_cash = pm * fee
            # Servicing charged for every account performing at the start of the period.
            servicing = survival * cost
            net_recovery = dm * outstanding * (1 - lgd)
            discounted_loss = dm * outstanding * lgd * rdf
            pv = _sum([(interest + principal_cash + prepay_cash - servicing) * df, net_recovery * rdf])
            contributions.append(pv)
            el_terms.append(discounted_loss)
            cash.append(dict(payment_date=p['payment_date'], recovery_date=p['recovery_date'], opening_survival=survival, default_mass=dm, prepayment_mass=pm, closing_survival=remaining, opening_principal=outstanding, coupon_if_nondefault=coupon, expected_interest=interest, expected_principal=principal_cash, expected_prepayment_fee=prepay_cash, expected_servicing_cost=servicing, expected_net_recovery=net_recovery, discounted_net_cashflow=pv, discounted_expected_loss=discounted_loss))
            default_mass += dm
            prepayment_mass += pm
            survival, outstanding, previous = remaining, max(0., outstanding - principal), p['payment_date']
        if (date.fromisoformat(previous) - date.fromisoformat(as_of)).days != row['tenor_days'] or outstanding > 1e-12 * max(1, row['amount']):
            _fail('DECISION_PRICING_CONTRACT', 'Full surviving amortization and calendar aligned to quote tenor required')
        booked_npv = _sum(contributions)
        per_offer = _sum([takeup * booked_npv, -row['offer_cost']])
        el = _sum(el_terms)
        out.append(dict(offer_id=row['offer_id'], version=row['version'], offer_hash=_fingerprint(row), amount=row['amount'], tenor_days=row['tenor_days'], annual_rate=rate, target=deepcopy(row['target']), takeup_probability=takeup, lifetime_pd=default_mass, prepayment_probability=prepayment_mass, terminal_performing_probability=survival, conditional_booked_npv=booked_npv, expected_npv_per_offer=per_offer, conditional_discounted_expected_loss=el, expected_loss_per_offer=takeup * el, booked_break_even=booked_npv >= 0, offer_break_even=per_offer >= 0, cashflows=cash))
    value = _bound(dict(schema_version='1.0', kind='PriceGrid', as_of=as_of, profile=profile, candidates=out))
    return _result(value, 'decisioning.compute_risk_based_price_grid', 'price_grid', ctx, limitations=('Supplied quote-conditioned default/prepayment hazards and takeup assumptions; no causal price elasticity is estimated. Fixed simple ACT/365F coupon; default before due misses coupon and exposes opening principal, prepayment after coupon returns remaining principal. Servicing charged on opening survivors. Funding enters only through supplied date-specific discount factors, not a second interest expense. Full surviving amortization required; no hidden terminal value. Risk scenarios can change with price, so break-even flags do not imply a monotone or unique break-even rate.',))


def evaluate_treatment_candidates(treatment_candidates, treatment_profile, *, as_of, context=None):
    """Compare explicit action-conditioned recovery plans with independent gates."""
    profile = deepcopy(treatment_profile)
    ctx = _context(profile, as_of, context)
    _required(profile, ['subject_id', 'baseline_id', 'baseline_present_value', 'identification', 'allowed_actions'])
    for k in ('subject_id', 'baseline_id'):
        _text(profile[k], k)
    _num(profile['baseline_present_value'], 'baseline_present_value')
    if profile['identification'] != 'CONDITIONAL_SCENARIO_NOT_CAUSAL' or not isinstance(profile['allowed_actions'], list) or any(not isinstance(x, str) or not x for x in profile['allowed_actions']):
        _fail('DECISION_TREATMENT_CONTRACT', 'Explicit allowed actions and noncausal conditional-scenario interpretation required')
    rows, ids, out = _rows(treatment_candidates, 'treatment_candidates'), set(), []
    for row in sorted(rows, key=lambda x: x.get('treatment_id', '')):
        _required(row, ['treatment_id', 'version', 'known_at', 'subject_id', 'currency', 'action', 'prediction_reference', 'support_status', 'affordability_status', 'eligibility_status', 'uptake_probability', 'offer_cost', 'accepted_upfront_cost', 'rejected_present_value', 'cashflows'])
        for k in ('treatment_id', 'version', 'action', 'prediction_reference'):
            _text(row[k], k)
        if row['treatment_id'] in ids:
            _fail('DECISION_DUPLICATE', 'Unique treatment IDs required')
        ids.add(row['treatment_id'])
        if row['currency'] != profile['currency'] or row['subject_id'] != profile['subject_id']:
            _fail('TARGET_MISMATCH', 'Treatment subject/currency must match profile')
        if _date(row['known_at'], 'known_at') > _date(as_of, 'as_of'):
            _fail('DECISION_FUTURE_INPUT', 'Treatment assumptions known by decision date required')
        if row['support_status'] not in ('SUPPORTED', 'UNSUPPORTED') or row['affordability_status'] not in ('AFFORDABLE', 'UNAFFORDABLE', 'NOT_REQUIRED', 'UNKNOWN') or row['eligibility_status'] not in ('ELIGIBLE', 'INELIGIBLE', 'UNKNOWN'):
            _fail('DECISION_TREATMENT_CONTRACT', 'Explicit support/affordability/eligibility states required')
        for k in ('offer_cost', 'accepted_upfront_cost'):
            _num(row[k], k, 0)
        uptake = _num(row['uptake_probability'], 'uptake_probability', 0, 1)
        reject = _num(row['rejected_present_value'], 'rejected_present_value')
        flows, fids, pv, nominal = _rows(row['cashflows'], 'cashflows'), set(), [], []
        for f in flows:
            _required(f, ['cashflow_id', 'date', 'expected_recovery', 'expected_cost', 'discount_factor'])
            _text(f['cashflow_id'], 'cashflow_id')
            if f['cashflow_id'] in fids or _date(f['date'], 'date') < _date(as_of, 'as_of'):
                _fail('DECISION_TREATMENT_CONTRACT', 'Unique future/as-of conditional cash flows required')
            fids.add(f['cashflow_id'])
            recovery, cost = _num(f['expected_recovery'], 'expected_recovery', 0), _num(f['expected_cost'], 'expected_cost', 0)
            df = _num(f['discount_factor'], 'discount_factor', 0)
            if df == 0:
                _fail('DECISION_TREATMENT_CONTRACT', 'Positive discount factor required')
            nominal.append(recovery - cost)
            pv.append((recovery - cost) * df)
        reasons = []
        if row['action'] not in profile['allowed_actions']:
            reasons.append('ACTION_NOT_ALLOWED')
        if row['support_status'] != 'SUPPORTED':
            reasons.append('UNSUPPORTED_PREDICTION')
        if row['affordability_status'] not in ('AFFORDABLE', 'NOT_REQUIRED'):
            reasons.append('AFFORDABILITY_NOT_ESTABLISHED')
        if row['eligibility_status'] != 'ELIGIBLE':
            reasons.append('ELIGIBILITY_NOT_ESTABLISHED')
        supported = row['support_status'] == 'SUPPORTED'
        conditional = _sum(pv + [-row['accepted_upfront_cost']]) if supported else None
        expected = _sum([uptake * conditional, (1 - uptake) * reject, -row['offer_cost']]) if supported else None
        out.append(dict(treatment_id=row['treatment_id'], version=row['version'], treatment_hash=_fingerprint(row), feasible=not reasons, failures=reasons, nominal_net_recovery=_sum(nominal) if supported else None, accepted_present_value=conditional, expected_present_value=expected, scenario_difference_from_baseline=_sum([expected, -profile['baseline_present_value']]) if supported else None))
    ranking = [r['treatment_id'] for r in sorted((r for r in out if r['feasible']), key=lambda r: (-r['expected_present_value'], r['treatment_id']))]
    value = _bound(dict(schema_version='1.0', kind='TreatmentEvaluation', as_of=as_of, profile=profile, candidates=out, feasible_ranking=ranking, status='FEASIBLE' if ranking else 'INFEASIBLE'))
    return _result(value, 'decisioning.evaluate_treatment_candidates', 'treatment_evaluation', ctx, limitations=('Action-conditioned cashflow scenarios, not identified causal treatment effects. Historical action selection and missing counterfactuals are not repaired. Expected recovery/cost amounts must already include within-acceptance timing and event probabilities; uptake is applied exactly once. Rejected value and baseline are explicit separate assumptions. Ranking is local and never executes a treatment.',))
