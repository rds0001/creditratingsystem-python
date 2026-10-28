"""Application evidence, selection-aware targets and conditional risk inference."""
from copy import deepcopy
import math

from .core import CalcResult, FitResult, _fail, _result, _fingerprint, _model_artifact
from .data import _rows, _date, _required, _unique, _days, build_observation_windows


def _object(x, name):
    if not isinstance(x, dict):
        _fail('APPLICATION_SCHEMA', f'{name} must be a mapping')
    return deepcopy(x)


def _string(x, name):
    if not isinstance(x, str) or not x.strip():
        _fail('APPLICATION_SCHEMA', f'{name} must be a nonempty string')
    return x


def _finite(x, name, nonnegative=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or (nonnegative and x < 0):
        _fail('APPLICATION_VALUE', f'{name} must be finite' + (' and nonnegative' if nonnegative else ''))
    return x


def _applications(applications):
    apps = _rows(applications, 'applications')
    _unique(apps, ['application_id'], 'applications')
    for a in apps:
        _required(a, ['application_id', 'customer_id', 'product_id', 'population_id', 'application_at', 'information_cutoff', 'requested_terms'])
        if _date(a['application_at'], 'application_at') > _date(a['information_cutoff'], 'information_cutoff'):
            _fail('APPLICATION_DATE_ORDER', 'Application is after information cutoff')
        _object(a['requested_terms'], 'requested_terms')
    return apps


def _snapshot(snapshot):
    if isinstance(snapshot, CalcResult):
        snapshot = snapshot.value
    s = _object(snapshot, 'snapshot')
    expected = s.pop('snapshot_hash', None)
    if expected is None or _fingerprint(s) != expected:
        _fail('APPLICATION_SNAPSHOT_INTEGRITY', 'Snapshot missing hash or altered after freezing')
    if s.get('schema_version') != '1.0':
        _fail('APPLICATION_SCHEMA', 'Unknown snapshot version')
    s['snapshot_hash'] = expected
    return s


def build_application_snapshot(applications, feature_records, term_events=None, required_features=None):
    """Freeze available feature revisions and each separately dated contract stage."""
    apps = _applications(applications)
    features = _rows(feature_records, 'feature_records')
    terms = _rows([] if term_events is None else term_events, 'term_events')
    app_ids = {a['application_id'] for a in apps}
    if required_features is not None and (not isinstance(required_features, (list, tuple)) or any(not isinstance(x, str) or not x for x in required_features)):
        _fail('APPLICATION_SCHEMA', 'required_features must be a sequence of names')
    required_features = list(required_features or [])
    if len(set(required_features)) != len(required_features):
        _fail('APPLICATION_SCHEMA', 'required_features must contain distinct names')
    for r in features:
        _required(r, ['application_id', 'feature', 'observed_at', 'known_at', 'source_id'])
        _string(r['feature'], 'feature')
        if 'value' not in r:
            _fail('APPLICATION_SCHEMA', 'Feature record requires an explicit value, possibly null')
        if r['application_id'] not in app_ids:
            _fail('APPLICATION_FOREIGN_KEY', 'Feature refers to unknown application')
        if _date(r['known_at'], 'known_at') < _date(r['observed_at'], 'observed_at'):
            _fail('APPLICATION_DATE_ORDER', 'Feature known before observation')
        _fingerprint(r['value'])
    for r in terms:
        _required(r, ['application_id', 'stage', 'terms', 'effective_at', 'known_at'])
        if r['application_id'] not in app_ids:
            _fail('APPLICATION_FOREIGN_KEY', 'Terms refer to unknown application')
        if r['stage'] not in ('offered', 'accepted', 'booked'):
            _fail('APPLICATION_SCHEMA', 'Term stage must be offered, accepted or booked')
        _object(r['terms'], 'terms')
        if _date(r['known_at'], 'known_at') < _date(r['effective_at'], 'effective_at'):
            _fail('APPLICATION_DATE_ORDER', 'Term event known before effective date')
    output = []
    for a in apps:
        cutoff = _date(a['information_cutoff'], 'information_cutoff')
        candidates = [r for r in features if r['application_id'] == a['application_id'] and _date(r['known_at'], 'known_at') <= cutoff and _date(r['observed_at'], 'observed_at') <= cutoff]
        values, sources = {}, {}
        for name in sorted({r['feature'] for r in candidates}):
            records = [r for r in candidates if r['feature'] == name]
            newest = max((r['observed_at'], r['known_at']) for r in records)
            selected = [r for r in records if (r['observed_at'], r['known_at']) == newest]
            if len(selected) != 1:
                _fail('APPLICATION_AMBIGUOUS_FEATURE', f'Ambiguous feature revision {name}')
            row = selected[0]
            values[name] = deepcopy(row['value'])
            sources[name] = {k: row[k] for k in ('source_id', 'observed_at', 'known_at')}
        stages, stage_sources = {'requested': deepcopy(a['requested_terms']), 'offered': None, 'accepted': None, 'booked': None}, {}
        for stage in ('offered', 'accepted', 'booked'):
            records = [r for r in terms if r['application_id'] == a['application_id'] and r['stage'] == stage and _date(r['known_at'], 'known_at') <= cutoff and _date(r['effective_at'], 'effective_at') <= cutoff]
            if records:
                newest = max((r['effective_at'], r['known_at']) for r in records)
                selected = [r for r in records if (r['effective_at'], r['known_at']) == newest]
                if len(selected) != 1:
                    _fail('APPLICATION_AMBIGUOUS_TERMS', 'Ambiguous term-stage revision')
                stages[stage] = deepcopy(selected[0]['terms'])
                stage_sources[stage] = {k: selected[0][k] for k in ('effective_at', 'known_at')}
        frozen = {'schema_version': '1.0', **{k: a[k] for k in ('application_id', 'customer_id', 'product_id', 'population_id', 'application_at', 'information_cutoff')}, 'features': values, 'feature_sources': sources, 'terms': stages, 'term_sources': stage_sources, 'missing_features': [f for f in required_features if values.get(f) is None]}
        frozen['snapshot_hash'] = _fingerprint(frozen)
        output.append(frozen)
    return _result(output, 'application.build_application_snapshot', diagnostics={'hash_format': 'cr-canonical-v1', 'information_rule': 'observed_at and known_at <= information_cutoff'}, limitations=('Hashes detect mutation; returned in-memory objects are not write-protected.',))


def assemble_joint_application(application_id, parties, income_items, obligation_items, currency):
    """Deduplicate economic items while retaining individual legal liability roles."""
    _string(application_id, 'application_id'); _string(currency, 'currency')
    parties = _rows(parties, 'parties')
    _unique(parties, ['party_id'], 'parties')
    pmap = {}
    for p in parties:
        _required(p, ['party_id', 'role', 'liability_share'])
        if p['role'] not in ('borrower', 'co_borrower', 'guarantor'):
            _fail('APPLICATION_SCHEMA', 'Unsupported party role')
        if not 0 <= _finite(p['liability_share'], 'liability_share') <= 1:
            _fail('APPLICATION_VALUE', 'Liability share must lie in [0,1]')
        pmap[p['party_id']] = p
    borrowers = {p['party_id'] for p in parties if p['role'] != 'guarantor'}
    if not borrowers:
        _fail('APPLICATION_SCHEMA', 'At least one borrower is required')
    def aggregate(items, name):
        unique, excluded = {}, []
        for row in _rows(items, name):
            _required(row, ['economic_id', 'party_id', 'amount_monthly', 'currency'])
            if row['party_id'] not in pmap:
                _fail('APPLICATION_FOREIGN_KEY', 'Economic item refers to unknown party')
            amount = _finite(row['amount_monthly'], 'amount_monthly', nonnegative=True)
            if row['currency'] != currency:
                _fail('APPLICATION_CURRENCY', 'Economic item currency mismatch')
            if row['party_id'] not in borrowers:
                excluded.append({'economic_id': row['economic_id'], 'party_id': row['party_id'], 'reason': 'guarantor_only'})
                continue
            key = row['economic_id']
            if key in unique and unique[key]['amount_monthly'] != amount:
                _fail('APPLICATION_AMBIGUOUS_ECONOMIC_ITEM', 'Shared economic ID has conflicting total amounts')
            unique.setdefault(key, {'economic_id': key, 'amount_monthly': amount, 'party_ids': []})
            if row['party_id'] not in unique[key]['party_ids']:
                unique[key]['party_ids'].append(row['party_id'])
        rows = sorted(unique.values(), key=lambda r: r['economic_id'])
        for r in rows:
            r['party_ids'].sort()
        return rows, excluded
    income, income_excluded = aggregate(income_items, 'income_items')
    obligations, debt_excluded = aggregate(obligation_items, 'obligation_items')
    households = [p.get('household_id') for p in parties if p.get('household_id')]
    return _result({'application_id': application_id, 'currency': currency, 'monthly_income': sum(r['amount_monthly'] for r in income), 'monthly_debt_service': sum(r['amount_monthly'] for r in obligations), 'parties': deepcopy(parties), 'income_contributions': income, 'obligation_contributions': obligations, 'excluded_items': income_excluded + debt_excluded, 'dependency_flags': {'shared_household': len(set(households)) < len(households), 'shared_income': any(len(r['party_ids']) > 1 for r in income), 'shared_obligation': any(len(r['party_ids']) > 1 for r in obligations)}}, 'application.assemble_joint_application', limitations=('Item amounts are total economic amounts, not liability-share-weighted amounts; guarantor resources are excluded.', 'No aggregation of individual PDs; no inferred household identities.'))


def build_application_targets(applications, decisions, bookings, default_events, exits, horizon_days, evaluation_cutoff, performance_origin='first_draw_at'):
    """Link selection stages to observable funded-facility first-default labels."""
    apps = _applications(applications)
    decisions, bookings = _rows(decisions, 'decisions'), _rows(bookings, 'bookings')
    cutoff = _date(evaluation_cutoff, 'evaluation_cutoff')
    _days(horizon_days, 'horizon_days', 1)
    if performance_origin not in ('booked_at', 'first_draw_at'):
        _fail('APPLICATION_SCHEMA', 'performance_origin must be booked_at or first_draw_at')
    app_ids = {a['application_id'] for a in apps}
    app_map = {a['application_id']: a for a in apps}
    if any(_date(a['information_cutoff'], 'information_cutoff') > cutoff for a in apps):
        _fail('APPLICATION_COHORT_CUTOFF', 'Application population includes information cutoffs after evaluation cutoff')
    build_observation_windows([], default_events, exits, horizon_days, evaluation_cutoff, entity_key='facility_id')
    _unique(decisions, ['decision_id'], 'decisions'); _unique(bookings, ['application_id'], 'bookings'); _unique(bookings, ['facility_id'], 'bookings')
    for d in decisions:
        _required(d, ['decision_id', 'application_id', 'status', 'decided_at', 'known_at'])
        if d['application_id'] not in app_ids or d['status'] not in ('approved', 'rejected', 'pending', 'withdrawn'):
            _fail('APPLICATION_SCHEMA', 'Invalid decision application or status')
        if _date(d['decided_at'], 'decided_at') < _date(app_map[d['application_id']]['application_at'], 'application_at'):
            _fail('APPLICATION_DATE_ORDER', 'Decision precedes application')
        if _date(d['known_at'], 'known_at') < _date(d['decided_at'], 'decided_at'):
            _fail('APPLICATION_DATE_ORDER', 'Decision known before it occurred')
        if d.get('accepted_at') is not None:
            if d['status'] != 'approved':
                _fail('APPLICATION_SELECTION_CONFLICT', 'Acceptance requires an approved decision')
            _required(d, ['accepted_known_at'])
            if _date(d['accepted_at'], 'accepted_at') < _date(d['decided_at'], 'decided_at') or _date(d['accepted_known_at'], 'accepted_known_at') < _date(d['accepted_at'], 'accepted_at'):
                _fail('APPLICATION_DATE_ORDER', 'Invalid acceptance dates')
    for b in bookings:
        _required(b, ['application_id', 'facility_id', 'booked_at', 'known_at'])
        if b['application_id'] not in app_ids:
            _fail('APPLICATION_FOREIGN_KEY', 'Booking refers to unknown application')
        if _date(b['known_at'], 'known_at') < _date(b['booked_at'], 'booked_at'):
            _fail('APPLICATION_DATE_ORDER', 'Booking known before occurrence')
        if b.get('first_draw_at') is not None:
            _required(b, ['first_draw_known_at'])
            if _date(b['first_draw_at'], 'first_draw_at') < _date(b['booked_at'], 'booked_at') or _date(b['first_draw_known_at'], 'first_draw_known_at') < _date(b['first_draw_at'], 'first_draw_at'):
                _fail('APPLICATION_DATE_ORDER', 'Invalid first-draw dates')
    out = []
    for a in apps:
        ds = [d for d in decisions if d['application_id'] == a['application_id'] and _date(d['known_at'], 'known_at') <= cutoff]
        decision = None
        if ds:
            newest = max((d['decided_at'], d['known_at']) for d in ds)
            selected = [d for d in ds if (d['decided_at'], d['known_at']) == newest]
            if len(selected) != 1:
                _fail('APPLICATION_AMBIGUOUS_DECISION', 'Multiple decisions tie on effective/knowledge dates')
            decision = selected[0]
        bs = [b for b in bookings if b['application_id'] == a['application_id'] and _date(b['known_at'], 'known_at') <= cutoff]
        booking = bs[0] if bs else None
        approved = decision is not None and decision['status'] == 'approved'
        accepted = approved and decision.get('accepted_at') is not None and _date(decision['accepted_known_at'], 'accepted_known_at') <= cutoff
        if booking and not approved:
            _fail('APPLICATION_SELECTION_CONFLICT', 'Known booking requires a known approved decision')
        origin_known = booking is not None and (performance_origin == 'booked_at' or booking.get('first_draw_at') is not None and _date(booking['first_draw_known_at'], 'first_draw_known_at') <= cutoff)
        if booking and not accepted:
            _fail('APPLICATION_SELECTION_CONFLICT', 'Known booking requires a known acceptance')
        if booking and (_date(booking['booked_at'], 'booked_at') < _date(decision['accepted_at'], 'accepted_at') or _date(booking['known_at'], 'known_at') < _date(decision['accepted_known_at'], 'accepted_known_at')):
            _fail('APPLICATION_DATE_ORDER', 'Booking precedes occurrence or knowledge of acceptance')
        selection = 'funded' if origin_known else 'booked_not_drawn' if booking else 'accepted_not_booked' if accepted else 'approved_not_taken_up' if approved else decision['status'] if decision else 'pending_decision'
        base = {'application_id': a['application_id'], 'customer_id': a['customer_id'], 'product_id': a['product_id'], 'information_cutoff': a['information_cutoff'], 'selection_status': selection, 'approved': approved, 'accepted': accepted, 'booked': booking is not None, 'funded': bool(origin_known), 'performance_origin': performance_origin, 'facility_id': booking['facility_id'] if booking else None}
        if origin_known:
            rating = {'observation_id': a['application_id'], 'facility_id': booking['facility_id'], 'as_of': a['information_cutoff'], 'performance_start': booking[performance_origin], 'funded': True}
            for key in ('followup_end', 'outcome_available_at'):
                if key in booking:
                    rating[key] = booking[key]
            window = build_observation_windows([rating], default_events, exits, horizon_days, evaluation_cutoff, entity_key='facility_id').value[0]
            base.update({k: window[k] for k in ('performance_start', 'outcome_end', 'observed_until', 'target', 'censored', 'eligible', 'label_status', 'event_id')})
            base['mature'] = _date(window['outcome_end'], 'outcome_end') <= cutoff
        else:
            base.update({'performance_start': None, 'outcome_end': None, 'observed_until': None, 'target': None, 'censored': False, 'eligible': False, 'label_status': 'not_funded', 'event_id': None, 'mature': False})
        base['label_observed'] = base['target'] is not None
        out.append(base)
    return _result(out, 'application.build_application_targets', unit='binary', diagnostics={'target': 'funded_facility_first_default', 'performance_origin': performance_origin, 'horizon_days': horizon_days}, limitations=('Unfunded counterfactual loan outcomes remain missing; first-default excludes previously defaulted facilities.',))


def _design(snapshot, model, target, override_terms=None):
    s = _snapshot(snapshot)
    target = _object(target, 'target')
    if isinstance(model, FitResult):
        model = model.model
    model = _model_artifact(model)
    contract = model.get('application_contract')
    if not isinstance(contract, dict):
        _fail('APPLICATION_MODEL_CONTRACT', 'Model requires explicit application_contract metadata')
    for name in ('product_ids', 'population_ids'):
        if not isinstance(contract.get(name), list) or not contract[name] or any(not isinstance(x, str) or not x for x in contract[name]):
            _fail('APPLICATION_MODEL_CONTRACT', f'{name} must list applicable IDs')
    if s['product_id'] not in contract['product_ids'] or s['population_id'] not in contract['population_ids']:
        _fail('APPLICATION_APPLICABILITY', 'Product or population outside model applicability')
    if contract.get('performance_origin') not in ('booked_at', 'first_draw_at') or target.get('performance_origin') != contract['performance_origin']:
        _fail('APPLICATION_TARGET', 'Target performance origin must match application contract')
    if target != model['target']:
        _fail('APPLICATION_TARGET', 'Complete target must match model')
    if target.get('probability_measure') != 'PHYSICAL':
        _fail('APPLICATION_TARGET', 'Application default inference requires physical probability')
    stage = contract.get('term_stage')
    if stage not in ('requested', 'offered', 'accepted', 'booked'):
        _fail('APPLICATION_MODEL_CONTRACT', 'Explicit term_stage is required')
    term_features = contract.get('term_features')
    if not isinstance(term_features, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in term_features.items()) or len(set(term_features.values())) != len(term_features):
        _fail('APPLICATION_MODEL_CONTRACT', 'term_features must map distinct term columns to feature names')
    if any(v not in model['feature_order'] for v in term_features.values()):
        _fail('APPLICATION_MODEL_CONTRACT', 'Term feature absent from fitted model')
    terms = override_terms if override_terms is not None else s['terms'].get(stage)
    if term_features and not isinstance(terms, dict):
        _fail('APPLICATION_MISSING_TERMS', f'No known {stage} terms available')
    values = deepcopy(s['features'])
    for field, feature in term_features.items():
        if field not in terms or terms[field] is None:
            _fail('APPLICATION_MISSING_TERMS', f'Missing term {field}')
        if feature in values:
            _fail('APPLICATION_FEATURE_COLLISION', 'Contract term features must be absent from customer feature namespace')
        values[feature] = terms[field]
    missing = [f for f in model['feature_order'] if values.get(f) is None]
    if missing:
        _fail('APPLICATION_MISSING_FEATURES', 'Required model features are missing', {'features': missing})
    x = [_finite(values[f], f) for f in model['feature_order']]
    return s, model, contract, x


def _predict(snapshot, model, target, calibrator=None, offset=None, override_terms=None):
    from .classification import predict_pd
    s, model, contract, x = _design(snapshot, model, target, override_terms)
    raw = float(predict_pd(model, [x], feature_order=model['feature_order'], expected_target=target, offset=None if offset is None else [offset]).value[0])
    pd = raw
    if calibrator is not None:
        from .calibration import apply_calibrator
        artifact = calibrator.value if isinstance(calibrator, CalcResult) else calibrator
        artifact = _object(artifact, 'calibrator')
        if artifact.get('calibration_type') == 'isotonic' and artifact.get('metadata', {}).get('input_scale') != 'probability':
            _fail('APPLICATION_CALIBRATION_SCALE', 'Isotonic calibrator must declare input_scale=probability')
        pd = float(apply_calibrator(calibrator, [raw], target=target, base_model_version=model['model_version']).value[0])
    return {'application_id': s['application_id'], 'snapshot_hash': s['snapshot_hash'], 'pd': pd, 'unadjusted_pd': raw, 'calibrated': calibrator is not None, 'model_id': model['model_id'], 'model_version': model['model_version'], 'target': deepcopy(target), 'product_id': s['product_id'], 'population_id': s['population_id'], 'term_stage': contract['term_stage'], 'feature_coverage': {'required': len(model['feature_order']), 'observed': len(x), 'missing': []}}


def predict_application_risk(snapshot, model, target, calibrator=None, offset=None):
    """Predict an explicitly applicable physical default target from frozen evidence."""
    value = _predict(snapshot, model, target, calibrator, offset)
    return _result(value, 'application.predict_application_risk', unit='probability', limitations=('Risk inference is separate from affordability, eligibility and credit approval.',))


def compare_offer_risk(snapshot, offers, model, target, support_bounds, calibrator=None, lgd=None, offset=None):
    """Compare conditional offered-term scenarios, with explicit extrapolation flags."""
    s = _snapshot(snapshot)
    artifact = model.model if isinstance(model, FitResult) else model
    artifact = _model_artifact(artifact)
    contract = artifact.get('application_contract', {})
    mapping = contract.get('term_features')
    if contract.get('term_stage') != 'offered' or not isinstance(mapping, dict) or not mapping:
        _fail('APPLICATION_OFFER_MODEL', 'Offer comparison requires a model conditional on explicit offered term features')
    bounds = _object(support_bounds, 'support_bounds')
    if set(bounds) != set(mapping):
        _fail('APPLICATION_SUPPORT', 'Supply support bounds for every modelled term')
    for field, pair in bounds.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or _finite(pair[0], field) > _finite(pair[1], field):
            _fail('APPLICATION_SUPPORT', 'Support bounds require ordered [lower,upper]')
    offers = _rows(offers, 'offers'); _unique(offers, ['offer_id'], 'offers')
    if lgd is not None and not 0 <= _finite(lgd, 'lgd') <= 1:
        _fail('APPLICATION_VALUE', 'LGD must lie in [0,1]')
    output = []
    for offer in offers:
        _required(offer, ['offer_id', 'terms'])
        terms = _object(offer['terms'], 'offer.terms')
        if set(terms) != set(mapping):
            _fail('APPLICATION_OFFER_MODEL', 'Every scenario term must be explicitly modelled; no silent ignored offer dimensions')
        extrapolated = [f for f, pair in bounds.items() if not pair[0] <= _finite(terms[f], f) <= pair[1]]
        risk = _predict(s, artifact, target, calibrator, offset, override_terms=terms)
        el = None
        if lgd is not None:
            if 'amount' not in terms:
                _fail('APPLICATION_OFFER_MODEL', 'EL requires an explicit amount term used as scenario EAD')
            _string(s['terms']['requested'].get('currency'), 'requested_terms.currency for EL')
            el = risk['pd'] * lgd * _finite(terms['amount'], 'amount', nonnegative=True)
        output.append({'offer_id': offer['offer_id'], 'terms': terms, 'pd': risk['pd'], 'unadjusted_pd': risk['unadjusted_pd'], 'expected_loss': el, 'currency': s['terms']['requested'].get('currency') if lgd is not None else None, 'extrapolation': bool(extrapolated), 'outside_support_terms': extrapolated, 'snapshot_hash': s['snapshot_hash']})
    return _result(output, 'application.compare_offer_risk', diagnostics={'support_bounds': bounds, 'comparison': 'fixed_customer_conditional_scenarios', 'ead_assumption': 'offered amount' if lgd is not None else None}, limitations=('Offer changes are conditional model scenarios, not identified causal effects or take-up predictions.', 'Bounds are user-supplied empirical support; marginal bounds do not establish joint support.', 'Optional EL treats offered amount as fixed EAD and supplied LGD as constant.'))
