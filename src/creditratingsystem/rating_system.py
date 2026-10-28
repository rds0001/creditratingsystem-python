"""Local dated rating objects, immutable model registry and deterministic replay.

Artifacts contain data only. Hashes bind content, not institutional approval.
"""
from copy import deepcopy
import math

from .core import CalcResult, CreditRiskError, _fail, _fingerprint, _model_artifact, _result
from .data import _date, _days, _required, _rows, _unique

_TYPES = ('customer', 'obligor', 'facility', 'issue', 'transaction', 'portfolio')
_RELATIONS = ('legal_obligor', 'joint_liability', 'economic_dependency', 'default_correlation',
              'data_identity', 'contains', 'pd_reference')


def _object(value, name):
    if isinstance(value, CalcResult):
        value = value.value
    if not isinstance(value, dict):
        _fail('RATING_SCHEMA', f'{name} must be an object')
    _fingerprint(value)
    return deepcopy(value)


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        _fail('RATING_SCHEMA', f'{name} must be a nonempty string')
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail('RATING_SCHEMA', f'{name} must be finite numeric')
    return value


def _dates(as_of, known_at):
    a, k = _date(as_of, 'as_of'), _date(known_at, 'known_at')
    if a > k:
        _fail('RATING_DATE_ORDER', 'Economic as_of must not follow knowledge cutoff')
    return a, k


def _interval(row):
    _required(row, ['effective_from', 'known_at'])
    start = _date(row['effective_from'], 'effective_from')
    end = None if row.get('effective_to') is None else _date(row['effective_to'], 'effective_to')
    _date(row['known_at'], 'known_at')
    if end is not None and end <= start:
        _fail('RATING_DATE_ORDER', 'Effective interval must be nonempty [from,to)')
    return start, end


def _effective(row, as_of, known_at):
    start, end = _interval(row)
    return _date(row['known_at'], 'known_at') <= known_at and start <= as_of and (end is None or as_of < end)


def _seal(value, key):
    value = deepcopy(value)
    value.pop(key, None)
    value[key] = _fingerprint(value)
    return value


def _sealed(value, key, name):
    value = _object(value, name)
    expected = value.pop(key, None)
    if expected is None or _fingerprint(value) != expected:
        _fail('RATING_INTEGRITY', f'{name} has missing or mismatched content hash')
    value[key] = expected
    if value.get('schema_version') != '1.0':
        _fail('RATING_SCHEMA', f'Unsupported {name} schema')
    return value


def _snapshot(snapshot, as_of, known_at):
    s = _object(snapshot, 'subject snapshot')
    _required(s, ['subject_id', 'subject_type', 'as_of', 'known_at', 'attributes'])
    if s['subject_type'] not in _TYPES:
        _fail('RATING_SCHEMA', 'Unsupported subject type')
    a, k = _dates(as_of, known_at)
    if _date(s['as_of'], 'snapshot.as_of') != a or _date(s['known_at'], 'snapshot.known_at') > k:
        _fail('RATING_SNAPSHOT_DATE', 'Snapshot must match economic date and be known by cutoff')
    attrs = _object(s['attributes'], 'attributes')
    values, findings = {'subject_type': s['subject_type'], 'subject_id': s['subject_id']}, []
    for field, fact in attrs.items():
        if field in values:
            _fail('RATING_SCHEMA', 'Attribute conflicts with a subject identity field')
        fact = _object(fact, field)
        _required(fact, ['observed_at', 'known_at'])
        if 'value' not in fact:
            _fail('RATING_SCHEMA', 'Attribute requires explicit value, possibly null')
        observed, available = _date(fact['observed_at'], 'observed_at'), _date(fact['known_at'], 'known_at')
        if available < observed:
            _fail('RATING_DATE_ORDER', 'Attribute known before observed')
        if observed > a or available > _date(s['known_at'], 'snapshot.known_at'):
            findings.append({'field': field, 'code': 'UNAVAILABLE'})
            values[field] = None
        else:
            values[field] = fact['value']
    return s, values, findings


def build_rating_subjects(tables, links, as_of, known_at):
    """Select dated subject versions and preserve explicit typed relationships."""
    tables = _object(tables, 'tables')
    a, k = _dates(as_of, known_at)
    if any(t not in _TYPES for t in tables):
        _fail('RATING_SCHEMA', 'Unknown subject table')
    nodes, conflicts = [], []
    for typ in sorted(tables):
        rows = _rows(tables[typ], typ)
        _unique(rows, ['subject_id', 'version'], typ)
        for row in rows:
            _required(row, ['subject_id', 'version', 'attributes'])
            _string(row['version'], 'version'); _object(row['attributes'], 'attributes'); _interval(row)
            if 'subject_type' in row:
                _fail('RATING_SCHEMA', 'Subject type is defined by table name, not duplicated in rows')
        for identity in sorted({r['subject_id'] for r in rows}):
            active = [r for r in rows if r['subject_id'] == identity and _effective(r, a, k)]
            if len(active) > 1:
                conflicts.append({'subject_type': typ, 'subject_id': identity, 'code': 'OVERLAPPING_VERSIONS', 'versions': sorted(r['version'] for r in active)})
            elif active:
                nodes.append({'subject_type': typ, **deepcopy(active[0])})
    node_keys = {(n['subject_type'], n['subject_id']) for n in nodes}
    edges, unresolved = [], []
    links = _rows(links, 'links'); _unique(links, ['link_id'], 'links')
    for edge in sorted(links, key=lambda r: r['link_id']):
        _required(edge, ['link_id', 'relation', 'from_type', 'from_id', 'to_type', 'to_id'])
        if edge['relation'] not in _RELATIONS or edge['from_type'] not in _TYPES or edge['to_type'] not in _TYPES:
            _fail('RATING_SCHEMA', 'Unsupported edge relationship or subject type')
        if not _effective(edge, a, k):
            continue
        if (edge['from_type'], edge['from_id']) not in node_keys or (edge['to_type'], edge['to_id']) not in node_keys:
            unresolved.append({'link_id': edge['link_id'], 'code': 'UNRESOLVED_ENDPOINT'})
        else:
            edges.append(deepcopy(edge))
    return _result({'schema_version': '1.0', 'as_of': as_of, 'known_at': known_at, 'subjects': nodes, 'links': edges, 'conflicts': conflicts, 'unresolved_links': unresolved}, 'rating_system.build_rating_subjects', limitations=('Relationships never copy PD, grade, LGD or loss between subject levels.',))


def _condition(condition, values):
    c = _object(condition, 'condition'); _required(c, ['field', 'operator'])
    _string(c['field'], 'condition.field')
    op = c['operator']; value = values.get(c['field'])
    if op not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'in', 'not_in', 'present', 'absent'):
        _fail('RATING_RULE', 'Unsupported condition operator')
    if op == 'present':
        return value is not None
    if op == 'absent':
        return value is None
    if 'value' not in c:
        _fail('RATING_RULE', 'Comparison condition requires value')
    expected = c['value']
    if op in ('in', 'not_in') and not isinstance(expected, list):
        _fail('RATING_RULE', 'Membership comparison requires a list')
    if op in ('lt', 'le', 'gt', 'ge'):
        _number(expected, 'condition.value')
        if value is not None:
            _number(value, c['field'])
    if value is None:
        return False
    if op in ('eq', 'ne'):
        same = _fingerprint(value) == _fingerprint(expected)
        return same if op == 'eq' else not same
    if op in ('in', 'not_in'):
        member = any(_fingerprint(value) == _fingerprint(x) for x in expected)
        return member if op == 'in' else not member
    return {'lt': lambda: value < expected, 'le': lambda: value <= expected,
            'gt': lambda: value > expected, 'ge': lambda: value >= expected}[op]()


def route_rating_profile(snapshot, rules, as_of, known_at):
    """Evaluate dated declarative rules; equal best priorities are ambiguous."""
    _, values, unavailable = _snapshot(snapshot, as_of, known_at)
    a, k = _dates(as_of, known_at)
    rules = _rows(rules, 'rules'); _unique(rules, ['rule_id'], 'rules')
    trace, matches = [], []
    for r in rules:
        _required(r, ['rule_id', 'profile_id', 'profile_version', 'priority', 'conditions'])
        _string(r['profile_version'], 'profile_version'); _number(r['priority'], 'priority')
        conditions = _rows(r['conditions'], 'conditions')
        evaluated = [_condition(c, values) for c in conditions]
        active = _effective(r, a, k)
        matched = active and all(evaluated)
        trace.append({'rule_id': r['rule_id'], 'priority': r['priority'], 'effective': active, 'condition_matches': evaluated, 'matched': matched})
        if matched:
            matches.append(r)
    best = [] if not matches else [r for r in matches if r['priority'] == min(v['priority'] for v in matches)]
    status = 'NO_ROUTE' if not best else 'AMBIGUOUS' if len(best) > 1 else 'ROUTED'
    return _result({'status': status, 'profile_id': best[0]['profile_id'] if len(best) == 1 else None, 'profile_version': best[0]['profile_version'] if len(best) == 1 else None, 'rule_id': best[0]['rule_id'] if len(best) == 1 else None, 'candidate_rule_ids': sorted(r['rule_id'] for r in best), 'trace': sorted(trace, key=lambda r: (r['priority'], r['rule_id'])), 'unavailable_features': unavailable}, 'rating_system.route_rating_profile')


def assess_rating_applicability(snapshot, contract, as_of, known_at):
    """Separate incompatible purpose/population from incomplete or stale evidence."""
    s, values, findings = _snapshot(snapshot, as_of, known_at)
    c = _object(contract, 'applicability contract')
    _required(c, ['contract_id', 'version', 'required_features', 'allowed_values', 'support_bounds', 'maximum_age_days'])
    required = c['required_features']
    if not isinstance(required, list) or any(not isinstance(x, str) for x in required) or len(set(required)) != len(required):
        _fail('RATING_SCHEMA', 'required_features must be distinct names')
    allowed, bounds, ages = (_object(c[x], x) for x in ('allowed_values', 'support_bounds', 'maximum_age_days'))
    incompatible = False
    for name, options in allowed.items():
        if not isinstance(options, list) or not options:
            _fail('RATING_SCHEMA', 'Allowed values must be nonempty lists')
        if values.get(name) is None:
            findings.append({'field': name, 'code': 'MISSING_APPLICABILITY_FIELD'})
        elif not any(_fingerprint(values[name]) == _fingerprint(v) for v in options):
            incompatible = True; findings.append({'field': name, 'code': 'OUTSIDE_APPLICABILITY'})
    for name in required:
        if values.get(name) is None:
            findings.append({'field': name, 'code': 'MISSING_REQUIRED'})
    for name, pair in bounds.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or _number(pair[0], name) > _number(pair[1], name):
            _fail('RATING_SCHEMA', 'Support requires ordered numeric bounds')
        if values.get(name) is not None and not pair[0] <= _number(values[name], name) <= pair[1]:
            findings.append({'field': name, 'code': 'OUTSIDE_TRAINING_SUPPORT'})
    for name, limit in ages.items():
        _days(limit, name)
        if name in s['attributes'] and values.get(name) is not None and (_date(as_of, 'as_of') - _date(s['attributes'][name]['observed_at'], 'observed_at')).days > limit:
            findings.append({'field': name, 'code': 'STALE'})
    for name, fact in s['attributes'].items():
        if fact.get('conflict', False) is not False:
            findings.append({'field': name, 'code': 'SOURCE_CONFLICT'})
    return _result({'status': 'NOT_APPLICABLE' if incompatible else 'REFER' if findings else 'APPLICABLE', 'findings': sorted(findings, key=lambda f: (f['field'], f['code'])), 'required_count': len(required), 'observed_required_count': sum(values.get(f) is not None for f in required), 'contract_id': c['contract_id'], 'contract_version': c['version']}, 'rating_system.assess_rating_applicability', limitations=('Applicability and marginal support checks are not credit approval or empirical model validation.',))


def _registry(registry):
    if registry is None:
        return _seal({'schema_version': '1.0', 'entries': []}, 'registry_hash')
    r = _sealed(registry, 'registry_hash', 'registry')
    entries = _rows(r.get('entries'), 'registry.entries')
    _unique(entries, ['model_id', 'model_version'], 'registry entries')
    for e in entries:
        _required(e, ['model_id', 'model_version', 'registered_at', 'model_hash', 'bundle'])
        _date(e['registered_at'], 'registered_at')
        if _fingerprint(e['bundle']) != e['model_hash']:
            _fail('RATING_INTEGRITY', 'Registered bundle hash mismatch')
        bundle = _object(e['bundle'], 'registered bundle')
        model = _model_artifact(bundle.get('model'))
        if model['model_id'] != e['model_id'] or model['model_version'] != e['model_version']:
            _fail('RATING_MODEL_IDENTITY', 'Registry model identity mismatch')
    return r


def register_rating_model_version(registry, bundle, registration):
    """Return a new content-addressed registry; identical registration is idempotent."""
    registry = _registry(registry)
    bundle = _object(bundle, 'bundle'); registration = _object(registration, 'registration')
    _required(bundle, ['model', 'dependencies', 'configuration'])
    model = _model_artifact(bundle['model']); bundle['model'] = model
    _object(bundle['dependencies'], 'dependencies'); _object(bundle['configuration'], 'configuration')
    _required(registration, ['model_id', 'model_version', 'registered_at'])
    if registration['model_id'] != model['model_id'] or registration['model_version'] != model['model_version']:
        _fail('RATING_MODEL_IDENTITY', 'Registration must match model identity')
    when = _date(registration['registered_at'], 'registered_at')
    if _date(model['training']['known_at'], 'model.training.known_at') > when:
        _fail('RATING_DATE_ORDER', 'Registration cannot precede model training knowledge')
    digest = _fingerprint(bundle)
    matches = [e for e in registry['entries'] if (e['model_id'], e['model_version']) == (model['model_id'], model['model_version'])]
    if matches:
        if matches[0]['model_hash'] != digest or matches[0]['registration'] != registration:
            _fail('RATING_VERSION_CONFLICT', 'Existing model version has different content or registration metadata')
        return _result(registry, 'rating_system.register_rating_model_version', diagnostics={'idempotent': True})
    registry['entries'].append({'model_id': model['model_id'], 'model_version': model['model_version'], 'registered_at': registration['registered_at'], 'model_hash': digest, 'registration': registration, 'bundle': bundle})
    registry['entries'].sort(key=lambda e: (e['model_id'], e['model_version']))
    return _result(_seal(registry, 'registry_hash'), 'rating_system.register_rating_model_version', diagnostics={'idempotent': False})


def select_effective_model(registry, assignments, profile_id, purpose, as_of, known_at):
    """Select a historical assignment, never an implicit latest model version."""
    registry = _registry(registry); a, k = _dates(as_of, known_at)
    if isinstance(assignments, CalcResult) or isinstance(assignments, dict):
        from .monitoring import _assignment_projection
        assignments = _assignment_projection(assignments, known_at, role='champion')
    _string(profile_id, 'profile_id'); _string(purpose, 'purpose')
    assignments = _rows(assignments, 'assignments'); _unique(assignments, ['assignment_id'], 'assignments')
    scoped, active = [], []
    for row in assignments:
        _required(row, ['assignment_id', 'profile_id', 'purpose', 'model_id', 'model_version', 'model_hash'])
        _interval(row)
        if row['profile_id'] == profile_id and row['purpose'] == purpose and _date(row['known_at'], 'known_at') <= k:
            scoped.append(row)
            if _effective(row, a, k):
                active.append(row)
    status, selected = 'MISSING_MODEL', None
    if len(active) > 1:
        status = 'AMBIGUOUS_MODEL'
    elif len(active) == 1:
        assignment = active[0]
        entries = [e for e in registry['entries'] if e['model_id'] == assignment['model_id'] and e['model_version'] == assignment['model_version'] and _date(e['registered_at'], 'registered_at') <= k]
        if entries:
            if entries[0]['model_hash'] != assignment['model_hash']:
                _fail('RATING_INTEGRITY', 'Assignment model hash differs from registered bundle')
            status, selected = 'SELECTED', {'assignment': deepcopy(assignment), **deepcopy(entries[0])}
    elif scoped:
        status = 'NOT_YET_EFFECTIVE' if all(_date(r['effective_from'], 'effective_from') > a for r in scoped) else 'EXPIRED_MODEL'
    return _result({'status': status, 'selected': selected, 'candidate_assignment_ids': sorted(r['assignment_id'] for r in active), 'profile_id': profile_id, 'purpose': purpose, 'as_of': as_of, 'known_at': known_at}, 'rating_system.select_effective_model')


def compose_risk_assessment(assessment_id, subject, components, binding, as_of, known_at, calculated_at, effects=None):
    """Preserve typed risk quantities and variants with explicit conflicts and lineage."""
    _string(assessment_id, 'assessment_id'); a, k = _dates(as_of, known_at)
    if _date(calculated_at, 'calculated_at') < k:
        _fail('RATING_DATE_ORDER', 'Calculation date precedes knowledge cutoff')
    subject, binding = _object(subject, 'subject'), _object(binding, 'binding')
    _required(subject, ['subject_id', 'subject_type'])
    if subject['subject_type'] not in _TYPES:
        _fail('RATING_SCHEMA', 'Unsupported subject type')
    _required(binding, ['profile_id', 'profile_version', 'model_id', 'model_version', 'model_hash', 'input_hash'])
    for key in binding:
        _string(binding[key], key)
    components = _rows(components, 'components'); _unique(components, ['component_id'], 'components')
    effects = _rows([] if effects is None else effects, 'effects'); _unique(effects, ['effect_id'], 'effects')
    conflicts, definitions, currencies = [], {}, set()
    ids = {c['component_id'] for c in components}
    for c in components:
        _required(c, ['component_id', 'quantity', 'value', 'subject_id', 'subject_type', 'variant', 'definition'])
        if c['subject_type'] not in _TYPES or c['quantity'] not in ('PD', 'GRADE', 'LGD', 'EAD', 'EL', 'SCORE'):
            _fail('RATING_SCHEMA', 'Unsupported quantity or component subject type')
        _string(c['variant'], 'component.variant')
        d = _object(c['definition'], 'component.definition')
        _required(d, ['event', 'horizon', 'time_unit', 'probability_measure', 'conditioning'])
        if _number(d['horizon'], 'horizon') <= 0 or d['time_unit'] not in ('days', 'months', 'years') or d['probability_measure'] not in ('PHYSICAL', 'RISK_NEUTRAL'):
            _fail('RATING_SCHEMA', 'Invalid component target definition')
        if c['quantity'] == 'GRADE':
            _string(c['value'], 'grade'); _required(c, ['scale_id', 'scale_version', 'graded_quantity'])
            if c['graded_quantity'] not in ('PD', 'EL', 'SCORE'):
                _fail('RATING_SCHEMA', 'Grade must name its graded quantity')
        else:
            value = _number(c['value'], 'component.value')
            if c['quantity'] == 'PD' and not 0 <= value <= 1:
                _fail('RATING_SCHEMA', 'PD outside [0,1]')
            if c['quantity'] == 'EAD' and value < 0:
                _fail('RATING_SCHEMA', 'EAD cannot be negative')
        if c['quantity'] in ('EAD', 'EL'):
            _required(c, ['currency']); currencies.add(_string(c['currency'], 'currency'))
        if (c['subject_id'], c['subject_type']) != (subject['subject_id'], subject['subject_type']):
            if not c.get('relationship_id'):
                conflicts.append({'component_id': c['component_id'], 'code': 'UNLINKED_SUBJECT_LEVEL'})
        key = (c['subject_type'], c['subject_id'], c['variant'])
        definitions.setdefault(key, []).append(c)
    for group in definitions.values():
        if len({_fingerprint(c['definition']) for c in group}) > 1:
            conflicts.append({'component_ids': sorted(c['component_id'] for c in group), 'code': 'TARGET_DEFINITION_CONFLICT'})
        if len({c['quantity'] for c in group}) < len(group):
            conflicts.append({'component_ids': sorted(c['component_id'] for c in group), 'code': 'DUPLICATE_QUANTITY_VARIANT'})
    if len(currencies) > 1:
        conflicts.append({'code': 'MIXED_CURRENCY', 'currencies': sorted(currencies)})
    orders = set(); edges = {}; producers = {}
    component_map = {c['component_id']: c for c in components}
    for effect in effects:
        _required(effect, ['effect_id', 'kind', 'order', 'level', 'input_component_id', 'output_component_id', 'reason'])
        if effect['kind'] not in ('support', 'transfer', 'override', 'cashflow') or effect['level'] not in _TYPES:
            _fail('RATING_SCHEMA', 'Unsupported adjustment effect')
        _days(effect['order'], 'effect.order'); _string(effect['reason'], 'effect.reason')
        if effect['order'] in orders or effect['input_component_id'] not in ids or effect['output_component_id'] not in ids or effect['input_component_id'] == effect['output_component_id']:
            _fail('RATING_EFFECT_CONFLICT', 'Effect needs unique order and distinct existing components')
        orders.add(effect['order'])
        if effect['output_component_id'] in edges:
            _fail('RATING_EFFECT_CONFLICT', 'Several effects overwrite the same output component')
        if component_map[effect['output_component_id']]['subject_type'] != effect['level']:
            _fail('RATING_EFFECT_CONFLICT', 'Effect level must match its output component')
        edges[effect['output_component_id']] = effect['input_component_id']
        producers[effect['output_component_id']] = effect['order']
    for effect in effects:
        if effect['input_component_id'] in producers and producers[effect['input_component_id']] >= effect['order']:
            _fail('RATING_EFFECT_CONFLICT', 'Effect order must follow its input producer')
    for node in edges:
        seen = set(); cursor = node
        while cursor in edges:
            if cursor in seen:
                _fail('RATING_EFFECT_CONFLICT', 'Adjustment effects contain a cycle')
            seen.add(cursor); cursor = edges[cursor]
    value = {'schema_version': '1.0', 'assessment_id': assessment_id, 'subject': subject, 'binding': binding, 'as_of': as_of, 'known_at': known_at, 'calculated_at': calculated_at, 'components': sorted(components, key=lambda c: c['component_id']), 'effects': sorted(effects, key=lambda e: e['order']), 'conflicts': conflicts, 'composition_status': 'CONFLICT' if conflicts else 'CONSISTENT', 'state': 'DRAFT', 'history': []}
    return _result(_seal(value, 'assessment_hash'), 'rating_system.compose_risk_assessment', limitations=('Composition preserves supplied quantities; it does not infer adjustments, copy PD across levels or approve credit.',))


def check_assessment_freshness(assessment, policy, events, as_of, known_at):
    """Assess elapsed information age, expiry and known material trigger events."""
    assessment = _sealed(assessment, 'assessment_hash', 'assessment'); policy = _object(policy, 'freshness policy')
    a, k = _dates(as_of, known_at)
    _required(policy, ['maximum_age_days', 'maximum_information_age_days', 'trigger_types'])
    for key in ('maximum_age_days', 'maximum_information_age_days'):
        _days(policy[key], key)
    if not isinstance(policy['trigger_types'], list) or any(not isinstance(t, str) for t in policy['trigger_types']):
        _fail('RATING_SCHEMA', 'trigger_types must be a list of event types')
    if _date(assessment['as_of'], 'assessment.as_of') > a or _date(assessment['known_at'], 'assessment.known_at') > k:
        _fail('RATING_DATE_ORDER', 'Assessment unavailable at requested historical date')
    if any(_date(h['event']['known_at'], 'history.known_at') > k for h in assessment['history']):
        _fail('RATING_DATE_ORDER', 'Lifecycle state contains events after knowledge cutoff')
    findings = []
    if (a - _date(assessment['as_of'], 'assessment.as_of')).days > policy['maximum_age_days']:
        findings.append({'code': 'ASSESSMENT_AGE'})
    if (k - _date(assessment['known_at'], 'assessment.known_at')).days > policy['maximum_information_age_days']:
        findings.append({'code': 'INFORMATION_AGE'})
    if policy.get('effective_to') is not None and a >= _date(policy['effective_to'], 'effective_to'):
        findings.append({'code': 'PROFILE_EXPIRED'})
    superseded = assessment['state'] == 'SUPERSEDED'
    events = _rows(events, 'events'); _unique(events, ['event_id'], 'events')
    for event in sorted(events, key=lambda e: e['event_id']):
        _required(event, ['event_id', 'subject_id', 'type', 'occurred_at', 'known_at'])
        occurrence, available = _date(event['occurred_at'], 'occurred_at'), _date(event['known_at'], 'known_at')
        if available < occurrence:
            _fail('RATING_DATE_ORDER', 'Event known before occurrence')
        if event['subject_id'] == assessment['subject']['subject_id'] and occurrence <= a and available <= k and available > _date(assessment['known_at'], 'assessment.known_at'):
            if event.get('supersedes_assessment_id') == assessment['assessment_id']:
                superseded = True; findings.append({'code': 'SUPERSEDED', 'event_id': event['event_id']})
            elif event['type'] in policy['trigger_types']:
                findings.append({'code': 'MATERIAL_EVENT', 'event_id': event['event_id']})
    return _result({'status': 'SUPERSEDED' if superseded else 'STALE' if findings else 'CURRENT', 'assessment_id': assessment['assessment_id'], 'findings': findings, 'as_of': as_of, 'known_at': known_at}, 'rating_system.check_assessment_freshness')


def transition_assessment_state(assessment, event, transitions):
    """Apply one declared lifecycle transition to a copied, content-bound assessment."""
    a = _sealed(assessment, 'assessment_hash', 'assessment'); event = _object(event, 'event')
    _required(event, ['event_id', 'type', 'occurred_at', 'known_at', 'reason', 'gates'])
    _string(event['reason'], 'reason'); gates = _object(event['gates'], 'gates')
    when, known = _dates(event['occurred_at'], event['known_at'])
    if when < _date(a['calculated_at'], 'calculated_at') or known < _date(a['known_at'], 'assessment.known_at'):
        _fail('RATING_DATE_ORDER', 'Lifecycle event precedes assessment availability')
    if any(h['event']['event_id'] == event['event_id'] for h in a['history']):
        _fail('RATING_EVENT_CONFLICT', 'Lifecycle event already applied')
    if a['history'] and (when < _date(a['history'][-1]['event']['occurred_at'], 'occurred_at') or known < _date(a['history'][-1]['event']['known_at'], 'known_at')):
        _fail('RATING_DATE_ORDER', 'Lifecycle history must remain ordered')
    transitions = _rows(transitions, 'transitions')
    for t in transitions:
        _required(t, ['from_state', 'to_state', 'event_type', 'required_gates'])
        if t['from_state'] not in ('DRAFT', 'SUBMITTED', 'REVIEWED', 'FINAL', 'SUPERSEDED') or t['to_state'] not in ('DRAFT', 'SUBMITTED', 'REVIEWED', 'FINAL', 'SUPERSEDED') or not isinstance(t['required_gates'], list):
            _fail('RATING_SCHEMA', 'Malformed transition table')
    matches = [t for t in transitions if t['from_state'] == a['state'] and t['event_type'] == event['type']]
    if len(matches) != 1:
        _fail('RATING_TRANSITION', 'Transition is absent or ambiguous')
    t = matches[0]
    if any(gates.get(g) is not True for g in t['required_gates']):
        _fail('RATING_GATE', 'Required lifecycle gates are not all explicitly true')
    if t['to_state'] == 'FINAL' and a['composition_status'] != 'CONSISTENT':
        _fail('RATING_GATE', 'Conflicted composition cannot be locally finalized')
    if a['state'] == 'SUPERSEDED' or (a['state'] == 'FINAL' and t['to_state'] != 'SUPERSEDED'):
        _fail('RATING_TRANSITION', 'Final history is immutable except an explicit superseding transition')
    a['history'].append({'from_state': a['state'], 'to_state': t['to_state'], 'previous_assessment_hash': a['assessment_hash'], 'event': event})
    a['state'] = t['to_state']
    return _result(_seal(a, 'assessment_hash'), 'rating_system.transition_assessment_state', limitations=('FINAL is a local processing status, not institutional or regulatory approval.',))


def _execute(request, registry, cutoff):
    _required(request, ['row_id', 'known_at', 'model_id', 'model_version', 'model_hash', 'recipe', 'input'])
    if _date(request['known_at'], 'request.known_at') > cutoff:
        _fail('RATING_REQUEST_UNAVAILABLE', 'Request unavailable at batch cutoff')
    entries = [e for e in registry['entries'] if e['model_id'] == request['model_id'] and e['model_version'] == request['model_version']]
    if not entries or _date(entries[0]['registered_at'], 'registered_at') > _date(request['known_at'], 'request.known_at'):
        _fail('RATING_MISSING_MODEL', 'Pinned historical model unavailable')
    entry = entries[0]
    if entry['model_hash'] != request['model_hash']:
        _fail('RATING_INTEGRITY', 'Pinned historical model hash mismatch')
    recipe = _object(request['recipe'], 'recipe'); inputs = _object(request['input'], 'input')
    _required(recipe, ['method', 'version', 'parameters'])
    parameters = _object(recipe['parameters'], 'recipe.parameters')
    if recipe['method'] == 'assessment_pipeline' and recipe['version'] == '2.0':
        from ._rating_execution import _execute_pipeline
        return _execute_pipeline(request, entry)
    from ._rating_execution import _dated
    _dated(parameters, _date(request['known_at'], 'request.known_at'), 'recipe parameters')
    if recipe['version'] != '1.0':
        _fail('RATING_RECIPE', 'Unsupported recipe version')
    if recipe['method'] == 'application_risk':
        from .application import predict_application_risk
        if set(inputs) != {'snapshot'} or set(parameters) - {'target', 'offset', 'calibrator'} or 'target' not in parameters:
            _fail('RATING_RECIPE', 'Application recipe requires snapshot input and target parameters')
        snapshot = _object(inputs['snapshot'], 'application snapshot')
        _required(snapshot, ['information_cutoff'])
        if _date(snapshot['information_cutoff'], 'snapshot.information_cutoff') > _date(request['known_at'], 'request.known_at'):
            _fail('RATING_REQUEST_UNAVAILABLE', 'Application snapshot exceeds request knowledge')
        result = predict_application_risk(inputs['snapshot'], entry['bundle']['model'], **parameters)
    elif recipe['method'] == 'behavioral_risk':
        from .behavioral import predict_behavioral_risk
        if set(inputs) != {'snapshot'} or set(parameters) - {'target', 'offset', 'calibrator'} or 'target' not in parameters:
            _fail('RATING_RECIPE', 'Behavioral recipe requires snapshot input and target parameters')
        snapshot = _object(inputs['snapshot'], 'behavioral snapshot')
        _required(snapshot, ['known_at'])
        if _date(snapshot['known_at'], 'snapshot.known_at') > _date(request['known_at'], 'request.known_at'):
            _fail('RATING_REQUEST_UNAVAILABLE', 'Snapshot exceeds request knowledge')
        result = predict_behavioral_risk(snapshot, entry['bundle']['model'], **parameters)
    elif recipe['method'] in ('binary_glm', 'pd_model'):
        from .classification import predict_pd
        if recipe['method'] == 'binary_glm' and entry['bundle']['model']['method_id'] != 'classification.fit_binary_glm':
            _fail('RATING_RECIPE', 'Binary GLM recipe requires a binary GLM artifact')
        if set(inputs) != {'X', 'feature_order'} or set(parameters) - {'expected_target', 'offset'} or 'expected_target' not in parameters:
            _fail('RATING_RECIPE', 'GLM recipe requires X/feature_order and expected_target')
        result = predict_pd(entry['bundle']['model'], inputs['X'], feature_order=inputs['feature_order'], **parameters)
    else:
        _fail('RATING_RECIPE', 'Unsupported declarative recipe method')
    value = result.to_dict()['value']
    return {'row_id': request['row_id'], 'status': 'OK', 'value': value, 'error_code': None, 'input_hash': _fingerprint(inputs), 'recipe_hash': _fingerprint(recipe), 'model_hash': entry['model_hash'], 'result_hash': _fingerprint(value)}


def rate_subject_batch(requests, registry, batch_id, known_at):
    """Run fixed declarative recipes with stable row keys and visible row errors."""
    _string(batch_id, 'batch_id'); cutoff = _date(known_at, 'known_at'); registry = _registry(registry)
    requests = _rows(requests, 'requests'); _unique(requests, ['row_id'], 'requests')
    output = []
    for request in sorted(requests, key=lambda r: r['row_id']):
        try:
            output.append(_execute(request, registry, cutoff))
        except CreditRiskError as exc:
            output.append({'row_id': request['row_id'], 'status': 'ERROR', 'value': None, 'error_code': exc.code, 'input_hash': _fingerprint(request.get('input')), 'recipe_hash': _fingerprint(request.get('recipe')), 'model_hash': request.get('model_hash'), 'result_hash': None})
    value = {'schema_version': '1.0', 'batch_id': batch_id, 'known_at': known_at, 'rows': output, 'request_hash': _fingerprint(sorted(requests, key=lambda r: r['row_id'])), 'registry_hash': registry['registry_hash']}
    return _result(_seal(value, 'batch_hash'), 'rating_system.rate_subject_batch', diagnostics={'success_count': sum(r['status'] == 'OK' for r in output), 'error_count': sum(r['status'] == 'ERROR' for r in output)}, limitations=('Pinned data-only pipelines support native calculation, pricing and explicit simulation, frozen empirical artifacts and local lifecycle replay; no callbacks or executable deserialization.',))


def apply_assessment_event(state, event, dependencies, known_at):
    """Project deduplicated event revisions onto explicitly declared dependencies."""
    state = _seal({'schema_version': '1.0', 'processed': [], 'applications': []}, 'state_hash') if state is None else _sealed(state, 'state_hash', 'event state')
    event = _object(event, 'event'); _required(event, ['event_id', 'revision', 'occurred_at', 'known_at', 'dependencies', 'payload'])
    _string(event['revision'], 'revision'); _dates(event['occurred_at'], event['known_at'])
    cutoff = _date(known_at, 'known_at'); digest = _fingerprint(event)
    if any(_date(r['event']['known_at'], 'processed.known_at') > cutoff for r in state['processed']):
        _fail('RATING_DATE_ORDER', 'Event state includes history after cutoff')
    previous = [r for r in state['processed'] if r['event_id'] == event['event_id'] and r['revision'] == event['revision']]
    if previous:
        if previous[0]['event_hash'] != digest:
            _fail('RATING_EVENT_CONFLICT', 'Same event ID/revision has different content')
        application = next(r for r in state['applications'] if r['event_id'] == event['event_id'] and r['revision'] == event['revision'])
        return _result({'status': 'DUPLICATE', 'state': state, 'affected_assessment_ids': application['affected_assessment_ids']}, 'rating_system.apply_assessment_event')
    if _date(event['known_at'], 'event.known_at') > cutoff:
        return _result({'status': 'DEFERRED', 'state': state, 'affected_assessment_ids': []}, 'rating_system.apply_assessment_event')
    revisions = [r for r in state['processed'] if r['event_id'] == event['event_id']]
    if revisions:
        if event.get('corrects_revision') != revisions[-1]['revision']:
            _fail('RATING_EVENT_CONFLICT', 'Correction must refer to latest previously processed revision')
        if _date(event['known_at'], 'known_at') < _date(revisions[-1]['event']['known_at'], 'known_at'):
            _fail('RATING_DATE_ORDER', 'Correction known before earlier revision')
    elif event.get('corrects_revision') is not None:
        _fail('RATING_EVENT_CONFLICT', 'Correction refers to missing earlier event')
    event_deps = _rows(event['dependencies'], 'event.dependencies')
    for d in event_deps:
        _required(d, ['type', 'id']); _string(d['type'], 'dependency.type'); _string(d['id'], 'dependency.id')
    keys = {(d['type'], d['id']) for d in event_deps}
    dependencies = _rows(dependencies, 'dependencies')
    for d in dependencies:
        _required(d, ['assessment_id', 'dependency_type', 'dependency_id'])
    affected = sorted({d['assessment_id'] for d in dependencies if (d['dependency_type'], d['dependency_id']) in keys})
    state['processed'].append({'event_id': event['event_id'], 'revision': event['revision'], 'event_hash': digest, 'event': event})
    state['applications'].append({'event_id': event['event_id'], 'revision': event['revision'], 'affected_assessment_ids': affected, 'dependency_hash': _fingerprint(dependencies)})
    return _result({'status': 'APPLIED', 'state': _seal(state, 'state_hash'), 'affected_assessment_ids': affected}, 'rating_system.apply_assessment_event', limitations=('Projection identifies affected assessments; it does not mutate risk values, accounts or external systems.',))


def replay_assessment_history(requests, registry, expected, known_at, replay_id='replay'):
    """Recompute pinned historical recipes and compare recorded content hashes."""
    expected = _rows(expected, 'expected'); _unique(expected, ['row_id'], 'expected')
    for row in expected:
        _required(row, ['row_id', 'input_hash', 'recipe_hash', 'model_hash', 'result_hash'])
    expectations = {r['row_id']: r for r in expected}
    try:
        batch = rate_subject_batch(requests, registry, replay_id, known_at).value
    except CreditRiskError as exc:
        return _result({'status':'NOT_REPRODUCIBLE','batch':None,'verified_row_ids':[], 'findings':[{'row_id':None,'code':exc.code}]}, 'rating_system.replay_assessment_history', limitations=('Invalid supplied artifact/history content prevents historical reproduction.',))
    findings, verified = [], []
    for row in batch['rows']:
        reference = expectations.get(row['row_id'])
        if row['status'] != 'OK':
            findings.append({'row_id': row['row_id'], 'code': row['error_code']})
        elif reference is None:
            findings.append({'row_id': row['row_id'], 'code': 'MISSING_REFERENCE'})
        else:
            fields = ['input_hash', 'recipe_hash', 'model_hash', 'result_hash']
            if isinstance(row.get('value'), dict) and row['value'].get('schema_version') == '2.0':
                fields += ['assessment_hash', 'event_state_hash', 'artifact_manifest_hash', 'innovation_hash', 'decision_hash']
            mismatches = [f for f in fields if f not in reference or row.get(f) != reference.get(f)]
            if mismatches:
                findings.append({'row_id': row['row_id'], 'code': 'REPLAY_MISMATCH', 'fields': mismatches})
            else:
                verified.append(row['row_id'])
    found = {r['row_id'] for r in batch['rows']}
    for identity in sorted(set(expectations) - found):
        findings.append({'row_id': identity, 'code': 'MISSING_REQUEST'})
    return _result({'status': 'REPRODUCED' if not findings else 'NOT_REPRODUCIBLE', 'batch': batch, 'verified_row_ids': verified, 'findings': findings}, 'rating_system.replay_assessment_history', limitations=('Replay uses only supplied pinned artifacts and historical cutoff; source names without matching hashes are insufficient.',))
