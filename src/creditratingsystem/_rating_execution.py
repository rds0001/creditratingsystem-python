"""Data-only, content-addressed local assessment execution.

Empirical estimators run in model development; historical assessments consume their
frozen outputs. Numerical pricing and explicitly driven simulations remain executable.
"""
from copy import deepcopy
from importlib import import_module
from inspect import signature
import math
from .core import CreditRiskError, _fail, _fingerprint, _json_data
from .data import _date, _required, _rows, _unique
from ._rating_operations import OPERATIONS


def _dated(value, cutoff, label):
    """Every declared information timestamp must be available, including nested artifacts."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('known_at', 'information_cutoff', 'registered_at') and item is not None:
                if _date(item, label+'.'+key) > cutoff:
                    _fail('RATING_REQUEST_UNAVAILABLE', label+' contains information after its cutoff')
            _dated(item, cutoff, label+'.'+key)
    elif isinstance(value, list):
        for item in value:
            _dated(item, cutoff, label)


def _path(value, path):
    if not isinstance(path, list):
        _fail('RATING_REFERENCE', 'Reference path must be a list of keys or zero-based indices')
    for key in path:
        if isinstance(value, dict) and isinstance(key, str) and key in value:
            value = value[key]
        elif isinstance(value, list) and isinstance(key, (int, float)) and not isinstance(key, bool) and math.isfinite(key) and key == int(key) and 0 <= key < len(value):
            value = value[int(key)]
        else:
            _fail('RATING_REFERENCE', 'Missing or incompatible reference path')
    return deepcopy(value)


def _references(value):
    found = []
    if isinstance(value, dict):
        if '$ref' in value:
            if set(value) != {'$ref'} or not isinstance(value['$ref'], dict):
                _fail('RATING_REFERENCE', 'Reference must be an exact data-only object')
            ref = value['$ref']
            if set(ref) != {'source', 'id', 'path'} or ref['source'] not in ('input', 'artifact', 'step', 'model') or not isinstance(ref['id'], str):
                _fail('RATING_REFERENCE', 'Reference requires source, id and path')
            found.append(ref)
        else:
            for item in value.values(): found.extend(_references(item))
    elif isinstance(value, list):
        for item in value: found.extend(_references(item))
    return found


def _resolve(value, sources):
    _references(value)
    if isinstance(value, dict):
        if '$ref' in value:
            ref = value['$ref']; table = sources[ref['source']]
            if ref['id'] not in table: _fail('RATING_REFERENCE', 'Reference identity is unavailable')
            return _path(table[ref['id']], ref['path'])
        return {k: _resolve(v, sources) for k, v in value.items()}
    if isinstance(value, list): return [_resolve(v, sources) for v in value]
    return deepcopy(value)


def _typed(value, kind):
    if kind == 'numeric_matrix':
        if not isinstance(value, list) or not value or any(not isinstance(r, list) for r in value) or len({len(r) for r in value}) != 1:
            _fail('RATING_ARGUMENT', 'numeric_matrix requires a nonempty rectangular array')
        flat = [x for row in value for x in row]
    elif kind in ('numeric_vector', 'character_vector', 'logical_vector'):
        if not isinstance(value, list): _fail('RATING_ARGUMENT', 'Typed vectors require arrays')
        flat = value
    else: _fail('RATING_ARGUMENT', 'Unsupported argument type')
    if kind.startswith('numeric'):
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in flat): _fail('RATING_ARGUMENT', 'Numeric arguments require finite numbers')
    elif kind == 'character_vector':
        if any(not isinstance(x, str) for x in flat): _fail('RATING_ARGUMENT', 'Character vector contains another type')
    elif any(not isinstance(x, bool) for x in flat): _fail('RATING_ARGUMENT', 'Logical vector contains another type')
    return value


def _output(value, shape):
    value = _json_data(value)
    if shape == 'scalar':
        if isinstance(value, list) and len(value) == 1: value = value[0]
        if value is None or isinstance(value, (dict, list)): _fail('RATING_OUTPUT', 'Expected a scalar output')
    elif shape == 'vector':
        if value is not None and not isinstance(value, (dict, list)): value = [value]
        if not isinstance(value, list) or any(isinstance(v, (dict, list)) for v in value): _fail('RATING_OUTPUT', 'Expected vector output')
    elif shape == 'matrix':
        if not isinstance(value, list) or any(not isinstance(v, list) for v in value) or len({len(v) for v in value}) > 1: _fail('RATING_OUTPUT', 'Expected matrix output')
    elif shape == 'object':
        if not isinstance(value, dict): _fail('RATING_OUTPUT', 'Expected object output')
    elif shape == 'records':
        if not isinstance(value, list) or any(not isinstance(v, dict) for v in value): _fail('RATING_OUTPUT', 'Expected record array')
    else: _fail('RATING_OUTPUT', 'An explicit output shape is required')
    _fingerprint(value)
    return value


def _invoke(method, arguments, argument_types):
    spec = OPERATIONS.get(method)
    if spec is None: _fail('RATING_RECIPE', 'Operation is outside the static supported catalogue')
    if spec['stage'] in ('PRE_FITTED', 'LIFECYCLE'):
        _fail('RATING_RECIPE', 'This operation requires a frozen development artifact or dedicated lifecycle stage')
    args = deepcopy(arguments)
    for key, kind in argument_types.items():
        if key not in args: _fail('RATING_ARGUMENT', 'Typed argument is absent')
        args[key] = _typed(args[key], kind)
    if method == 'explainability.estimate_credit_shapley_values':
        predictor = args.get('predictor')
        if not isinstance(predictor, dict) or set(predictor) != {'kind','model','feature_order','expected_target'} or predictor['kind'] != 'FROZEN_PD':
            _fail('RATING_RECIPE', 'Explanation requires a data-only FROZEN_PD predictor contract')
        if args.get('model_contract', {}).get('output_scale') != 'probability': _fail('RATING_RECIPE', 'Frozen PD adapter outputs probability only')
        model_contract = args['model_contract']; model = predictor['model']
        if any(model_contract.get(k) != model.get(k) for k in ('model_id','model_version','feature_order','target')): _fail('RATING_INTEGRITY', 'Explanation contract differs from pinned predictor')
        from .classification import predict_pd
        args['predictor'] = lambda X: predict_pd(model, X, feature_order=predictor['feature_order'], expected_target=predictor['expected_target'])
    module = import_module('.'+spec['module'], 'creditratingsystem')
    function = getattr(module, spec['function'], None)
    if function is None: _fail('RATING_RECIPE', 'Operation is unavailable in this installed version')
    try: signature(function).bind(**args)
    except TypeError: _fail('RATING_ARGUMENT', 'Arguments do not match operation signature')
    try:
        return function(**args)
    except CreditRiskError:
        raise
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as exc:
        _fail('RATING_EXECUTION', 'Operation rejected malformed data or failed numerically: '+type(exc).__name__)


def _execute_pipeline(request, entry):
    from . import rating_system as rs
    recipe = request['recipe']; p = rs._object(recipe['parameters'], 'pipeline parameters')
    if set(p) != {'routing_rules', 'applicability_contract', 'steps', 'components', 'effects', 'transitions', 'event_dependencies', 'decision'}:
        _fail('RATING_RECIPE', 'Pipeline requires all explicit routing, execution, composition, lifecycle and decision fields')
    if entry['bundle']['configuration'].get('assessment_recipe') != recipe:
        _fail('RATING_INTEGRITY', 'Pipeline recipe must be pinned inside model bundle configuration')
    inputs = rs._object(request['input'], 'pipeline input')
    _required(inputs, ['snapshot', 'datasets', 'assessment_id', 'assessment_known_at', 'calculated_at', 'lifecycle_events', 'events'])
    snapshot = rs._object(inputs['snapshot'], 'snapshot'); _required(snapshot, ['as_of', 'known_at'])
    base_known = _date(inputs['assessment_known_at'], 'assessment_known_at'); request_known = _date(request['known_at'], 'request.known_at')
    calculated = _date(inputs['calculated_at'], 'calculated_at')
    if not base_known <= calculated <= request_known: _fail('RATING_DATE_ORDER', 'Calculation must follow assessment knowledge and precede history cutoff')
    rs._snapshot(snapshot, snapshot['as_of'], inputs['assessment_known_at'])
    _dated(snapshot, base_known, 'snapshot')
    if _date(entry['registered_at'], 'registered_at') > base_known: _fail('RATING_MISSING_MODEL', 'Model was unavailable at assessment calculation cutoff')
    _dated(entry['bundle']['model'], base_known, 'model')
    datasets = rs._object(inputs['datasets'], 'datasets')
    for key, dataset in datasets.items():
        _required(dataset, ['value', 'as_of', 'known_at']); _dated(dataset, base_known, 'dataset.'+key)
        if _date(dataset['as_of'], 'dataset.as_of') > _date(snapshot['as_of'], 'snapshot.as_of'): _fail('RATING_SNAPSHOT_DATE', 'Dataset economic date follows assessment')
    artifacts = rs._object(entry['bundle']['dependencies'].get('artifacts', {}), 'artifacts'); content = {}; manifest = []
    for identity, artifact in sorted(artifacts.items()):
        _required(artifact, ['kind', 'content', 'sha256', 'known_at'])
        if artifact['sha256'] != _fingerprint(artifact['content']): _fail('RATING_INTEGRITY', 'Frozen dependency hash mismatch')
        _dated(artifact, base_known, 'artifact.'+identity)
        content[identity] = artifact['content']
        manifest.append({'artifact_id': identity, 'kind': artifact['kind'], 'sha256': artifact['sha256'], 'known_at': artifact['known_at']})
    sources = {'input': datasets, 'artifact': content, 'step': {}, 'model': {'primary': entry['bundle']['model']}}
    rules = _resolve(p['routing_rules'], sources); contract = _resolve(p['applicability_contract'], sources)
    route = rs.route_rating_profile(snapshot, rules, snapshot['as_of'], inputs['assessment_known_at']).value
    applicable = rs.assess_rating_applicability(snapshot, contract, snapshot['as_of'], inputs['assessment_known_at']).value
    if route['status'] != 'ROUTED' or applicable['status'] != 'APPLICABLE': _fail('RATING_APPLICABILITY', 'Assessment lacks one applicable profile route')
    steps = _rows(p['steps'], 'steps'); _unique(steps, ['step_id'], 'steps'); pending = {s['step_id']:s for s in steps}; trace = []
    all_ids = set(pending)
    for step in steps:
        _required(step, ['step_id', 'method_id', 'arguments', 'argument_types', 'output_contract'])
        if step['method_id'] not in OPERATIONS: _fail('RATING_RECIPE', 'Unknown operation')
        for ref in _references(step['arguments']):
            if ref['source'] == 'step' and ref['id'] not in all_ids: _fail('RATING_REFERENCE', 'Unknown predecessor step')
    while pending:
        ready = sorted(identity for identity,step in pending.items() if all(ref['source'] != 'step' or ref['id'] in sources['step'] for ref in _references(step['arguments'])))
        if not ready: _fail('RATING_RECIPE', 'Execution graph contains a cycle')
        for identity in ready:
            step = pending.pop(identity); method = step['method_id']; op = OPERATIONS[method]
            arguments = rs._object(_resolve(step['arguments'], sources), 'step arguments'); _dated(arguments, base_known, 'step arguments')
            output_contract = rs._object(step['output_contract'], 'output contract'); _required(output_contract, ['shape', 'unit'])
            if 'target' not in output_contract: _fail('RATING_OUTPUT', 'Explicit nullable target required')
            if op['stage'] == 'PRE_FITTED':
                if set(arguments) != {'artifact'} or len(_references(step['arguments'])) != 1 or _references(step['arguments'])[0]['source'] != 'artifact': _fail('RATING_RECIPE', 'Empirical fit requires exactly one frozen artifact reference')
                frozen = arguments['artifact']
                if not isinstance(frozen, dict) or frozen.get('method_id') != method: _fail('RATING_RECIPE', 'Frozen development artifact method differs')
                raw_value = frozen; unit = 'artifact'; result_context = {}; diagnostics = {'execution': 'FROZEN_DEVELOPMENT_ARTIFACT'}
            else:
                result = _invoke(method, arguments, rs._object(step['argument_types'], 'argument types'))
                if not hasattr(result, 'to_dict'): _fail('RATING_OUTPUT', 'Operation must return a structured result')
                raw = result.to_dict(); raw_value = raw.get('value'); unit = raw.get('unit'); result_context = raw.get('context', {}); diagnostics = raw.get('diagnostics', {})
            if unit != output_contract['unit']: _fail('RATING_OUTPUT', 'Actual result unit differs from pinned contract')
            target = output_contract['target']
            if target is not None:
                if not isinstance(target, dict): _fail('RATING_OUTPUT', 'Target must be explicit object or null')
                actual_target = diagnostics.get('target') or (raw_value.get('target') if isinstance(raw_value, dict) else None)
                if actual_target is not None and actual_target != target: _fail('RATING_OUTPUT', 'Actual target differs from pinned contract')
                for key in ('probability_measure', 'horizon', 'time_unit', 'event'):
                    if diagnostics.get(key) is not None and key in target and diagnostics[key] != target[key]: _fail('RATING_OUTPUT', 'Method diagnostics differ from target')
                for key,val in target.items():
                    if result_context.get(key) is not None and result_context[key] != val: _fail('RATING_OUTPUT', 'Result context differs from target')
            value = _output(raw_value, output_contract['shape'])
            fields = output_contract.get('fields', [])
            for field in _rows(fields, 'output fields'):
                _required(field, ['path', 'unit'])
                if not {'target', 'target_path'} <= set(field): _fail('RATING_OUTPUT', 'Explicit field target and target_path required')
                _path(value, field['path'])
                if field['target_path'] is not None and _path(value, field['target_path']) != field['target']: _fail('RATING_OUTPUT', 'Nested target differs from declared field contract')
            if len({_fingerprint(f['path']) for f in fields}) != len(fields): _fail('RATING_OUTPUT', 'Duplicate field contract path')
            sources['step'][identity] = {'value':value, 'unit':unit, 'target':target, 'fields':fields}
            trace.append({'step_id':identity, 'method_id':method, 'stage':op['stage'], 'arguments_hash':_fingerprint(arguments), 'output_contract':output_contract, 'value_hash':_fingerprint(value), 'value':value})
    components = _resolve(p['components'], sources); effects = _resolve(p['effects'], sources)
    for component in components:
        # Bound quantity references must carry an explicitly matching target.
        original = next(c for c in p['components'] if c['component_id'] == component['component_id'])
        for ref in _references(original.get('value')):
            if ref['source'] == 'step':
                produced = sources['step'][ref['id']]
                field = next((f for f in produced['fields'] if ref['path'] == ['value']+f['path']), produced)
                target, field_unit = field['target'], field['unit']
                required_unit = {'PD':'probability', 'LGD':'loss_fraction', 'GRADE':'grade', 'SCORE':'score'}.get(component['quantity'], component.get('currency'))
                if field_unit != required_unit: _fail('RATING_OUTPUT', 'Component quantity or currency differs from producing field unit')
                if component['quantity'] in ('PD','LGD','EAD','EL') and target != component['definition']: _fail('RATING_OUTPUT', 'Component definition differs from producing step target')
    binding = {'profile_id':route['profile_id'], 'profile_version':route['profile_version'], 'model_id':request['model_id'], 'model_version':request['model_version'], 'model_hash':entry['model_hash'], 'input_hash':_fingerprint(inputs), 'recipe_hash':_fingerprint(recipe), 'artifact_manifest_hash':_fingerprint(manifest)}
    assessment = rs.compose_risk_assessment(inputs['assessment_id'], {'subject_id':snapshot['subject_id'],'subject_type':snapshot['subject_type']}, components,binding,snapshot['as_of'],inputs['assessment_known_at'],inputs['calculated_at'],effects=effects).value
    lifecycle = _rows(inputs['lifecycle_events'], 'lifecycle events')
    applied_lifecycle = {}
    for event in lifecycle:
        _dated(event, request_known, 'lifecycle event')
        if event.get('event_id') in applied_lifecycle:
            if applied_lifecycle[event['event_id']] != _fingerprint(event): _fail('RATING_EVENT_CONFLICT', 'Lifecycle event ID has different content')
            continue
        assessment = rs.transition_assessment_state(assessment,event,_resolve(p['transitions'],sources)).value
        applied_lifecycle[event['event_id']] = _fingerprint(event)
    event_state = None; event_trace = []
    for event in _rows(inputs['events'], 'events'):
        applied = rs.apply_assessment_event(event_state,event,_resolve(p['event_dependencies'],sources),request['known_at']).value
        event_state = applied['state']; event_trace.append({'event_id':event['event_id'],'revision':event['revision'],'status':applied['status'],'affected_assessment_ids':applied['affected_assessment_ids']})
    if event_state is None: event_state = rs._seal({'schema_version':'1.0','processed':[],'applications':[]},'state_hash')
    decision = None; decision_hash = None
    if p['decision'] is not None:
        from .decisioning import apply_credit_decision_policy
        arguments = _resolve(p['decision'], sources); _dated(arguments,request_known,'decision')
        if not isinstance(arguments, dict): _fail('RATING_ARGUMENT', 'Decision arguments must be an object')
        try: signature(apply_credit_decision_policy).bind(**arguments)
        except TypeError: _fail('RATING_ARGUMENT', 'Decision arguments do not match operation signature')
        risk = arguments.get('risk', {})
        if not isinstance(risk, dict): _fail('RATING_DECISION', 'Decision risk must be an object')
        if not any(c['quantity']=='PD' and c['value']==risk.get('pd') and c['definition']==risk.get('target') for c in components): _fail('RATING_DECISION', 'Decision risk must equal an explicitly composed PD and target')
        try: decision = apply_credit_decision_policy(**arguments).to_dict()['value']
        except CreditRiskError: raise
        except Exception: _fail('RATING_EXECUTION', 'Decision operation rejected malformed data or failed numerically')
        decision_hash = _fingerprint(decision)
    output = {'schema_version':'2.0','assessment':assessment,'routing':route,'applicability':applicable,'steps':trace,'artifact_manifest':manifest,'event_state':event_state,'event_trace':event_trace,'decision':decision,'execution_role':'LOCAL_ASSESSMENT_NOT_EMPIRICAL_VALIDATION'}
    return {'row_id':request['row_id'],'status':'OK','value':output,'error_code':None,'input_hash':_fingerprint(inputs),'recipe_hash':_fingerprint(recipe),'model_hash':entry['model_hash'],'result_hash':_fingerprint(output),'assessment_hash':assessment['assessment_hash'],'event_state_hash':event_state['state_hash'],'artifact_manifest_hash':_fingerprint(manifest),'innovation_hash':_fingerprint([r for r in manifest if r['kind']=='innovations']),'decision_hash':decision_hash}
