"""Evidence-based system coverage without implicit Cartesian completeness claims."""
from collections import Counter
from copy import deepcopy
from .core import _fail, _fingerprint, _result
from ._expert_common import _ex_object, _ex_ids
from ._validation_common import _vb_string
from ._validation_retail import _vr_rows

_AXES = ('customer_segment', 'product', 'lifecycle', 'method', 'granularity')
_CELL_EVIDENCE = ('data_target_contract', 'independent_oracle', 'positive_routing',
                  'negative_routing', 'decision_trace', 'monitoring', 'source_review', 'parity')


def _vs_index(records, key, name):
    rows = _vr_rows(records, name)
    _ex_ids([r.get(key) for r in rows], name + ' identifiers', empty=True)
    return rows, {r[key]: r for r in rows}


def _vs_finish(method, payload, violations, inputs, context):
    counts = dict(sorted(Counter(v['code'] for v in violations).items()))
    value = dict(artifact_type='ValidationResult', **payload, violations=violations,
                 violation_counts=counts, decision=dict(status='BLOCKED' if violations else 'PASS',
                 scope='DECLARED_EVIDENCE_CONTRACT'), source_provenance_authenticated=False,
                 empirical_performance_assessed=False, input_fingerprint=_fingerprint(inputs))
    return _result(value, 'validation.' + method, 'evidence_audit', context,
                   limitations=('Evaluates supplied evidence and explicit scope. It does not authenticate provenance, '
                                'execute tests, grant empirical or regulatory approval, or publish a release.',))


def validate_system_coverage(cells, function_inventory, journeys, evidence, *, scope, context=None):
    """Audit explicit five-axis obligations and their function/journey evidence."""
    rows, index = _vs_index(cells, 'cell_id', 'cells')
    funcs, fi = _vs_index(function_inventory, 'function_id', 'function_inventory')
    trips, ji = _vs_index(journeys, 'journey_id', 'journeys')
    proofs, ei = _vs_index(evidence, 'evidence_id', 'evidence')
    contract = deepcopy(_ex_object(scope, 'scope'))
    for key in ('scope_id', 'version', 'reference'):
        _vb_string(contract.get(key), key)
    axes = _ex_object(contract.get('axes'), 'scope axes')
    if set(axes) != set(_AXES):
        _fail('SYSTEM_COVERAGE_AXES', 'Exactly the five declared system axes are required')
    levels = {a: _ex_ids(axes[a], a) for a in _AXES}
    required = _ex_ids(contract.get('required_cell_ids'), 'required_cell_ids')
    violations, traces = [], []
    def issue(code, unit, detail):
        violations.append(dict(code=code, unit_id=unit, detail=detail))
    for cid in required:
        if cid not in index:
            issue('MISSING_REQUIRED_CELL', cid, 'Required scoped cell has no applicability record')
    for f in funcs:
        if f.get('status') not in ('IMPLEMENTED', 'PLANNED', 'NOT_IMPLEMENTED'):
            _fail('SYSTEM_COVERAGE_STATUS', 'Unknown function implementation status')
        _ex_ids(f.get('languages'), 'function languages', empty=True)
    for trip in trips:
        _ex_ids(trip.get('function_ids'), 'journey function_ids', empty=True)
    for proof in proofs:
        for key in ('kind', 'language', 'status', 'reference', 'content_hash', 'cell_id'):
            _vb_string(proof.get(key), 'evidence ' + key)
        if proof['cell_id'] not in index:
            issue('ORPHAN_EVIDENCE', proof['evidence_id'], 'Evidence references an unknown cell')
    seen = set()
    for row in rows:
        cid = row['cell_id']
        before = len(violations)
        coordinates = _ex_object(row.get('axes'), 'cell axes')
        if set(coordinates) != set(_AXES) or any(coordinates[a] not in levels[a] for a in _AXES):
            _fail('SYSTEM_COVERAGE_COORDINATES', 'Cell coordinates lie outside the explicit five-axis scope')
        key = tuple(coordinates[a] for a in _AXES)
        if key in seen:
            issue('DUPLICATE_COORDINATE', cid, 'Two applicability records describe the same five-axis cell')
        seen.add(key)
        status = row.get('applicability')
        if status not in ('REQUIRED', 'NOT_APPLICABLE', 'GAP'):
            _fail('SYSTEM_COVERAGE_APPLICABILITY', 'Use REQUIRED, NOT_APPLICABLE or GAP')
        if cid in required and status != 'REQUIRED':
            issue('REQUIRED_SCOPE_DOWNGRADED', cid, 'A declared required cell cannot be waived by its evidence record')
        if status == 'REQUIRED' and cid not in required:
            issue('UNREGISTERED_REQUIRED_CELL', cid, 'Required applicability must be included in the frozen scope obligations')
        if status == 'NOT_APPLICABLE':
            decision = row.get('applicability_decision', {})
            for k in ('reason', 'version', 'authority', 'reference'):
                if not isinstance(decision.get(k), str) or not decision[k].strip():
                    issue('UNJUSTIFIED_NONAPPLICABILITY', cid, 'Non-applicability requires reason, version, authority and reference')
                    break
            if decision.get('reason_code') not in ('OUTSIDE_BUSINESS_SCOPE', 'INCOMPATIBLE_TARGET', 'INAPPLICABLE_LIFECYCLE'):
                issue('INVALID_NONAPPLICABILITY_REASON', cid, 'Missing implementation is not a non-applicability reason')
        elif status == 'GAP':
            issue('EXPLICIT_COVERAGE_GAP', cid, 'Applicable cell remains unresolved')
        else:
            declared = row.get('contracts', {})
            for k in ('data', 'target', 'result', 'measure', 'horizon', 'model_profile', 'policy_profile'):
                if not isinstance(declared.get(k), str) or not declared[k].strip():
                    issue('MISSING_CELL_CONTRACT', cid, k + ' definition/reference missing')
            fnids = _ex_ids(row.get('function_ids'), 'cell function_ids', empty=True)
            jids = _ex_ids(row.get('journey_ids'), 'cell journey_ids', empty=True)
            eids = _ex_ids(row.get('evidence_ids'), 'cell evidence_ids', empty=True)
            if not fnids:
                issue('MISSING_CELL_FUNCTION', cid, 'Required cell has no applicable function')
            if not jids:
                issue('MISSING_CELL_JOURNEY', cid, 'Required cell has no journey')
            for fid in fnids:
                f = fi.get(fid)
                if f is None or f['status'] != 'IMPLEMENTED' or not {'python', 'r'} <= set(f['languages']):
                    issue('FUNCTION_NOT_NATIVE_COMPLETE', cid, fid + ' lacks implemented Python and native R')
            for jid in jids:
                j = ji.get(jid)
                if j is None or not set(fnids) <= set(j['function_ids']):
                    issue('JOURNEY_FUNCTION_COVERAGE', cid, jid + ' does not cover the declared applicable functions')
            local = []
            for eid in eids:
                e = ei.get(eid)
                if e is None or e['cell_id'] != cid:
                    issue('INVALID_CELL_EVIDENCE_LINK', cid, eid + ' missing or belongs to another cell')
                else:
                    local.append(e)
            for kind in _CELL_EVIDENCE:
                languages = ('both',) if kind == 'parity' else ('python', 'r')
                for language in languages:
                    if not any(e['kind'] == kind and e['language'] == language and e['status'] == 'PASS' for e in local):
                        issue('MISSING_PASS_EVIDENCE', cid, kind + '/' + language)
        traces.append(dict(cell_id=cid, applicability=status, complete=len(violations) == before))
    covered = {a: sorted({r['axes'][a] for r in rows}) for a in _AXES}
    for a in _AXES:
        for level in levels[a]:
            if level not in covered[a]:
                issue('UNCONSIDERED_AXIS_LEVEL', a + ':' + level, 'Declared axis level has no explicit applicability cell')
    payload = dict(scope=contract, metrics=dict(declared_cells=len(rows), required_cells=len(required),
        complete_required_cells=sum(t['complete'] and t['applicability'] == 'REQUIRED' for t in traces),
        not_applicable_cells=sum(t['applicability'] == 'NOT_APPLICABLE' for t in traces)),
        cell_findings=traces, covered_axis_levels=covered, applicability='APPLICABLE',
        cartesian_completeness_claimed=False)
    return _vs_finish('validate_system_coverage', payload, violations,
        dict(cells=rows, function_inventory=funcs, journeys=trips, evidence=proofs, scope=contract), context)
