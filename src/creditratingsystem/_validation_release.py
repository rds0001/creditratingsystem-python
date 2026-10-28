"""Read-only release evidence assessment bound to an exact candidate."""
from copy import deepcopy
import re
from .core import _fail, _fingerprint
from ._expert_common import _ex_object, _ex_ids
from ._validation_common import _vb_number, _vb_string
from ._validation_system import _vs_index, _vs_finish, _CELL_EVIDENCE


def _vs_hashes(value, name, empty=False):
    value = deepcopy(_ex_object(value, name))
    if not value and not empty:
        _fail('RELEASE_HASHES', name + ' cannot be empty')
    for key, digest in value.items():
        _vb_string(key, name + ' key')
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            _fail('RELEASE_HASHES', name + ' requires lowercase SHA-256 values')
    return value


def release_candidate_assessment(candidate, evidence, *, requirements, context=None):
    """Evaluate supplied exact-candidate evidence; never run or publish a release."""
    c = deepcopy(_ex_object(candidate, 'candidate'))
    candidate_id = _vb_string(c.get('candidate_id'), 'candidate_id')
    artifacts = _vs_hashes(c.get('artifact_hashes'), 'artifact_hashes')
    resources = _vs_hashes(c.get('resource_hashes'), 'resource_hashes', empty=True)
    runtimes = ('python_wheel', 'python_sdist', 'native_r')
    if not set(runtimes) <= set(artifacts):
        _fail('RELEASE_ARTIFACTS', 'Candidate must identify wheel, sdist and native R artifacts')
    req = deepcopy(_ex_object(requirements, 'requirements'))
    for key in ('requirements_id', 'version', 'reference'):
        _vb_string(req.get(key), key)
    scopes = {kind: _ex_ids(req.get(key), key, empty=kind in ('GENERATOR', 'PROFILE'))
              for kind, key in (('CELL', 'required_cell_ids'), ('JOURNEY', 'required_journey_ids'),
                                ('GENERATOR', 'required_generator_ids'), ('PROFILE', 'required_profile_ids'))}
    expected_resources = _vs_hashes(req.get('resource_hashes'), 'required resource_hashes', empty=True)
    limits = _ex_object(req.get('limits'), 'limits')
    max_seconds = _vb_number(limits.get('max_runtime_seconds'), 'max_runtime_seconds', 0)
    max_mb = _vb_number(limits.get('max_peak_memory_mb'), 'max_peak_memory_mb', 0)
    if max_seconds == 0 or max_mb == 0:
        _fail('RELEASE_BUDGET', 'Strictly positive predeclared execution budgets required')
    rows, _ = _vs_index(evidence, 'evidence_id', 'evidence')
    violations, gates = [], []
    def issue(code, unit, detail):
        violations.append(dict(code=code, unit_id=unit, detail=detail))
    if resources != expected_resources:
        issue('RESOURCE_MANIFEST_MISMATCH', candidate_id, 'Candidate resources differ from frozen requirements')
    eligible = []
    for e in rows:
        eid = e['evidence_id']
        for key in ('scope_type', 'scope_id', 'kind', 'language', 'status', 'reference', 'data_origin'):
            _vb_string(e.get(key), 'evidence ' + key)
        _vs_hashes({'content': e.get('content_hash')}, 'evidence content_hash')
        if e['data_origin'] not in ('SYNTHETIC', 'EMPIRICAL', 'SOFTWARE', 'MIXED'):
            _fail('RELEASE_ORIGIN', 'Explicit evidence data origin required')
        if e['scope_type'] not in (*scopes, 'PACKAGE'):
            _fail('RELEASE_SCOPE', 'Unknown evidence scope type')
        if e['scope_type'] == 'PACKAGE':
            known = e['scope_id'] in runtimes
        else:
            known = e['scope_id'] in scopes[e['scope_type']]
        if not known:
            issue('UNDECLARED_EVIDENCE_SCOPE', eid, 'Evidence is outside the frozen release obligations')
        ea = _vs_hashes(e.get('artifact_hashes'), 'evidence artifact_hashes')
        er = _vs_hashes(e.get('resource_hashes'), 'evidence resource_hashes', empty=True)
        if e.get('candidate_id') != candidate_id or ea != artifacts or er != resources:
            issue('WRONG_CANDIDATE_EVIDENCE', eid, 'Evidence is not bound to this exact artifact/resource manifest')
            continue
        eligible.append(e)
    obligations = []
    for cid in scopes['CELL']:
        for kind in _CELL_EVIDENCE:
            for language in (('both',) if kind == 'parity' else ('python', 'r')):
                obligations.append(('CELL', cid, kind, language))
    for kind in ('JOURNEY', 'GENERATOR', 'PROFILE'):
        for sid in scopes[kind]:
            for runtime in runtimes:
                obligations.append((kind, sid, 'installed_execution', runtime))
    obligations.extend(('PACKAGE', runtime, 'installed_execution', runtime) for runtime in runtimes)
    for scope_type, sid, kind, language in obligations:
        key = '/'.join((scope_type, sid, kind, language))
        matching = [e for e in eligible if (e['scope_type'], e['scope_id'], e['kind'], e['language']) ==
                    (scope_type, sid, kind, language)]
        before = len(violations)
        if not matching:
            issue('MISSING_REQUIRED_EVIDENCE', key, 'No exact-candidate evidence for this mandatory gate')
        for e in matching:
            if e['status'] != 'PASS':
                issue('NONPASS_REQUIRED_EVIDENCE', e['evidence_id'], 'Mandatory failed, skipped or non-applicable evidence cannot pass')
            if kind == 'installed_execution':
                flags = {'executed': True, 'installed': True, 'offline': True, 'source_tree_access': False}
                if any(type(e.get(k)) is not bool or e[k] != v for k, v in flags.items()):
                    issue('INVALID_INSTALLED_EXECUTION', e['evidence_id'], 'Require executed installed offline run without source-tree access')
                for metric, limit in (('runtime_seconds', max_seconds), ('peak_memory_mb', max_mb)):
                    value = e.get(metric)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= limit:
                        issue('EXECUTION_BUDGET_UNPROVEN', e['evidence_id'], metric + ' missing, nonfinite or exceeds declared budget')
        gates.append(dict(gate_id=key, status='PASS' if len(violations) == before else 'BLOCKED',
                          evidence_ids=[e['evidence_id'] for e in matching]))
    payload = dict(candidate=c, requirements=req, applicability='APPLICABLE', gates=gates,
        metrics=dict(required_gate_count=len(gates), passed_gate_count=sum(g['status'] == 'PASS' for g in gates)),
        assessment_axes=dict(software_system='BLOCKED' if violations else 'PASS',
            synthetic_method_evidence='PRESENT' if any(e['data_origin'] == 'SYNTHETIC' for e in eligible) else 'NOT_ASSESSED',
            empirical_model_performance='NOT_ASSESSED', regulatory_approval='NOT_ASSESSED'),
        publication_performed=False, tests_executed_by_assessor=False,
        candidate_manifest_hash=_fingerprint(dict(candidate_id=candidate_id, artifact_hashes=artifacts, resource_hashes=resources)))
    return _vs_finish('release_candidate_assessment', payload, violations,
                      dict(candidate=c, evidence=rows, requirements=req), context)
