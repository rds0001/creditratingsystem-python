"""Predeclared criterion assessment and traceable validation report bundles."""
from copy import deepcopy
import math
from .core import _fail, _fingerprint
from .data import _date
from ._expert_common import _ex_object, _ex_ids
from ._validation_common import _vb_string, _vb_number
from ._validation_release import _vs_hashes
from ._validation_system import _vs_index, _vs_finish


def validation_report(results, *, acceptance_policy, manifests, context=None):
    """Apply frozen separate criteria; missing/NA evidence never becomes PASS."""
    rows, index = _vs_index(results, 'result_id', 'results')
    policy = deepcopy(_ex_object(acceptance_policy, 'acceptance_policy'))
    manifests = deepcopy(_ex_object(manifests, 'manifests'))
    for key in ('policy_id', 'version', 'reference'):
        _vb_string(policy.get(key), key)
    started = _date(manifests.get('evaluation_started_at'), 'evaluation_started_at')
    if _date(policy.get('declared_at'), 'policy declared_at') > started:
        _fail('VALIDATION_POLICY_LOOKAHEAD', 'Acceptance policy must precede evaluation')
    for key in ('model_manifest', 'data_manifest', 'split_manifest'):
        _vs_hashes(manifests.get(key), key)
    required = _ex_ids(policy.get('required_result_ids'), 'required_result_ids')
    criteria, _ = _vs_index(policy.get('criteria'), 'criterion_id', 'criteria')
    violations, findings, bundles = [], [], []
    def issue(code, unit, detail):
        violations.append(dict(code=code, unit_id=unit, detail=detail))
    for row in rows:
        rid = row['result_id']
        result = row.get('result')
        if hasattr(result, 'value'):
            result = result.value
        result = deepcopy(_ex_object(result, 'validation result'))
        row['result'] = result
        if result.get('artifact_type') != 'ValidationResult':
            _fail('VALIDATION_REPORT_RESULT', 'Only explicit ValidationResult payloads can be bundled')
        for key in ('reference', 'data_origin'):
            _vb_string(row.get(key), key)
        if row['data_origin'] not in ('SYNTHETIC', 'EMPIRICAL', 'SOFTWARE', 'MIXED'):
            _fail('VALIDATION_REPORT_ORIGIN', 'Explicit separate data-origin tag required')
        bindings = _vs_hashes(row.get('manifest_hashes'), 'result manifest_hashes')
        expected = {key: _fingerprint(manifests[key]) for key in ('model_manifest', 'data_manifest', 'split_manifest')}
        if bindings != expected:
            issue('RESULT_MANIFEST_MISMATCH', rid, 'Result does not match declared model/data/split manifests')
        bundles.append(dict(result_id=rid, result=result, reference=row['reference'], data_origin=row['data_origin'], result_hash=_fingerprint(result)))
    for rid in required:
        if rid not in index:
            issue('MISSING_REQUIRED_RESULT', rid, 'Required result is absent')
        elif index[rid]['result'].get('applicability') == 'NOT_APPLICABLE' or index[rid]['result'].get('decision', {}).get('status') in ('FAIL', 'BLOCKED', 'NOT_APPLICABLE'):
            issue('NONPASS_REQUIRED_RESULT', rid, 'Required result is failed, blocked or non-applicable')
        if not any(c.get('result_id') == rid for c in criteria):
            issue('UNEVALUATED_REQUIRED_RESULT', rid, 'Required result has no predeclared acceptance criterion')
    for c in criteria:
        cid = c['criterion_id']; rid = _vb_string(c.get('result_id'), 'criterion result_id')
        path = c.get('path')
        if not isinstance(path, (list, tuple)) or not path or any(not isinstance(k, str) or not k for k in path):
            _fail('VALIDATION_REPORT_PATH', 'Criterion path is a nonempty ordered array of object keys')
        op = c.get('operator')
        if op not in ('LE', 'GE', 'EQ', 'BETWEEN'):
            _fail('VALIDATION_REPORT_OPERATOR', 'Use LE, GE, EQ or BETWEEN')
        _vb_string(c.get('rationale'), 'criterion rationale')
        threshold = c.get('threshold')
        if op == 'BETWEEN':
            if not isinstance(threshold, (list, tuple)) or len(threshold) != 2:
                _fail('VALIDATION_REPORT_THRESHOLD', 'BETWEEN requires two numeric bounds')
            lo, hi = [_vb_number(t, 'criterion bound') for t in threshold]
            if lo > hi:
                _fail('VALIDATION_REPORT_THRESHOLD', 'Bounds must be ordered')
        elif op != 'EQ':
            _vb_number(threshold, 'criterion threshold')
        elif not isinstance(threshold, (str, bool, int, float)) or isinstance(threshold, float) and not math.isfinite(threshold):
            _fail('VALIDATION_REPORT_THRESHOLD', 'EQ requires a finite JSON scalar')
        value = index.get(rid, {}).get('result')
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        status = 'NOT_ESTIMABLE'
        if value is not None:
            if op == 'EQ':
                comparable = (isinstance(value, bool) and isinstance(threshold, bool)) or (isinstance(value, str) and isinstance(threshold, str)) or (type(value) in (int, float) and type(threshold) in (int, float) and math.isfinite(value))
                if comparable:
                    status = 'PASS' if value == threshold else 'FAIL'
            elif type(value) in (int, float) and math.isfinite(value):
                passed = lo <= value <= hi if op == 'BETWEEN' else value <= threshold if op == 'LE' else value >= threshold
                status = 'PASS' if passed else 'FAIL'
        if status != 'PASS':
            issue('CRITERION_' + status, cid, rid + '/' + '/'.join(path))
        findings.append(dict(criterion_id=cid, result_id=rid, path=list(path), observed=value, operator=op, threshold=threshold, status=status, rationale=c['rationale']))
    payload = dict(acceptance_policy=policy, manifests=manifests, applicability='APPLICABLE',
        findings=findings, results=bundles, metrics=dict(required_results=len(required), criteria=len(criteria),
        passed_criteria=sum(c['status'] == 'PASS' for c in findings)),
        p_values_combined=False, acceptance_scope='PREDECLARED_VALIDATION_CRITERIA_ONLY',
        renderable_report=dict(title='Validation acceptance report', result_order=[r['result_id'] for r in bundles],
        open_issues=[v['code'] + ':' + v['unit_id'] for v in violations]))
    return _vs_finish('validation_report', payload, violations,
                      dict(results=rows, acceptance_policy=policy, manifests=manifests), context)
