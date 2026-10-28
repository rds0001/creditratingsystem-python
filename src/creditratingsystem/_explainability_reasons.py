"""Deterministic reasons from stored same-scale attribution and policy evidence."""
from copy import deepcopy
import math
from .core import _fail,_result,_fingerprint,_numeric,_context_value
from .data import _rows,_required
from .portfolio import _ids,_sum
from .decisioning import _checked


def rank_credit_reason_codes(contributions,reference_basis,reason_dictionary,*,policy_decision=None,max_reasons=None,context=None):
    """Consolidate signed model effects; retain separate immutable policy reasons."""
    rows=_rows(contributions,'contributions');basis=deepcopy(reference_basis);ctx=_context_value(context)
    if not isinstance(basis,dict)or not isinstance(reason_dictionary,dict):_fail('REASON_CONTRACT','Versioned reference and dictionary objects required')
    _required(basis,['attribution_id','model_id','model_version','reference_id','explained_quantity','output_scale','baseline','prediction','adverse_direction'])
    _required(reason_dictionary,['dictionary_id','version','model_reasons','policy_reasons'])
    for k in ('attribution_id','model_id','model_version','reference_id','explained_quantity','output_scale'):_ids([basis[k]],k)
    for k in ('dictionary_id','version'):_ids([reason_dictionary[k]],k)
    if basis['output_scale']not in ('probability','log_odds','score','currency','linear_predictor')or basis['adverse_direction']not in ('HIGHER','LOWER'):_fail('REASON_CONTRACT','Explicit common contribution scale and adverse direction required')
    baseline=_numeric(basis['baseline'],'baseline');prediction=_numeric(basis['prediction'],'prediction')
    if baseline.ndim or prediction.ndim:_fail('REASON_CONTRACT','Scalar baseline and prediction required')
    if basis['output_scale']=='probability'and not (0<=baseline<=1 and 0<=prediction<=1):_fail('REASON_CONTRACT','Probability reference values must lie in [0,1]')
    values={}
    for row in rows:
        _required(row,['feature_id','contribution']);fid=row['feature_id'];_ids([fid],'feature_id');value=_numeric(row['contribution'],'contribution')
        if fid in values or value.ndim:_fail('REASON_DUPLICATE','One scalar stored contribution per transformed feature required')
        values[fid]=float(value)
    residual=float(prediction)-float(baseline)-_sum(values.values())
    if abs(residual)>1e-10*max(1,abs(float(prediction)),abs(float(baseline))):_fail('REASON_RECONSTRUCTION','Stored contributions must reconstruct the declared prediction scale')
    if max_reasons is not None and (type(max_reasons)is not int or max_reasons<1):_fail('REASON_LIMIT','Positive integer reason limit required')
    codes=set();assigned=set();ranked=[];suppressed=[];direction=1 if basis['adverse_direction']=='HIGHER'else -1
    def check_entry(entry):
        _required(entry,['reason_code','description','priority']);code=entry['reason_code'];_ids([code],'reason_code');_ids([entry['description']],'description')
        if code in codes or type(entry['priority'])is not int or entry['priority']<0:_fail('REASON_DICTIONARY','Distinct codes and nonnegative integer priorities required')
        codes.add(code);return code
    for entry in _rows(reason_dictionary['model_reasons'],'model_reasons'):
        code=check_entry(entry);_required(entry,['feature_ids']);features=_ids(entry['feature_ids'],'feature_ids')
        if set(features)-values.keys()or assigned.intersection(features):_fail('REASON_GROUPS','Model reason groups must use known features without duplication')
        assigned.update(features);net=_sum(values[f]for f in features);adverse=direction*net
        row=dict(reason_code=code,description=entry['description'],priority=entry['priority'],feature_ids=sorted(features),contribution=net,adverse_contribution=adverse,source_contributions=[dict(feature_id=f,contribution=values[f])for f in sorted(features)])
        if adverse>0:ranked.append(row)
        else:suppressed.append(dict(**row,suppression='NON_ADVERSE_GROUP_NET'))
    ranked.sort(key=lambda r:(-r['adverse_contribution'],r['priority'],r['reason_code']));omitted=[]
    if max_reasons is not None:omitted=ranked[max_reasons:];ranked=ranked[:max_reasons]
    policy_dictionary={}
    for entry in _rows(reason_dictionary['policy_reasons'],'policy_reasons'):
        code=check_entry(entry);_required(entry,['rule_ids']);_ids(entry['rule_ids'],'rule_ids');policy_dictionary[code]=entry
    policy=[];decision_hash=None;findings=[];action=None
    if policy_decision is not None:
        decision=_checked(policy_decision,'DecisionArtifact');decision_hash=decision['sha256'];action=decision.get('action');findings=deepcopy(decision.get('findings',[]));seen=set();aggregated={}
        for trace in decision.get('rule_trace',[]):
            if trace.get('matched')is not True:continue
            code=trace.get('reason_code');rule=trace.get('rule_id')
            if rule in seen:_fail('REASON_POLICY','Policy trace contains a duplicate matched rule')
            seen.add(rule)
            if code not in policy_dictionary or rule not in policy_dictionary[code]['rule_ids']:_fail('REASON_POLICY','Matched rule requires an explicitly bound dictionary reason')
            entry=policy_dictionary[code]
            if code not in aggregated:aggregated[code]=dict(reason_code=code,description=entry['description'],priority=entry['priority'],rules=[])
            aggregated[code]['rules'].append(deepcopy(trace))
        policy=sorted(aggregated.values(),key=lambda r:(r['priority'],r['reason_code']))
        for row in policy:row['rules'].sort(key=lambda r:r['rule_id'])
    absolute=_sum(abs(v)for v in values.values());mapped=_sum(abs(values[f])for f in assigned)
    return _result(dict(reference_basis=basis,model_reasons=ranked,policy_reasons=policy,suppressed_groups=sorted(suppressed,key=lambda r:r['reason_code']),omitted_by_limit=omitted,unmapped_contributions=[dict(feature_id=f,contribution=values[f])for f in sorted(values.keys()-assigned)],coverage=dict(feature_count=len(values),mapped_feature_count=len(assigned),absolute_contribution_mass=absolute,mapped_absolute_mass=mapped,mapped_absolute_fraction=None if absolute==0 else mapped/absolute),reconstruction_residual=residual,policy_decision_hash=decision_hash,policy_action=action,policy_findings=findings),'explainability.rank_credit_reason_codes','reason_codes',ctx,diagnostics=dict(dictionary_hash=_fingerprint(reason_dictionary),contribution_hash=_fingerprint(sorted(rows,key=lambda r:r['feature_id']))),limitations=('Model reasons consolidate same-scale stored contributions relative to the declared reference. Policy reasons reproduce matched stored rules separately and are never added to model effects. Netting transformed features within a reason can suppress individually adverse terms. Ranking is descriptive, not a causal claim or automatically a legally sufficient adverse-action notice.',))
