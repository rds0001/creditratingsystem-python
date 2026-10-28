"""Explicit feasible finite local domains for frozen credit model inspection."""
from copy import deepcopy
import numpy as np
from .core import _fail,_numeric,_result,_fingerprint,map_pd_to_grade,CreditRiskError
from .decisioning import _checked
from .data import _required
from .portfolio import _ids
from .explainability import _ex_contract,_ex_predict


def _local_constraints(contract,features):
    if not isinstance(contract,dict):_fail('LOCAL_CONSTRAINTS','Versioned feasibility contract required')
    _required(contract,['contract_id','version','bounds','actionable_features','immutable_features','integer_features','one_hot_groups','linear_constraints','derived_relations'])
    for k in ('contract_id','version'):_ids([contract[k]],k)
    def subset(key):
        v=contract[key]
        if not isinstance(v,list)or any(not isinstance(f,str)for f in v)or len(v)!=len(set(v))or set(v)-set(features):_fail('LOCAL_CONSTRAINTS','Explicit distinct known features required: '+key)
        return set(v)
    actionable=subset('actionable_features');immutable=subset('immutable_features');subset('integer_features')
    if actionable&immutable or actionable|immutable!=set(features):_fail('LOCAL_CONSTRAINTS','Actionable and immutable features must partition model features')
    bounds=contract['bounds']
    if not isinstance(bounds,dict)or set(bounds)!=set(features):_fail('LOCAL_CONSTRAINTS','Finite declared bounds required for every feature')
    for f in features:
        b=_numeric(bounds[f],'bounds',vector_only=True)
        if b.shape!=(2,)or b[0]>b[1]:_fail('LOCAL_CONSTRAINTS','Ordered finite bound pair required')
    seen=set();ids=set()
    for g in contract['one_hot_groups']:
        members=_ids(g,'one_hot_group')
        if set(members)-set(features)or seen.intersection(members):_fail('LOCAL_CONSTRAINTS','One-hot groups must be disjoint known features')
        seen.update(members)
    for rule in contract['linear_constraints']:
        _required(rule,['constraint_id','coefficients','operator','rhs']);rid=rule['constraint_id'];_ids([rid],'constraint_id')
        if rid in ids or rule['operator']not in ('<=','>=','==')or not isinstance(rule['coefficients'],dict)or not rule['coefficients']or set(rule['coefficients'])-set(features):_fail('LOCAL_CONSTRAINTS','Distinct identified linear rules on known features required')
        ids.add(rid)
        for v in list(rule['coefficients'].values())+[rule['rhs']]:
            if _numeric(v,'linear scalar').ndim:_fail('LOCAL_CONSTRAINTS','Scalar linear coefficients required')
    for rule in contract['derived_relations']:
        _required(rule,['constraint_id','output_feature','input_features','operation']);rid=rule['constraint_id'];_ids([rid],'constraint_id');members=rule['input_features']
        if rid in ids or rule['output_feature']not in features or not isinstance(members,list)or len(members)!=2 or any(f not in features for f in members)or rule['operation']not in ('product','ratio'):_fail('LOCAL_CONSTRAINTS','Distinct two-input product/ratio identity required')
        ids.add(rid)
    return contract


def _local_feasible(x,baseline,features,contract):
    findings=[];index={f:i for i,f in enumerate(features)};tol=1e-10
    for j,f in enumerate(features):
        lo,hi=contract['bounds'][f]
        if x[j]<lo or x[j]>hi:findings.append('BOUND:'+f)
        if f in contract['immutable_features']and x[j]!=baseline[j]:findings.append('IMMUTABLE:'+f)
        if f in contract['integer_features']and x[j]!=np.rint(x[j]):findings.append('INTEGER:'+f)
    for i,group in enumerate(contract['one_hot_groups']):
        v=x[[index[f]for f in group]]
        if np.any((v!=0)&(v!=1))or v.sum()!=1:findings.append('ONE_HOT:'+str(i))
    for rule in contract['linear_constraints']:
        terms=[float(c)*x[index[f]]for f,c in rule['coefficients'].items()];lhs=sum(terms);rhs=rule['rhs'];eps=tol*max(1,abs(lhs),abs(rhs))
        valid=np.isfinite(lhs)and (lhs<=rhs+eps if rule['operator']=='<='else lhs>=rhs-eps if rule['operator']=='>='else abs(lhs-rhs)<=eps)
        if not valid:findings.append('LINEAR:'+rule['constraint_id'])
    for rule in contract['derived_relations']:
        a,b=(x[index[f]]for f in rule['input_features']);expected=a*b if rule['operation']=='product'else a/b if b!=0 else np.nan;actual=x[index[rule['output_feature']]]
        if not np.isfinite(expected)or abs(actual-expected)>tol*max(1,abs(actual),abs(expected)):findings.append('DERIVED:'+rule['constraint_id'])
    return sorted(findings)


def compute_credit_ice(predictor,record,grid,model_contract,feasibility_contract,*,varying_features,thresholds=None,rating_scale=None,policy_evaluator=None,policy_contract=None,context=None):
    """Evaluate a supplied ordered feature path, retaining infeasibility and turns."""
    features,ctx=_ex_contract(model_contract,context);base=_numeric(record,'record',vector_only=True);vary=_ids(varying_features,'varying_features');values=_numeric(grid,'grid')
    if base.shape!=(len(features),)or set(vary)-set(features)or values.ndim!=2 or values.shape[1]!=len(vary):_fail('LOCAL_SHAPE','Record and grid must match frozen feature and varying-feature orders')
    contract=_local_constraints(feasibility_contract,features);thresholds=[]if thresholds is None else deepcopy(thresholds);seen=set()
    for t in thresholds:
        _required(t,['threshold_id','value','operator']);tid=t['threshold_id'];_ids([tid],'threshold_id')
        if tid in seen or t['operator']not in ('<=','>=')or _numeric(t['value'],'threshold').ndim:_fail('LOCAL_THRESHOLD','Distinct finite scalar prediction boundaries required')
        seen.add(tid)
    rows=np.tile(base,(len(values),1));rows[:,[features.index(f)for f in vary]]=values
    findings=[_local_feasible(row,base,features,contract)for row in rows];valid=[i for i,f in enumerate(findings)if not f];pred=[None]*len(rows)
    if valid:
        for i,v in zip(valid,_ex_predict(predictor,rows[valid],model_contract)):pred[i]=float(v)
    base_findings=_local_feasible(base,base,features,contract);base_prediction=None if base_findings else float(_ex_predict(predictor,base[None,:],model_contract)[0])
    if rating_scale is not None and model_contract['output_scale']!='probability':_fail('LOCAL_GRADE','Rating scale requires probability outputs')
    grades=[None if v is None or rating_scale is None else map_pd_to_grade(v,rating_scale).value for v in pred]
    decisions=[None]*len(rows)
    if policy_evaluator is not None or policy_contract is not None:
        if not callable(policy_evaluator)or not isinstance(policy_contract,dict):_fail('LOCAL_POLICY','Frozen full-policy evaluator and contract required together')
        _required(policy_contract,['policy_hash','source_mode','fixed_source_hashes'])
        if policy_contract['source_mode']!='REPLAY_FULL_POLICY'or not isinstance(policy_contract['fixed_source_hashes'],dict):_fail('LOCAL_POLICY','Full policy replay with explicit fixed input hashes required')
        for source in policy_contract['fixed_source_hashes']:
            if source not in ('risk','affordability','eligibility','operations'):_fail('LOCAL_POLICY','Unknown fixed policy source')
        for i in valid:
            try:replayed=policy_evaluator(rows[i].tolist(),pred[i])
            except CreditRiskError:raise
            except Exception as e:_fail('LOCAL_POLICY','Full policy replay failed: '+type(e).__name__)
            if not isinstance(replayed,dict):_fail('LOCAL_POLICY','Replay must return decision and explained_input_hash')
            _required(replayed,['decision','explained_input_hash'])
            if replayed['explained_input_hash']!=_fingerprint(dict(record=rows[i].tolist(),prediction=pred[i])):_fail('LOCAL_POLICY','Replay input binding does not match evaluated record and prediction')
            decision=_checked(replayed['decision'],'DecisionArtifact')
            if decision.get('policy_hash')!=policy_contract['policy_hash']or decision.get('action')not in ('APPROVE','DECLINE','REFER','BLOCKED'):_fail('LOCAL_POLICY','Replay must preserve policy hash and known action')
            for source,h in policy_contract['fixed_source_hashes'].items():
                if decision.get('input_hashes',{}).get(source)!=h:_fail('LOCAL_POLICY','Replay changed an explicitly fixed source')
            decisions[i]=deepcopy(decision)
    def matches(v):return None if v is None or not thresholds else {t['threshold_id']:bool(v<=t['value']if t['operator']=='<='else v>=t['value'])for t in thresholds}
    states=[matches(v)for v in pred];segments=[];directions=[]
    for i in range(1,len(rows)):
        supported=pred[i-1]is not None and pred[i]is not None;delta=pred[i]-pred[i-1]if supported else None
        sign=None if delta is None else int(np.sign(delta));directions.append(sign)
        segments.append(dict(from_index=i-1,to_index=i,status='EVALUATED'if supported else'NON_ESTIMABLE',prediction_change=delta,threshold_crossings=[]if not supported else [t['threshold_id']for t in thresholds if states[i-1][t['threshold_id']]!=states[i][t['threshold_id']]]))
    for i,segment in enumerate(segments):
        segment['grade_change']=None if grades[i]is None or grades[i+1]is None else grades[i]!=grades[i+1]
        segment['policy_action_change']=None if decisions[i]is None or decisions[i+1]is None else decisions[i]['action']!=decisions[i+1]['action']
    # A gap terminates a contiguous path: no monotonicity inference across it.
    turns=[];last_sign=None
    for i,sign in enumerate(directions):
        if sign is None:last_sign=None
        elif sign!=0:
            if last_sign is not None and sign!=last_sign:turns.append(i)
            last_sign=sign
    return _result(dict(feature_order=features,varying_features=vary,baseline_record=base.tolist(),baseline_prediction=base_prediction,baseline_findings=base_findings,grid_values=values.tolist(),evaluated_records=rows.tolist(),predictions=pred,feasibility_findings=findings,threshold_states=states,segments=segments,turn_indices=turns,nonmonotone_on_evaluated_path=bool(turns),thresholds=thresholds,grades=grades,policy_decisions=decisions),'explainability.compute_credit_ice',model_contract['output_scale'],ctx,diagnostics=dict(model_contract_hash=_fingerprint(model_contract),feasibility_hash=_fingerprint(contract),rating_scale_hash=None if rating_scale is None else _fingerprint(rating_scale),policy_contract_hash=None if policy_contract is None else _fingerprint(policy_contract)),limitations=('The supplied grid is an ordered finite path, not a continuous threshold search. Feasibility uses only declared constraints and is not empirical support or causal intervention validity. Scalar boundaries, explicit rating-scale mapping and sealed full-policy replay are separate. Replay input binding is a caller assertion, not independent authentication. Infeasible records are not evaluated; gaps are never interpolated.',))
