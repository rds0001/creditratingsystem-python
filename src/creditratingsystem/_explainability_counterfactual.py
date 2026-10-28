"""Finite-domain feasible credit counterfactual search with explicit search limits."""
from copy import deepcopy
import numpy as np
from .core import _fail,_numeric,_result,_fingerprint
from .data import _required
from .portfolio import _ids
from .explainability import _ex_contract
from ._explainability_local import compute_credit_ice


def _cf_target(target):
    if not isinstance(target,dict):_fail('COUNTERFACTUAL_TARGET','Explicit versioned target required')
    _required(target,['target_id','version','kind']);_ids([target['target_id']],'target_id');_ids([target['version']],'target version')
    if target['kind']=='MODEL_OUTPUT':
        _required(target,['operator','value']);v=_numeric(target['value'],'target value')
        if target['operator']in ('<=','>='):
            if v.ndim:_fail('COUNTERFACTUAL_TARGET','Scalar boundary required')
        elif target['operator']=='INTERVAL':
            if v.shape!=(2,)or v[0]>v[1]:_fail('COUNTERFACTUAL_TARGET','Ordered closed interval required')
        else:_fail('COUNTERFACTUAL_TARGET','Supported scalar target operator required')
    elif target['kind']=='POLICY_ACTION':
        _required(target,['actions']);actions=_ids(target['actions'],'actions')
        if set(actions)-{'APPROVE','DECLINE','REFER','BLOCKED'}:_fail('COUNTERFACTUAL_TARGET','Known policy actions required')
    else:_fail('COUNTERFACTUAL_TARGET','Target is MODEL_OUTPUT or POLICY_ACTION')
    return target


def _cf_success(prediction,decision,target):
    if target['kind']=='POLICY_ACTION':return decision is not None and decision['action']in target['actions']
    if prediction is None:return False
    if target['operator']=='<=':return prediction<=target['value']
    if target['operator']=='>=':return prediction>=target['value']
    return target['value'][0]<=prediction<=target['value'][1]


def find_feasible_credit_counterfactual(predictor,record,candidates,model_contract,feasibility_contract,cost_contract,target,*,candidate_ids,max_candidates=10000,policy_evaluator=None,policy_contract=None,context=None):
    """Enumerate a supplied finite domain in cost/ID order; never claim continuous optimality."""
    features,ctx=_ex_contract(model_contract,context);base=_numeric(record,'record',vector_only=True);x=_numeric(candidates,'candidates');ids=_ids(candidate_ids,'candidate_ids');target=_cf_target(deepcopy(target))
    if base.shape!=(len(features),)or x.ndim!=2 or x.shape!=(len(ids),len(features)):_fail('COUNTERFACTUAL_SHAPE','One candidate row per ID in frozen feature order required')
    if type(max_candidates)is not int or max_candidates<1:_fail('COUNTERFACTUAL_BUDGET','Positive integer candidate budget required')
    if not isinstance(cost_contract,dict):_fail('COUNTERFACTUAL_COST','Explicit cost contract required')
    _required(cost_contract,['cost_id','version','metric','feature_weights','feature_scales'])
    for key in ('cost_id','version'):_ids([cost_contract[key]],key)
    for key in ('feature_weights','feature_scales'):
        if not isinstance(cost_contract[key],dict)or set(cost_contract[key])!=set(features):_fail('COUNTERFACTUAL_COST','Cost weights/scales must cover every feature')
    w=_numeric([cost_contract['feature_weights'][f]for f in features],'feature_weights',vector_only=True);scale=_numeric([cost_contract['feature_scales'][f]for f in features],'feature_scales',vector_only=True)
    if np.any(w<0)or not np.any(w>0)or np.any(scale<=0):_fail('COUNTERFACTUAL_COST','Nonnegative nonzero cost weights and positive feature scales required')
    with np.errstate(over='ignore',invalid='ignore'):delta=(x-base)/scale
    metric=cost_contract['metric']
    if metric=='WEIGHTED_L1':costs=np.sum(np.abs(delta)*w,axis=1)
    elif metric=='WEIGHTED_L2':costs=np.sqrt(np.sum(delta*delta*w,axis=1))
    elif metric=='WEIGHTED_CHANGE_COUNT':costs=np.sum((x!=base)*w,axis=1)
    else:_fail('COUNTERFACTUAL_COST','Explicit weighted L1, L2 or change-count metric required')
    if not np.all(np.isfinite(costs)):_fail('NUMERICAL_FAILURE','Candidate cost exceeds finite representation')
    order=sorted(range(len(ids)),key=lambda i:(float(costs[i]),ids[i]));chosen=order[:max_candidates];remaining=order[max_candidates:]
    if target['kind']=='POLICY_ACTION'and policy_evaluator is None:_fail('COUNTERFACTUAL_TARGET','Policy action target requires full policy replay')
    ice=compute_credit_ice(predictor,base,x[chosen],model_contract,feasibility_contract,varying_features=features,policy_evaluator=policy_evaluator,policy_contract=policy_contract,context=context).value
    evaluated=[];successful=[]
    for j,i in enumerate(chosen):
        feasible=not ice['feasibility_findings'][j];success=feasible and _cf_success(ice['predictions'][j],ice['policy_decisions'][j],target)
        row=dict(candidate_id=ids[i],record=x[i].tolist(),cost=float(costs[i]),changed_features=[f for k,f in enumerate(features)if x[i,k]!=base[k]],feasible=feasible,findings=ice['feasibility_findings'][j],prediction=ice['predictions'][j],policy_decision=ice['policy_decisions'][j],target_satisfied=bool(success));evaluated.append(row)
        if success:successful.append(deepcopy(row))
    status='SEARCH_LIMIT'if remaining else'FEASIBLE_SOLUTION'if successful else'NO_FEASIBLE_SOLUTION'
    return _result(dict(status=status,domain_complete=not remaining,optimality_scope='SUPPLIED_FINITE_DOMAIN'if not remaining else'EVALUATED_SUBSET_ONLY',target=target,cost_contract=deepcopy(cost_contract),baseline_record=base.tolist(),baseline_prediction=ice['baseline_prediction'],baseline_findings=ice['baseline_findings'],evaluated_candidates=evaluated,successful_candidates=successful,best_candidate=None if not successful else successful[0],unevaluated_candidate_ids=[ids[i]for i in remaining],domain_size=len(ids),evaluated_count=len(chosen),search_budget=max_candidates),'explainability.find_feasible_credit_counterfactual','counterfactual',ctx,diagnostics=dict(model_contract_hash=_fingerprint(model_contract),feasibility_hash=_fingerprint(feasibility_contract),domain_hash=_fingerprint([dict(candidate_id=ids[i],record=x[i].tolist())for i in sorted(range(len(ids)),key=lambda i:ids[i])]),cost_contract_hash=_fingerprint(cost_contract)),limitations=('Optimality is limited to the explicitly supplied finite candidate domain and declared costs. SEARCH_LIMIT remains visible even when a feasible witness was found. Constraints and model or policy replay do not establish causal intervention success or promise credit approval. Every failed evaluated candidate remains in the report.',))
