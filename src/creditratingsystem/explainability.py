"""Frozen local credit-model explanations with explicit reference distributions."""
from copy import deepcopy
import itertools
import math
import numpy as np
from scipy.special import expit,ndtr
from .core import _fail,_result,_numeric,_context_value,_fingerprint,CalcResult,CreditRiskError
from .data import _required
from .portfolio import _ids,_weights,_sum


def _ex_contract(contract,context=None):
    if not isinstance(contract,dict):_fail('EXPLANATION_CONTRACT','Explicit frozen model contract required')
    _required(contract,['model_id','model_version','feature_order','output_scale','target','deterministic_frozen'])
    features=_ids(contract['feature_order'],'feature_order')
    if contract['output_scale']not in ('probability','log_odds','score','currency')or contract['deterministic_frozen']is not True:_fail('EXPLANATION_CONTRACT','Declare output scale and deterministic frozen prediction')
    target=contract['target']
    if not isinstance(target,dict):_fail('EXPLANATION_CONTRACT','Explicit prediction target required')
    _required(target,['event','horizon','time_unit','probability_measure','conditioning'])
    ctx=_context_value(context)
    for key in ('horizon','time_unit','probability_measure','conditioning'):
        if ctx.get(key)is not None and ctx[key]!=target[key]:_fail('TARGET_MISMATCH','Explanation target differs from context')
    return features,ctx


def _ex_predict(predictor,x,contract):
    if not callable(predictor):_fail('EXPLANATION_PREDICTOR','Frozen local prediction callable required')
    try:raw=predictor(np.asarray(x,dtype=float).copy())
    except CreditRiskError:raise
    except Exception as error:_fail('EXPLANATION_PREDICTOR','Prediction failed: '+type(error).__name__)
    value=raw.value if isinstance(raw,CalcResult)else raw;y=_numeric(value,'predictions',vector_only=True)
    if y.shape!=(len(x),):_fail('EXPLANATION_PREDICTION','One finite prediction per input row required')
    if contract['output_scale']=='probability'and np.any((y<0)|(y>1)):_fail('EXPLANATION_PREDICTION','Probability outputs must be in [0,1]')
    return y


def _ex_groups(groups,features):
    if not isinstance(groups,dict)or not groups:_fail('EXPLANATION_GROUPS','Explicit nonoverlapping feature groups required')
    ids=sorted(groups);flat=[];indices=[]
    for gid in ids:
        _ids([gid],'group_id');members=_ids(groups[gid],'group features')
        if set(members)-set(features):_fail('EXPLANATION_GROUPS','Unknown grouped feature')
        flat+=members;indices.append([features.index(f)for f in members])
    if len(flat)!=len(set(flat))or set(flat)!=set(features):_fail('EXPLANATION_GROUPS','Every feature belongs to exactly one group')
    return ids,indices


def explain_linear_credit_prediction(linear_model,transformed_inputs,baseline,*,feature_groups,offset=None,baseline_offset=0,context=None):
    """Exact linear-scale contributions; nonlinear response deltas are separate."""
    features,ctx=_ex_contract(linear_model,context);_required(linear_model,['coefficients','intercept','link','transformed_feature_lineage'])
    x=_numeric(transformed_inputs,'transformed_inputs');base=_numeric(baseline,'baseline',vector_only=True);beta=_numeric(linear_model['coefficients'],'coefficients',vector_only=True);intercept=_numeric(linear_model['intercept'],'intercept')
    if x.ndim!=2 or x.shape[1]!=len(features)or base.shape!=(len(features),)or beta.shape!=base.shape or intercept.ndim!=0:_fail('EXPLANATION_SHAPE','Transformed matrix, baseline and coefficients must match feature order')
    lineage=linear_model['transformed_feature_lineage']
    if not isinstance(lineage,dict)or set(lineage)!=set(features)or any(not isinstance(v,list)or not v for v in lineage.values()):_fail('EXPLANATION_LINEAGE','Every transformed feature needs explicit raw-source lineage')
    off=np.zeros(len(x))if offset is None else _numeric(offset,'offset',vector_only=True);bo=_numeric(baseline_offset,'baseline_offset')
    if off.shape!=(len(x),)or bo.ndim!=0:_fail('EXPLANATION_SHAPE','Row offsets and scalar baseline offset required')
    ids,indices=_ex_groups(feature_groups,features);contributions=(x-base)*beta;grouped=np.column_stack([contributions[:,idx].sum(axis=1)for idx in indices]);baseline_eta=float(intercept+base@beta+bo);eta=float(intercept)+x@beta+off;offset_contribution=off-float(bo)
    if not np.all(np.isfinite(eta))or not np.all(np.isfinite(contributions))or not math.isfinite(baseline_eta):_fail('NUMERICAL_FAILURE','Linear explanation exceeds finite range')
    link=linear_model['link']
    if link=='identity':response=eta;reference=baseline_eta
    elif link=='logit':response=expit(eta);reference=float(expit(baseline_eta))
    elif link=='probit':response=ndtr(eta);reference=float(ndtr(baseline_eta))
    elif link=='cloglog':
        response=-np.expm1(-np.exp(np.minimum(eta,709)));reference=float(-math.expm1(-math.exp(min(baseline_eta,709))))
    else:_fail('EXPLANATION_LINK','Supported identity/logit/probit/cloglog link required')
    if link!='identity'and linear_model['output_scale']!='probability':_fail('EXPLANATION_CONTRACT','Binary probability link requires probability output scale')
    if linear_model['output_scale']=='probability'and (np.any((response<0)|(response>1))or not 0<=reference<=1):_fail('EXPLANATION_PREDICTION','Probability outputs and baseline must be in [0,1]')
    residual=eta-(baseline_eta+contributions.sum(axis=1)+offset_contribution)
    return _result(dict(feature_order=features,group_ids=ids,baseline=base.tolist(),baseline_linear_predictor=baseline_eta,linear_predictor=eta.tolist(),feature_contributions=contributions.tolist(),group_contributions=grouped.tolist(),offset_contributions=offset_contribution.tolist(),response=response.tolist(),baseline_response=reference,response_difference=(response-reference).tolist(),reconstruction_residual=residual.tolist(),contribution_scale='LINEAR_PREDICTOR',response_scale=linear_model['output_scale'],link=link,transformed_feature_lineage=deepcopy(lineage)),'explainability.explain_linear_credit_prediction','linear_predictor',ctx,diagnostics=dict(model_hash=_fingerprint(linear_model),input_hash=_fingerprint(x.tolist())),limitations=('Contributions are additive on the linear predictor scale. Logit/probit/cloglog contributions are not additive probability-point changes. Baseline and transformed-feature lineage are explicit supplied references, not causal interventions.',))


def estimate_credit_shapley_values(predictor,record,background,model_contract,*,feature_groups,masking_contract,method='exact',permutations=None,max_coalitions=4096,context=None):
    """Grouped exact or permutation Shapley values for a declared masking game."""
    features,ctx=_ex_contract(model_contract,context);x=_numeric(record,'record',vector_only=True)
    if x.shape!=(len(features),):_fail('EXPLANATION_SHAPE','One record in frozen feature order required')
    groups,indices=_ex_groups(feature_groups,features);m=len(groups)
    if not isinstance(masking_contract,dict):_fail('EXPLANATION_MASK','Explicit masking contract required')
    _required(masking_contract,['masking','reference_id','reference_version'])
    maskkind=masking_contract['masking']
    if maskkind not in ('INTERVENTIONAL_EMPIRICAL','SUPPLIED_CONDITIONAL_DISTRIBUTIONS'):_fail('EXPLANATION_MASK','Interventional and supplied conditional masking must be distinguished')
    if not isinstance(max_coalitions,int)or max_coalitions<1:_fail('EXPLANATION_BUDGET','Positive explicit coalition budget required')
    cache={};evaluated=0
    def game(mask):
        nonlocal evaluated
        key=','.join(str(i)for i in range(m)if mask&(1<<i))or 'EMPTY'
        if key in cache:return cache[key]
        if len(cache)>=max_coalitions:_fail('EXPLANATION_BUDGET','Coalition budget exceeded; provide a smaller explicit permutation plan or larger budget')
        source=background if maskkind=='INTERVENTIONAL_EMPIRICAL'else masking_contract.get('coalition_backgrounds',{}).get(key)
        if not isinstance(source,dict):_fail('EXPLANATION_MASK','Each requested reference distribution needs values and weights')
        _required(source,['values','weights']);z=_numeric(source['values'],'background values')
        if z.ndim!=2 or z.shape[1]!=len(features):_fail('EXPLANATION_SHAPE','Background matrix must match feature order')
        w=_weights(source['weights'],len(z));z=z.copy()
        for j,idx in enumerate(indices):
            if mask&(1<<j):
                if maskkind=='SUPPLIED_CONDITIONAL_DISTRIBUTIONS'and np.any(z[:,idx]!=x[idx]):_fail('EXPLANATION_MASK','Conditional background must preserve every observed coalition feature')
                z[:,idx]=x[idx]
        predictions=_ex_predict(predictor,z,model_contract);value=_sum(w*predictions);cache[key]=value;evaluated+=len(z);return value
    full=(1<<m)-1;baseline=game(0);prediction=float(_ex_predict(predictor,x[None,:],model_contract)[0]);endpoint=game(full)
    if not math.isclose(prediction,endpoint,rel_tol=1e-12,abs_tol=1e-14):_fail('EXPLANATION_PREDICTOR','Frozen predictor gives inconsistent full-coalition prediction')
    se=None;draw_contributions=None
    if method=='exact':
        if permutations is not None or 2**m>max_coalitions:_fail('EXPLANATION_BUDGET','Exact enumeration requires every coalition within budget and no permutation plan')
        phi=np.zeros(m)
        for j in range(m):
            terms=[]
            for mask in range(1<<m):
                if not mask&(1<<j):terms.append((game(mask|(1<<j))-game(mask))/(m*math.comb(m-1,mask.bit_count())))
            phi[j]=_sum(terms)
    elif method=='permutation':
        if not isinstance(permutations,list)or not permutations:_fail('EXPLANATION_PERMUTATIONS','Explicit shared permutation paths required')
        draws=[]
        for order in permutations:
            if not isinstance(order,list)or len(order)!=m or set(order)!=set(groups):_fail('EXPLANATION_PERMUTATIONS','Every permutation must contain each group exactly once')
            contributions=np.zeros(m);mask=0;previous=baseline
            for gid in order:
                j=groups.index(gid);mask|=1<<j;current=game(mask);contributions[j]=current-previous;previous=current
            draws.append(contributions)
        draws=np.asarray(draws);phi=draws.mean(axis=0);draw_contributions=draws.tolist()
        design=masking_contract.get('permutation_design')
        if design not in ('IID_RANDOM_PERMUTATIONS','FIXED_PERMUTATIONS'):_fail('EXPLANATION_PERMUTATIONS','Declare IID random or fixed permutation design')
        if design=='IID_RANDOM_PERMUTATIONS'and len(draws)>1:se=(np.std(draws,axis=0,ddof=1)/math.sqrt(len(draws))).tolist()
    else:_fail('EXPLANATION_METHOD','Method must be exact or permutation')
    residual=prediction-baseline-_sum(phi)
    if not math.isclose(residual,0,rel_tol=0,abs_tol=1e-10*max(1,abs(prediction),abs(baseline))):_fail('NUMERICAL_FAILURE','Shapley local-accuracy identity failed')
    return _result(dict(group_ids=groups,values=phi.tolist(),baseline=baseline,prediction=prediction,local_accuracy_residual=residual,output_scale=model_contract['output_scale'],masking=maskkind,method=method,permutation_standard_error=se,permutation_contributions=draw_contributions,coalition_values=[dict(coalition_key=k,value=v)for k,v in sorted(cache.items())],evaluated_coalitions=len(cache),evaluated_background_rows=evaluated),'explainability.estimate_credit_shapley_values',model_contract['output_scale'],ctx,diagnostics=dict(model_contract_hash=_fingerprint(model_contract),reference_hash=_fingerprint(dict(background=background,masking_contract=masking_contract))),limitations=('Attributions explain the frozen masking game, not causal changes or credit approval. Interventional background breaks feature dependence; conditional distributions must be supplied and validated independently. Supplied fixed permutations have no IID uncertainty estimate. Grouping changes the game and must preserve transformed-feature meaning.',))

from ._explainability_reasons import rank_credit_reason_codes

from ._explainability_local import compute_credit_ice

from ._explainability_counterfactual import find_feasible_credit_counterfactual

from ._explainability_robustness import assess_counterfactual_robustness

from ._explainability_permutation import credit_permutation_importance

from ._explainability_ale import compute_credit_ale
