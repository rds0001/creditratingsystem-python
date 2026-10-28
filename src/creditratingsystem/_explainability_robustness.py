"""Shared finite model/measurement scenarios for candidate robustness."""
from copy import deepcopy
import numpy as np
from .core import _fail,_numeric,_result,_fingerprint,CreditRiskError
from .data import _required,_rows
from .portfolio import _ids,_weights
from .explainability import _ex_contract
from ._explainability_local import compute_credit_ice,_local_constraints
from ._explainability_counterfactual import _cf_target,_cf_success


def _robust_margin(prediction,target):
    if prediction is None or target['kind']=='POLICY_ACTION':return None
    if target['operator']=='<=':return float(target['value']-prediction)
    if target['operator']=='>=':return float(prediction-target['value'])
    return float(min(prediction-target['value'][0],target['value'][1]-prediction))


def assess_counterfactual_robustness(candidates,record,model_variants,scenarios,feasibility_contract,target,*,candidate_ids,nominal_variant_id,context=None):
    """Keep every supplied scenario, including infeasible or failed evaluations."""
    if not isinstance(model_variants,dict)or nominal_variant_id not in model_variants:_fail('ROBUSTNESS_MODELS','Named local model variants including the nominal variant required')
    _ids(list(model_variants),'model_variant_ids');nominal=model_variants[nominal_variant_id]
    if not isinstance(nominal,dict):_fail('ROBUSTNESS_MODELS','Each variant needs predictor and model_contract')
    _required(nominal,['predictor','model_contract']);features,ctx=_ex_contract(nominal['model_contract'],context);target=_cf_target(deepcopy(target));base=_numeric(record,'record',vector_only=True);x=_numeric(candidates,'candidates');ids=_ids(candidate_ids,'candidate_ids')
    if base.shape!=(len(features),)or x.ndim!=2 or x.shape!=(len(ids),len(features)):_fail('ROBUSTNESS_SHAPE','Aligned candidate matrix and baseline required')
    _local_constraints(feasibility_contract,features);contracts={}
    for vid,v in model_variants.items():
        if not isinstance(v,dict):_fail('ROBUSTNESS_MODELS','Each variant needs predictor and model_contract')
        _required(v,['predictor','model_contract']);vf,_=_ex_contract(v['model_contract'],context)
        if vf!=features or v['model_contract']['target']!=nominal['model_contract']['target']or v['model_contract']['output_scale']!=nominal['model_contract']['output_scale']or not callable(v['predictor']):_fail('ROBUSTNESS_MODELS','Variants must share feature order, output scale and target with callable predictions')
        if target['kind']=='POLICY_ACTION'and (not callable(v.get('policy_evaluator'))or not isinstance(v.get('policy_contract'),dict)):_fail('ROBUSTNESS_MODELS','Policy targets require full replay for every variant')
        contracts[vid]=dict(model_contract=deepcopy(v['model_contract']),policy_contract=deepcopy(v.get('policy_contract')))
    rows=_rows(scenarios,'scenarios');_ids([r.get('scenario_id')for r in rows],'scenario_ids');weights=_weights([r.get('weight')for r in rows],len(rows));deltas=[]
    for row in rows:
        _required(row,['scenario_id','model_variant_id','delta','weight'])
        if row['model_variant_id']not in model_variants:_fail('ROBUSTNESS_MODELS','Unknown scenario model variant')
        delta=_numeric(row['delta'],'delta',vector_only=True)
        if delta.shape!=base.shape:_fail('ROBUSTNESS_SHAPE','One additive measurement perturbation per feature required')
        deltas.append(delta)
    def evaluate(vid,delta):
        v=model_variants[vid]
        try:
            with np.errstate(over='ignore'):perturbed_base=base+delta;perturbed=x+delta
            ice=compute_credit_ice(v['predictor'],perturbed_base,perturbed,v['model_contract'],feasibility_contract,varying_features=features,policy_evaluator=v.get('policy_evaluator'),policy_contract=v.get('policy_contract'),context=context).value
            return [dict(candidate_id=cid,feasible=not ice['feasibility_findings'][i],findings=ice['feasibility_findings'][i],prediction=ice['predictions'][i],policy_decision=ice['policy_decisions'][i],target_satisfied=not ice['feasibility_findings'][i]and bool(_cf_success(ice['predictions'][i],ice['policy_decisions'][i],target)),target_margin=_robust_margin(ice['predictions'][i],target),error_code=None)for i,cid in enumerate(ids)]
        except CreditRiskError as error:
            return [dict(candidate_id=cid,feasible=None,findings=[],prediction=None,policy_decision=None,target_satisfied=False,target_margin=None,error_code=error.code)for cid in ids]
    nominal_results=evaluate(nominal_variant_id,np.zeros(len(features)));reports=[]
    for row,delta,w in zip(rows,deltas,weights):reports.append(dict(scenario_id=row['scenario_id'],model_variant_id=row['model_variant_id'],delta=delta.tolist(),normalized_weight=float(w),candidates=evaluate(row['model_variant_id'],delta)))
    summaries=[]
    for i,cid in enumerate(ids):
        results=[r['candidates'][i]for r in reports];success=np.asarray([r['target_satisfied']for r in results],dtype=float);known=np.asarray([r['error_code']is None and r['feasible']is True for r in results]);margins=[r['target_margin']for r in results];all_margins=all(m is not None for m in margins)
        summaries.append(dict(candidate_id=cid,nominal=nominal_results[i],all_scenarios_succeed=all(bool(v)for v in success),scenario_success_fraction=float(weights@success),scenario_feasibility_lower=float(weights@np.asarray([r['feasible']is True for r in results],dtype=float)),scenario_feasibility_upper=float(weights@np.asarray([r['feasible']is not False for r in results],dtype=float)),unevaluable_weight=float(weights@np.asarray([r['error_code']is not None for r in results],dtype=float)),failed_scenario_ids=[row['scenario_id']for row,r in zip(rows,results)if not r['target_satisfied']],worst_target_margin=min(margins)if all_margins else None,best_target_margin=max(margins)if all_margins else None,evaluated_prediction_count=int(known.sum())))
    fragile=[]
    for j,f in enumerate(features):
        vals=[float(d[j])for d in deltas];failed=[r['scenario_id']for r in reports if any(not c['target_satisfied']for c in r['candidates'])]
        fragile.append(dict(feature_id=f,minimum_delta=min(vals),maximum_delta=max(vals),varies_across_scenarios=min(vals)!=max(vals),failed_scenarios_with_nonzero_delta=[r['scenario_id']for r in reports if r['scenario_id']in failed and r['delta'][j]!=0]))
    return _result(dict(target=target,nominal_variant_id=nominal_variant_id,candidate_ids=ids,candidate_summaries=summaries,scenarios=reports,feature_perturbation_report=fragile,scenario_count=len(rows),weight_interpretation='DECLARED_FINITE_SCENARIO_MASS_NOT_INTERVENTION_PROBABILITY'),'explainability.assess_counterfactual_robustness','robustness',ctx,diagnostics=dict(variant_contracts_hash=_fingerprint(contracts),scenario_hash=_fingerprint(rows),feasibility_hash=_fingerprint(feasibility_contract)),limitations=('Measurement perturbations are added to both baseline and candidates, so immutable features mean no action, not zero measurement uncertainty. Every supplied scenario remains in denominators, including failed evaluations. Reported feasibility bounds assign unknown feasibility to neither/both endpoints. Success fractions are finite scenario summaries, not real intervention probabilities or continuous worst-case guarantees. Feature/failure co-occurrence is descriptive, not causal attribution.',))
