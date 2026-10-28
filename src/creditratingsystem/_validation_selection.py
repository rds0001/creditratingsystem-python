"""Fixed-candidate selection sensitivity with separated evidence populations."""
from copy import deepcopy
from collections import Counter
import numpy as np
from .core import _fail,_fingerprint
from .data import _date
from ._validation_common import _vb_design,_vb_records,_vb_number,_vb_draws,_vb_bands,_vb_result
from ._validation_system import _vs_index
from ._validation_affordability import _va_differences
from ._monitoring_metrics import _metrics
from . import selection_bias as sb


def selection_bias_sensitivity(candidates,evaluation_rows,selection_results,*,design,reference_model_id,uncertainty='none',cluster_resamples=None,confidence_level=.95,outcome_comparability=None,tolerance=1e-10,context=None):
    """Compare fixed predictions; replay selection arithmetic without fitting models."""
    d=_vb_design(design,context);rows=_vb_records(evaluation_rows,d);candidates,cm=_vs_index(candidates,'model_id','candidates')
    if not candidates or reference_model_id not in cm:_fail('VALIDATION_CANDIDATES','Reference candidate must exist')
    tol=_vb_number(tolerance,'tolerance',0);ids=[r.get('application_id')for r in rows];_,rm=_vs_index(rows,'application_id','evaluation rows')
    if not rows:_fail('VALIDATION_POPULATION','Nonempty through-the-door population required')
    draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);prepared=deepcopy(rows);exclusions=[];pre_reasons={};weights=[]
    for r,p in zip(rows,prepared):
        w=_vb_number(r.get('weight',1),'weight',0);weights.append(w)
        if type(r.get('funded'))is not bool:_fail('VALIDATION_SELECTION','funded must explicitly be Boolean')
        known=r.get('known_at');reason=None
        if r['horizon_end']>d['evaluation_cutoff']:reason='IMMATURE_HORIZON'
        elif known is None or _date(known,'known_at')>_date(d['evaluation_cutoff'],'cutoff'):reason='OUTCOME_NOT_YET_KNOWN'
        elif r.get('label_origin')in ('OBSERVED_OWN_CONTRACT','AUTHORIZED_EXPLORATION','EXTERNAL_CONTRACT','SYNTHETIC_TRUTH') and r.get('outcome')==0 and (r.get('complete_through')is None or _date(r['complete_through'],'complete_through')<_date(r['horizon_end'],'end')):reason='INCOMPLETE_NEGATIVE_COVERAGE'
        if reason:
            pre_reasons[r['application_id']]=reason;p.update(label_origin='UNOBSERVED',known_at=d['evaluation_cutoff'],outcome=None)
        p['mature']=r['horizon_end']<=d['evaluation_cutoff']
    if sum(weights)<=0:_fail('VALIDATION_WEIGHTS','Positive target population mass required')
    ev=sb.prepare_reject_evaluation(candidates,prepared,as_of=d['evaluation_cutoff'],target=d['target'],outcome_comparability=outcome_comparability).value
    scopes={r['application_id']:r for r in ev['rows']};flags={}
    for f in ev['contamination_flags']:flags.setdefault(f['application_id'],[]).append(f['reason'])
    for r in rows:
        key=r['application_id'];s=scopes[key]
        if not s['validation_eligible']:exclusions.append(dict(prediction_id=r['prediction_id'],reason=pre_reasons.get(key,(flags.get(key)or['NO_OBSERVED_TRUTH'])[0])))
    predictions={}
    for c in candidates:
        pred,pm=_vs_index(c.get('predictions'),'application_id','candidate predictions')
        if set(pm)!=set(ids):_fail('VALIDATION_POPULATION','Every frozen candidate must cover the exact same through-the-door cases')
        values=[]
        for r in rows:
            f=pm[r['application_id']];p=_vb_number(f.get('pd'),'pd',0)
            if p>1:_fail('VALIDATION_PROBABILITY','PD exceeds one')
            if _date(f.get('known_at'),'prediction known_at')>_date(r['as_of'],'as_of'):_fail('VALIDATION_LOOKAHEAD','Candidate prediction became known after application index')
            if f['known_at']<c['training_known_at']:_fail('VALIDATION_LOOKAHEAD','Prediction predates frozen training')
            values.append(p)
        predictions[c['model_id']]=values
    def assess(mult=None):
        result={};contrasts={}
        for scope in sorted({s['evidence_scope']for s in scopes.values()if s['validation_eligible']}):
            result[scope]={}
            for model,pred in predictions.items():
                metric_rows=[dict(eligible=scopes[r['application_id']]['validation_eligible']and scopes[r['application_id']]['evidence_scope']==scope,weight=weights[i]*(1 if mult is None else mult[i]),target=r.get('outcome'),pd=pred[i],cluster_id=r['cluster_id'])for i,r in enumerate(rows)]
                result[scope][model]=_metrics(metric_rows)
            ref=result[scope][reference_model_id]
            for model,m in result[scope].items():
                for key in ('brier','auc','calibration_error','log_loss'):
                    a,b=m[key],ref[key];contrasts[scope+'|'+model+'|'+key]=None if a is None or b is None else a-b
        return result,contrasts
    metrics,contrasts=assess();bands=_vb_bands(None if draws is None else[assess(x)[1]for x in draws],confidence_level)
    factual=[int(scopes[r['application_id']]['validation_eligible']and scopes[r['application_id']]['evidence_scope']in ('FUNDED_OBSERVED_TEST','OBSERVED_EXPLORATION'))for r in rows]
    outcomes=[r['outcome']if s else None for r,s in zip(rows,factual)];scenarios=[];violations=[]
    selection_results,_=_vs_index(selection_results,'result_id','selection results')
    methods={'IPW':sb.selection_ipw_weights,'BOUNDS':sb.reject_inference_bounds,'MNAR_SHIFT':sb.reject_inference_sensitivity,'AIPW':sb.estimate_population_risk_aipw}
    for record in selection_results:
        kind=record.get('kind');model=record.get('model_id')
        if kind not in methods or model not in cm:_fail('VALIDATION_SELECTION','Known arithmetic kind and frozen candidate required')
        if record.get('population_id')!=d['population_id']or record.get('target')!=d['target']or record.get('application_ids')!=ids:_fail('VALIDATION_POPULATION','Selection result must bind exact ordered population and target')
        inputs=deepcopy(record.get('inputs'))
        if not isinstance(inputs,dict)or 'context'in inputs:_fail('VALIDATION_SELECTION','Fixed arithmetic inputs required; context inherited from validation')
        if inputs.get('observed')!=factual or inputs.get('target_weights')!=weights:_fail('VALIDATION_SELECTION','Arithmetic sample and target weights must match actual mature funded evidence')
        if kind!='IPW'and inputs.get('outcomes')!=outcomes:_fail('VALIDATION_SELECTION','Pseudo/synthetic/proxy labels cannot replace funded observed truth in transport arithmetic')
        if kind=='MNAR_SHIFT'and inputs.get('baseline_pd')!=predictions[model]:_fail('VALIDATION_SELECTION','MNAR baseline must equal frozen candidate predictions')
        if kind=='AIPW':
            lineage=inputs.get('fold_lineage',{}).get('rows',[])
            if len(lineage)!=len(rows)or any(m.get('entity_id')!=r['entity_id']or m.get('as_of')!=r['as_of']or m.get('features_known_at')>r['as_of']or m.get('label_known_at')!=r['known_at']for m,r in zip(lineage,rows)):_fail('VALIDATION_SELECTION','OOF nuisance lineage must bind actual ordered evaluation entities and dates')
        overlap=None;computed=None;status='ASSUMPTION_CONDITIONAL'
        if kind in ('IPW','AIPW'):
            prop=np.asarray(inputs.get('propensities'),float)
            if prop.ndim not in (1,2)or len(prop)!=len(rows)or not np.all(np.isfinite(prop))or np.any((prop<0)|(prop>1)):_fail('VALIDATION_SELECTION','Aligned finite propensity probabilities required')
            joint=prop if prop.ndim==1 else np.prod(prop,axis=1)
            if np.any((joint==0)&np.array(factual,dtype=bool)):_fail('VALIDATION_SELECTION','Observed cases cannot have zero observation propensity')
            missing=float(sum(w for w,e in zip(weights,joint)if e==0));overlap=dict(minimum_propensity=float(min(joint)),maximum_propensity=float(max(joint)),zero_support_target_mass=missing,full_population_identified_under_mar=missing==0)
            trim=inputs.get('trim');retained=(joint>=trim[0])&(joint<=trim[1])if kind=='IPW'and trim is not None else np.ones(len(rows),bool)
            if np.any((joint==0)&(np.array(weights)>0)&retained):status='NOT_IDENTIFIED_ZERO_SUPPORT'
        if status!='NOT_IDENTIFIED_ZERO_SUPPORT':computed=methods[kind](**inputs).value
        mismatch=_va_differences(computed,record.get('result'),tol)
        if mismatch:violations.append(dict(result_id=record['result_id'],code='SELECTION_ARTIFACT_RECONCILIATION',paths=mismatch))
        transported=None
        if kind=='IPW'and computed is not None:
            transported=_metrics([dict(eligible=bool(factual[i]),weight=computed['weights'][i],target=outcomes[i],pd=predictions[model][i],cluster_id=r['cluster_id'])for i,r in enumerate(rows)])
        scenarios.append(dict(result_id=record['result_id'],kind=kind,model_id=model,status=status,verified=not mismatch,recomputed_result=computed,overlap=overlap,ipw_metrics_conditional_mar=transported,empirical_full_population_validation=False))
    value=_vb_result('selection_bias_sensitivity',dict(evidence_scopes=metrics,paired_candidate_minus_reference=contrasts,through_the_door_count=len(rows),funded_count=sum(r['funded']for r in rows),mature_factual_count=sum(factual)),d,rows,exclusions,bands,context,diagnostics=dict(evaluation_plan=ev,transport_scenarios=scenarios,violations=violations,candidate_hashes={c['model_id']:c['model_hash']for c in candidates},fitted_any_model=False,counterfactual_policy_performance_identified=False)).value
    value['decision']=dict(status='BLOCKED'if violations else'NOT_ASSESSED',rule='ARTIFACT_RECONCILIATION_ONLY')
    from .core import _result
    return _result(value,'validation.selection_bias_sensitivity','validation_metrics',context,limitations=('Observed funded, authorized exploration, synthetic engineering truth and transported external proxies remain separate. MAR, MNAR shifts, AIPW nuisance assumptions and partial bounds are assumptions, not observed rejected outcomes. Zero support does not identify a new lending policy. Uncertainty is paired cluster sampling conditional on frozen candidates, not refit or selection-model uncertainty.',))
