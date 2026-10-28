"""Shared contracts for dated validation, explicit applicability and cluster draws."""
from copy import deepcopy
from collections import Counter
import numpy as np
from .core import _result,_numeric,_fail,_fingerprint,_context_value
from .data import _date
from ._expert_common import _ex_object,_ex_ids,_ex_target
from .hybrid import _hy_horizon_end


def _vb_string(x,name):
    if not isinstance(x,str)or not x:_fail('VALIDATION_CONTRACT',name+' must be a nonempty string')
    return x


def _vb_number(x,name,minimum=None):
    v=_numeric(x,name)
    if v.ndim or(minimum is not None and v<minimum):_fail('VALIDATION_CONTRACT',name+' outside scalar domain')
    return float(v)


def _vb_design(design,context=None,physical_only=True):
    d=deepcopy(_ex_object(design,'validation design'))
    for key in ('model_id','model_version','parameter_version','data_fingerprint','population_id','split_id'):_vb_string(d.get(key),key)
    if d.get('prediction_origin')not in ('HOLDOUT','EXTERNAL'):_fail('VALIDATION_DESIGN','Frozen HOLDOUT or EXTERNAL predictions are required')
    _date(d.get('evaluation_cutoff'),'evaluation_cutoff');d['target']=_ex_target(d.get('target'),context,allowed_events=('default','first_default'))
    if physical_only and d['target']['probability_measure']!='PHYSICAL':_fail('VALIDATION_TARGET','Observed outcomes validate physical target probabilities')
    ctx=_context_value(context)
    if d.get('currency')is not None:
        _vb_string(d['currency'],'design currency')
        if ctx.get('currency')is not None and ctx['currency']!=d['currency']:_fail('VALIDATION_CURRENCY','Design and context currencies must agree')
    if ctx.get('known_at')is not None and ctx['known_at']!=d['evaluation_cutoff']:_fail('VALIDATION_CONTEXT','Evaluation cutoff must equal context knowledge date')
    return d


def _vb_records(records,design):
    if not isinstance(records,(list,tuple))or any(not isinstance(r,dict)for r in records):_fail('VALIDATION_CONTRACT','Records must be an array of objects')
    rows=deepcopy(list(records));_ex_ids([r.get('prediction_id')for r in rows],'prediction_ids',empty=True)
    for row in rows:
        for key in ('entity_id','cluster_id','segment'):_vb_string(row.get(key),key)
        a=_date(row.get('as_of'),'as_of');known=_date(row.get('forecast_known_at'),'forecast_known_at');end=_date(row.get('horizon_end'),'horizon_end')
        if known>a:_fail('VALIDATION_LOOKAHEAD','Forecast known after its observation date')
        if end!=_hy_horizon_end(row['as_of'],design['target']):_fail('VALIDATION_HORIZON','Row horizon end differs from declared target')
        for key in ('model_id','model_version','parameter_version'):
            if row.get(key)!=design[key]:_fail('VALIDATION_VERSION','Forecast identity differs from declared design')
    return rows


def _vb_observed(row,design):
    if _date(row['horizon_end'],'horizon_end')>_date(design['evaluation_cutoff'],'evaluation_cutoff'):return 'IMMATURE_HORIZON'
    if row.get('outcome_known_at')is None:return 'OUTCOME_UNKNOWN'
    if _date(row['outcome_known_at'],'outcome_known_at')>_date(design['evaluation_cutoff'],'evaluation_cutoff'):return 'OUTCOME_NOT_YET_KNOWN'
    return None


def _vb_draws(uncertainty,cluster_resamples,rows,confidence_level):
    level=_vb_number(confidence_level,'confidence_level')
    if not 0<level<1:_fail('VALIDATION_UNCERTAINTY','Confidence level must be interior')
    if uncertainty=='none':
        if cluster_resamples is not None:_fail('VALIDATION_UNCERTAINTY','Draws require cluster_bootstrap uncertainty')
        return None
    if uncertainty!='cluster_bootstrap':_fail('VALIDATION_UNCERTAINTY','Use none or supplied cluster_bootstrap')
    units=sorted(set(r['cluster_id']for r in rows))
    if len(units)<2:_fail('VALIDATION_UNCERTAINTY','Cluster bootstrap requires at least two distinct sampling clusters')
    if not isinstance(cluster_resamples,(list,tuple))or len(cluster_resamples)<2:_fail('VALIDATION_UNCERTAINTY','At least two shared cluster draws required')
    out=[]
    for draw in cluster_resamples:
        if not isinstance(draw,(list,tuple))or len(draw)!=len(units)or any(v not in units for v in draw):_fail('VALIDATION_UNCERTAINTY','Each draw samples the original number of entire known clusters')
        counts=Counter(draw);out.append(np.array([counts[r['cluster_id']]for r in rows],dtype=float))
    return out


def _vb_bands(samples,confidence_level):
    if samples is None:return dict(status='NOT_REQUESTED',method=None,intervals=None)
    keys=sorted(set(k for s in samples for k in s));intervals={}
    for key in keys:
        values=[s.get(key)for s in samples];valid=[float(x)for x in values if x is not None and np.isfinite(x)]
        good=len(valid)==len(samples)
        intervals[key]=dict(lower=float(np.quantile(valid,(1-confidence_level)/2))if good else None,upper=float(np.quantile(valid,(1+confidence_level)/2))if good else None,applicability='APPLICABLE'if good else'NOT_APPLICABLE',reason=None if good else'NON_ESTIMABLE_RESAMPLE',identified_resamples=len(valid),degenerate=bool(good and min(valid)==max(valid)))
    return dict(status='ESTIMATED_CONDITIONAL_ON_FROZEN_MODELS',method='SUPPLIED_CLUSTER_PERCENTILE_BOOTSTRAP_TYPE7',confidence_level=confidence_level,replicates=len(samples),intervals=intervals)


def _vb_errors(predicted,observed,weights):
    p=np.asarray(predicted,float);y=np.asarray(observed,float);w=np.asarray(weights,float);active=w>0
    empty=dict(n=int(active.sum()),weight_sum=float(w.sum()),effective_sample_size=None,mean_prediction=None,mean_outcome=None,bias=None,mae=None,rmse=None,bias_contributions=[None]*len(w),applicability='NOT_APPLICABLE',reason='NO_POSITIVE_WEIGHT_OBSERVED_OUTCOMES')
    if not active.any():return empty
    if not np.isfinite(w.sum()):_fail('VALIDATION_NUMERICAL','Weight total exceeds numerical range')
    a=w/w.sum();e=p-y;bias=float(a@e);mae=float(a@np.abs(e));scale=float(np.max(np.abs(e)))
    rmse=0. if scale==0 else float(scale*np.sqrt(a@((e/scale)**2)))
    values=[bias,mae,rmse,float(a@p),float(a@y)]
    if not np.all(np.isfinite(values)):_fail('VALIDATION_NUMERICAL','Error metric exceeds numerical range')
    return dict(n=int(active.sum()),weight_sum=float(w.sum()),effective_sample_size=float(1/(a@a)),mean_prediction=values[3],mean_outcome=values[4],bias=bias,mae=mae,rmse=rmse,bias_contributions=(a*e).tolist(),applicability='APPLICABLE',reason=None)


def _vb_result(method,metrics,design,rows,exclusions,uncertainty,context=None,diagnostics=None,unit='validation_metrics'):
    counts=Counter(e['reason']for e in exclusions);included=len(rows)-len(exclusions)
    value=dict(artifact_type='ValidationResult',metrics=metrics,uncertainty=uncertainty,p_value=None,decision=dict(status='NOT_ASSESSED',rule=None),applicability='APPLICABLE'if included else'NOT_APPLICABLE',reason=None if included else'NO_ELIGIBLE_OBSERVATIONS',design=deepcopy(design),coverage=dict(input_count=len(rows),included_count=included,excluded_count=len(exclusions),exclusions=exclusions,exclusion_counts=dict(sorted(counts.items())),repeated_entities=len(rows)-len({r['entity_id']for r in rows}),cluster_count=len({r['cluster_id']for r in rows})),diagnostics=diagnostics or {},input_fingerprint=_fingerprint(rows))
    return _result(value,'validation.'+method,unit,context,limitations=('Descriptive validation does not imply approval or empirical calibration on other populations. Shared cluster draws must represent the sampling dependence; intervals condition on frozen prediction, censoring and selection models. Declared provenance is checked but not authenticated.',))
