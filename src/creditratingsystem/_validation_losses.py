"""Default-conditional LGD and EAD validation without invented completed workouts."""
from collections import Counter
import numpy as np
from .core import _fail
from .data import _date
from ._expert_common import _ex_object
from ._validation_common import (_vb_design,_vb_records,_vb_observed,_vb_draws,_vb_bands,_vb_errors,_vb_result,_vb_number,_vb_string)


def _vl_default(row,design,at_default_allowed):
    default=_date(row.get('default_date'),'default_date');start=_date(row['as_of'],'as_of');end=_date(row['horizon_end'],'horizon_end')
    if default<start or(default==start and not at_default_allowed)or default>end:_fail('VALIDATION_DEFAULT_DATE','Default date outside the forecast conditioning window')
    reason=_vb_observed(row,design)
    if reason is None and _date(row['outcome_known_at'],'outcome_known_at')<default:_fail('VALIDATION_LOOKAHEAD','Outcome cannot be known before default')
    return reason


def _vl_summary(p,y,weightings,rows,draws,level):
    metrics={name:_vb_errors(p,y,w)for name,w in weightings.items()};segments={}
    for segment in sorted(set(r['segment']for r in rows)):
        keep=np.array([r['segment']==segment for r in rows]);segments[segment]={name:_vb_errors(p,y,w*keep)for name,w in weightings.items()}
    samples=None
    if draws is not None:
        samples=[]
        for draw in draws:
            values={}
            for segment in [None]+sorted(segments):
                mask=np.ones(len(rows))if segment is None else np.array([r['segment']==segment for r in rows])
                for name,w in weightings.items():
                    scores=_vb_errors(p,y,w*draw*mask);prefix=''if segment is None else'segment/'+segment+'/'
                    for metric in ('bias','mae','rmse','mean_prediction','mean_outcome'):values[prefix+name+'.'+metric]=scores[metric]
            samples.append(values)
    return dict(overall=metrics,by_segment=segments),_vb_bands(samples,level)


def lgd_backtest(records,*,design,workout_policy,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Report separate count/EAD-weighted loss errors with explicit closure selection."""
    d=_vb_design(design,context)
    if d['target']['conditioning']!='default':_fail('VALIDATION_TARGET','LGD validation is default-conditioned')
    rows=_vb_records(records,d);policy=_ex_object(workout_policy,'workout_policy');method=policy.get('method');_vb_string(policy.get('reference'),'workout reference')
    if method not in ('closed_only','ipcw'):_fail('VALIDATION_WORKOUT','Use closed_only or explicitly modeled closure IPCW')
    draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);n=len(rows);p=np.zeros(n);y=np.zeros(n);w=np.zeros(n);ead=np.zeros(n);exclusions=[];open_count=0
    if method=='ipcw':
        if policy.get('identification')!='closure_independent_of_final_lgd_given_recorded_covariates':_fail('VALIDATION_SELECTION','Explicit conditional closure identification is required')
        _vb_string(policy.get('model_fingerprint'),'closure model fingerprint');known=_date(policy.get('model_known_at'),'closure model known_at');minimum=_vb_number(policy.get('minimum_observation_probability'),'minimum observation probability')
        if not 0<minimum<1:_fail('VALIDATION_SELECTION','Positive interior positivity threshold required')
        if any(known>_date(r['as_of'],'as_of')for r in rows):_fail('VALIDATION_LOOKAHEAD','Closure model must be frozen before forecasts')
    for i,row in enumerate(rows):
        p[i]=_vb_number(row.get('predicted_lgd'),'predicted_lgd');reason=_vl_default(row,d,True);status=row.get('workout_status')
        if status not in ('OPEN','CLOSED'):_fail('VALIDATION_WORKOUT','Declare OPEN or CLOSED workout')
        open_count+=status=='OPEN';prob=1.
        if method=='ipcw':
            prob=_vb_number(row.get('observation_probability'),'observation_probability')
            if not 0<=prob<=1:_fail('VALIDATION_SELECTION','Observation probabilities must be in [0,1]')
            if prob<minimum:reason=reason or'INSUFFICIENT_SELECTION_SUPPORT'
        if reason is None and status=='OPEN':reason='OPEN_WORKOUT'
        if reason is None:
            closed=_date(row.get('workout_closed_at'),'workout_closed_at')
            if closed<_date(row['default_date'],'default_date')or closed>_date(row['outcome_known_at'],'outcome_known_at'):_fail('VALIDATION_WORKOUT','Closure must follow default and precede observed final outcome')
            y[i]=_vb_number(row.get('realized_lgd'),'realized_lgd');ead[i]=_vb_number(row.get('ead_at_default'),'ead_at_default',0);w[i]=1/prob
        else:exclusions.append(dict(prediction_id=row['prediction_id'],reason=reason))
    metrics,bands=_vl_summary(p,y,dict(count_weighted=w,ead_weighted=w*ead),rows,draws,confidence_level)
    diagnostics=dict(selection_support_exclusions=sum(e['reason']=='INSUFFICIENT_SELECTION_SUPPORT'for e in exclusions),population_estimand='OBSERVED_CLOSED_WORKOUTS'if method=='closed_only'else'POSITIVE_SUPPORT_SUBPOPULATION_ONLY'if any(e['reason']=='INSUFFICIENT_SELECTION_SUPPORT'for e in exclusions)else'DEFAULT_POPULATION_UNDER_DECLARED_CLOSURE_IDENTIFICATION',error_direction='PREDICTED_MINUS_OBSERVED',workout_policy=policy,open_workout_count=open_count,closed_only_selection=method=='closed_only',closed_observation_weights=w.tolist(),unweighted_closed_sample_not_full_default_population=method=='closed_only',ipcw_estimator='SELF_NORMALIZED_HAJEK_CONDITIONAL_ON_SUPPLIED_CLOSURE_MODEL'if method=='ipcw'else None,raw_lgd_outside_unit_interval_allowed=True)
    return _vb_result('lgd_backtest',metrics,d,rows,exclusions,bands,context,diagnostics,'loss_fraction_error')


def ead_backtest(records,*,design,weighting_policy,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Evaluate EAD and reference-undrawn CCF; fully drawn lines have no CCF label."""
    d=_vb_design(design,context)
    if d['target']['conditioning']!='default':_fail('VALIDATION_TARGET','EAD validation is default-conditioned')
    currency=_vb_string(d.get('currency'),'currency');rows=_vb_records(records,d);policy=_ex_object(weighting_policy,'weighting_policy');method=policy.get('method');_vb_string(policy.get('reference'),'weighting reference')
    if method not in ('equal','reference_limit','supplied'):_fail('VALIDATION_WEIGHT','Use equal, reference_limit or supplied weights')
    draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);n=len(rows);p=np.zeros(n);y=np.zeros(n);w=np.zeros(n);pc=np.zeros(n);yc=np.zeros(n);ccw=np.zeros(n);exclusions=[];denominators=[]
    for i,row in enumerate(rows):
        if row.get('currency')!=currency:_fail('VALIDATION_CURRENCY','Exposure currency must match the design')
        p[i]=_vb_number(row.get('predicted_ead'),'predicted_ead',0);drawn=_vb_number(row.get('drawn_at_reference'),'drawn_at_reference',0);limit=_vb_number(row.get('limit_at_reference'),'limit_at_reference',0);undrawn=limit-drawn
        denominator_status='VALID'if undrawn>0 else'FULLY_DRAWN'if undrawn==0 else'OVERLIMIT'
        reason=_vl_default(row,d,False)
        if reason is None:
            y[i]=_vb_number(row.get('realized_ead'),'realized_ead',0)
            if row.get('realized_ead_date')!=row['default_date']:_fail('VALIDATION_EAD_DATE','Realized EAD must use the matched default date')
            w[i]=1. if method=='equal'else limit if method=='reference_limit'else _vb_number(row.get('weight'),'weight',0)
            if undrawn>0:pc[i]=(p[i]-drawn)/undrawn;yc[i]=(y[i]-drawn)/undrawn;ccw[i]=w[i]
        else:exclusions.append(dict(prediction_id=row['prediction_id'],reason=reason))
        denominators.append(dict(prediction_id=row['prediction_id'],undrawn_at_reference=undrawn,status=denominator_status,predicted_ccf=pc[i]if reason is None and undrawn>0 else None,observed_ccf=yc[i]if reason is None and undrawn>0 else None))
    included=np.array([r['prediction_id']not in {e['prediction_id']for e in exclusions}for r in rows],float)
    ead_metrics,bands=_vl_summary(p,y,dict(unweighted=included,selected_weighting=w),rows,draws,confidence_level)
    ccf_metrics,ccf_bands=_vl_summary(pc,yc,dict(unweighted=included*np.array([v['status']=='VALID'for v in denominators]),selected_weighting=ccw),rows,draws,confidence_level)
    metrics=dict(ead=ead_metrics,ccf=ccf_metrics);uncertainty_result=dict(ead=bands,ccf=ccf_bands)
    diagnostics=dict(currency=currency,error_direction='PREDICTED_MINUS_OBSERVED',weighting_policy=policy,denominators=denominators,ccf_undefined_count=sum(v['status']!='VALID'for v in denominators),ccf_formula='(EAD_at_default-drawn_at_reference)/(limit_at_reference-drawn_at_reference)',ccf_clipping=False)
    return _vb_result('ead_backtest',metrics,d,rows,exclusions,uncertainty_result,context,diagnostics,'exposure_and_ccf_error')
