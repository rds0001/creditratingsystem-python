"""Frozen-curve right-censoring validation with explicit IPCW boundaries."""
from copy import deepcopy
from datetime import timedelta
import math
import numpy as np
from .core import _numeric,_fail,_fingerprint
from .data import _date
from ._expert_common import _ex_object
from ._validation_common import (_vb_design,_vb_records,_vb_draws,_vb_bands,_vb_result,_vb_number,_vb_string)


def survival_backtest(predicted_event_curves,observations,*,horizons,censoring,design,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """IPCW Brier, cumulative/dynamic AUC and truncated cross-cluster concordance.

    Horizon controls require observed time strictly above the horizon. Event
    priority at tied event/censoring times gives event weights G(T-)^{-1}.
    Censoring model estimation uncertainty is not silently treated as known.
    """
    d=_vb_design(design,context)
    if d['target']['conditioning']!='performing':_fail('VALIDATION_TARGET','Event curves require a performing cohort')
    factor=_vb_number(d.get('days_per_time_unit'),'days_per_time_unit',0)
    allowed={'days':[1.],'months':[30.,365/12,365.25/12],'years':[365.,365.25]}
    if factor not in allowed[d['target']['time_unit']]:_fail('VALIDATION_TIME','Declare a supported elapsed-day basis for the time unit')
    dated=deepcopy(d);dated['target']['time_unit']='days';dated['target']['horizon']=math.ceil(d['target']['horizon']*factor);rows=_vb_records(observations,dated)
    h=_numeric(horizons,'horizons',vector_only=True);p=_numeric(predicted_event_curves,'predicted_event_curves',allow_empty=True)
    if not rows and p.size==0:p=p.reshape(0,len(h))
    if np.any(h<=0)or np.any(np.diff(h)<=0)or h[-1]>d['target']['horizon']or p.shape!=(len(rows),len(h))or np.any((p<0)|(p>1))or np.any(np.diff(p,axis=1)<0):_fail('VALIDATION_SURVIVAL','Increasing supported horizons and aligned nondecreasing cumulative PD curves required')
    spec=_ex_object(censoring,'censoring');method=spec.get('method');_vb_string(spec.get('reference'),'censoring reference')
    if method not in ('complete_followup','frozen_ipcw'):_fail('VALIDATION_CENSORING','Use complete_followup or frozen_ipcw')
    times=np.array([_vb_number(r.get('observed_time'),'observed_time',0)for r in rows]);events=np.array([r.get('event')for r in rows],object)
    if any(not isinstance(e,(bool,np.bool_))for e in events):_fail('VALIDATION_EVENT','Event indicators must be Boolean; censoring is not a nondefault label')
    events=events.astype(bool);eligible=np.ones(len(rows));exclusions=[]
    for i,row in enumerate(rows):
        known=row.get('outcome_known_at');reason=None
        if known is None:reason='OUTCOME_UNKNOWN'
        elif _date(known,'outcome_known_at')>_date(d['evaluation_cutoff'],'evaluation_cutoff'):reason='OUTCOME_NOT_YET_KNOWN'
        elif _date(known,'outcome_known_at')<_date(row['as_of'],'as_of')+timedelta(days=math.ceil(times[i]*factor)):_fail('VALIDATION_LOOKAHEAD','Observed followup cannot be known before its elapsed time')
        if reason is not None:eligible[i]=0;exclusions.append(dict(prediction_id=row['prediction_id'],reason=reason))
    if method=='frozen_ipcw':
        if spec.get('independence_assumption')!='censoring_independent_of_event_features_and_across_clusters'or spec.get('tie_rule')!='event_before_censoring':_fail('VALIDATION_CENSORING','Declare marginal independent censoring and event-before-censoring ties')
        _vb_string(spec.get('model_fingerprint'),'censoring model fingerprint');known=_date(spec.get('model_known_at'),'censoring model known_at')
        if any(known>_date(r['as_of'],'as_of')for r in rows):_fail('VALIDATION_LOOKAHEAD','Censoring model must be frozen before forecasts')
        knots=_numeric(spec.get('times'),'censoring times',allow_empty=True,vector_only=True);survival=_numeric(spec.get('survival'),'censoring survival',allow_empty=True,vector_only=True)
        support=_vb_number(spec.get('support_end'),'censoring support_end',0);minimum=_vb_number(spec.get('minimum_survival'),'minimum censoring survival')
        if not 0<minimum<1 or knots.shape!=survival.shape or np.any(knots<0)or np.any(np.diff(knots)<=0)or np.any((survival<0)|(survival>1))or np.any(np.diff(survival)>0)or(len(knots)and knots[-1]>support):_fail('VALIDATION_CENSORING','Valid nonincreasing censoring survival, positive support and positivity threshold required')
        def G(t,left=False):
            index=np.searchsorted(knots,np.asarray(t),side='left'if left else'right')-1
            return np.r_[1.,survival][index+1]
    else:
        support=float(h[-1]);minimum=0.
        if np.any((~events)&(times<=h[-1])&(eligible>0)):_fail('VALIDATION_CENSORING','Complete followup requires every nonevent to be observed strictly beyond the last horizon')
        def G(t,left=False):return np.ones_like(np.asarray(t),dtype=float)
    gl=G(times,True);gr=G(times,False);gh=G(h,False);draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);clusters=np.array([r['cluster_id']for r in rows]);n=len(rows)
    def calculate(multiplicity):
        mass=eligible*multiplicity;N=float(mass.sum());out=[]
        for j,tau in enumerate(h):
            cases=events&(times<=tau)&(mass>0);controls=(times>tau)&(mass>0);supported=tau<=support and gh[j]>minimum and np.all(gl[cases]>minimum)
            reason='NO_ELIGIBLE_OBSERVATIONS'if N==0 else'INSUFFICIENT_CENSORING_SUPPORT'if not supported else None
            if reason is not None:out.append(dict(horizon=float(tau),brier=None,auc=None,applicability='NOT_APPLICABLE',reason=reason,case_count=int(cases.sum()),control_count=int(controls.sum()),censored_by_horizon=int(np.sum((~events)&(times<=tau)&(mass>0))),ipcw_effective_sample_size=None,brier_contributions=None,auc_pair_mass=None));continue
            weights=np.zeros(n);weights[cases]=mass[cases]/gl[cases];weights[controls]=mass[controls]/gh[j]
            if not np.all(np.isfinite(weights))or not np.isfinite(weights.sum()):_fail('VALIDATION_NUMERICAL','IPCW exceeds finite range; review positivity support')
            if weights.sum()==0:out.append(dict(horizon=float(tau),brier=None,auc=None,applicability='NOT_APPLICABLE',reason='NO_OBSERVED_HORIZON_LABELS',case_count=0,control_count=0,censored_by_horizon=int(np.sum((~events)&(times<=tau)&(mass>0))),ipcw_effective_sample_size=None,brier_contributions=None,auc_pair_mass=None));continue
            contributions=weights*(cases.astype(float)-p[:,j])**2/N;num=den=0.
            for i in np.flatnonzero(cases):
                js=np.flatnonzero(controls&(clusters!=clusters[i]));pair=weights[i]*weights[js];num+=float(pair@((p[i,j]>p[js,j])+.5*(p[i,j]==p[js,j])));den+=float(pair.sum())
            if not np.isfinite(num+den):_fail('VALIDATION_NUMERICAL','AUC IPCW pair weights exceed finite range')
            normalized=weights/weights.sum();out.append(dict(horizon=float(tau),brier=float(contributions.sum()),auc=None if den==0 else num/den,applicability='APPLICABLE',reason=None,case_count=int(cases.sum()),control_count=int(controls.sum()),censored_by_horizon=int(np.sum((~events)&(times<=tau)&(mass>0))),ipcw_effective_sample_size=float(1/(normalized@normalized)),brier_contributions=contributions.tolist(),auc_pair_mass=den,auc_applicability='APPLICABLE'if den>0 else'NOT_APPLICABLE',auc_reason=None if den>0 else'NO_CROSS_CLUSTER_CASE_CONTROL_PAIRS'))
        tau=float(h[-1]);case=events&(times<=tau)&(mass>0);good=tau<=support and np.all(gl[case]>minimum)and np.all(gr[case]>minimum);num=den=0.;pairs=0
        if good:
            for i in np.flatnonzero(case):
                js=np.flatnonzero((times>times[i])&(mass>0)&(clusters!=clusters[i]));weight=mass[i]*mass[js]/(gl[i]*gr[i]);num+=float(weight@((p[i,-1]>p[js,-1])+.5*(p[i,-1]==p[js,-1])));den+=float(weight.sum());pairs+=len(js)
            if not np.isfinite(num+den):_fail('VALIDATION_NUMERICAL','Concordance IPCW exceeds numerical support')
        concordance=dict(value=num/den if good and den>0 else None,applicability='APPLICABLE'if good and den>0 else'NOT_APPLICABLE',reason=None if good and den>0 else'INSUFFICIENT_CENSORING_SUPPORT'if not good else'NO_CROSS_CLUSTER_COMPARABLE_PAIRS',tau=tau,pair_count=pairs,pair_mass=den if good else None,risk_score_horizon=tau)
        values=[r['brier']for r in out];integrated=float(np.sum(np.diff(h)*(np.asarray(values)[1:]+np.asarray(values)[:-1])/2)/(h[-1]-h[0]))if len(h)>1 and all(v is not None for v in values)else None
        return dict(by_horizon=out,concordance=concordance,integrated_brier=integrated,integration_interval=[float(h[0]),float(h[-1])],integration_rule='TRAPEZOID_ON_SUPPLIED_GRID_NOT_EXACT_CONTINUOUS_INTEGRAL')
    metrics=calculate(np.ones(n));samples=None
    if draws is not None:
        samples=[]
        for draw in draws:
            m=calculate(draw);flat={'integrated_brier':m['integrated_brier'],'concordance':m['concordance']['value']}
            for j,r in enumerate(m['by_horizon']):flat['brier_'+str(j)]=r['brier'];flat['auc_'+str(j)]=r['auc']
            samples.append(flat)
    diagnostics=dict(censoring=spec,days_per_time_unit=factor,horizon_censoring_rule='CONTROLS_REQUIRE_OBSERVED_TIME_STRICTLY_GREATER_THAN_HORIZON',event_weight='1/G(event_time_left)',control_weight='1/G(horizon_right)',concordance_pair_weight='1/(G(event_time_left)*G(event_time_right)); continuous-censoring limit is Uno 1/G^2',pair_population='DIFFERENT_DECLARED_SAMPLING_CLUSTERS',score_ties='HALF_CREDIT',event_time_ties='NOT_COMPARABLE',brier_denominator='TOTAL_ELIGIBLE_MULTIPLICITY_NOT_SUM_OF_IPCW',censoring_estimation_uncertainty='NOT_INCLUDED',prediction_fingerprint=_fingerprint(p.tolist()))
    result=_vb_result('survival_backtest',metrics,d,rows,exclusions,_vb_bands(samples,confidence_level),context,diagnostics,'survival_validation')
    if all(r['brier']is None for r in metrics['by_horizon'])and metrics['concordance']['value']is None:result.value.update(applicability='NOT_APPLICABLE',reason='NO_SUPPORTED_SURVIVAL_METRIC')
    return result
