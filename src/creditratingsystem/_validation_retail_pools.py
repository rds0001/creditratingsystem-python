"""Independent holdout diagnostics for dated frozen retail risk pools."""
from copy import deepcopy
from collections import Counter
from .core import _fail,_result,_fingerprint,CalcResult
from .data import _date
from ._retail_pools import _checked,assign_retail_risk_pool,estimate_pool_risk_parameters
from ._validation_common import _vb_design,_vb_records,_vb_number,_vb_string,_vb_draws,_vb_bands,_vb_errors
from ._validation_system import _vs_index
from ._validation_affordability import _va_differences
from ._monitoring_metrics import _metrics


def _vrp_assignment(value):
    a=value.value if isinstance(value,CalcResult)else deepcopy(value)
    if not isinstance(a,dict)or a.get('artifact_type')!='RETAIL_POOL_ASSIGNMENTS':_fail('VALIDATION_POOL','Sealed retail assignments required')
    payload={k:v for k,v in a.items()if k!='artifact_hash'}
    if a.get('artifact_hash')!=_fingerprint(payload):_fail('VALIDATION_POOL','Assignment content binding differs')
    return a


def retail_pool_diagnostics(fitted_model,pool_assignments,assignment_inputs,predictions,cohort_outcomes,workouts,links,estimation_profile,*,design,previous_assignments=None,uncertainty='none',cluster_resamples=None,confidence_level=.95,tolerance=1e-10,context=None):
    """Replay pool assignment, then diagnose separate mature PD/LGD/EAD evidence."""
    d=_vb_design(design,context);model=_checked(fitted_model);a=_vrp_assignment(pool_assignments);tol=_vb_number(tolerance,'tolerance',0)
    if any(model['target'].get(k)!=v for k,v in d['target'].items())or model['target']['currency']!=d.get('currency'):_fail('VALIDATION_POOL','Pool target/currency differs from validation design')
    if not isinstance(assignment_inputs,dict)or any(k in assignment_inputs for k in ('context','fitted_model')):_fail('VALIDATION_POOL','Original assignment inputs without model/context required')
    replay=assign_retail_risk_pool(fitted_model=model,**assignment_inputs).value;differences=_va_differences(a,replay,tol);violations=[]
    if differences:violations.append(dict(code='POOL_ASSIGNMENT_RECONCILIATION',observation_id=None,paths=differences))
    rows=_vb_records(predictions,d);_,fm=_vs_index(rows,'observation_id','predictions');_,am=_vs_index(replay['assignments'],'observation_id','assignments');links,lm=_vs_index(links,'observation_id','links');cohort,om=_vs_index(cohort_outcomes,'observation_id','cohort outcomes')
    if set(fm)!=set(am)or set(lm)!=set(am)or not set(om)<=set(am):_fail('VALIDATION_POOL','Predictions/links must cover exactly assigned population; outcomes may be missing')
    if not rows:_fail('VALIDATION_POOL','Nonempty assigned population required')
    cut=d['evaluation_cutoff'];connected={};cluster_conflicts=[];extra_exclusions=[];units=set()
    for r in rows:
        key=r['observation_id'];link=lm[key];assignment=am[key]
        for field in ('facility_id','obligor_id','household_id','risk_group_id','diagnostic_group'):_vb_string(link.get(field),field)
        unit=link[model['target']['default_unit']+'_id']
        if unit in units:_fail('VALIDATION_POOL','One forecast per declared default unit required; aggregate facilities for obligor targets')
        units.add(unit)
        if r['entity_id']!=unit or r['as_of']!=assignment['index_date']:_fail('VALIDATION_POOL','Prediction must identify the declared default unit and assignment index')
        if _date(link.get('known_at'),'link known_at')>_date(r['as_of'],'index')or model['training']['known_at']>=r['as_of']:_fail('VALIDATION_LOOKAHEAD','Pool model and relationships must precede held-out index')
        if r.get('excluded_from_training')is not True or r.get('excluded_from_calibration')is not True:_fail('VALIDATION_POOL','Explicit held-out model and calibration exclusion required')
        p=_vb_number(r.get('pd'),'pd',0)
        if p>1:_fail('VALIDATION_POOL','PD exceeds one')
        _vb_number(r.get('reference_exposure'),'reference_exposure',0)
        for parameter in ('lgd','ead'):
            if r.get(parameter)is not None:_vb_number(r[parameter],parameter,0 if parameter=='ead'else None)
        for field in ('obligor_id','household_id','risk_group_id'):
            k=(field,link[field]);old=connected.get(k)
            if old is not None and old!=r['cluster_id']:cluster_conflicts.append(dict(relation=field,group_id=link[field]))
            connected[k]=r['cluster_id']
        o=om.get(key)
        if o is not None:
            if o.get('default_unit_id')!=unit or o.get('index_date')!=r['as_of']or o.get('reference_exposure')!=r['reference_exposure']:_fail('VALIDATION_POOL','Outcome unit/date/exposure must match frozen prediction')
            if o.get('defaulted')is False and(o.get('complete_through')is None or _date(o['complete_through'],'coverage')<_date(o['label_window_end'],'end')):
                o['outcome_status']='CENSORED';extra_exclusions.append(dict(observation_id=key,reason='INCOMPLETE_NEGATIVE_COVERAGE'))
    if cluster_conflicts and uncertainty!='none':_fail('VALIDATION_DEPENDENCE','Declared connected units cannot cross bootstrap clusters')
    if estimation_profile.get('known_at')>min(r['as_of']for r in rows):_fail('VALIDATION_LOOKAHEAD','Diagnostic weighting profile must precede evaluation indices')
    # Arithmetic observed summaries only: no model fit and no reuse as calibration.
    observed=estimate_pool_risk_parameters(replay,cohort,workouts,estimation_profile,evaluation_cutoff=cut,calibration_id='validation-observed-only',calibration_version='1')
    excluded={x['observation_id']:x['reason']for x in observed.diagnostics['excluded_cohort_outcomes']};excluded.update({x['observation_id']:x['reason']for x in extra_exclusions});episodes=observed.diagnostics['episode_results'];draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level)
    for key in set(am)-set(om):excluded[key]='MISSING_OUTCOME'
    pool_ids=replay['pool_ids']
    def assess(mult=None):
        summaries=[];flat={};global_count=[];global_ead=[];pool_aggregated=[]
        for pool in pool_ids:
            idx=[i for i,r in enumerate(rows)if am[r['observation_id']]['pool_id']==pool];cohort_rows=[];ead_rows=[];residuals={};lp=[];ly=[];lw=[];ep=[];ey=[];ew=[]
            for i in idx:
                r=rows[i];key=r['observation_id'];o=om.get(key);m=1 if mult is None else mult[i];eligible=key not in excluded and o is not None
                rr=dict(eligible=eligible,weight=m,target=int(o['defaulted'])if eligible else None,pd=r['pd'],cluster_id=r['cluster_id']);cohort_rows.append(rr);ead_rows.append(dict(rr,weight=m*r['reference_exposure']))
                if eligible:
                    group=lm[key]['diagnostic_group'];residuals.setdefault(group,[]).append((o['defaulted']-r['pd'],m));episode=episodes.get(key)
                    if episode is not None:
                        loss=episode['lgd'];actual_ead=loss['ead_at_default']
                        if r.get('ead')is not None:ep.append(r['ead']);ey.append(actual_ead);ew.append(m*(1 if estimation_profile['ead_weighting']=='episode'else r['reference_exposure']))
                        if loss['episode_status']=='CLOSED'and loss['economic_lgd']is not None and r.get('lgd')is not None:lp.append(r['lgd']);ly.append(loss['economic_lgd']);lw.append(m*(1 if estimation_profile['lgd_weighting']=='episode'else actual_ead))
            count=_metrics(cohort_rows);exposure=_metrics(ead_rows);global_count+=cohort_rows;global_ead+=ead_rows
            active=[r for r in cohort_rows if r['eligible']and r['weight']>0];variable=len({r['pd']for r in active})>1
            within=dict(auc=count['auc']if variable else None,gini=count['gini']if variable else None,applicability='APPLICABLE'if variable and count['auc']is not None else'NOT_APPLICABLE',reason='CONSTANT_POOL_PD_NO_WITHIN_RANKING'if not variable else'ONE_OBSERVED_CLASS'if count['auc']is None else None)
            if active:
                mass=sum(x['weight']for x in active);mean=sum(x['weight']*x['pd']for x in active)/mass
                pool_aggregated += [dict(x,pd=mean)for x in active]
            group_means={k:sum(v*w for v,w in vals)/sum(w for v,w in vals)if sum(w for v,w in vals)>0 else None for k,vals in sorted(residuals.items())};valid=[v for v in group_means.values()if v is not None];spread=max(valid)-min(valid)if len(valid)>=2 else None
            lgd=_vb_errors(lp,ly,lw);ead=_vb_errors(ep,ey,ew);summary=dict(pool_id=pool,assigned_count=len(idx),reference_exposure_sum=sum(rows[i]['reference_exposure']for i in idx),population_count_share=len(idx)/len(rows),population_exposure_share=sum(rows[i]['reference_exposure']for i in idx)/sum(r['reference_exposure']for r in rows)if sum(r['reference_exposure']for r in rows)>0 else None,frozen_pd_distribution=[dict(pd=p,count=sum(rows[i]['pd']==p for i in idx))for p in sorted({rows[i]['pd']for i in idx})],pd_count_weighted=count,pd_exposure_weighted=exposure,within_pool_discrimination=within,residual_group_means=group_means,residual_group_spread=spread,lgd_error=lgd,ead_error=ead,homogeneity_decision='NOT_ASSESSED');summaries.append(summary)
            for key,value in [('pd_count_error',count['calibration_error']),('pd_exposure_error',exposure['calibration_error']),('residual_group_spread',spread),('lgd_bias',lgd['bias']),('ead_bias',ead['bias'])]:flat[pool+'|'+key]=value
        between=_metrics(pool_aggregated)
        between['ranking_applicability']='APPLICABLE'if len({r['pd']for r in pool_aggregated})>1 and between['auc']is not None else'NOT_APPLICABLE'
        if len({r['pd']for r in pool_aggregated})<2:between.update(auc=None,gini=None)
        return dict(pools=summaries,overall_count=_metrics(global_count),overall_exposure=_metrics(global_ead),between_pool_discrimination=between),flat
    metrics,point=assess();bands=_vb_bands(None if draws is None else[assess(x)[1]for x in draws],confidence_level)
    def concentration(field):
        groups={}
        for r in rows:
            group=lm[r['observation_id']][field];v=groups.setdefault(group,dict(count=0,exposure=0.));v['count']+=1;v['exposure']+=r['reference_exposure']
        n=len(rows);total=sum(v['exposure']for v in groups.values());table=[dict(group_id=k,count=v['count'],count_share=v['count']/n,exposure=v['exposure'],exposure_share=v['exposure']/total if total else None)for k,v in sorted(groups.items())]
        return dict(groups=table,count_hhi=sum(v['count_share']**2 for v in table),exposure_hhi=sum(v['exposure_share']**2 for v in table)if total else None)
    migration=[]
    if previous_assignments is not None:
        prev=_vrp_assignment(previous_assignments);_,old=_vs_index(prev['assignments'],'observation_id','previous assignments')
        if prev['target']!=replay['target']:_fail('VALIDATION_POOL','Previous assignment target differs')
        for r in rows:
            key=r['observation_id'];prior=lm[key].get('previous_observation_id')
            if prior is None:migration.append(dict(observation_id=key,status='NO_PRIOR_OBSERVATION',from_pool=None,to_pool=am[key]['pool_id'],changed=None));continue
            if prior not in old or old[prior]['index_date']>=r['as_of']or prev['information_cutoff']>r['as_of']:_fail('VALIDATION_POOL','Prior assignment must be known and strictly earlier')
            comparable=prev['model_hash']==replay['model_hash'];covered=old[prior]['status']=='COVERED'and am[key]['status']=='COVERED'
            migration.append(dict(observation_id=key,status='DEFINITION_CHANGED'if not comparable else'UNASSIGNED_ENDPOINT'if not covered else'COMPARABLE',from_pool=old[prior]['pool_id'],to_pool=am[key]['pool_id'],changed=old[prior]['pool_id']!=am[key]['pool_id']if comparable and covered else None))
    actual=deepcopy(observed.value['pool_risk_table'])
    for row in actual:
        row.pop('unweighted_pd_exact_interval');row.pop('unweighted_pd_one_sided_upper')
        for key in ('pd','closed_workout_lgd','default_conditioned_ead','default_conditioned_ccf'):row[key].pop('independent_episode_standard_error',None)
    counts=Counter(v['code']for v in violations);value=dict(artifact_type='ValidationResult',design=d,metrics=metrics,uncertainty=bands,p_value=None,decision=dict(status='BLOCKED'if violations else'NOT_ASSESSED',rule='FROZEN_ASSIGNMENT_RECONCILIATION'),violations=violations,violation_counts=dict(sorted(counts.items())),coverage=dict(input_count=len(rows),mature_eligible_count=len(rows)-len(excluded),exclusions=[dict(observation_id=k,reason=v)for k,v in sorted(excluded.items())],assignment_status_counts=dict(sorted(Counter(r['status']for r in replay['assignments']).items()))),concentration={k:concentration(k)for k in ('obligor_id','household_id','risk_group_id')},migration=migration,observed_parameter_summaries=actual,workout_exclusions=observed.diagnostics['excluded_workouts'],dependence=dict(connected_cluster_conflicts=cluster_conflicts,independent_binomial_inference_used=False,household_default_propagation=False),frozen_model_hash=model['artifact_hash'],assignments_hash=replay['artifact_hash'],input_fingerprint=_fingerprint(dict(predictions=predictions,cohort_outcomes=cohort_outcomes,workouts=workouts,links=links)),fitted_any_model=False)
    return _result(value,'validation.retail_pool_diagnostics','validation_metrics',context,limitations=('Frozen pool assignment is replayed, never fitted on evaluation labels. Observed risk summaries are diagnostic only and cannot replace frozen predictions. PD, closed-workout LGD and observed default EAD have different denominators. Missing/open workouts remain unobserved; connected household relationships never propagate default labels. Cluster intervals condition on frozen pools and forecasts; sparse or constant-PD pools do not establish homogeneity or within-pool ranking.',))
