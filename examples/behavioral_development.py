"""Prior-only behavioral fits, monthly linked customers and actual validation evidence."""
from copy import deepcopy
from datetime import date,timedelta
from importlib.resources import files
import json
import math
import os
import numpy as np
import creditratingsystem as cr


def _panel(c):return cr.build_account_month_panel(c['accounts'],c['statements'],c['events'],observation_dates=c['observation_dates'],obligations=c['obligations'],payments=c['payments']).value['accounts']
def _labels(panel,c,cutoff):return cr.build_behavioral_targets(panel,c['events'],horizon_months=2,target_event='DEFAULT',competing_events=['CLOSED'],exit_treatment='competing',evaluation_cutoff=cutoff,coverage=c['coverage']).value

def run(data=None,verify_reference=True,measured_fit=None,return_artifacts=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('behavioral_development.json').read_text())
    if measured_fit is None:measured_fit=os.environ.get('CREDITRATINGSYSTEM_EXTENDED_TESTS')=='true'
    d=deepcopy(data);tr=d['train'];hold=d['holdout'];target=d['target'];groups=[{r['customer_id']for r in tr['accounts']},{r['customer_id']for r in d['calibration']},{r['customer_id']for r in hold['accounts']}]
    if any(groups[i]&groups[j]for i in range(3)for j in range(i+1,3)):raise ValueError('Customer groups must be disjoint across training, calibration and holdout')
    if not d['training']['known_at']<min(r['index_date']for r in d['calibration'])<=d['pool_training']['known_at']<min(hold['observation_dates']):raise ValueError('Training and calibration must be known before later forecasts')
    trainpanel=_panel(tr);trainjan=[r for r in trainpanel if r['as_of']=='2020-01-31'];labels=_labels(trainjan,tr,d['training']['known_at'])
    if not all(r['calendar_mature']and r['eligible']for r in labels):raise ValueError('Training needs fully mature eligible outcomes')
    y=[r['label']for r in labels]
    kw=dict(penalty='ridge',lambda_path=[.1],selected_lambda=.1,scale=False,tolerance=1e-12,target=target,training=d['training'],model_version='1')
    anchor=cr.fit_penalized_glm(np.empty((len(y),0)),y,feature_order=[],model_id='behavioral-anchor',**kw)
    model=deepcopy(anchor.model);measured=None
    if measured_fit:
        fitted=cr.fit_penalized_glm([[r['features']['utilization']]for r in trainjan],y,feature_order=['utilization'],model_id='behavioral-utilization',**kw)
        model=deepcopy(fitted.model)
        measured=dict(coefficients=model['parameters']['coefficients'],training_utilization=[r['features']['utilization']for r in trainjan],training_labels=y)
    model['behavioral_contract']=dict(risk_unit='account',event_policy='first_default',product_ids=['card'],allowed_states=['ACTIVE','DEFAULT'],maximum_source_age_days=0)
    intervals=[]
    for i,row in enumerate(trainjan):
        aid=row['account_id'];event=next((e['occurred_at']for e in tr['events']if e['account_id']==aid),None);early=event is not None and event<='2020-02-29'
        intervals.append(dict(entity_id=aid,start=0,stop=29,event=int(early),period=0,interval_end_type='scheduled'))
        if not early:intervals.append(dict(entity_id=aid,start=29,stop=60,event=int(event is not None and event<='2020-03-31'),period=1,interval_end_type='scheduled'))
    hazard=cr.fit_discrete_hazard(intervals,np.empty((len(intervals),0)),feature_order=[],period_grid=[0,29,60],link='logit',partial_period_policy='exclude_censored',target=target,training=d['training'],model_id='behavioral-discrete',model_version='1')
    curve=cr.predict_event_curves(hazard,[dict(path_id='account',start=0,stop=60,covariates=[],covariate_time=0,stratum='all')],[29,60],feature_order=[],target=target).value['curves'][0]
    pool=cr.fit_retail_risk_pools(d['calibration'],d['pooling_spec'],target=d['pool_target'],training=d['pool_training'],model_id='behavioral-pools',model_version='1')
    calobs=[{k:v for k,v in r.items()if k in ['observation_id','index_date','feature_observed_at','feature_known_at','score',*d['strata']]}for r in d['calibration']]
    calassign=cr.assign_retail_risk_pool(calobs,pool,as_of=d['pool_training']['known_at'],information_cutoff=d['pool_training']['known_at'],expected_target=d['pool_target'],uncovered_policy='error')
    outcomes=[dict(observation_id=r['observation_id'],default_unit_id=r['default_unit_id'],index_date=r['index_date'],label_window_end=r['label_window_end'],label_known_at=r['label_known_at'],outcome_status='MATURE',reference_exposure=100,reference_known_at=r['index_date'],defaulted=r['defaulted'],default_date=r['default_date'])for r in d['calibration']]
    parameters=cr.estimate_pool_risk_parameters(calassign,outcomes,[],d['estimation_profile'],evaluation_cutoff=d['pool_training']['known_at'],calibration_id='behavioral-prior-calibration',calibration_version='1').value
    grades=['A','B','C'];counts=[sum(r['grade']==g for r in d['calibration'])for g in grades];defaults=[sum(r['grade']==g and r['defaulted']for r in d['calibration'])for g in grades]
    grade=cr.calibrate_grade_pd(counts,defaults,grades=grades).value
    panel=_panel(hold);active=[r for r in panel if r['as_of']in ['2024-01-31','2024-02-29']and r['at_risk']]
    predictions=[];assignments=[];history=[]
    for snap in active:
        predictions.append(cr.predict_behavioral_risk(snap,model,target).value)
        o=dict(observation_id=snap['account_id']+'-'+snap['as_of'],index_date=snap['as_of'],feature_observed_at=snap['as_of'],feature_known_at=snap['as_of'],score=100*snap['features']['utilization'],**d['strata'])
        assignments.append(cr.assign_retail_risk_pool([o],pool,as_of=snap['as_of'],information_cutoff=snap['known_at'],expected_target=d['pool_target'],uncovered_policy='error').value)
    for snap in [r for r in active if r['as_of']=='2024-01-31']:
        aid=snap['account_id'];obs=[r for r in hold['obligations']if r['account_id']==aid];pays=[r for r in hold['payments']if r['account_id']==aid]
        dpd=cr.compute_days_past_due(obs,pays,as_of=snap['as_of'],known_at=snap['known_at'],currency='EUR').value
        state=cr.assign_delinquency_state(dpd['dpd'],dpd['past_due_amount'],snap['features']['balance'],{'utp':False},d['state_profile']).value
        features=cr.build_payment_history_features(obs,pays,dict(status='COMPLETE',known_at=snap['as_of'],observed_from='2024-01-01',complete_through=snap['as_of']),account_id=aid,as_of=snap['as_of'],information_cutoff=snap['known_at'],lookback_days=30,allocation_policy='fifo',currency='EUR',grace_days=0).value
        history.append(dict(account_id=aid,dpd=dpd,state=state,features=features))
    poolpd={r['pool_id']:r['pd']['estimate']for r in parameters['pool_risk_table']};gradepd={r['grade']:r['pd']for r in grade['grades']}
    risk_outputs=[dict(account_id=s['account_id'],as_of=s['as_of'],model_pd=round(p['pd'],8),prior_pool_pd=poolpd[a['assignments'][0]['pool_id']],utilization_grade=('A'if 100*s['features']['utilization']<25 else'B'if 100*s['features']['utilization']<60 else'C'),prior_grade_pd=gradepd['A'if 100*s['features']['utilization']<25 else'B'if 100*s['features']['utilization']<60 else'C'],calibration_known_at=d['pool_training']['known_at'])for s,p,a in zip(active,predictions,assignments)]
    heldlabels=_labels(active,hold,'2024-05-01');earlylabels=_labels(active,hold,'2024-02-20')
    observations=[];forecasts=[];survrows=[]
    for snap,label,pred in zip(active,heldlabels,predictions):
        aid=snap['account_id'];oid=aid+'-'+snap['as_of'];observations.append(dict(observation_id=oid,facility_id=aid,as_of=snap['as_of'],performance_start=snap['as_of'],outcome_end=label['horizon_end'],eligible=label['eligible'],prior_default=False,target=label['label'],outcome_known_at='2024-05-01',default_definition=d['pool_target']['default_definition'],cohort=snap['as_of']))
        forecasts.append(dict(observation_id=oid,pd=round(pred['pd'],8),forecast_at=snap['as_of'],known_at=snap['as_of'],default_definition=d['pool_target']['default_definition'],horizon_days=60,unit='facility',probability_measure='PHYSICAL',model_id=model['model_id'],model_version='1'))
        event=next((e for e in hold['events']if e['account_id']==aid and snap['as_of']<e['occurred_at']<=label['horizon_end']),None)
        survrows.append(dict(prediction_id=oid,entity_id=aid,cluster_id=snap['customer_id'],segment='card',as_of=snap['as_of'],forecast_known_at=snap['as_of'],horizon_end=label['horizon_end'],model_id='behavioral-discrete',model_version='1',parameter_version='1',outcome_known_at='2024-05-01',observed_time=(date.fromisoformat(event['occurred_at'])-date.fromisoformat(snap['as_of'])).days if event else 61,event=event is not None))
    count=cr.default_count_backtest(observations,forecasts,['cohort'],unit='facility',horizon_days=60,default_definition=d['pool_target']['default_definition'],redefault_policy='exclude_previous_default',evaluation_cutoff='2024-05-01').value
    cids=sorted(groups[2]);draws=[cids,list(reversed(cids)),[cids[0]]*10,[cids[-1]]*10]
    design=dict(model_id='behavioral-discrete',model_version='1',parameter_version='1',data_fingerprint='actual-two-overlapping-holdout-months',population_id='active-card-account-months',split_id='holdout-customers-2024',prediction_origin='HOLDOUT',evaluation_cutoff='2024-05-01',days_per_time_unit=1,target=target)
    cumulative=[round(1-s,12)for s in curve['survival']]
    survival=cr.survival_backtest([cumulative]*len(survrows),survrows,horizons=[29,60],censoring=dict(method='complete_followup',reference='Separate actual complete target windows; default remains absorbing'),design=design,uncertainty='cluster_bootstrap',cluster_resamples=draws)
    rolls=cr.estimate_retail_roll_rates(trainpanel,states=['ACTIVE','DEFAULT'],from_date='2020-02-29',to_date='2020-03-31',group_by=[],weighting='count',missing_successor='error').value
    matrix=rolls['groups'][0]['transition_matrix'];projection=cr.project_retail_roll_rates([1,0],[matrix,matrix],states=['ACTIVE','DEFAULT'],default_states=['DEFAULT'],closure_states=[],exposure_by_period=[[100,100],[80,80]]).value
    vintage=cr.build_vintage_performance(hold['accounts'],panel,hold['events'],months_on_book=[0,1,2,5],evaluation_cutoff='2024-05-01',coverage=hold['coverage'],default_event='DEFAULT',delinquency_states=[],exit_events=['CLOSED']).value
    w=d['open_workout'];event=next(e for e in hold['events']if e['account_id']=='hold0');default_snapshot=next(s for s in panel if s['account_id']=='hold0'and s['as_of']=='2024-02-29')
    if w['episode']['episode_id']!='hold0-default'or w['episode']['default_date']!=event['occurred_at']or default_snapshot['at_risk']or not default_snapshot['ever_default']:raise ValueError('Recovery must bind the same observed defaulted account and date')
    recovery=cr.estimate_open_workout_lgd([w['episode']],w['cashflows'],d['residual_model'],[[]],[w['valuation']],w['future_costs'],feature_order=[],expected_target=d['recovery_target'],stage_spec=d['stage_spec'],as_of='2024-03-29',currency='EUR',extrapolation='parametric').value
    manifests=dict(evaluation_started_at='2024-05-01',model_manifest={'discrete_training':cr.content_fingerprint(dict(rows=intervals,target=target,training=d['training'])).value},data_manifest={'holdout':cr.content_fingerprint(hold).value},split_manifest={'customer_groups':cr.content_fingerprint([sorted(g)for g in groups]).value})
    policy=dict(policy_id='synthetic-behavioral-review',version='1',reference='Illustrative predeclared criterion; not bank acceptance',declared_at='2023-12-01',required_result_ids=['survival'],criteria=[dict(criterion_id='brier',result_id='survival',path=['metrics','integrated_brier'],operator='LE',threshold=.3,rationale='Illustrative test threshold only')])
    report=cr.validation_report([dict(result_id='survival',result=survival,reference='Actual grouped holdout Brier calculation',data_origin='SYNTHETIC',manifest_hashes={k:cr.content_fingerprint(manifests[k]).value for k in ['model_manifest','data_manifest','split_manifest']})],acceptance_policy=policy,manifests=manifests).value
    out=dict(data_origin=d['data_origin'],split_groups=[sorted(g)for g in groups],training_labels=labels,anchor_pd=[round(v,8)for v in cr.predict_pd(anchor,[[]],feature_order=[],expected_target=target).value],measured_fit=measured,discrete_curve=curve,pool_parameters=[{k:r[k]for k in ['pool_id','mature_default_units','defaults','parameter_status','missing_default_workouts']}|{'pd':{k:r['pd'][k]for k in ['estimate','numerator','denominator','count']}}for r in parameters['pool_risk_table']],grade_calibration=grade,risk_outputs=risk_outputs,predictions=[dict(p,pd=round(p['pd'],8),unadjusted_pd=round(p['unadjusted_pd'],8))for p in predictions],prediction_reporting_decimal_places=8,pool_assignments=assignments,payment_history=history,heldout_labels=heldlabels,immature_labels=earlylabels,default_count=count,survival_validation=survival.value,prior_roll_rates=rolls,roll_projection=projection,vintage=vintage,default_recovery=recovery,validation_report={k:report[k]for k in ['decision','findings','metrics','acceptance_scope','manifests']},raw_artifacts_available=True)
    if verify_reference:
        assert np.allclose(curve['survival'],[.8,.6],atol=1e-8)
        assert count['overall']['observed_count']==18 and count['overall']['observed_defaults']==8
        assert np.allclose([r['brier']for r in survival.value['metrics']['by_horizon']],[3.12/18,4.48/18])
        assert matrix==[[.75,.25],[0,1]]and projection['periods'][1]['state_probabilities']==[.5625,.4375]
        assert not any(r['calendar_mature']for r in earlylabels)
        assert all(not r['ever_default']for r in active)
        assert report['decision']['status']=='PASS'
    if return_artifacts:out['raw_artifacts']=dict(behavioral_predictions=predictions,anchor_prediction=cr.predict_pd(anchor,[[]],feature_order=[],expected_target=target).value,validation_report=report,survival_validation=survival.value,pool_parameters=parameters,model=model,hazard_model=hazard.model,pool_model=pool.model,report_results=[dict(result_id='survival',result=survival.value,reference='Actual grouped holdout Brier calculation',data_origin='SYNTHETIC',manifest_hashes={k:cr.content_fingerprint(manifests[k]).value for k in ['model_manifest','data_manifest','split_manifest']})],report_policy=policy)
    return out


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
