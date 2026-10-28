"""Dated corporate lifecycle, observed outcomes, residual workouts and local replay."""
from importlib.resources import files
from copy import deepcopy
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_corporate_lifecycle.json').read_text())
    target=data['target'];tr=data['development']
    model=cr.fit_binary_glm(tr['X'],tr['y'],weights=tr['weights'],feature_order=tr['feature_order'],link='logit',fit_intercept=True,target=target,training=tr['training'],model_id='corporate-lifecycle',model_version='1')
    prediction=cr.predict_pd(model,[[2]],feature_order=tr['feature_order'],expected_target=target)
    policy=cr.compile_credit_decision_policy(data['decision_policy'])
    timing=dict(as_of='2024-01-01',known_at='2024-01-01')
    decision=cr.apply_credit_decision_policy(policy,risk=dict(timing,pd=prediction.value[0],target=target,freshness='CURRENT'),affordability=dict(timing,status='AFFORDABLE'if data['credit_gate_inputs']['available_cashflow']>=data['credit_gate_inputs']['debt_service_due']else'NOT_AFFORDABLE'),eligibility=dict(timing,status='ELIGIBLE'if data['credit_gate_inputs']['legal_eligibility']else'INELIGIBLE'),operations=dict(timing,status=data['credit_gate_inputs']['operations_status']),**timing).value
    governance=_governance(data,model)
    monitor=cr.build_monitoring_cohorts(**data['monitoring'])
    summary=cr.monitor_rating_cohort(monitor,group_by=['cohort'],reference_window=dict(start='2024-01-01',end='2024-12-31'),minimum_count=10,uncertainty='iid_homogeneous_binomial').to_dict()['value']
    feature_drift=cr.detect_credit_feature_drift(**data['feature_drift']).to_dict()['value']
    drift_args=deepcopy(data['pd_drift']);drift_sources=[];drift_args['residuals']=[]
    for observed in data['drift_observed_cohorts']:
        cohort=cr.build_monitoring_cohorts(**observed)
        rows=[x for x in cohort.value['rows']if x['eligible']]
        residual=sum(x['target']-x['pd']for x in rows)/len(rows)
        drift_args['residuals'].append(dict(time=observed['evaluation_cutoff'],value=residual,mature=True))
        drift_sources.append(dict(evaluation_cutoff=observed['evaluation_cutoff'],eligible_count=len(rows),defaults=sum(x['target']for x in rows),mean_pd=sum(x['pd']for x in rows)/len(rows),residual=residual))
    pd_drift=cr.detect_pd_calibration_drift(**drift_args).to_dict()['value']
    windows=cr.build_observation_windows(**data['observation_windows']).value
    sm=data['survival_metadata']
    multi=cr.estimate_multistate_transitions(data['multistate_history'],states=['performing','default','cured'],allowed_transitions=[['performing','default'],['default','cured'],['cured','default']],initial_state='performing',**sm,model_id='corporate-multistate',model_version='1')
    multi_curves=cr.predict_event_curves(multi,[],[0,1,2,3],feature_order=[],target=sm['target'],event_states=['default']).to_dict()['value']
    competing=cr.fit_cause_specific_hazards(data['cause_panel'],[[]for _ in data['cause_panel']],data['cause_events'],feature_order=[],causes=['default','exit'],ties='breslow',**sm,model_id='corporate-competing',model_version='1')
    competing_curves=cr.predict_event_curves(competing,[dict(path_id='corporate-exit',start=0,stop=2,covariates=[],stratum='all',covariate_time=0)],[0,1,2],feature_order=[],target=sm['target']).to_dict()['value']
    treatment=cr.evaluate_treatment_candidates(**data['treatment']).to_dict()['value']
    exposure=cr.exposure_schedule(**data['exposure']).value
    episodes=cr.build_recovery_episodes(**data['recovery_episodes']).value
    recovery=cr.discount_recoveries(**data['recovery_value'])
    closed=cr.workout_lgd(80,recovery,episode_status='CLOSED',default_date='2024-05-01',evaluation_cutoff='2024-09-03',currency='EUR').value
    incomplete=cr.workout_lgd(80,recovery,episode_status='OPEN',default_date='2024-05-01',evaluation_cutoff='2024-09-03',currency='EUR').value
    workout_model=cr.fit_recovery_hurdle(**data['workout_fit'])
    open_workout=cr.estimate_open_workout_lgd(fitted_recovery_model=workout_model,**data['open_workout']).to_dict()['value']
    frozen_workout_replay={}
    for language in ['python','r']:
        frozen=cr.load_fitted_model(files('creditratingsystem.datasets').joinpath('wholesale_corporate_workout_'+language+'.json'))
        frozen_workout_replay[language]=dict(model_hash=frozen.diagnostics['sha256'],forecast=cr.estimate_open_workout_lgd(fitted_recovery_model=frozen,**data['open_workout']).to_dict()['value'])
    loss=cr.event_conditioned_loss(**data['event_loss']).value
    graph=cr.build_entity_graph([dict(entity_id=x['entity_id'])for x in data['positions']],[],data['graph_policy'],as_of='2024-01-01',known_at='2024-01-01')
    portfolio=cr.prepare_credit_portfolio(data['positions'],graph,data['default_link_policy'],data['portfolio_contract']).value
    concentration=cr.credit_concentration_measures(data['positions'],weight_field='ead',group_fields=['sector'],weight_unit='EUR').value
    panel=[dict(row,prior_default=row['label_status']=='prior_default',default_definition='first_default',outcome_known_at='2025-01-01',grade='B')for row in windows]
    forecasts=[dict(observation_id=row['observation_id'],pd=.1,forecast_at='2024-01-01',known_at='2024-01-01',default_definition='first_default',horizon_days=365,unit='obligor',probability_measure='PHYSICAL',model_id='corporate-lifecycle',model_version='1')for row in panel]
    counts=cr.default_count_backtest(panel,forecasts,['grade'],unit='obligor',horizon_days=365,default_definition='first_default',redefault_policy='exclude_previous_default',evaluation_cutoff='2025-01-02').value
    lgdtest=cr.lgd_backtest(data['loss_records'],design=data['loss_backtest_design'],workout_policy=dict(method='closed_only',reference='Actual closed corporate settlements; open episodes excluded'),uncertainty='none').value
    eadtest=cr.ead_backtest(data['loss_records'],design=data['loss_backtest_design'],weighting_policy=dict(method='equal',reference='Each defaulted facility counted once'),uncertainty='none').value
    if verify_reference:
        assert decision['action']=='APPROVE'
        assert summary[0]['metrics']['n']==20 and math.isclose(summary[0]['metrics']['brier'],.09,abs_tol=1e-12)
        assert feature_drift[0]['statistic']==3
        assert all(math.isclose(a,b,abs_tol=1e-12)for a,b in zip(pd_drift['statistic_path'],[.05,.125,.2125]))
        assert sum(w['target']for w in windows)==2
        curve=multi_curves['curves'][0]
        assert curve['first_hit_probability']==[0,.25,.25,.25]
        assert [x[1]for x in curve['state_occupancy']]==[0,.25,0,.25]
        expected=1-math.exp(-3/8)
        assert math.isclose(competing_curves['curves'][0]['cumulative_incidence']['default'][1],expected/3,abs_tol=1e-12)
        assert math.isclose(competing_curves['curves'][0]['cumulative_incidence']['exit'][1],expected*2/3,abs_tol=1e-12)
        assert exposure['schedule'][0]['conditional_ead']==80
        assert recovery.value['discounted_net_recoveries']==56 and closed['economic_lgd']==.3 and incomplete['economic_lgd']is None
        expected_open=1-(10+80*(1+math.exp(.125))/3)/100
        assert math.isclose(open_workout['episodes'][0]['expected_final_lgd'],expected_open,abs_tol=1e-7)
        assert math.isclose(loss['expected_loss'],3.4,abs_tol=1e-12)
        assert math.isclose(concentration['dimensions'][0]['hhi'],41/81,abs_tol=1e-12)
        assert counts['overall']['observed_count']==20 and counts['overall']['observed_defaults']==2
        assert math.isclose(lgdtest['metrics']['overall']['ead_weighted']['bias'],1/30,abs_tol=1e-12)
        assert math.isclose(eadtest['metrics']['ead']['overall']['selected_weighting']['bias'],-5,abs_tol=1e-12)
    return dict(product=data['product'],data_origin='SYNTHETIC',target=target,credit_gate_inputs=data['credit_gate_inputs'],drift_sources=drift_sources,origination_decision=decision['action'],governance=governance,cohort_summary=summary,feature_drift=feature_drift,pd_drift=pd_drift,observation_windows=windows,multistate=multi_curves,competing_exit=competing_curves,treatment=treatment,exposure=exposure,recovery_episodes=episodes,discounted_recoveries=recovery.value,closed_workout=closed,incomplete_workout=incomplete,estimated_open_workout=open_workout,frozen_workout_replay=frozen_workout_replay,event_loss=loss,portfolio_position_count=len(portfolio['positions']),concentration=concentration,default_backtest=counts,lgd_backtest=lgdtest,ead_backtest=eadtest)


def _governance(data,model,return_audit_artifacts=False):
    # Bound, dated inference uses a real fitted model and immutable local requests.
    target=data['target'];sid=data['subject_id']
    rule=dict(rule_id='corporate',profile_id='corporate',profile_version='1',priority=1,effective_from='2023-01-01',effective_to=None,known_at='2023-01-01',conditions=[dict(field='financial_statements',operator='eq',value=True)])
    contract=dict(contract_id='corporate',version='1',required_features=['financial_statements'],allowed_values={'financial_statements':[True]},support_bounds={},maximum_age_days={'financial_statements':90})
    ref=lambda source,identity,*path:{'$ref':dict(source=source,id=identity,path=list(path))}
    recipe=dict(method='assessment_pipeline',version='2.0',parameters=dict(routing_rules=[rule],applicability_contract=contract,steps=[dict(step_id='pd',method_id='classification.predict_pd',arguments=dict(fitted_model=ref('model','primary'),X=ref('input','features','value'),feature_order=data['development']['feature_order'],expected_target=target),argument_types={},output_contract=dict(shape='vector',unit='probability',target=target))],components=[dict(component_id='pd',quantity='PD',value=ref('step','pd','value',0),subject_id=sid,subject_type='obligor',variant='standalone',definition=target)],effects=[],transitions=[dict(from_state='DRAFT',to_state='REVIEWED',event_type='review',required_gates=['checked']),dict(from_state='REVIEWED',to_state='FINAL',event_type='finalize',required_gates=['approved']),dict(from_state='FINAL',to_state='SUPERSEDED',event_type='exit',required_gates=['settled'])],event_dependencies=[dict(assessment_id='corporate-assessment',dependency_type='subject',dependency_id=sid)],decision=None))
    registry=cr.register_rating_model_version(None,dict(model=model.model,dependencies={},configuration=dict(assessment_recipe=recipe)),dict(model_id='corporate-lifecycle',model_version='1',registered_at='2023-12-01')).value
    assignment=dict(assignment_id='corporate-champion-1',profile_id='corporate',purpose='PD',model_id='corporate-lifecycle',model_version='1',model_hash=registry['entries'][0]['model_hash'],effective_from='2024-01-01',effective_to=None,known_at='2023-12-01')
    selection=cr.select_effective_model(registry,[assignment],'corporate','PD','2024-01-01','2024-01-01').value
    snapshot=dict(subject_id=sid,subject_type='obligor',as_of='2024-01-01',known_at='2024-01-01',attributes={'financial_statements':dict(value=True,observed_at='2023-12-31',known_at='2024-01-01')})
    subject=dict(subject_id=sid,version='1',attributes=snapshot['attributes'],effective_from='2024-01-01',effective_to=None,known_at='2024-01-01')
    subjects=cr.build_rating_subjects({'obligor':[subject]},[],'2024-01-01','2024-01-01').value
    request=dict(row_id='origination',known_at='2024-01-02',model_id='corporate-lifecycle',model_version='1',model_hash=assignment['model_hash'],recipe=recipe,input=dict(snapshot=snapshot,datasets={'features':dict(value=[[2]],as_of='2024-01-01',known_at='2024-01-01')},assessment_id='corporate-assessment',assessment_known_at='2024-01-01',calculated_at='2024-01-02',lifecycle_events=[],events=[]))
    original=cr.rate_subject_batch([request],registry,'corporate-origination','2024-01-02').value['rows'][0]
    assert original['status']=='OK',original
    assessment=original['value']['assessment']
    fp=dict(maximum_age_days=90,maximum_information_age_days=90,trigger_types=['financial_deterioration'])
    freshness=cr.check_assessment_freshness(assessment,fp,[],'2024-03-01','2024-03-01').value
    trigger=dict(event_id='review-trigger',subject_id=sid,type='financial_deterioration',occurred_at='2024-04-01',known_at='2024-04-02')
    stale=cr.check_assessment_freshness(assessment,fp,[trigger],'2024-04-02','2024-04-02').value
    history=deepcopy(request);history['row_id']='closed-history';history['known_at']='2024-09-03'
    history['input']['lifecycle_events']=[dict(event_id='review',type='review',occurred_at='2024-01-03',known_at='2024-01-03',reason='Financial statements checked',gates={'checked':True}),dict(event_id='approval',type='finalize',occurred_at='2024-01-04',known_at='2024-01-04',reason='Local approval',gates={'approved':True}),dict(event_id='exit',type='exit',occurred_at='2024-09-03',known_at='2024-09-03',reason='Settled corporate claim',gates={'settled':True})]
    event=dict(event_id='credit-review',revision='1',occurred_at='2024-04-01',known_at='2024-04-02',dependencies=[dict(type='subject',id=sid)],payload={'review_required':True})
    default_event=dict(event_id='credit-default',revision='1',occurred_at='2024-05-01',known_at='2024-05-02',dependencies=[dict(type='subject',id=sid)],payload={'default_observed':True})
    delinquency=dict(event_id='credit-delinquency',revision='1',occurred_at='2024-04-15',known_at='2024-04-16',dependencies=[dict(type='subject',id=sid)],payload=dict(due_date='2024-03-15',scheduled_amount=8,received_amount=0,days_past_due=31))
    proposal=dict(event_id='restructuring-proposal',revision='1',occurred_at='2024-04-20',known_at='2024-04-20',dependencies=[dict(type='subject',id=sid)],payload=dict(treatment_id='extend',proposal_only=True,contract_changed=False))
    history['input']['events']=[event,event,delinquency,proposal,default_event]
    row=cr.rate_subject_batch([history],registry,'corporate-exit','2024-09-03').value['rows'][0]
    assert row['status']=='OK',row
    replay=cr.replay_assessment_history([request,history],registry,[original,row],'2025-01-02').value
    tr=data['development'];later_training=dict(tr['training'],known_at='2025-01-01',split_id='train_2023',data_fingerprint='later-observed-corporate-cohort')
    later_model=cr.fit_binary_glm(tr['X'],tr['y'],weights=[8,2,6,4],feature_order=tr['feature_order'],link='logit',fit_intercept=True,target=target,training=later_training,model_id='corporate-lifecycle',model_version='2')
    later_registry=cr.register_rating_model_version(registry,dict(model=later_model.model,dependencies={},configuration=dict(assessment_recipe=recipe)),dict(model_id='corporate-lifecycle',model_version='2',registered_at='2025-02-01')).value
    later_replay=cr.replay_assessment_history([request,history],later_registry,[original,row],'2025-03-01').value
    assert later_replay['status']=='REPRODUCED'
    tampered=deepcopy(row);tampered['event_state_hash']='changed'
    rejected=cr.replay_assessment_history([history],registry,[tampered],'2025-01-02').value
    assert replay['status']=='REPRODUCED'and rejected['status']=='NOT_REPRODUCIBLE'
    assert row['value']['assessment']['state']=='SUPERSEDED'
    assert [x['status']for x in row['value']['event_trace']]==['APPLIED','DUPLICATE','APPLIED','APPLIED','APPLIED']
    assert freshness['status']=='CURRENT'and stale['status']=='STALE'
    report=dict(subject_count=len(subjects['subjects']),selection=selection['status'],initial_state=assessment['state'],final_state=row['value']['assessment']['state'],current=freshness['status'],review_required=stale['status'],event_statuses=[x['status']for x in row['value']['event_trace']],replay=replay['status'],replay_after_new_champion=later_replay['status'],tampered_replay=rejected['status'],origination_pd=assessment['components'][0]['value'],historical_pd=row['value']['assessment']['components'][0]['value'])
    return dict(report=report,registry=registry,requests=[request,history],results=[original,row])if return_audit_artifacts else report


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
