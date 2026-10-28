"""Four dated revolving account journeys with independent prior fits and observed workout ledgers."""
from copy import deepcopy
from datetime import date,timedelta
import math,json
import numpy as np
import creditratingsystem as cr


def origin(case,product):
    t=case['application_target'];tr=case['application_training'];ca=case['application_calibration']
    fit=cr.fit_binary_glm(tr['X'],tr['y'],weights=tr['weights'],link='logit',feature_order=['recent_arrears'],fit_intercept=True,target=t,training=tr['metadata'],model_id=product+'-application',model_version='1')
    raw=cr.predict_pd(fit,ca['X'],feature_order=['recent_arrears'],expected_target=t)
    calibrator=cr.calibrate_logistic(raw.value,ca['y'],weights=ca['weights'],metadata=ca['metadata'])
    calibrated=cr.apply_calibrator(calibrator.value,raw.value,target=t,base_model_version='1').value
    assert abs(calibrated[0]-case['expected_application_pd'])<1e-8
    model=deepcopy(fit.model);model['application_contract']=dict(product_ids=[product],population_ids=[product+'-new-funded'],performance_origin='first_draw_at',term_stage='requested',term_features={});model['calibration_artifact']=calibrator.value
    features=[dict(r,source_id=product+'-prior-bureau')for r in case['features']]
    snapshots=cr.build_application_snapshot(case['applications'],features,required_features=['recent_arrears']).value
    risks=[cr.predict_application_risk(s,model,t,calibrator=calibrator.value).value for s in snapshots]
    policy=cr.compile_credit_decision_policy(case['application_policy'])
    due=30 if case['kind']=='CARD'else 600 if case['kind']in ('CHARGE','OVERDRAFT')else 1000
    residual=cr.residual_income(case['monthly_income'],case['monthly_expenses'],case['existing_debt_service'],due,period='monthly',currency='EUR').value
    ratio=cr.debt_service_to_income(case['existing_debt_service']+due,case['monthly_income'],period='monthly',currency='EUR').value
    capacities=capacity(case,product)
    decisions=[]
    for risk,cap in zip(risks,capacities):
        dated=dict(as_of='2024-01-01',known_at='2024-01-01')
        decisions.append(cr.apply_credit_decision_policy(policy,dict(risk,freshness='CURRENT',**dated),dict(status=cap['assessment']['assessment'],**dated),dict(status='ELIGIBLE',**dated),dict(status='CLEAR',**dated),**dated).value)
    defaults=[dict(event_id=e['event_id'],facility_id=e['account_id'],default_at=e['occurred_at'],known_at=e['known_at'])for e in case['events']if e['event_type']=='DEFAULT']
    exits=[dict(event_id=e['event_id'],facility_id=e['account_id'],exit_at=e['occurred_at'],known_at=e['known_at'],exit_type='closed')for e in case['events']if e['event_type']=='CLOSED']
    targets=cr.build_application_targets(case['applications'],case['decisions'],case['bookings'],defaults,exits,365,'2025-01-02').value
    for risk in risks:assert abs(risk['pd']-case['expected_application_pd'])<1e-8
    assert all(d['action']=='APPROVE'for d in decisions)
    return dict(model=model,calibrator=calibrator.value,snapshots=snapshots,risks=risks,policy=policy.value,decisions=decisions,targets=targets,residual=residual,dsti=ratio,capacities=capacities)




def design(product,name,target,cutoff='2025-01-02'):
    return dict(model_id=product+'-'+name,model_version='1',parameter_version='1',data_fingerprint=product+'-actual-dated-ledger',population_id=product+'-funded',split_id=product+'-2024-holdout',prediction_origin='HOLDOUT',evaluation_cutoff=cutoff,target=target)


def account_history(c,product,original):
    panel=cr.build_account_month_panel(c['accounts'],c['statements'],c['events'],observation_dates=c['observation_dates'],obligations=c['obligations'],payments=c['payments']).value['accounts']
    labels=cr.build_behavioral_targets(panel,c['events'],horizon_months=2,target_event='DEFAULT',competing_events=['CLOSED'],exit_treatment='competing',evaluation_cutoff='2025-01-02',coverage=c['coverage']).value
    profile=dict(profile_id=product+'-first-default',version='1',reference='First own-facility default, no household contagion; behavioral default before competing closure',known_at='2023-12-01',default_definition='first-default-before-exit',default_group_rule='facility-only',default_unit='facility',first_default=True,exit_treatment='competing',default_event='DEFAULT',chargeoff_event='CHARGEOFF',cure_event='CURED',competing_events=['CLOSED'],risk_set_by_product={product:dict(active_states=['ACTIVE','CURED'],include_zero_balance=True)},split_isolation_keys=['customer_id','household_id'],application_horizon_days=365,behavioral_horizon_months=2)
    sources=[dict(source_id=f['feature_record_id'],prediction_key='application:'+f['application_id'],source_hash=cr.content_fingerprint(f).value,kind='BUREAU',observed_at=f['observed_at'],available_at=f['known_at'])for f in c['features']]
    for snap in panel:
        record=next(s for s in c['statements']if s['account_id']==snap['account_id']and s['observed_at']==snap['as_of'])
        sources.append(dict(source_id=record['statement_id'],prediction_key='account:'+snap['account_id']+'@'+snap['as_of'],source_hash=cr.content_fingerprint(record).value,kind='STATEMENT',observed_at=record['observed_at'],available_at=record['known_at']))
    target=dict(event='default',horizon=2,time_unit='months',probability_measure='PHYSICAL',conditioning='performing')
    args=dict(application_targets=original['targets'],account_panel=panel,behavioral_targets=labels,target_profile=profile,split_memberships=[dict(customer_id=a['customer_id'],household_id=a['customer_id']+'-household',role='test')for a in c['accounts']],source_lineage=sources,design=design(product,'behavioral',target))
    # Public positional arguments preserve the operational builder artifacts.
    cohort=cr.validate_retail_cohort(original['targets'],panel,labels,**{k:v for k,v in args.items()if k not in ['application_targets','account_panel','behavioral_targets']}).value
    return dict(panel=panel,labels=labels,cohort=cohort,cohort_input=args)


def workout_history(c,product,risk):
    args=deepcopy(c['execution_inputs']['collections_history'])
    episodes=cr.build_collections_episodes(**args).value
    targets=cr.build_cure_targets(episodes,horizon_days=90,redefault_horizon_days=90,evaluation_cutoff='2024-08-31').value
    workouts=[]
    for ep,witness in zip(episodes['episodes'],c['execution_inputs']['workout_claims']):
        w=deepcopy(witness)
        if w['episode_id']!=ep['episode_id']:raise ValueError('Workout must bind actual collection episode')
        pv=cr.discount_recoveries(w['cashflows'],w['discount_curve'],w['fx_contract'],default_date=w['default_date'],evaluation_cutoff='2024-08-31',reporting_currency='EUR')
        w['discounted_recovery']=pv.value
        w['loss']=cr.workout_lgd(w['ead'],pv,episode_status='CLOSED',default_date=w['default_date'],evaluation_cutoff='2024-08-31',currency='EUR').value
        workouts.append(w)
    va=deepcopy(c['execution_inputs']['collections_validation']);va['episodes']=episodes
    for forecast,w in zip(va['forecasts'],workouts):
        forecast['predicted_cashflows'][0]['amount']=(1-risk['lgd']['mean_lgd'][0])*w['ead']
        forecast['predicted_cashflows'][0]['occurred_at']=(date.fromisoformat(w['default_date'])+timedelta(days=25)).isoformat()
    validation=cr.collections_backtest(**va).value
    assert workouts[0]['loss']['economic_lgd']==0 and abs(workouts[1]['loss']['economic_lgd']-c['second_default']['expected_lgd'])<1e-12
    assert targets[0]['cure_at']=='2024-05-20'and targets[0]['redefault_at']=='2024-06-20'
    return dict(episodes=episodes,cure_targets=targets,workouts=workouts,validation=validation,episode_input=args,validation_input=va)


def vintage_review(c,product,history):
    mobs=[0,1,2,3,4,5,6,7];ends=c['observation_dates'];table=cr.build_vintage_performance(c['accounts'],history['panel'],c['events'],months_on_book=mobs,evaluation_cutoff='2025-01-02',coverage=c['coverage'],default_event='DEFAULT',delinquency_states=[],exit_events=['CLOSED']).value
    # Distinct historical frequency sample for every product, observed in2019.
    # The first-event hazard uses actual independent prior groups at each month.
    intervals=[];n=10;failures=1 if c['kind']!='OVERDRAFT'else 2
    for month in range(8):
        for i in range(n):intervals.append(dict(entity_id=product+'-vintage-prior-'+str(month)+'-'+str(i),start=month,stop=month+1,event=int(i<failures),period=month,interval_end_type='scheduled'))
    target=dict(event='default',horizon=8,time_unit='months',probability_measure='PHYSICAL',conditioning='not_defaulted_and_not_exited_at_month_start')
    training=dict(data_fingerprint=cr.content_fingerprint(intervals).value,known_at='2019-12-31',split_id=product+'-vintage-prior-2018',data_origin='SYNTHETIC')
    fit=cr.fit_discrete_hazard(intervals,np.empty((len(intervals),0)),feature_order=[],period_grid=list(range(9)),link='logit',partial_period_policy='exclude_censored',target=target,training=training,model_id=product+'-vintage',model_version='1')
    curve=cr.predict_event_curves(fit,[dict(path_id=product,start=0,stop=8,covariates=[],stratum='all',covariate_time=0)],list(range(1,9)),feature_order=[],target=target).value['curves'][0]
    cumulative=[1-s for s in curve['survival']];predictions=[];outcomes=[]
    for a in c['accounts']:
        aid=a['account_id'];first=next((e for e in sorted(c['events'],key=lambda e:e['occurred_at'])if e['account_id']==aid and e['event_type']in ['DEFAULT','CLOSED']),None)
        for mob,end,pc in zip(mobs,ends,cumulative):
            hit=first is not None and first['occurred_at']<=end;default=hit and first['event_type']=='DEFAULT';exited=hit and first['event_type']=='CLOSED'
            predictions.append(dict(account_id=aid,months_on_book=mob,model_id=product+'-vintage',model_version='1',parameter_version='1',forecast_at=a['opened_at'],known_at=a['known_at'],cumulative_pd=pc,interval_pd=pc-(cumulative[mob-1]if mob else 0)))
            row=dict(account_id=aid,months_on_book=mob,period_end=end,known_at='2025-01-02',default_definition='first-default-before-exit',complete=True,exited=exited,cumulative_default=int(default),interval_default=int(default and (mob==0 or first['occurred_at']>ends[mob-1])),complete_through=end)
            if hit:row.update(event_id=first['event_id'],first_event_at=first['occurred_at'])
            outcomes.append(row)
    args=dict(vintage_table=table,account_predictions=predictions,account_outcomes=outcomes,origin_cohorts=c['accounts'],denominator_contract=dict(method='fixed_origination',weighting='count',exit_treatment='first_competing_exit',reference='Own facility first default or terminal exit; three original facilities remain denominator',default_definition='first-default-before-exit',horizon_convention='origination_month_end_plus_mob'),design=design(product,'vintage',dict(target,horizon=7,conditioning='performing')),uncertainty='none')
    result=cr.vintage_backtest(**args).value
    assert all(abs(p-(1-(1-failures/n)**(j+1)))<1e-8 for j,p in enumerate(cumulative))
    assert result['metrics']['cells'][3]['cumulative']['mean_outcome']==1/3
    return dict(table=table,curve=curve,validation=result,input=args,model=fit.model)


def roll_review(c,product,history):
    states=['ACTIVE','DEFAULT','CURED','CLOSED'];rows=[]
    for r in history['panel']:
        aid=r['account_id'];second=r['as_of']>='2024-05-20'and aid.endswith('-main');state=r['state'];ep=aid+'-default-'+('2'if second else'1')if state=='DEFAULT'else None
        rows.append(dict(account_id=aid,as_of=r['as_of'],known_at=r['known_at'],risk_episode_id=aid+('-after-cure'if second else'-initial'),default_episode_id=ep,state=state))
    # Prior product-specific observed endpoint pairs, not heldout transitions.
    prior=[]
    for j,origin in enumerate(states):
        destinations=[origin,origin,origin,states[min(j+1,3)]]
        for k,dest in enumerate(destinations):
            aid=product+'-prior-roll-'+str(j)+'-'+str(k)
            prior.extend([dict(account_id=aid,as_of='2022-01-31',state=origin,balance=100),dict(account_id=aid,as_of='2022-02-28',state=dest,balance=100)])
    estimated=cr.estimate_retail_roll_rates(prior,states=states,from_date='2022-01-31',to_date='2022-02-28',group_by=[],weighting='count',missing_successor='error').value;matrix=estimated['groups'][0]['transition_matrix']
    first_hit=deepcopy(matrix);first_hit[1]=[0,1,0,0]
    projection=cr.project_retail_roll_rates([1,0,0,0],[first_hit,first_hit],states=states,default_states=['DEFAULT'],closure_states=['CLOSED'],exposure_by_period=[[600*c['scale']]*3+[0]]*2).value
    forecasts=[]
    for r in history['panel']:
        if r['as_of'] not in ['2024-01-31','2024-04-30']:continue
        row=next(x for x in rows if x['account_id']==r['account_id']and x['as_of']==r['as_of'])
        forecasts.append(dict(prediction_id=r['account_id']+'@'+r['as_of'],account_id=r['account_id'],entity_id=r['customer_id'],cluster_id=r['customer_id'],segment=product,as_of=r['as_of'],forecast_known_at=r['as_of'],horizon_end='2024-03-31'if r['as_of']=='2024-01-31'else'2024-06-30',model_id=product+'-roll',model_version='1',parameter_version='1',risk_episode_id=row['risk_episode_id'],default_episode_id=row['default_episode_id'],origin_state=row['state'],opening_balance=r['features']['balance'],transition_matrix=matrix))
    args=dict(forecasts=forecasts,account_panel=rows,states=states,state_contract=dict(scale_id=product+'-states',scale_version='1',reference='New risk and default episode after operational cure; closure absorbing',known_at='2023-12-01',default_states=['DEFAULT'],chargeoff_states=[],cure_states=['CURED'],exit_states=['CLOSED'],absorbing_states=['CLOSED']),design=design(product,'roll',dict(event='default',horizon=2,time_unit='months',probability_measure='PHYSICAL',conditioning='origin_state')),weighting='count',uncertainty='none')
    result=cr.roll_rate_backtest(**args).value
    assert not result['diagnostics']['episode_violations']
    assert any(r['redefault_transitions']==1 for r in result['diagnostics']['endpoint_traces'])
    return dict(prior=prior,estimate=estimated,projection=projection,validation=result,input=args)


def pool_review(c,product,history,workout):
    target=dict(event='default',horizon=60,time_unit='days',probability_measure='PHYSICAL',conditioning='not_defaulted_at_index',default_unit='facility',default_definition='first-default-before-exit',currency='EUR');strata=dict(product=product,borrower_type=c['customer_segment'],facility_type='revolving',delinquency_state='current')
    prior=[]
    for i in range(10):
        hit=i<2
        prior.append(dict(observation_id=product+'-pool-prior-'+str(i),default_unit_id=product+'-pool-prior-'+str(i),role='TRAIN',index_date='2022-01-01',feature_observed_at='2022-01-01',feature_known_at='2022-01-01',label_window_end='2022-03-02',label_known_at='2022-06-01',defaulted=hit,default_date='2022-02-01'if hit else None,lgd_known_at='2022-06-01'if hit else None,ead_known_at='2022-02-01'if hit else None,lgd_status='CLOSED'if hit else'NOT_DEFAULTED',lgd=.4 if hit else None,ead=600*c['scale']if hit else None,score=80 if i<5 else 10,**strata))
    spec=dict(spec_id=product+'-pool',version='1',known_at='2021-12-01',stratum_fields=list(strata),score_field='score',segments=[dict(segment_id=product,values=strata,cutpoints=[],score_min=0,score_max=100,monotonicity=dict(pd='none',lgd='none',ead='none'))],minimum_cases=2,minimum_defaults=0,minimum_lgd_count=0,minimum_ead_count=0,maximum_bin_mean_spread=dict(pd=None,lgd=None,ead=None))
    training=dict(role='TRAIN',known_at='2022-06-01',split_id=product+'-pool-prior-2022',data_fingerprint=cr.content_fingerprint(prior).value,data_origin='SYNTHETIC')
    model=cr.fit_retail_risk_pools(prior,spec,target=target,training=training,model_id=product+'-pool',model_version='1')
    prior_means=model.model['parameters']['segments'][0]['pools'][0]['statistics']['means']
    active=[r for r in history['panel']if r['as_of']=='2024-02-29'and r['at_risk']];observations=[dict(observation_id=r['account_id'],index_date=r['as_of'],feature_observed_at=r['as_of'],feature_known_at=r['as_of'],score=10,**strata)for r in active]
    ai=dict(observations=observations,as_of='2024-02-29',information_cutoff='2024-02-29',expected_target=target,uncovered_policy='error');assignment=cr.assign_retail_risk_pool(fitted_model=model,**ai)
    outcomes=[];predictions=[];links=[]
    for r in active:
        aid=r['account_id'];hit=aid.endswith('-main');balance=r['features']['balance']
        outcomes.append(dict(observation_id=aid,default_unit_id=aid,index_date='2024-02-29',label_window_end='2024-04-29',label_known_at='2024-04-30',outcome_status='MATURE',reference_exposure=balance,reference_known_at='2024-02-29',defaulted=hit,default_date='2024-04-15'if hit else None,complete_through='2024-04-29'))
        predictions.append(dict(prediction_id=aid,observation_id=aid,entity_id=aid,cluster_id=r['customer_id'],segment=product,as_of='2024-02-29',forecast_known_at='2024-02-29',horizon_end='2024-04-29',model_id=product+'-pool',model_version='1',parameter_version='1',excluded_from_training=True,excluded_from_calibration=True,reference_exposure=balance,pd=prior_means['pd'],lgd=prior_means['lgd'],ead=prior_means['ead']))
        links.append(dict(observation_id=aid,facility_id=aid,obligor_id=r['customer_id'],household_id=r['customer_id']+'-household',risk_group_id=r['customer_id'],diagnostic_group='zero-balance'if balance==0 else'drawn',known_at='2024-01-02'))
    first=workout['workouts'][0];aid=next(r['account_id']for r in active if r['account_id'].endswith('-main'))
    observed=dict(episode_id=first['episode_id'],observation_id=aid,default_date=first['default_date'],default_known_at=first['default_date'],ead_at_default=first['ead'],episode_status='CLOSED',valuation_known_at='2024-08-31',reference_date='2024-02-29',reference_known_at='2024-02-29',reference_drawn=600*c['scale'],reference_limit=600*c['scale'],recovery_cashflows=first['cashflows'],discount_curve=first['discount_curve'],fx_contract=first['fx_contract'])
    profile=dict(profile_id=product+'-pool-estimation',version='1',known_at='2023-12-01',pd_weighting='reference_exposure',lgd_weighting='ead',ead_weighting='episode',ccf_weighting='undrawn',confidence=.95,independence_assumption='independent_default_units',minimum_pool_cases=5)
    d=design(product,'pool',{k:target[k]for k in ['event','horizon','time_unit','probability_measure','conditioning']});d['currency']='EUR'
    args=dict(fitted_model=model.model,pool_assignments=assignment.value,assignment_inputs=ai,predictions=predictions,cohort_outcomes=outcomes,workouts=[observed],links=links,estimation_profile=profile,design=d)
    validation=cr.retail_pool_diagnostics(**args).value
    estimated=cr.estimate_pool_risk_parameters(assignment,outcomes,[observed],profile,evaluation_cutoff='2025-01-02',calibration_id=product+'-holdout-descriptive',calibration_version='1').value
    assert validation['metrics']['pools'][0]['pd_count_weighted']['observed_rate']==.5
    assert validation['metrics']['pools'][0]['pd_exposure_weighted']['observed_rate']==1
    return dict(model=model.model,prior=prior,spec=spec,training=training,assignment=assignment.value,validation=validation,input=args,estimated=estimated)


def risk_models(c,product,history):
    train=dict(data_fingerprint=product+'-prior-2021-eight-accounts',split_id=product+'-train-2021',known_at='2022-06-01',data_origin='SYNTHETIC')
    target=dict(event='default',horizon=2,time_unit='months',probability_measure='PHYSICAL',conditioning='performing')
    y=[0,0,0,0,0,0,1,1]
    fitted=cr.fit_penalized_glm(np.empty((8,0)),y,feature_order=[],penalty='ridge',lambda_path=[.1],selected_lambda=.1,scale=False,tolerance=1e-12,target=target,training=train,model_id=product+'-behavioral',model_version='1')
    model=deepcopy(fitted.model);model['behavioral_contract']=dict(risk_unit='account',event_policy='first_default',product_ids=[product],allowed_states=['ACTIVE','CURED'],maximum_source_age_days=0)
    active=[r for r in history['panel']if r['as_of']in ['2024-01-31','2024-02-29']and r['at_risk']]
    predictions=[cr.predict_behavioral_risk(r,model,target).value for r in active]
    et=dict(event='ead',horizon=60,time_unit='days',probability_measure='PHYSICAL',conditioning='given_default',currency='EUR');rows=[]
    for i in range(2):
        drawn=100*c['scale'];limit=200*c['scale'];value=drawn+c['ccf']*(limit-drawn)if c['ccf']is not None else [400,600][i]
        rows.append(dict(episode_id=product+'-prior-ead-'+str(i),reference_date='2021-01-01',reference_known_at='2021-01-01',default_date='2021-03-02',default_known_at='2021-03-02',features_observed_at='2021-01-01',features_known_at='2021-01-01',reference_drawn=drawn,reference_limit=limit,ead_at_default=value,currency='EUR'))
    mode='ccf'if c['ccf']is not None else'direct_ead';spec=dict(mean_model='identity',covariance='HC0')if mode=='ccf'else dict(positive_distribution='lognormal',zero_model='none')
    ef=cr.fit_ead_model(rows,np.empty((2,0)),feature_order=[],target_mode=mode,model_spec=spec,target=et,training=train,model_id=product+'-ead',model_version='1',currency='EUR')
    aid=next(a['account_id']for a in c['accounts']if a['account_id'].endswith('-main'));drawn=sum((1 if r['kind']=='DRAW'else-1)*r['amount']for r in c['transactions']if r['account_id']==aid and r['date']<='2024-02-15'and r['known_at']<='2024-02-15');limit=c['regime']['original_limit']
    ead=cr.predict_ead(ef,np.empty((1,0)),feature_order=[],expected_target=et,reference_dates=['2024-02-15'],default_dates=['2024-04-15'],reference_drawn=[drawn]if mode=='ccf'else None,reference_limit=[limit]if mode=='ccf'else None,currency='EUR').value
    observed=cr.realized_ccf(drawn,limit,600*c['scale'],reference_date='2024-02-15',reference_known_at='2024-02-15',default_date='2024-04-15',default_known_at='2024-04-15',evaluation_cutoff='2024-08-31',currency='EUR').value if mode=='ccf'else None
    lt=dict(event='economic_loss',horizon=60,time_unit='days',probability_measure='PHYSICAL',conditioning='given_default');lf=cr.fit_fractional_lgd(np.empty((4,0)),[.2,.4,.6,.4],feature_order=[],link='logit',covariance='HC0',target=lt,training=dict(train,workout_population='closed_defaulted_episodes'),model_id=product+'-lgd',model_version='1')
    lgd=cr.predict_lgd(lf,np.empty((1,0)),feature_order=[],expected_target=lt,default_condition='given_default').value
    # Operational monitoring uses actual first-default dates, known complete-through
    # coverage and terminal exit censoring, never a repayment proxy for survival.
    mt=dict(event='default',probability_measure='PHYSICAL',default_definition='first-default-before-exit',unit='facility',horizon_days=60,default_grade='D');forecasts=[]
    for r,p in zip(active,predictions):forecasts.append(dict(forecast_id=r['account_id']+'@'+r['as_of'],case_id=r['account_id']+'@'+r['as_of'],entity_id=r['account_id'],cluster_id=r['customer_id'],cohort=r['as_of'],segment=product,model_id=product+'-behavioral',model_version='1',forecast_at=r['as_of'],known_at=r['known_at'],pd=round(p['pd'],8),target=mt,origin_grade='A',decision=None,override=False,prediction_origin='OUT_OF_TIME'))
    events=[dict(event_id=e['event_id'],entity_id=e['account_id'],event_at=e['occurred_at'],known_at=e['known_at'],event='default'if e['event_type']=='DEFAULT'else'exit',default_definition=mt['default_definition'])for e in c['events']if e['event_type']in ['DEFAULT','CLOSED']]
    coverage=[dict(entity_id=r['account_id'],complete_through=r['observed_through'],known_at=r['known_at'])for r in c['coverage']]
    cohorts=cr.build_monitoring_cohorts(forecasts,events,coverage,target_contract=mt,evaluation_cutoff='2025-01-02')
    monitor=cr.monitor_rating_cohort(cohorts,group_by=['cohort'],reference_window=dict(start='2024-01-01',end='2024-01-31'),minimum_count=5,uncertainty='none').value
    assert all(abs(p['pd']-.25)<1e-9 for p in predictions)
    return dict(model=model,ead_model=ef.model,lgd_model=lf.model,ead=ead,realized_ccf=observed,lgd=lgd,predictions=predictions,monitor=monitor,cohorts=cohorts.value,reference_drawn=drawn,reference_limit=limit,prior_ead=rows,prior_target=y,training=train,ead_spec=spec,ead_target=et,lgd_target=lt,behavioral_target=target)


def replay_review(c,product,original):
    aid=next(a['account_id']for a in c['accounts']if a['account_id'].endswith('-main'))
    event0=dict(event_id=aid+'-opening-limit',idempotency_key=aid+'-opening-limit',occurred_at='2024-01-02',known_at='2024-01-02',sequence=0,payload=dict(limit=c['regime']['original_limit']),revision_of=None)
    event1=dict(event_id=aid+'-revised-limit',idempotency_key=aid+'-revised-limit',occurred_at=c['regime']['effective_from'],known_at=c['regime']['known_at'],sequence=1,payload=dict(limit=c['regime']['revised_limit']),revision_of=event0['event_id'])
    for e in [event0,event1]:e['payload_hash']=cr.content_fingerprint(e['payload']).value
    events=[event0,deepcopy(event0),event1];snapshots=[]
    for kind,content in [('MODEL',original['model']),('CALIBRATION',original['calibrator']),('POLICY',original['policy']),('SOURCE',c['features'])]:snapshots.append(dict(snapshot_id=product+'-'+kind,kind=kind,content_hash=cr.content_fingerprint(content).value,known_at='2023-12-31',effective_from='2023-12-31',effective_until=None))
    def execute():
        checkpoints=[]
        for cutoff in ['2024-02-16','2024-02-29']:
            state=None;seen=set();applied=[];limit=None
            for e in sorted(events,key=lambda x:(x['known_at'],x['sequence'],x['event_id'])):
                if e['known_at']>cutoff or e['occurred_at']>cutoff:continue
                event=dict(event_id=e['event_id'],revision='1',occurred_at=e['occurred_at'],known_at=e['known_at'],dependencies=[dict(type='account',id=aid)],payload=e['payload'])
                result=cr.apply_assessment_event(state,event,[dict(assessment_id=aid+'-assessment',dependency_type='account',dependency_id=aid)],cutoff).value;state=result['state']
                if e['idempotency_key']in seen:
                    assert result['status']=='DUPLICATE';continue
                seen.add(e['idempotency_key']);applied.append(e['event_id']);limit=e['payload']['limit']
            # Execute frozen fit and calibrator again, not copied historical output.
            risk=cr.predict_application_risk(original['snapshots'][1],original['model'],c['application_target'],calibrator=original['calibrator']).value
            checkpoints.append(dict(checkpoint_id=aid+'@'+cutoff,as_of=cutoff,known_at=cutoff,event_log_hash=cr.content_fingerprint(events).value,snapshot_hashes={r['snapshot_id']:r['content_hash']for r in snapshots},applied_event_ids=applied,state=dict(limit=limit,original_application_pd=risk['pd'],assessment_event_state=state)))
        return checkpoints
    historical=execute();replayed=execute()
    for r in replayed:r['mode']='REPLAY'
    args=dict(event_log=events,original_checkpoints=historical,replayed_checkpoints=replayed,snapshot_manifest=snapshots,replay_contract=dict(contract_id=product+'-replay',version='1',reference='Independent execution of original frozen model/calibrator and dated limit reducer',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=[r['checkpoint_id']for r in historical],absolute_tolerance=1e-10,relative_tolerance=0))
    result=cr.replay_consistency(**args).value
    assert result['decision']['status']=='PASS'
    return dict(validation=result,input=args,known_limits=[r['state']['limit']for r in historical])


def capacity(c,product):
    results=[];calendar=dict(monthly=12,annual=1,weekly=52)
    for a in c['applications']:
        cid=a['customer_id'];sid=cid+'-bureau';D='2024-01-01'
        bureau=cr.build_bureau_features([dict(snapshot_id=sid,release_seq=1,customer_id=cid,provider_id='synthetic-bureau',observed_at='2023-12-30',reported_at='2023-12-31',known_at='2023-12-31',coverage_status='COMPLETE',currency='EUR')],[dict(snapshot_id=sid,snapshot_release_seq=1,tradeline_id=cid+'-old-loan',opened_at='2021-01-01',closed_at=None,status='OPEN',balance=12*c['existing_debt_service'],limit=12*c['existing_debt_service'],max_dpd=0)],[],dict(mapping_id='synthetic-v1',version='1',known_at='2023-01-01',snapshots={},tradelines={},inquiries={}),customer_id=cid,provider_id='synthetic-bureau',as_of=D,information_cutoff=D,lookback_days=30,max_age_days=30,currency='EUR').value
        row=dict(evidence_id=cid+'-income',release_seq=1,known_at='2023-12-31',customer_id=cid,component_id='income',component_type='NET_BUSINESS_INCOME'if c['kind']=='OVERDRAFT'else'SALARY',observed_from='2023-12-01',observed_until='2023-12-30',amount=c['monthly_income'],basis='NET',period='month',currency='EUR',evidence_reference='Separate synthetic net-income document; owner draw after business costs')
        policy=dict(policy_id=product+'-income',version='1',known_at='2023-01-01',source_priority=['DOCUMENT','TRANSACTION','DECLARED'],included_component_types=[row['component_type']],expected_component_ids=['income'],coverage_status='COMPLETE',maximum_age_days=30,allow_stale=False,relative_discrepancy_threshold=.05)
        reconciled=cr.reconcile_income_evidence([dict(row,evidence_id=cid+'-declared')],[row],[],policy,customer_id=cid,as_of=D,information_cutoff=D,currency='EUR',period='month',basis='NET').value
        normalized=cr.normalize_income_period([dict(economic_id=cid+'-income',amount=reconciled['total_selected_income'],frequency='monthly',basis='net',currency='EUR',kind='recurring')],target_period='monthly',target_basis='net',currency='EUR',payments_per_year=calendar,variable_income_factor=1).value
        expenses=cr.estimate_recurring_expenses([dict(economic_id=cid+'-living',category='living',amount=c['monthly_expenses'],frequency='monthly',currency='EUR',kind='essential')],target_period='monthly',currency='EUR',payments_per_year=calendar,household_profile=dict(profile_id=cid+'-household',required_categories=['living']),benchmarks=[]).value
        due=30 if c['kind']=='CARD'else 600 if c['kind']in ['CHARGE','OVERDRAFT']else 1000
        residual=cr.residual_income(normalized['total_income'],expenses['recurring_expenses'],c['existing_debt_service'],due,period='monthly',currency='EUR').value
        dsti=cr.debt_service_to_income(c['existing_debt_service']+due,normalized['total_income'],period='monthly',currency='EUR').value
        ap=dict(schema_version='1.0',policy_id=product+'-capacity',version='1',required_evidence=['income_verified'],rules=[dict(rule_id='service',metric='dsti',operator='<=',threshold=.45),dict(rule_id='residual',metric='residual',operator='>',threshold=0)])
        evidence=dict(income_verified=reconciled['document_supported_current_income']==normalized['total_income'])
        assess=cr.assess_affordability(dict(dsti=dsti['ratio'],residual=residual['residual']),ap,evidence).value
        missing=cr.assess_affordability(dict(dsti=dsti['ratio'],residual=residual['residual']),ap,dict(income_verified=False)).value
        results.append(dict(customer_id=cid,bureau=bureau,income=reconciled,normalization=normalized,expenses=expenses,residual=residual,dsti=dsti,assessment=assess,missing_evidence=missing))
    return results


def governance(c,product,original):
    model=original['model'];D='2024-01-01';registry=cr.register_rating_model_version(None,dict(model=model,dependencies={},configuration=dict(population=product+'-new-funded')),dict(model_id=model['model_id'],model_version='1',registered_at='2023-12-31')).value
    mh=registry['entries'][0]['model_hash'];results=[]
    for app,snap,risk in zip(c['applications'],original['snapshots'],original['risks']):
        sid=app['customer_id'];aid=app['application_id'];subject=dict(subject_id=sid,subject_type='customer')
        subjects=cr.build_rating_subjects(dict(customer=[dict(subject_id=sid,version='1',effective_from=D,effective_to=None,known_at=D,attributes=dict(product=product))]),[],D,D).value
        rs=dict(subject,as_of=D,known_at=D,attributes=dict(product=dict(value=product,observed_at=D,known_at=D),recent_arrears=dict(value=0,observed_at='2023-12-31',known_at='2023-12-31')))
        route=cr.route_rating_profile(rs,[dict(rule_id=product,profile_id=product+'-application',profile_version='1',priority=1,effective_from=D,effective_to=None,known_at='2023-12-31',conditions=[dict(field='product',operator='eq',value=product)])],D,D).value
        applicability=cr.assess_rating_applicability(rs,dict(contract_id=product+'-support',version='1',required_features=['product','recent_arrears'],allowed_values=dict(product=[product]),support_bounds=dict(recent_arrears=[0,1]),maximum_age_days=dict(recent_arrears=31)),D,D).value
        assignments=[dict(assignment_id=product+'-champion',profile_id=route['profile_id'],purpose='application',model_id=model['model_id'],model_version='1',model_hash=mh,effective_from=D,effective_to='2026-01-01',known_at='2023-12-31')]
        selected=cr.select_effective_model(registry,assignments,route['profile_id'],'application',D,D).value
        # Explicit calibrator dependency in batch inputs, matching direct forecast.
        request=dict(row_id=aid,known_at=D,model_id=model['model_id'],model_version='1',model_hash=mh,recipe=dict(method='application_risk',version='1.0',parameters=dict(target=c['application_target'],calibrator=original['calibrator'])),input=dict(snapshot=snap))
        batch=cr.rate_subject_batch([request],registry,product+'-'+aid,D).value
        binding=dict(profile_id=route['profile_id'],profile_version='1',model_id=model['model_id'],model_version='1',model_hash=mh,input_hash=batch['rows'][0]['input_hash'])
        assessment=cr.compose_risk_assessment(aid+'-assessment',subject,[dict(component_id=aid+'-pd',quantity='PD',value=risk['pd'],**subject,variant='application_given_funding',definition=c['application_target'])],binding,D,D,D).value
        replay=cr.replay_assessment_history([request],registry,batch['rows'],D).value
        results.append(dict(subjects=subjects,route=route,applicability=applicability,selected=selected,assessment=assessment,batch=batch,replay=replay))
    return dict(registry=registry,rows=results)


def phase_details(c,product):
    aid=next(a['account_id']for a in c['accounts']if a['account_id'].endswith('-main'));obs=[r for r in c['obligations']if r['account_id']==aid];payments=[r for r in c['payments']if r['account_id']==aid]
    arrears=cr.compute_days_past_due(obs,payments,as_of='2024-03-15',known_at='2024-03-15',currency='EUR').value
    state=cr.assign_delinquency_state(arrears['dpd'],arrears['past_due_amount'],600*c['scale'],dict(closed=False,confirmed_default=False),c['state_profile']).value
    payment=cr.build_payment_history_features(obs,payments,dict(status='COMPLETE',known_at='2024-03-15',observed_from='2024-02-01',complete_through='2024-03-15'),account_id=aid,as_of='2024-03-15',information_cutoff='2024-03-15',lookback_days=30,allocation_policy='fifo',currency='EUR',grace_days=0).value
    arrangement=cr.simulate_repayment_arrangement(**c['restructuring']).value
    action=cr.build_collections_action_history(**c['action_history']).value
    assert not arrangement['contract_changed']and not arrangement['default_status_changed']and not action['rows'][0]['accepted']and not action['rows'][0]['performed']
    assert arrears['dpd']==14 and state['state']=='ARREARS'
    revolving=cr.build_revolving_features(**c['revolving_history']).value
    transactions=cr.build_transaction_features(**c['transaction_history']).value if c['kind']=='OVERDRAFT'else None
    household=None
    if c['kind']=='HELOC':
        joint=cr.assemble_joint_application(**c['household_joint']).value
        graph=cr.build_customer_household_graph(**c['household_graph'])
        aggregate=cr.aggregate_household_obligations(household_graph=graph,**c['household_obligations']).value
        cltv=cr.combined_loan_to_value(**c['cltv']).value
        assert joint['monthly_income']==5000 and joint['monthly_debt_service']==1000
        assert aggregate['households'][0]['total_budget_debt_service']==1000
        av=deepcopy(c['affordability_validation'])
        for row in av['cases']:row['assessment']=cr.stress_affordability(row['baseline'],row['scenario']).value
        validation=cr.affordability_stress_backtest(**av).value
        assert validation['decision']['status']=='PASS'
        household=dict(joint=joint,graph=graph.value,aggregate=aggregate,combined_ltv=cltv,affordability_validation=validation,affordability_inputs=av)
    return dict(delinquency=arrears,delinquency_state=state,payment_history=payment,arrangement=arrangement,action_history=action,household=household,revolving_features=revolving,transaction_features=transactions)


def verify_input_ledgers(c):
    """Audit independently supplied validator witnesses against dated raw events."""
    ids={a['account_id']for a in c['accounts']}
    if {b['facility_id']for b in c['bookings']}!=ids:raise ValueError('Journey booking/account identities differ')
    for row in c['statements']:
        balance=sum((1 if r['kind']=='DRAW'else-1)*r['amount']for r in c['transactions']if r['account_id']==row['account_id']and r['date']<=row['observed_at']and r['known_at']<=row['known_at'])
        if abs(balance-row['balance'])>1e-10:raise ValueError('Statement does not reconcile principal ledger')
    for row in c['execution_inputs']['vintage']['account_outcomes']:
        events=sorted([e for e in c['events']if e['account_id']==row['account_id']and e['event_type']in ['DEFAULT','CLOSED']],key=lambda e:e['occurred_at']);first=events[0]if events else None;hit=first is not None and first['occurred_at']<=row['period_end']
        if row['cumulative_default']!=int(hit and first['event_type']=='DEFAULT')or row['exited']!=bool(hit and first['event_type']=='CLOSED'):raise ValueError('Vintage labels must derive from same first event ledger')
    claims=c['execution_inputs']['workout_claims']
    for w in claims:
        expected=sum(r['amount']for r in c['transactions']if r['account_id'].endswith('-main')and r['kind']in ['REPAYMENT','RECOVERY']and r['date']>=w['default_date']and r['date']==w['cashflows'][0]['cashflow_date'])
        actual=sum(f['amount']for f in w['cashflows']if f['kind']=='RECOVERY')
        if abs(expected-actual)>1e-10:raise ValueError('Recovery label is not the actual episode receipt')
    if c['new_draw_contract']['known_at']>'2024-06-01' or c['new_draw_contract']['effective_from']!='2024-06-01':raise ValueError('Post-cure draw requires prior known historical amendment')


def run(data=None,verify_reference=True,product_ids=None,return_artifacts=False,accounts_path=None,releases_path=None):
    from importlib.resources import files
    from pathlib import Path
    import importlib.util
    resource=files('creditratingsystem.datasets')
    if data is None:data=json.loads(resource.joinpath('revolving_lifecycle.json').read_text())
    spec=importlib.util.spec_from_file_location('_revolving_input_helper',Path(__file__).with_name('input_lineage.py'));helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    paths=(accounts_path or resource.joinpath('revolving_accounts.csv'),releases_path or resource.joinpath('revolving_releases.csv'))
    source={name:helper.prepare_snapshot(*paths,data['lineage'],dict(known_at=cutoff,revision=revision))for name,cutoff,revision in [('original','2024-02-15','0'),('revised','2024-02-29','1')]}
    if any(s['audit']['decision']['status']!='PASS'for s in source.values()):raise ValueError('Historical local source ledger failed')
    output={};artifacts={}
    for product,c in data['cases'].items():
        if product_ids is not None and product not in product_ids:continue
        verify_input_ledgers(c)
        o=origin(c,product);h=account_history(c,product,o);risk=risk_models(c,product,h);w=workout_history(c,product,risk);v=vintage_review(c,product,h);r=roll_review(c,product,h);pool=pool_review(c,product,h,w);replay=replay_review(c,product,o);g=governance(c,product,o);ph=phase_details(c,product);supplement=linked_diagnostics(c,product,h,risk,v,pool)
        # Validate the native/static witness state table against the actual builder.
        lookup={(x['account_id'],x['as_of']):x for x in h['panel']}
        if any(lookup[(x['account_id'],x['as_of'])]['state']!=x['state']for x in c['execution_inputs']['roll']['account_panel']):raise ValueError('Roll witness state differs from actual account panel')
        aid=next(a['account_id']for a in c['accounts']if a['account_id'].endswith('-main'))
        if next(s for s in source['original']['rows']if s['record_id']==aid)['credit_limit']!=risk['reference_limit']:raise ValueError('EAD forecast used a future limit revision')
        reasons=[cr.decompose_policy_decision(d,{rule['reason_code']:rule['reason_code'].replace('_',' ').lower()for rule in c['application_policy']['rules']}).value for d in o['decisions']]
        validation=dict(retail_cohort=h['cohort'],vintage=v['validation'],roll_rate=r['validation'],collections=w['validation'],retail_pool=pool['validation'],replay=replay['validation'])
        if ph['household'] is not None:
            lgd_inputs=heloc_lgd_review(c,product,risk,w)
            w['lgd_validation_input']=lgd_inputs;validation['lgd']=cr.lgd_backtest(**lgd_inputs).value
            validation['affordability']=ph['household'].pop('affordability_validation')
            affordability_inputs=ph['household'].pop('affordability_inputs')
            ph['household']['affordability_metrics']=validation['affordability']['metrics']
            ph['household']['stress_projection']=affordability_inputs['cases'][0]['assessment']
        else:affordability_inputs=None
        output[product]=dict(account_ids=[a['account_id']for a in c['accounts']],application_ids=[a['application_id']for a in c['applications']],application_pd=[round(x['pd'],8)for x in o['risks']],decision_actions=[d['action']for d in o['decisions']],application_targets=o['targets'],capacity=o['capacities'],panel=h['panel'],behavioral_targets=h['labels'],behavioral_pd=[round(x['pd'],8)for x in risk['predictions']],ead_forecast=risk['ead']['conditional_ead'],ead_reference=dict(drawn=risk['reference_drawn'],limit=risk['reference_limit']),observed_ccf=risk['realized_ccf'],lgd_forecast=risk['lgd']['mean_lgd'],phase_details=ph,episodes=w['episodes'],cure_targets=w['cure_targets'],workout_losses=[x['loss']for x in w['workouts']],vintage_table=v['table'],vintage_curve=v['curve'],roll_matrix=r['estimate']['groups'][0]['transition_matrix'],roll_projection=r['projection'],monitor=risk['monitor'],validation_metrics={k:z.get('metrics',{})for k,z in validation.items()},validation_coverage={k:z.get('coverage',{})for k,z in validation.items()},cohort_contract_status=h['cohort']['decision']['status'],replay_status=replay['validation']['decision']['status'],known_limits=replay['known_limits'],governance_replayed=[x['replay']['status']for x in g['rows']],phase_status=dict(ORIGINATION='APPROVED_AND_FUNDED',PERFORMING='ACTIVE_ZERO_RETAINED',REVIEW_EARLY_WARNING='DATED_LIMIT_AND_RISK_REVIEW',DELINQUENCY='OBSERVED_ARREARS',RESTRUCTURING='OFFERED_NOT_ACCEPTED',DEFAULT_WORKOUT='TWO_DISTINCT_DEFAULT_CLAIMS',EXIT='ACTUAL_CLOSURE',PORTFOLIO_MONITORING='SIX_LINKED_VALIDATORS'))
        output[product]['linked_diagnostics']={k:supplement[k]for k in ['lifetime_loss','path_loss','grade_calibration','grade_assignment','default_count','loss_assumption']}
        output[product]['linked_diagnostics']['collections']={k:z for k,z in supplement['collections'].items()if k!='raw_artifacts'}
        output[product]['linked_diagnostics']['survival_metrics']=supplement['survival_validation']['metrics']
        output[product]['linked_diagnostics']['report_decision']=supplement['validation_report']['decision']
        # Raw model/validation seals are never rounded or substituted by summaries.
        artifacts[product]=dict(origin=o,behavioral_and_loss_models=risk,governance=g,validation=validation,replay=replay,workout=w,pools=pool,vintage=v,roll=r,decision_reasons=reasons,phase_details=ph,affordability_inputs=affordability_inputs,linked_diagnostics=supplement)
        if verify_reference:
            assert h['cohort']['decision']['status']=='PASS'and replay['validation']['decision']['status']=='PASS'
            assert all(x['replay']['status']=='REPRODUCED'for x in g['rows'])
            assert len(h['panel'])==24 and [x['target']for x in o['targets']]==c['expected_application_targets']
            assert abs(w['workouts'][1]['loss']['economic_lgd']-c['second_default']['expected_lgd'])<1e-12
    result=dict(data_origin='SYNTHETIC',journey='J_REVOLVING_LIFECYCLE',products=output,source_lineage_status={k:v['audit']['decision']['status']for k,v in source.items()},prediction_reporting_decimal_places=8,raw_artifacts_available=True,interpretation='Tiny synthetic holdout evidence, not bank acceptance. Cured first-default accounts remain outside the original risk set. New draw is a separately observed historical amendment. Forecast receipt timing is an explicit25-day assumption; proposals never alter contractual cash or defaults.')
    if return_artifacts:result['raw_artifacts']=dict(products=artifacts,source=source)
    return result





def linked_diagnostics(c,product,h,risk,vintage,pool):
    from pathlib import Path
    import importlib.util
    spec=importlib.util.spec_from_file_location('_revolving_collections_helper',Path(__file__).with_name('retail_collections_review.py'));helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    chain=c['collections_chain'];aid=chain['claim_binding']['account_id']
    actual_default=next(e for e in c['events']if e['account_id']==aid and e['event_type']=='DEFAULT')
    if actual_default['occurred_at']!=chain['open_workout']['episode']['default_date']or chain['valuation_date']>='2024-05-10':raise ValueError('Open recovery must bind actual first default before its known full repayment')
    collections=helper.run(chain,verify_reference=False,return_artifacts=True)
    assert collections['cure_targets'][0]['cure_at']=='2024-05-20'and collections['cure_targets'][0]['redefault_at']=='2024-06-20'
    residual=600*c['scale'];expected=.25*(7/12*residual-5*c['scale'])+.75*.2*residual-2*c['scale']
    assert abs(collections['action_value']['expected_net_recovery']-expected)<1e-8
    survival=vintage['curve']['survival'];marginal=np.diff([0]+[1-x for x in survival]).tolist();lgd=risk['lgd']['mean_lgd'][0];conditional=[lgd*c['regime']['original_limit']]*8
    loss=cr.lifetime_expected_loss(marginal,conditional,[1]*8,[(i+1)/12 for i in range(8)],currency='EUR',probability_measure='PHYSICAL').value
    indicators=[[int(i==j)for j in range(8)]for i in range(9)]
    paths=cr.expected_loss_from_paths(indicators,[[lgd]*8]*9,[[c['regime']['original_limit']]*8]*9,marginal+[survival[-1]],[1]*8,scenario_ids=[product+'-first-event-bin-'+str(i)for i in range(9)],time_grid=[(i+1)/12 for i in range(8)],currency='EUR',probability_measure='PHYSICAL').value
    assert abs(loss['lifetime_expected_loss']-paths['expected_loss'])<1e-8
    grade_rows=pool['prior'];grade_counts=[sum((r['score']<50)==low for r in grade_rows)for low in [True,False]];grade_defaults=[sum(r['defaulted']and (r['score']<50)==low for r in grade_rows)for low in [True,False]]
    grade=cr.calibrate_grade_pd(grade_counts,grade_defaults,grades=['LOW_UTILIZATION','HIGH_UTILIZATION']).value
    active=[s for s in h['panel']if s['as_of']in ['2024-01-31','2024-02-29']and s['at_risk']];grade_assignment=[dict(account_id=r['account_id'],as_of=r['as_of'],grade=grade['grades'][int(r['features']['utilization']>=.5)]['grade'],pd=grade['grades'][int(r['features']['utilization']>=.5)]['pd'])for r in active];labelmap={(l['account_id'],l['as_of']):l for l in h['labels']};obs=[];forecast=[];surv=[]
    for s,p in zip(active,risk['predictions']):
        label=labelmap[(s['account_id'],s['as_of'])];oid=s['account_id']+'@'+s['as_of'];event=next((e for e in c['events']if e['account_id']==s['account_id']and s['as_of']<e['occurred_at']<=label['horizon_end']and e['event_type']in ['DEFAULT','CLOSED']),None)
        obs.append(dict(observation_id=oid,facility_id=s['account_id'],as_of=s['as_of'],performance_start=s['as_of'],outcome_end=label['horizon_end'],eligible=label['eligible'],prior_default=False,target=label['label'],outcome_known_at='2025-01-02',default_definition='first-default-before-exit',cohort=s['as_of']))
        forecast.append(dict(observation_id=oid,pd=round(p['pd'],8),forecast_at=s['as_of'],known_at=s['as_of'],default_definition='first-default-before-exit',horizon_days=60,unit='facility',probability_measure='PHYSICAL',model_id=product+'-behavioral',model_version='1'))
        hit=event is not None and event['event_type']=='DEFAULT';duration=(date.fromisoformat(event['occurred_at'])-date.fromisoformat(s['as_of'])).days if hit else 61
        surv.append(dict(prediction_id=oid,entity_id=s['account_id'],cluster_id=s['customer_id'],segment=product,as_of=s['as_of'],forecast_known_at=s['as_of'],horizon_end=label['horizon_end'],model_id=product+'-vintage',model_version='1',parameter_version='1',outcome_known_at='2025-01-02',observed_time=duration,event=hit,terminal_exit_at=event['occurred_at']if event is not None and not hit else None))
    count=cr.default_count_backtest(obs,forecast,['cohort'],unit='facility',horizon_days=60,default_definition='first-default-before-exit',redefault_policy='exclude_previous_default',evaluation_cutoff='2025-01-02').value
    target=dict(event='default',horizon=60,time_unit='days',probability_measure='PHYSICAL',conditioning='performing');design_args=design(product,'vintage',target);design_args['days_per_time_unit']=1
    sv=cr.survival_backtest([[1-survival[0],1-survival[1]]]*len(surv),surv,horizons=[29,60],censoring=dict(method='complete_followup',reference='Complete first-default-before-terminal-exit target; observed terminal closure permanently ends facility target, not issuer followup'),design=design_args,uncertainty='none').value
    manifests=dict(evaluation_started_at='2025-01-02',model_manifest={'prior_vintage_model':cr.content_fingerprint(vintage['model']).value},data_manifest={'actual_account_events':cr.content_fingerprint(c['events']).value},split_manifest={'heldout_customers':cr.content_fingerprint(sorted(a['customer_id']for a in c['accounts'])).value})
    policy=dict(policy_id=product+'-illustrative-validation',version='1',reference='Tiny fixture arithmetic threshold, not bank acceptance',declared_at='2023-12-01',required_result_ids=['survival'],criteria=[dict(criterion_id='brier',result_id='survival',path=['metrics','integrated_brier'],operator='LE',threshold=.3,rationale='Predeclared illustrative fixture criterion')])
    entries=[dict(result_id='survival',result=sv,reference='Same five actual overlapping facility windows; terminal first-exit definition',data_origin='SYNTHETIC',manifest_hashes={k:cr.content_fingerprint(manifests[k]).value for k in ['model_manifest','data_manifest','split_manifest']})]
    report=cr.validation_report(entries,acceptance_policy=policy,manifests=manifests).value
    return dict(collections=collections,lifetime_loss=loss,path_loss=paths,grade_calibration=grade,grade_assignment=grade_assignment,default_count=count,survival_validation=sv,validation_report=report,report_inputs=dict(results=entries,acceptance_policy=policy,manifests=manifests),loss_assumption='Full original authorized ceiling conditional exposure (charge authorization is not an undrawn commitment), LGD prior mean0.4, no future contract amendments and zero discount; first-event calendar buckets, not multiplying cumulative PD at each date')


def heloc_lgd_review(c,product,risk,workout):
    """Two actual closed claims, one borrower cluster; no independent-episode CI."""
    aid=next(a['account_id']for a in c['accounts']if a['account_id'].endswith('-main'));cid=next(a['customer_id']for a in c['accounts']if a['account_id']==aid)
    rows=[]
    for i,w in enumerate(workout['workouts']):
        rows.append(dict(prediction_id=w['episode_id'],entity_id=aid,cluster_id=cid+'-household',segment=product,as_of=w['default_date'],forecast_known_at=w['default_date'],horizon_end=(date.fromisoformat(w['default_date'])+timedelta(days=60)).isoformat(),model_id=product+'-lgd',model_version='1',parameter_version='1',default_date=w['default_date'],outcome_known_at='2024-08-31',workout_closed_at='2024-05-20'if i==0 else'2024-08-01',workout_status='CLOSED',predicted_lgd=risk['lgd']['mean_lgd'][0],realized_lgd=w['loss']['economic_lgd'],ead_at_default=w['ead']))
    target=dict(event='default',horizon=60,time_unit='days',probability_measure='PHYSICAL',conditioning='default')
    return dict(records=rows,design=design(product,'lgd',target),workout_policy=dict(method='closed_only',reference='Two actual same-borrower closed HELOC claims; prior frozen LGD, no independent-episode inference'),uncertainty='none')


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
