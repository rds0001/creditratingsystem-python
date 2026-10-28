"""Chronological applicant development with explicit missing-outcome identification."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import numpy as np
import creditratingsystem as cr


def fingerprint(value):return cr.content_fingerprint(value).value


def prepare(data):
    snapshots=cr.build_application_snapshot(data['applications'],data['features'],required_features=['x','instrument','cashflow_ratio']).value
    labels=cr.build_application_targets(data['applications'],data['decisions'],data['bookings'],data['defaults'],data['exits'],30,data['evaluation_cutoff']).value
    designs={r['application_id']:r for r in data['cohort_design']};apps={r['application_id']:r for r in data['applications']};decisions={r['application_id']:r for r in data['decisions']};bookings={r['application_id']:r for r in data['bookings']}
    selection_apps=[];selection_decisions=[];selection_bookings=[];selection_outcomes=[];panel=[]
    for label in labels:
        key=label['application_id'];app=apps[key];decision=decisions[key];design=designs[key]
        selection_apps.append(dict(application_id=key,entity_id=app['customer_id'],submitted_at=app['application_at']))
        selection_decisions.append(dict(application_id=key,as_of=decision['decided_at'],known_at=decision['known_at'],decision='offered'if decision['status']=='approved'else'rejected',policy_version='synthetic-fixed-policy'))
        if label['funded']:
            booking=bookings[key];selection_bookings.append(dict(application_id=key,as_of=booking['first_draw_at'],known_at=booking['first_draw_known_at'],taken_up=True,funded=True))
            selected=label['label_observed']and label['mature']
            selection_outcomes.append(dict(application_id=key,as_of=label['outcome_end']if selected else data['evaluation_cutoff'],known_at=f"{design['vintage']}-02-03"if selected else data['evaluation_cutoff'],outcome_scope='own_contract',status='observed'if selected else'immature',outcome=label['target']if selected else None,target=data['target']))
        panel.append(dict(observation_id=key,entity_id=app['customer_id'],group_id=design['group_id'],observed_at=app['application_at'],outcome_end=design['planned_window_end']))
    selection=cr.build_selection_dataset(selection_apps,selection_decisions,selection_bookings,selection_outcomes,as_of=data['evaluation_cutoff'],target_population='All dated applicants; unknown own-loan potential outcomes remain missing',target=data['target']).value
    split=cr.make_grouped_temporal_splits(panel,**data['split_contract']).value
    roles={r['observation_id']:r['split']for r in split['assignments']};snap={r['application_id']:r for r in snapshots};sel={r['application_id']:r for r in selection['rows']}
    rows=[]
    for app in data['applications']:
        key=app['application_id'];s=sel[key];rows.append(dict(application_id=key,entity_id=app['customer_id'],as_of=app['application_at'],features_known_at=app['information_cutoff'],role=roles[key],snapshot=snap[key],design=designs[key],selected=int(s['observed']),outcome=s['outcome']))
    return dict(snapshots=snapshots,labels=labels,selection=selection,splits=split,rows=rows)


def pool_analysis(data,train_obs,cal_obs,model):
    T=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='not_defaulted_at_index',default_unit='facility',default_definition='DECLARED_FIRST_DEFAULT',currency='EUR')
    strata=dict(product='RETAIL_UNSECURED_INSTALLMENT',borrower_type='person',facility_type='installment',delinquency_state='performing')
    event_dates={e['facility_id']:e['default_at']for e in data['defaults']}
    features=model.model['feature_order']
    def score(rows):return list(cr.predict_pd(model,[[r['snapshot']['features'][name]for name in features]for r in rows],feature_order=features,expected_target=data['target']).value)
    def assignment_row(r,p):
        year=r['design']['vintage'];return dict(observation_id=r['application_id'],index_date=f'{year}-01-03',feature_observed_at=r['as_of'],feature_known_at=r['as_of'],score=float(p),**strata)
    observations=[]
    for row,p in zip(train_obs,score(train_obs)):
        year=row['design']['vintage'];key=row['application_id'];observations.append(dict(assignment_row(row,p),default_unit_id='facility-'+key,role='TRAIN',label_window_end=row['design']['planned_window_end'],label_known_at=f'{year}-02-03',defaulted=bool(row['outcome']),default_date=event_dates.get('facility-'+key),lgd_status='OPEN'if row['outcome']else'NOT_DEFAULTED',lgd=None,ead=None,lgd_known_at=None,ead_known_at=None))
    spec=dict(spec_id='application-funded-pools',version='1',known_at='2015-12-31',stratum_fields=list(strata),score_field='score',segments=[dict(segment_id='funded-installment',values=strata,cutpoints=[.5],score_min=0,score_max=1,monotonicity=dict(pd='increasing',lgd='none',ead='none'))],minimum_cases=2,minimum_defaults=0,minimum_lgd_count=0,minimum_ead_count=0,maximum_bin_mean_spread=dict(pd=None,lgd=None,ead=None))
    pool_model=cr.fit_retail_risk_pools(observations,spec,target=T,training=dict(data['training'],role='TRAIN'),model_id='funded-pools',model_version='1')
    cal_rows=[assignment_row(row,p)for row,p in zip(cal_obs,score(cal_obs))]
    assigned=cr.assign_retail_risk_pool(cal_rows,pool_model,as_of='2020-01-03',information_cutoff='2020-01-03',expected_target=T,uncovered_policy='error')
    outcomes=[dict(observation_id=row['application_id'],default_unit_id='facility-'+row['application_id'],index_date='2020-01-03',label_window_end=row['design']['planned_window_end'],label_known_at='2020-02-03',outcome_status='MATURE',reference_exposure=100,reference_known_at='2020-01-03',defaulted=bool(row['outcome']),default_date=event_dates.get('facility-'+row['application_id']))for row in cal_obs]
    profile=dict(profile_id='heldout-funded',version='1',known_at='2019-12-31',pd_weighting='reference_exposure',lgd_weighting='ead',ead_weighting='episode',ccf_weighting='undrawn',confidence=.95,independence_assumption='independent_default_units',minimum_pool_cases=2)
    parameters=cr.estimate_pool_risk_parameters(assigned,outcomes,[],profile,evaluation_cutoff='2020-02-03',calibration_id='pool-calibration',calibration_version='1').value
    return dict(training_inputs=observations,model=pool_model.model,assignments=assigned.value,calibration_parameters=parameters)


def evaluate_holdout(data,prepared,test_obs,predictions,models,calibrators):
    base=lambda name:models['scorecard'if name=='calibrated'else'firth'if name=='thin_calibrated'else name]
    training_entities=[r['entity_id']for r in prepared['rows']if r['role']in ('train','calibration')]
    candidates=[dict(model_id=m,model_hash=fingerprint(dict(model=base(m),calibrator=calibrators.get(m))),training_known_at='2020-12-31',training_entities=training_entities)for m in predictions]
    evaluation_rows=[dict(application_id=r['application_id'],entity_id=r['entity_id'],as_of=r['as_of'],known_at='2022-02-03',label_origin='OBSERVED_OWN_CONTRACT',outcome=r['outcome'],target=data['target'],funded=True,mature=True)for r in test_obs]
    pseudo=dict(evaluation_rows[0],application_id='diagnostic-pseudo',entity_id='diagnostic-pseudo',label_origin='PSEUDO',outcome=1)
    evaluation=cr.prepare_reject_evaluation(candidates,evaluation_rows+[pseudo],as_of=data['evaluation_cutoff'],target=data['target']).value
    cohort_target=dict(event='default',probability_measure='PHYSICAL',default_definition='DECLARED_FIRST_DEFAULT',unit='facility',horizon_days=30,default_grade='D')
    entries={}
    defaults={e['facility_id']:e for e in data['defaults']}
    for name,values in predictions.items():
        forecasts=[];events=[];coverage=[]
        for row,p in zip(test_obs,values):
            key=row['application_id'];facility='facility-'+key
            # Re-publication at first draw carries the locked application forecast;
            # no new information or parameter refit is used to move the horizon.
            forecasts.append(dict(forecast_id=name+'-'+key,case_id=key,entity_id=facility,cluster_id=row['entity_id'],cohort='2022-01',segment='funded-installment',model_id=name,model_version='1',forecast_at='2022-01-03',known_at='2022-01-03',original_prediction_at=row['as_of'],pd=float(p),target=cohort_target,prediction_origin='HOLDOUT'))
            if facility in defaults:
                e=defaults[facility];events.append(dict(event_id=e['event_id'],entity_id=facility,event='default',event_at=e['default_at'],known_at=e['known_at'],default_definition='DECLARED_FIRST_DEFAULT'))
            coverage.append(dict(entity_id=facility,complete_through=row['design']['planned_window_end'],known_at='2022-02-03'))
        cohort=cr.build_monitoring_cohorts(forecasts,events,coverage,target_contract=cohort_target,evaluation_cutoff=data['evaluation_cutoff']).value
        entries[name]=dict(cohort=cohort,model_version='1',parameter_version='1',model_fingerprint=fingerprint(dict(model=base(name),calibrator=calibrators.get(name))),training_known_at='2020-12-31')
    design=dict(model_id='scorecard',model_version='1',parameter_version='1',data_fingerprint=fingerprint(evaluation_rows),population_id='observed-funded-test-only',split_id='untouched-2022',prediction_origin='HOLDOUT',evaluation_cutoff=data['evaluation_cutoff'],target=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='performing'))
    benchmark=cr.benchmark_predictions(entries,reference_model_id='scorecard',metrics=['brier','auc','log_loss'],matching_policy='strict',multiple_comparisons='none',design=design,uncertainty='none').value
    manifests=dict(evaluation_started_at='2022-01-01',model_manifest={k:v['model_fingerprint']for k,v in entries.items()},data_manifest=dict(test=fingerprint(evaluation_rows)),split_manifest=dict(assignments=fingerprint(prepared['splits'])))
    report=cr.validation_report([dict(result_id='paired-holdout',result=benchmark,reference='Actual independently held-out synthetic funded outcomes; not rejects',data_origin='SYNTHETIC',manifest_hashes={k:fingerprint(manifests[k])for k in ('model_manifest','data_manifest','split_manifest')})],acceptance_policy=data['predeclared_acceptance'],manifests=manifests).value
    return dict(evaluation_roles=evaluation,cohorts=entries,benchmark=benchmark,report=report)


def thin_analysis(data,train_obs,cal_obs,test_obs):
    X=lambda rr:[[r['snapshot']['features']['cashflow_ratio']]for r in rr]
    firth=cr.fit_firth_logit(X(train_obs),[r['outcome']for r in train_obs],feature_order=['cashflow_ratio'],target=data['target'],training=data['training'],model_id='firth',model_version='1')
    model=deepcopy(firth.model);model['application_contract']=dict(product_ids=['RETAIL_UNSECURED_INSTALLMENT'],population_ids=['synthetic_all_applicants'],performance_origin='first_draw_at',term_stage='requested',term_features={})
    cal_pd=cr.predict_pd(model,X(cal_obs),feature_order=['cashflow_ratio'],expected_target=data['target']).value
    groups=[r['snapshot']['features']['thin_file_stratum']for r in cal_obs]
    hierarchical=cr.fit_hierarchical_calibration(cal_pd,[r['outcome']for r in cal_obs],groups,metadata=data['calibration_metadata']).value
    test_pd=cr.predict_pd(model,X(test_obs),feature_order=['cashflow_ratio'],expected_target=data['target']).value
    test_cal=cr.apply_calibrator(hierarchical,test_pd,target=data['target'],base_model_version='1',groups=[r['snapshot']['features']['thin_file_stratum']for r in test_obs]).value
    cases={}
    for name,c in data['thin_file_cases'].items():
        bureau=cr.build_bureau_features(**c['bureau'])
        evidence=cr.build_thin_file_features(bureau,c['evidence'],c['eligibility_profile'],customer_id=c['customer_id'],as_of='2022-02-28',information_cutoff='2022-02-28').value
        transactions=cr.build_transaction_features(**c['transactions']).value
        income=cr.reconcile_income_evidence(**c['income']).value
        ratio=income['total_selected_income']/c['monthly_expense_budget']
        app=dict(application_id=c['customer_id'],customer_id=c['customer_id'],product_id='RETAIL_UNSECURED_INSTALLMENT',population_id='synthetic_all_applicants',application_at='2022-03-01',information_cutoff='2022-03-01',requested_terms=dict(amount=1000,currency='EUR',term_months=12))
        snapshot=cr.build_application_snapshot([app],[dict(application_id=c['customer_id'],feature='cashflow_ratio',value=ratio,observed_at='2022-02-28',known_at='2022-02-28',source_id='reconciled-dated-income-budget')],required_features=['cashflow_ratio']).value[0]
        support_snapshot=dict(subject_id=c['customer_id'],subject_type='customer',as_of='2022-03-01',known_at='2022-03-01',attributes=dict(cashflow_ratio=dict(value=ratio,observed_at='2022-02-28',known_at='2022-02-28')))
        support=cr.assess_rating_applicability(support_snapshot,dict(contract_id='firth-support',version='1',required_features=['cashflow_ratio'],support_bounds=dict(cashflow_ratio=[.5,1.5]),allowed_values={},maximum_age_days=dict(cashflow_ratio=30)),'2022-03-01','2022-03-01').value
        risk=None;calibrated=None
        if support['status']=='APPLICABLE':
            risk=cr.predict_application_risk(snapshot,model,data['target']).value
            calibrated=cr.apply_calibrator(hierarchical,[risk['pd']],target=data['target'],base_model_version='1',groups=[evidence['thin_file_stratum']]).value
        cases[name]=dict(bureau=bureau.value,evidence=evidence,transactions=transactions,income=income,cashflow_ratio=ratio,snapshot=snapshot,applicability=support,raw_application_risk=risk,calibrated_pd=calibrated)
    unknown=cr.apply_calibrator(hierarchical,[.5],target=data['target'],base_model_version='1',groups=['NEW_GROUP']).value
    return dict(calibration_inputs=dict(p=list(cal_pd),w=[1]*len(cal_obs),y=[r['outcome']for r in cal_obs],groups=groups),model=model,hierarchical_calibrator=hierarchical,cases=cases,test_raw=list(test_pd),test_calibrated=list(test_cal),unknown_group_reference=list(unknown))


def run(data=None,verify_reference=True,full_profiles=False,include_holdout=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('retail_development.json').read_text())
    data=deepcopy(data);prepared=prepare(data);rows=prepared['rows'];train=[r for r in rows if r['role']=='train'];cal=[r for r in rows if r['role']=='calibration'];test=[r for r in rows if r['role']=='test'];observed=lambda rr:[r for r in rr if r['selected']]
    train_obs=observed(train);cal_obs=observed(cal);test_obs=observed(test)
    X=lambda rr,names:np.array([[r['snapshot']['features'][n]for n in names]for r in rr],float)
    Y=lambda rr:[r['outcome']for r in rr]
    target=data['target'];training=data['training'];common=dict(target=target,training=training,model_version='1')
    stepwise=cr.fit_stepwise(X(train_obs,['x','instrument']),Y(train_obs),feature_order=['x','instrument'],candidate_groups=dict(signal=['x'],selection_instrument=['instrument']),direction='forward',criterion='AIC',estimator='GLM',training=training).value
    chosen=stepwise['selected_model']['feature_order']
    scorecard=cr.fit_scorecard(X(train_obs,chosen),Y(train_obs),feature_order=chosen,fit_spec=dict(neutral_profile=[0]*len(chosen),feature_kinds=['expert']*len(chosen)),model_id='scorecard',**common)
    gam=cr.fit_gam_pd(X(train_obs,['x']),Y(train_obs),feature_order=['x'],linear_features=[],smooth_specs=[dict(feature='x',boundary_knots=[-1,1],interior_knots=[0],degree=1,smoothing=.1,monotonicity='increasing',extrapolation='error')],variant='monotone',model_id='gam',**common)
    cal_raw=cr.predict_pd(scorecard,X(cal_obs,chosen),feature_order=chosen,expected_target=target).value
    calibrator=cr.calibrate_logistic(cal_raw,Y(cal_obs),metadata=data['calibration_metadata']).value
    test_raw=cr.predict_pd(scorecard,X(test_obs,chosen),feature_order=chosen,expected_target=target).value
    test_calibrated=cr.apply_calibrator(calibrator,test_raw,target=target,base_model_version='1').value
    test_gam=cr.predict_pd(gam,X(test_obs,['x']),feature_order=['x'],expected_target=target).value
    metadata=[dict(entity_id=r['entity_id'],as_of=r['as_of'],features_known_at=r['features_known_at'],label_known_at=str(r['design']['vintage'])+'-02-03')for r in train]
    warm=[i for i,r in enumerate(train)if r['design']['vintage']==2016];forward=[i for i,r in enumerate(train)if r['design']['vintage']==2018];folds=[dict(train=warm,validation=forward)]
    selection_target=dict(target,event='funding',conditioning='eligible_applicants')
    propensity=cr.fit_selection_propensity(X(train,['instrument']),[r['selected']for r in train],feature_order=['instrument'],mechanism='funding',folds=folds,row_metadata=metadata,target=selection_target,training=training,model_id='propensity',model_version='1')
    fold_training=dict(training,known_at='2016-02-03',data_fingerprint=fingerprint(dict(X=X([train[i]for i in warm],['instrument']).tolist(),selected=[train[i]['selected']for i in warm])))
    fold_model=cr.fit_binary_glm(X([train[i]for i in warm],['instrument']),[train[i]['selected']for i in warm],feature_order=['instrument'],link='logit',fit_intercept=True,target=selection_target,training=fold_training,model_id='propensity',model_version='1').model
    assert fingerprint(fold_model)==propensity.diagnostics['folds'][0]['model_hash']
    p_all=cr.predict_pd(propensity,X(train,['instrument']),feature_order=['instrument'],expected_target=selection_target).value.tolist()
    oof_p=[p_all[i]if v is None else v for i,v in enumerate(propensity.diagnostics['out_of_fold_propensity'])]
    warm_obs=[r for r in train_obs if r['design']['vintage']==2016]
    warm_model=cr.fit_scorecard(X(warm_obs,chosen),Y(warm_obs),feature_order=chosen,fit_spec=dict(neutral_profile=[0]*len(chosen),feature_kinds=['expert']*len(chosen)),target=target,training=dict(training,known_at='2016-12-31'),model_id='warmup-scorecard',model_version='1')
    oof_m=cr.predict_pd(warm_model,X(train,chosen),feature_order=chosen,expected_target=target).value.tolist();weights=[int(i in forward)for i in range(len(train))]
    ipw=cr.selection_ipw_weights([r['selected']for r in train],oof_p,target_weights=weights).value
    aipw=cr.estimate_population_risk_aipw(Y(train),[r['selected']for r in train],oof_m,oof_p,fold_lineage=dict(rows=metadata,folds=folds,nuisance_prediction_origin='OUT_OF_FOLD'),target_weights=weights).value
    fuzzy=cr.fuzzy_reject_augmentation(Y(train),[r['selected']for r in train],oof_m,application_ids=[r['application_id']for r in train]).value
    bands=[];counts=[];bads=[];rejects=[]
    for x in [-1,1]:
        band=[r for r in train if r['design']['x']==x];bands.append(str(x));counts.append(sum(r['selected']for r in band));bads.append(sum(r['outcome']or 0 for r in band));rejects.append(len(band)-counts[-1])
    parcel=cr.parcel_rejected_applications(bands,bads,counts,rejects,log_odds_shift=math.log(2)).value
    joint=cr.fit_joint_selection_model(X(train,['x']),X(train,['instrument']),[r['selected']for r in train],Y(train),outcome_features=['x'],selection_features=['instrument'],model_id='joint',rho_grid=[-.3,0,.3]if full_profiles else[0],selected_rho=0,**common)
    bounds=cr.reject_inference_bounds(Y(train),[r['selected']for r in train],target_weights=weights).value
    sensitivity=cr.reject_inference_sensitivity(oof_m,[r['selected']for r in train],Y(train),log_odds_shifts=[-math.log(2),0,math.log(2)],target_weights=weights,threshold=.55).value
    overlap_code=None
    try:cr.selection_ipw_weights([1,0],[1,0])
    except cr.CreditRiskError as e:overlap_code=e.code
    models=dict(scorecard=scorecard.model,gam=gam.model,propensity=propensity.model,warmup=warm_model.model,joint=joint.model)
    thin=thin_analysis(data,train_obs,cal_obs,test_obs)
    thin['population_bounds']={}
    for group in ['BUREAU_NO_HIT','BUREAU_STALE']:
        group_rows=[r for r in train if r['snapshot']['features']['thin_file_stratum']==group]
        thin['population_bounds'][group]=cr.reject_inference_bounds(Y(group_rows),[r['selected']for r in group_rows]).value
    models['firth']=thin['model']
    pools=pool_analysis(data,train_obs,cal_obs,scorecard)
    holdout=None if not include_holdout else evaluate_holdout(data,prepared,test_obs,dict(scorecard=list(test_raw),calibrated=list(test_calibrated),gam=list(test_gam),firth=thin['test_raw'],thin_calibrated=thin['test_calibrated']),models,dict(calibrated=calibrator,thin_calibrated=thin['hierarchical_calibrator']))
    output=dict(fit_inputs=dict(logistic=dict(values=list(cal_raw),weights=[1]*len(cal_obs),outcomes=Y(cal_obs)),propensity_fold_model=fold_model),thin_file=thin,pools=pools,holdout=holdout,data_origin=data['data_origin'],version=data['version'],product=data['product'],prepared=prepared,stepwise=stepwise,models=models,logistic_calibrator=calibrator,test_predictions=dict(scorecard=list(test_raw),calibrated=list(test_calibrated),gam=list(test_gam)),selection_analysis=dict(ipw=ipw,aipw=aipw,fuzzy=fuzzy,parcel=parcel,joint_diagnostics=joint.diagnostics,bounds=bounds,sensitivity=sensitivity,overlap_failure=overlap_code))
    if verify_reference:
        assert chosen==['x']and all(abs(p-(.25 if r['design']['x']==-1 else .75))<1e-7 for p,r in zip(test_raw,test_obs))
        assert abs(aipw['population_pd']-.5)<1e-8 and abs(ipw['effective_sample_size']-38.4)<1e-7
        assert abs(bounds['lower']-.3125)<1e-12 and abs(bounds['upper']-.6875)<1e-12 and overlap_code=='POSITIVITY_FAILURE'
    return json.loads(json.dumps(output,allow_nan=False,default=lambda value:value.tolist()))


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
