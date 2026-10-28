"""Dated PD and workout development on the same borrower/facility population."""
from copy import deepcopy
from datetime import date,timedelta
from importlib.resources import files
import importlib.util
from pathlib import Path
import json
import numpy as np
import creditratingsystem as cr


def upstream_module():
    spec=importlib.util.spec_from_file_location('retail_development_component',Path(__file__).with_name('retail_development.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def canonical(data,retail,prepared):
    applications={r['application_id']:r for r in retail['applications']}
    booked={r['application_id']:r for r in retail['bookings']}
    rows=[r for r in prepared['rows']if r['selected']]
    tables=dict(obligors=[dict(entity_id=r['entity_id'])for r in prepared['rows']],observations=[dict(entity_id=r['entity_id'],observed_at=r['as_of'],available_at=r['features_known_at'])for r in prepared['rows']],facilities=[dict(facility_id=b['facility_id'],entity_id=applications[key]['customer_id'],ead=100)for key,b in booked.items()],defaults=[dict(episode_id=r['episode_id'],entity_id=r['entity_id'],default_at=r['default_date'],closed_at=r['resolution_date'])for r in data['workouts']],recoveries=[dict(cashflow_id=f['cashflow_id'],episode_id=f['episode_id'],cashflow_at=f['cashflow_date'],amount=f['amount'],kind='recovery'if f['kind']=='RECOVERY'else'cost')for f in data['cashflows']])
    validation=cr.validate_credit_dataset(tables).value
    if not validation['valid']:raise ValueError('Canonical credit tables failed validation: '+str(validation['violations']))
    ratings=[dict(observation_id=r['application_id'],entity_id=r['entity_id'],as_of=r['as_of'],performance_start=booked[r['application_id']]['first_draw_at'],followup_end=booked[r['application_id']]['followup_end'],funded=True)for r in rows]
    defaults=[dict(event_id=r['episode_id'],entity_id=r['entity_id'],default_at=r['default_date'],known_at=r['default_known_at'])for r in data['workouts']]
    windows=cr.build_observation_windows(ratings,defaults,[],30,data['evaluation_cutoff']).value
    assert [w['target']for w in windows]==[r['outcome']for r in rows]
    return dict(tables=tables,validation=validation,windows=windows)


def realized_loss(data,row,cutoff):
    if row['resolution_known_at']is None or row['resolution_known_at']>cutoff:return None
    flows=[f for f in data['cashflows']if f['episode_id']==row['episode_id']and f['known_at']<=cutoff and f['cashflow_date']<=row['resolution_date']]
    net=sum(f['amount']if f['kind']=='RECOVERY'else-f['amount']for f in flows)
    value=(row['ead_at_default']-net-row['cure_value'])/row['ead_at_default']
    if row['realized_lgd']is not None and abs(value-row['realized_lgd'])>1e-10:raise ValueError('Closed loss does not reconcile to cash and explicit cure asset value')
    return value


def loss_development(data):
    train=[r for r in data['workouts']if r['vintage']in (2016,2018)and r['resolution_known_at']is not None and r['resolution_known_at']<=data['loss_training']['known_at']]
    y=[realized_loss(data,r,data['loss_training']['known_at'])for r in train];x=np.empty((len(train),0));common=dict(feature_order=[],target=data['loss_target'],training=data['loss_training'],model_version='1')
    fractional=cr.fit_fractional_lgd(x,y,link='logit',covariance='HC0',model_id='fractional-lgd',**common)
    mixture=cr.fit_zero_one_inflated_lgd(x,y,component_spec=dict(boundary_origin='structural',gate_link='logit'),model_id='mixture-lgd',**common)
    cutoff=data['residual_training_cutoff'];landmarks=deepcopy([r for r in data['workouts']if r['vintage']in (2016,2018)])
    for row in landmarks:row['observation_end']=min(row['observation_end'],cutoff)
    ids={r['episode_id']for r in landmarks};flows=[f for f in data['cashflows']if f['episode_id']in ids]
    training=dict(data['loss_training'],known_at=cutoff,split_id='residual-landmarks-before-2018-02-15',workout_population='open_and_closed_defaulted_landmarks')
    hurdle=cr.fit_recovery_hurdle(landmarks,flows,np.empty((len(landmarks),0)),feature_order=[],stage_spec=data['stage_spec'],target=data['residual_target'],training=training,model_id='residual-hurdle',model_version='1',currency='EUR')
    test=[r for r in data['workouts']if r['vintage']==2022];predictions={name:cr.predict_lgd(model,np.empty((len(test),0)),feature_order=[],expected_target=data['loss_target'],default_condition='given_default').value for name,model in [('fractional',fractional),('mixture',mixture)]}
    opened=[r for r in test if r['resolution_date']is None];assert len(opened)==1
    reference=opened[0];past=[f for f in data['cashflows']if f['episode_id']==reference['episode_id']]
    dates=['2022-01-15','2022-03-01','2022-04-01'];valuation=dict(episode_id=reference['episode_id'],discount_curve=dict(anchor_date=reference['default_date'],known_at='2022-02-01',discount_basis='supplied economic reference curve',points=[dict(cashflow_date=d,discount_factor=f)for d,f in zip(dates,[1,.9,.8])]),fx_contract=dict(reporting_currency='EUR',known_at='2022-02-01',conversion_basis='cashflow_date',rates=[dict(currency='EUR',date=d,rate=1)for d in dates]),future_resolution_dates=dates[1:],resolution_time_convention='RIGHT_ENDPOINT',tail_discount_factor=.7,tail_assumption_reference='User flat discounted terminal payout beyond grid; not empirically identified',unmodeled_future_costs_assumption='NONE_BEYOND_SUPPLIED')
    args=dict(feature_order=[],expected_target=data['residual_target'],stage_spec=data['stage_spec'],as_of='2022-02-01',currency='EUR',extrapolation='parametric')
    value=cr.estimate_open_workout_lgd(opened,past,hurdle,np.empty((1,0)),[valuation],[],**args).value
    no_tail=deepcopy(valuation);no_tail.update(tail_discount_factor=None,tail_assumption_reference=None)
    unresolved=cr.estimate_open_workout_lgd(opened,past,hurdle,np.empty((1,0)),[no_tail],[],**args).value
    return dict(models=dict(fractional=fractional.model,mixture=mixture.model,hurdle=hurdle.model),training_rows=train,hurdle_training_rows=landmarks,hurdle_training_cashflows=flows,test_rows=test,predictions=predictions,open_valuation=value,open_without_tail=unresolved,valuation_contract=valuation,open_inputs=dict(episodes=opened,cashflows=past,as_of='2022-02-01'))


def evaluate_losses(data,loss):
    records=[]
    for row,p in zip(loss['test_rows'],loss['predictions']['mixture']['mean_lgd']):
        opened=row['resolution_date']is None
        records.append(dict(prediction_id=row['episode_id'],entity_id=row['entity_id'],cluster_id=row['entity_id'],segment='installment',as_of=row['default_date'],forecast_known_at=row['default_date'],horizon_end=(date.fromisoformat(row['default_date'])+timedelta(days=180)).isoformat(),model_id='mixture-lgd',model_version='1',parameter_version='1',default_date=row['default_date'],outcome_known_at=data['evaluation_cutoff']if opened else row['resolution_known_at'],workout_closed_at=row['resolution_date'],workout_status='OPEN'if opened else'CLOSED',predicted_lgd=float(p),realized_lgd=realized_loss(data,row,data['evaluation_cutoff']),ead_at_default=row['ead_at_default']))
    design=dict(model_id='mixture-lgd',model_version='1',parameter_version='1',data_fingerprint=cr.content_fingerprint(records).value,population_id='same-facility-default-workouts',split_id='untouched-2022-workouts',prediction_origin='HOLDOUT',evaluation_cutoff=data['evaluation_cutoff'],currency='EUR',target=dict(event='default',horizon=180,time_unit='days',probability_measure='PHYSICAL',conditioning='default'))
    result=cr.lgd_backtest(records,design=design,workout_policy=dict(method='closed_only',reference='Actual closed ledger outcomes only; open terminal payout estimate is not truth'),uncertainty='none').value
    return dict(records=records,design=design,result=result)


def run(data=None,retail_data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('model_development.json').read_text())
    if retail_data is None:retail_data=json.loads(files('creditratingsystem.datasets').joinpath(data['upstream_dataset']).read_text())
    data=deepcopy(data);retail_data=deepcopy(retail_data);component=upstream_module()
    upstream=component.run(retail_data,verify_reference=verify_reference,include_holdout=False)
    prepared=upstream['prepared'];roles={role:[r for r in prepared['rows']if r['role']==role and r['selected']]for role in ['train','calibration','test']}
    relational=canonical(data,retail_data,prepared);target=retail_data['target'];training=retail_data['training']
    features=lambda rows:[dict(x=r['snapshot']['features']['x'])for r in rows]
    pre=cr.fit_preprocessor(features(roles['train']),[dict(name='x',type='numeric',imputation='error',scaling='standard',missing_indicator=False)],target=target,training=training,model_id='pd-preprocessor',model_version='1')
    transformed={role:cr.transform_features(features(rows),pre).value for role,rows in roles.items()}
    x={role:np.asarray(value['design_matrix'],float)for role,value in transformed.items()};y={role:[r['outcome']for r in rows]for role,rows in roles.items()}
    common=dict(feature_order=['x'],target=target,training=training,model_version='1')
    glm=cr.fit_binary_glm(x['train'],y['train'],link='logit',fit_intercept=True,model_id='glm',**common)
    ridge=cr.fit_penalized_glm(x['train'],y['train'],penalty='ridge',lambda_path=[1,.1,0],selected_lambda=.1,scale=False,model_id='ridge',**common)
    firth=cr.fit_firth_logit(x['train'],y['train'],model_id='firth-x',**common)
    models={name:fit.model for name,fit in [('glm',glm),('ridge',ridge),('firth-x',firth)]}
    pred={name:list(cr.predict_pd(model,x['test'],feature_order=['x'],expected_target=target).value)for name,model in models.items()}
    cal=list(cr.predict_pd(glm,x['calibration'],feature_order=['x'],expected_target=target).value)
    isotonic=cr.calibrate_isotonic(cal,y['calibration'],metadata=retail_data['calibration_metadata']).value
    grade=cr.calibrate_grade_pd([sum((r['snapshot']['features']['x']<0)==(value==-1)for r in roles['calibration'])for value in [-1,1]],[sum(r['outcome']for r in roles['calibration']if (r['snapshot']['features']['x']<0)==(value==-1))for value in [-1,1]],grades=['LOW','HIGH'],prior_alpha=0,prior_beta=0,monotonic=True).value
    pred['isotonic']=list(cr.apply_calibrator(isotonic,pred['glm'],target=target,base_model_version='1').value);models['isotonic']=glm.model
    models['scorecard']=upstream['models']['scorecard'];pred['scorecard']=upstream['test_predictions']['scorecard']
    pred['grade']=[grade['grades'][0 if row['snapshot']['features']['x']<0 else 1]['pd']for row in roles['test']]
    models['grade']=dict(artifact_type='FrozenGradePDMapping',model_id='grade',model_version='1',method_id='calibration.calibrate_grade_pd',target=target,cutpoint=0,feature='x',lower_grade='LOW',upper_grade='HIGH',known_at=retail_data['calibration_metadata']['calibration']['known_at'],calibration=grade)
    candidate_models={name:dict(estimator=model,preprocessor=None if name in ('scorecard','grade')else pre.model)for name,model in models.items()}
    holdout=component.evaluate_holdout(retail_data,prepared,roles['test'],pred,candidate_models,dict(isotonic=isotonic))
    pd=pred['glm'];ids=[r['application_id']for r in roles['test']]
    design=dict(forecast_fingerprint=cr.content_fingerprint(pd).value,prediction_origin='HOLDOUT',forecast_known_at='2021-12-31',evaluation_at=data['evaluation_cutoff'],split_id='untouched-2022',target=target,observation_ids=ids)
    spiegelhalter=cr.spiegelhalter_test(pd,y['test'],design=design,alternative='two-sided',independence_assumption='independent_fixed_bernoulli')
    null=data['dependent_null'];innovations=dict(draw_assumption='independent_standard_normal_rows',observation_ids=ids,factor_keys=[dict(cluster_id='shared-macro',time_id='2022')],factor_normals=null['factor_normals'],idiosyncratic_normals=null['idiosyncratic_normals'])
    dependent=cr.dependent_pd_backtest(pd,y['test'],['shared-macro']*len(pd),['2022']*len(pd),design=design,null_model=dict(family='gaussian_factor',asset_correlation=null['asset_correlation'],factor_correlation=[[1]],parameter_uncertainty='conditional_fixed'),innovations=innovations,alternative='greater')
    loss=loss_development(data);loss_validation=evaluate_losses(data,loss)
    policy=dict(data['predeclared_policy'],required_result_ids=['pd-heldout','lgd-heldout'],criteria=[dict(criterion_id='pd-brier',result_id='pd-heldout',path=['metrics','by_model','glm','brier'],operator='LE',threshold=.1875001,rationale='Independent matched-cell Brier3/16'),dict(criterion_id='closed-lgd-bias',result_id='lgd-heldout',path=['metrics','overall','count_weighted','bias'],operator='BETWEEN',threshold=[-.1,.1],rationale='Predeclared closed-only synthetic calibration check; excludes open outcome')])
    manifests=dict(evaluation_started_at='2022-01-01',model_manifest=dict(pd=cr.content_fingerprint(candidate_models).value,lgd=cr.content_fingerprint(loss['models']).value),data_manifest=dict(applicants=cr.content_fingerprint(retail_data).value,workouts=cr.content_fingerprint(data).value),split_manifest=dict(assignments=cr.content_fingerprint(prepared['splits']).value))
    bindings={key:cr.content_fingerprint(manifests[key]).value for key in ['model_manifest','data_manifest','split_manifest']}
    results=[dict(result_id=key,result=result,reference='Observed dated synthetic holdout ledger',data_origin='SYNTHETIC',manifest_hashes=bindings)for key,result in [('pd-heldout',holdout['benchmark']),('lgd-heldout',loss_validation['result'])]]
    report=cr.validation_report(results,acceptance_policy=policy,manifests=manifests).value
    upstream_summary=dict(component_id='RETAIL_APPLICATION_DEVELOPMENT',input_fingerprint=cr.content_fingerprint(retail_data).value,prepared=prepared,stepwise=upstream['stepwise'],logistic_calibrator=upstream['logistic_calibrator'],logistic_calibration_inputs=upstream['fit_inputs']['logistic'],selected_application_model=upstream['models']['scorecard'])
    output=dict(data_origin='SYNTHETIC',version=data['version'],upstream=upstream_summary,relational=relational,preprocessor=pre.model,transformed=transformed,pd_models=models,candidate_models=candidate_models,isotonic_calibrator=isotonic,isotonic_inputs=dict(values=cal,weights=[1]*len(cal),outcomes=y['calibration']),grade_calibration=grade,holdout=holdout,spiegelhalter=dict(value=spiegelhalter.value,diagnostics=spiegelhalter.diagnostics),dependent=dict(value=dependent.value,diagnostics=dependent.diagnostics,innovations=innovations),loss=loss,loss_validation=loss_validation,report=report)
    if verify_reference:
        assert abs(holdout['benchmark']['metrics']['by_model']['glm']['brier']-3/16)<1e-10
        assert abs(loss['predictions']['mixture']['mean_lgd'][0]-.5)<1e-7
        assert loss['open_without_tail']['episodes'][0]['expected_final_lgd']is None
    return json.loads(json.dumps(output,allow_nan=False,default=lambda value:value.tolist()))


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
