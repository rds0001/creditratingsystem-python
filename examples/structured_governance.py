"""Dated, domain-specific investment governance with immutable model replay."""
from copy import deepcopy
from importlib.resources import files
import itertools
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('structured_governance.json').read_text())
    data=deepcopy(data);out={}
    for product,case in data['variants'].items():
        sid=case['subject_id'];stype=case['subject_type'];model=case['model'];D='2024-01-01'
        subjects=cr.build_rating_subjects({stype:[dict(subject_id=sid,version='1',effective_from=D,effective_to=None,known_at=D,attributes=dict(product=product))]},[],D,D).value
        route=cr.route_rating_profile(case['snapshot'],case['rules'],D,D).value
        applicability=cr.assess_rating_applicability(case['snapshot'],case['applicability_contract'],D,D).value
        registry=cr.register_rating_model_version(None,dict(model=model,dependencies={},configuration=dict(source_recipe_reference=case['source_recipe_reference'])),dict(model_id=model['model_id'],model_version='v1',registered_at='2023-12-31')).value
        mh=registry['entries'][0]['model_hash'];assignment=[dict(assignment_id=product+'-champion',profile_id=route['profile_id'],purpose='PD',model_id=model['model_id'],model_version='v1',model_hash=mh,effective_from=D,effective_to='2025-01-01',known_at='2023-12-31')]
        selected=cr.select_effective_model(registry,assignment,route['profile_id'],'PD','2024-01-15','2024-01-15').value
        request=dict(row_id=product,known_at=D,model_id=model['model_id'],model_version='v1',model_hash=mh,recipe=dict(method='binary_glm',version='1.0',parameters=dict(expected_target=model['target'])),input=dict(X=[[0]],feature_order=['x']))
        batch=cr.rate_subject_batch([request],registry,product+'-origin',D).value;pd=batch['rows'][0]['value'][0]
        subject=dict(subject_id=sid,subject_type=stype);components=[dict(component_id=product+'-pd',quantity='PD',value=pd,**subject,variant='independent_30day_monitor',definition=model['target'])]
        binding=dict(profile_id=route['profile_id'],profile_version='1',model_id=model['model_id'],model_version='v1',model_hash=mh,input_hash=batch['rows'][0]['input_hash'])
        assessment=cr.compose_risk_assessment(product+'-governance',subject,components,binding,D,D,D).value
        contract=case['policy']['approval_contract'];timing=dict(as_of=D,known_at=D)
        decision_inputs=dict(risk=dict(timing,pd=pd,target=model['target'],freshness='CURRENT'),affordability=dict(timing,status='NOT_APPLICABLE',domain=contract['domain'],applicability_reference=contract['affordability']['reference'],valid_until='2024-02-01'),eligibility=dict(timing,status='ELIGIBLE'),operations=dict(timing,status='CLEAR',legal_status='VERIFIED'),as_of=D,known_at=D)
        policy=cr.compile_credit_decision_policy(case['policy']).value;decision=cr.apply_credit_decision_policy(policy,**decision_inputs).value
        invalid=deepcopy(decision_inputs);invalid['operations']['legal_status']='UNVERIFIED';blocked=cr.apply_credit_decision_policy(policy,**invalid).value
        fresh=cr.check_assessment_freshness(assessment,dict(maximum_age_days=30,maximum_information_age_days=30,trigger_types=['payment_delay']),[], '2024-01-15','2024-01-15').value
        performing_assessment=cr.compose_risk_assessment(product+'-performing',subject,components,binding,'2024-01-15','2024-01-15','2024-01-15').value
        trigger=dict(event_id=product+'-arrears',subject_id=sid,type='payment_delay',occurred_at='2024-01-15',known_at='2024-01-16')
        stale=cr.check_assessment_freshness(assessment,dict(maximum_age_days=30,maximum_information_age_days=30,trigger_types=['payment_delay']),[trigger],'2024-01-16','2024-01-16').value
        dependency=[dict(assessment_id=assessment['assessment_id'],dependency_type=stype,dependency_id=sid)]
        event=dict(event_id=trigger['event_id'],revision='1',occurred_at=trigger['occurred_at'],known_at=trigger['known_at'],dependencies=[dict(type=stype,id=sid)],payload=dict(contractual_payment_status='DELAYED'))
        updated=cr.apply_assessment_event(None,event,dependency,'2024-01-16').value
        duplicate=cr.apply_assessment_event(updated['state'],event,dependency,'2024-01-16').value
        cohort=cr.build_monitoring_cohorts(**case['cohort_input']).value
        monitoring=cr.monitor_rating_cohort(cohort,group_by=['cohort'],reference_window=dict(start=D,end='2024-01-31'),minimum_count=4,uncertainty='none').value
        immature_args=deepcopy(case['cohort_input']);immature_args['evaluation_cutoff']='2024-01-15';immature=cr.build_monitoring_cohorts(**immature_args).value
        feature_drift=cr.detect_credit_feature_drift([{'x':0},{'x':1}],[{'x':2},{'x':3}],feature_order=['x'],method='ks',permutations=list(itertools.permutations(range(4))),multiple_testing='none',dependence_assumption='exchangeable_rows').value
        # A degenerate reference null demonstrates sequential boundaries only.
        residuals=[dict(time='2024-02-01',value=monitoring[0]['metrics']['calibration_error'],mature=True)]
        null_spec=dict(model_id=model['model_id'],model_version='v1',calibration_id='zero-variation-analytical-reference',residual_scale='observed_minus_expected_rate',dependence='prespecified_dependent',probability_measure='PHYSICAL',calibrated_before='2024-01-31',validation_paths_independent=True)
        calibration_drift=cr.detect_pd_calibration_drift(residuals,method='ewma',parameters=dict(smoothing=.5),null_paths=[[0] for _ in range(19)],null_spec=null_spec,alpha=.05,validation_null_paths=[[0] for _ in range(20)]).value
        # Exit is a censoring event, never a fresh nondefault label.
        exit_args=deepcopy(case['cohort_input']);entity=exit_args['predictions'][0]['entity_id'];exit_args['outcome_events'].append(dict(event_id=product+'-sale',entity_id=entity,event_at='2024-01-15',known_at='2024-01-16',event='exit',default_definition=exit_args['target_contract']['default_definition']))
        after=deepcopy(exit_args['predictions'][0]);after.update(forecast_id=product+'-after-sale',case_id=product+'-after-sale',forecast_at='2024-02-01',known_at='2024-02-01');exit_args['predictions'].append(after)
        exit_cohort=cr.build_monitoring_cohorts(**exit_args).value
        transitions=[dict(from_state='DRAFT',to_state='FINAL',event_type='approve',required_gates=['local_review']),dict(from_state='FINAL',to_state='SUPERSEDED',event_type='close',required_gates=['position_closed'])]
        approved=cr.transition_assessment_state(assessment,dict(event_id='approved',type='approve',occurred_at='2024-01-02',known_at='2024-01-02',reason='Synthetic local review',gates=dict(local_review=True)),transitions).value
        retired=cr.transition_assessment_state(approved,dict(event_id='closed',type='close',occurred_at='2024-05-01',known_at='2024-05-01',reason='Workout settled; main claim assessment superseded',gates=dict(position_closed=True)),transitions).value
        newer=deepcopy(model);newer['model_version']='v2';newer['parameters']['coefficients'][0]=10
        later=cr.register_rating_model_version(registry,dict(model=newer,dependencies={},configuration={}),dict(model_id=model['model_id'],model_version='v2',registered_at='2024-02-01')).value
        replay=cr.replay_assessment_history([request],later,batch['rows'],D).value
        out[product]=dict(source_recipe_reference=case['source_recipe_reference'],subjects=subjects,route=route,applicability=applicability,registry=registry,selected=selected,batch=batch,assessment=assessment,performing_assessment=performing_assessment,policy=policy,decision=decision,failed_legal_gate=blocked,freshness=fresh,triggered_freshness=stale,event=updated,duplicate_event=duplicate,cohort=cohort,monitoring=monitoring,immature_cohort=immature,feature_drift=feature_drift,calibration_drift=calibration_drift,exit_cohort=exit_cohort,retired=retired,replay=replay)
    if verify_reference:
        for product,r in out.items():
            assert r['decision']['action']=='APPROVE'and r['failed_legal_gate']['action']=='REFER'
            assert r['applicability']['status']=='APPLICABLE'and r['selected']['status']=='SELECTED'
            assert r['freshness']['status']=='CURRENT'and r['triggered_freshness']['status']=='STALE'
            assert r['event']['affected_assessment_ids']==[r['assessment']['assessment_id']]and r['duplicate_event']['status']=='DUPLICATE'
            m=r['monitoring'][0]['metrics'];assert m['observed_rate']==.5 and m['brier']==.25
            assert not any(x['eligible']for x in r['immature_cohort']['rows'])
            assert r['exit_cohort']['rows'][0]['status']=='CENSORED'and r['exit_cohort']['rows'][-1]['status']=='EXITED_BEFORE_FORECAST'
            assert r['feature_drift'][0]['statistic']==1 and math.isclose(r['feature_drift'][0]['p_value'],9/25)
            assert r['calibration_drift']['statistic_path']==[0]
            assert r['retired']['state']=='SUPERSEDED'and len(r['retired']['history'])==2 and r['replay']['status']=='REPRODUCED'
    return dict(data_origin=data['data_origin'],version=data['version'],product=data['product'],variants=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
