"""Seven dated sector histories pinned to actual component models and calibrators."""
from importlib.resources import files
from copy import deepcopy
import json,math
import creditratingsystem as cr


def run(data=None,verify_reference=True,artifact_paths=None,return_audit_artifacts=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_sector_governance.json').read_text())
    rows=[];audits=[]
    for case in data['cases']:
        path=artifact_paths[case['model_resource']]if artifact_paths else files('creditratingsystem.datasets').joinpath(case['model_resource'])
        model=cr.load_fitted_model(path)
        result,audit=_journey(case,model)
        if verify_reference:
            assert math.isclose(result['origination_pd'],case['reference_pd'],abs_tol=1e-12,rel_tol=1e-12)
            expected=case.get('reference_action','APPROVE'if case['gate']['available_unrestricted_cashflow']>=case['gate']['debt_service_due']else'REFER')
            assert result['origination_decision']==expected
        rows.append(result);audits.append(audit)
    report=dict(data_origin='SYNTHETIC',scope=data['scope'],cases=rows)
    return dict(report=report,audits=audits)if return_audit_artifacts else report


def _journey(data,model):
    from datetime import date
    dates={'routing_start': '2023-01-01', 'origination': '2024-03-01', 'processing': '2024-03-02', 'review': '2024-03-03', 'finalize': '2024-03-04', 'performing_review': '2024-03-15', 'warning': '2024-04-01', 'warning_known': '2024-04-02', 'delinquency': '2024-04-15', 'delinquency_known': '2024-04-16', 'proposal': '2024-04-20', 'default': '2024-05-01', 'default_known': '2024-05-02', 'closure': '2024-09-03', 'replay': '2025-01-02', 'later_registration': '2025-02-01', 'later_replay': '2025-03-01', 'payment_due': '2024-03-15'}
    supplied=data.get('timeline',{})
    if set(supplied)-set(dates):raise ValueError('Unknown timeline date')
    dates.update(supplied)
    for value in dates.values():
        if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value:raise ValueError('Timeline requires ISO dates')
    for first,last in [('routing_start','origination'),('origination','processing'),('origination','review'),('review','finalize'),('finalize','closure'),('warning','warning_known'),('delinquency','delinquency_known'),('default','default_known'),('closure','replay'),('replay','later_registration'),('later_registration','later_replay'),('payment_due','delinquency')]:
        if dates[first]>dates[last]:raise ValueError('Inconsistent timeline chronology')
    for key in ['warning_known','delinquency_known','default_known','proposal']:
        if not dates['origination']<=dates[key]<=dates['closure']:raise ValueError('Event outside historical interval')

    # Bound, dated inference uses a real fitted model and immutable local requests.
    target=data['target'];sid=data['case_id'];mid=model.value['model_id'];facts=data['facts']
    rule=dict(rule_id=sid,profile_id=sid,profile_version='1',priority=1,effective_from=dates['routing_start'],effective_to=None,known_at=dates['routing_start'],conditions=[dict(field=k,operator='eq',value=v)for k,v in facts.items()])
    contract=dict(contract_id=sid,version='1',required_features=list(facts),allowed_values={k:[v]for k,v in facts.items()},support_bounds={},maximum_age_days={k:90 for k in facts})
    ref=lambda source,identity,*path:{'$ref':dict(source=source,id=identity,path=list(path))}
    recipe=dict(method='assessment_pipeline',version='2.0',parameters=dict(routing_rules=[rule],applicability_contract=contract,steps=[dict(step_id='pd',method_id='classification.predict_pd',arguments=dict(fitted_model=ref('model','primary'),X=ref('input','features','value'),feature_order=data['feature_order'],expected_target=target),argument_types={},output_contract=dict(shape='vector',unit='probability',target=target))],components=[dict(component_id='pd',quantity='PD',value=ref('step','pd','value',0),subject_id=sid,subject_type=data.get('subject_type','obligor'),variant='standalone',definition=target)],effects=[],transitions=[dict(from_state='DRAFT',to_state='REVIEWED',event_type='review',required_gates=['checked']),dict(from_state='REVIEWED',to_state='FINAL',event_type='finalize',required_gates=['approved']),dict(from_state='FINAL',to_state='SUPERSEDED',event_type='exit',required_gates=['settled'])],event_dependencies=[dict(assessment_id=sid+'-assessment',dependency_type='subject',dependency_id=sid)],decision=None))
    recipe['parameters']['steps'].append(dict(step_id='calibrated',method_id='calibration.apply_calibrator',arguments=dict(calibrator=ref('artifact','calibrator'),values=ref('step','pd','value'),target=target,base_model_version='1'),argument_types={'values':'numeric_vector'},output_contract=dict(shape='vector',unit='probability',target=target)))
    extra=data.get('calibrator_arguments',{})
    if set(extra)-{'groups'}:raise ValueError('Only explicit calibration groups are configurable')
    recipe['parameters']['steps'][1]['arguments'].update(extra)
    recipe['parameters']['components'][0]['value']=ref('step','calibrated','value',0)
    dependencies=dict(artifacts={'calibrator':dict(kind='calibrator',content=data['calibrator'],sha256=cr.content_fingerprint(data['calibrator']).value,known_at=dates['origination'])})
    policy=cr.compile_credit_decision_policy(data['decision_policy']).value
    dependencies['artifacts']['credit_policy']=dict(kind='parameters',content=policy,sha256=cr.content_fingerprint(policy).value,known_at=dates['origination'])
    timing=dict(as_of=dates['origination'],known_at=dates['origination']);gate=data['gate']
    recipe['parameters']['decision']=dict(policy=ref('artifact','credit_policy'),risk=dict(timing,pd=ref('step','calibrated','value',0),target=target,freshness='CURRENT'),affordability=dict(timing,status='AFFORDABLE'if gate['available_unrestricted_cashflow']>=gate['debt_service_due']else'NOT_AFFORDABLE'),eligibility=dict(timing,status='ELIGIBLE'if gate['legal_eligibility']else'INELIGIBLE'),operations=dict(timing,status=gate['operations']),**timing)
    registry=cr.register_rating_model_version(None,dict(model=model.value,dependencies=dependencies,configuration=dict(assessment_recipe=recipe)),dict(model_id=mid,model_version='1',registered_at=dates['origination'])).value
    assignment=dict(assignment_id='corporate-champion-1',profile_id=sid,purpose='PD',model_id=mid,model_version='1',model_hash=registry['entries'][0]['model_hash'],effective_from=dates['origination'],effective_to=None,known_at=dates['origination'])
    selection=cr.select_effective_model(registry,[assignment],sid,'PD',dates['origination'],dates['origination']).value
    snapshot=dict(subject_id=sid,subject_type=data.get('subject_type','obligor'),as_of=dates['origination'],known_at=dates['origination'],attributes={k:dict(value=v,observed_at=dates['origination'],known_at=dates['origination'])for k,v in facts.items()})
    subject=dict(subject_id=sid,version='1',attributes=snapshot['attributes'],effective_from=dates['origination'],effective_to=None,known_at=dates['origination'])
    subjects=cr.build_rating_subjects({data.get('subject_type','obligor'):[subject]},[],dates['performing_review'],dates['performing_review']).value
    request=dict(row_id='origination',known_at=dates['processing'],model_id=mid,model_version='1',model_hash=assignment['model_hash'],recipe=recipe,input=dict(snapshot=snapshot,datasets={'features':dict(value=data['features'],as_of=dates['origination'],known_at=dates['origination'])},assessment_id=sid+'-assessment',assessment_known_at=dates['origination'],calculated_at=dates['processing'],lifecycle_events=[],events=[]))
    original=cr.rate_subject_batch([request],registry,sid+'-origination',dates['processing']).value['rows'][0]
    assert original['status']=='OK',original
    assessment=original['value']['assessment']
    fp=dict(maximum_age_days=90,maximum_information_age_days=90,trigger_types=['financial_deterioration'])
    freshness=cr.check_assessment_freshness(assessment,fp,[],dates['performing_review'],dates['performing_review']).value
    trigger=dict(event_id='review-trigger',subject_id=sid,type='financial_deterioration',occurred_at=dates['warning'],known_at=dates['warning_known'])
    stale=cr.check_assessment_freshness(assessment,fp,[trigger],dates['warning_known'],dates['warning_known']).value
    history=deepcopy(request);history['row_id']='closed-history';history['known_at']=dates['closure']
    history['input']['lifecycle_events']=[dict(event_id='review',type='review',occurred_at=dates['review'],known_at=dates['review'],reason='Financial statements checked',gates={'checked':True}),dict(event_id='approval',type='finalize',occurred_at=dates['finalize'],known_at=dates['finalize'],reason='Local approval',gates={'approved':True}),dict(event_id='exit',type='exit',occurred_at=dates['closure'],known_at=dates['closure'],reason=data['observed']['exit_reason'],gates={'settled':True})]
    event=dict(event_id='credit-review',revision='1',occurred_at=dates['warning'],known_at=dates['warning_known'],dependencies=[dict(type='subject',id=sid)],payload={'review_required':True})
    default_event=dict(event_id='credit-default',revision='1',occurred_at=dates['default'],known_at=dates['default_known'],dependencies=[dict(type='subject',id=sid)],payload={'default_observed':True})
    delinquency=dict(event_id='credit-delinquency',revision='1',occurred_at=dates['delinquency'],known_at=dates['delinquency_known'],dependencies=[dict(type='subject',id=sid)],payload=dict(due_date=dates['payment_due'],scheduled_amount=data['observed']['scheduled_due'],received_amount=data['observed']['received'],days_past_due=(date.fromisoformat(dates['delinquency'])-date.fromisoformat(dates['payment_due'])).days))
    proposal=dict(event_id='restructuring-proposal',revision='1',occurred_at=dates['proposal'],known_at=dates['proposal'],dependencies=[dict(type='subject',id=sid)],payload=dict(treatment_id='extend',proposal_only=True,contract_changed=False))
    history['input']['events']=[event,event,delinquency,proposal,default_event]
    row=cr.rate_subject_batch([history],registry,sid+'-exit',dates['closure']).value['rows'][0]
    assert row['status']=='OK',row
    replay=cr.replay_assessment_history([request,history],registry,[original,row],dates['replay']).value
    tr=dict(X=[[0],[0],[1],[1]],y=[0,1,0,1],feature_order=data['feature_order']);later_training=data['later_training']
    later_model=cr.fit_binary_glm(tr['X'],tr['y'],weights=[8,2,6,4],feature_order=tr['feature_order'],link='logit',fit_intercept=True,target=target,training=later_training,model_id=mid,model_version='2')
    later_registry=cr.register_rating_model_version(registry,dict(model=later_model.model,dependencies=dependencies,configuration=dict(assessment_recipe=recipe)),dict(model_id=mid,model_version='2',registered_at=dates['later_registration'])).value
    later_replay=cr.replay_assessment_history([request,history],later_registry,[original,row],dates['later_replay']).value
    assert later_replay['status']=='REPRODUCED'
    tampered=deepcopy(row);tampered['event_state_hash']='changed'
    rejected=cr.replay_assessment_history([history],registry,[tampered],dates['replay']).value
    assert replay['status']=='REPRODUCED'and rejected['status']=='NOT_REPRODUCIBLE'
    assert row['value']['assessment']['state']=='SUPERSEDED'
    assert [x['status']for x in row['value']['event_trace']]==['APPLIED','DUPLICATE','APPLIED','APPLIED','APPLIED']
    assert freshness['status']=='CURRENT'and stale['status']=='STALE'
    summary=dict(origination_decision=original['value']['decision']['action'],credit_gate_inputs=data['gate'],product=data['product'],case_id=sid,source_component=data['source_component'],source_case_id=data['source_case_id'],assessed_scope='CALIBRATED_STANDALONE_BEFORE_SUPPORT_TRANSFER',subject_count=len(subjects['subjects']),selection=selection['status'],initial_state=assessment['state'],final_state=row['value']['assessment']['state'],current=freshness['status'],review_required=stale['status'],event_statuses=[x['status']for x in row['value']['event_trace']],replay=replay['status'],replay_after_new_champion=later_replay['status'],tampered_replay=rejected['status'],origination_pd=assessment['components'][0]['value'],historical_pd=row['value']['assessment']['components'][0]['value'])
    return summary,dict(registry=registry,requests=[request,history],results=[original,row])


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
