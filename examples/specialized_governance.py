"""Local frozen transaction replay and separately observed specialist servicing."""
from importlib.resources import files
from copy import deepcopy
import json
import math
import creditratingsystem as cr


def _ref(source,identity,*path):return {'$ref':dict(source=source,id=identity,path=list(path))}


def _journey(j,model,later_model,verify,return_audit_artifacts=False):
    sid=j['case_id'];origin='2023-01-01';aid=sid+'-assessment';target=j['simulation_target'];facts=dict(product=j['product'],variant=j['variant'],legal_verified=True)
    rules=[dict(rule_id=sid,profile_id=sid,profile_version='1',priority=1,effective_from=origin,effective_to=None,known_at=origin,conditions=[dict(field=k,operator='eq',value=v)for k,v in facts.items()])]
    applicability=dict(contract_id=sid,version='1',required_features=list(facts),allowed_values={k:[v]for k,v in facts.items()},support_bounds={},maximum_age_days={k:90 for k in facts})
    pdpath=['transactions',0,'risk','pd'];elpath=['transactions',0,'risk','expected_loss'];targetpath=['transactions',0,'risk','target']
    step=dict(step_id='transaction',method_id='simulation.simulate_transactions',arguments={k:_ref('artifact',k)for k in j['frozen_arguments']},argument_types={},output_contract=dict(shape='object',unit='transaction_paths',target=None,fields=[dict(path=pdpath,unit='probability',target=target,target_path=targetpath),dict(path=elpath,unit='EUR',target=target,target_path=targetpath)]))
    fields={'risk.pd':dict(type='number')}
    for name,values in [('affordability.status',['AFFORDABLE','NOT_AFFORDABLE']),('eligibility.status',['ELIGIBLE','INELIGIBLE']),('operations.status',['CLEAR','HOLD'])]:fields[name]=dict(type='category',values=values)
    policy_spec=dict(schema_version='1.1',policy_id=sid+'-policy',version='1',approval_reference='Synthetic local credit review',known_at=origin,effective_from=origin,effective_until='2024-01-01',default_action='REFER',missing_action='BLOCKED',target=target,fields=fields,approval_contract=dict(domain='WHOLESALE_CREDIT',affordability=dict(mode='REQUIRED'),required_gates=[dict(field=k,value=v)for k,v in [('affordability.status','AFFORDABLE'),('eligibility.status','ELIGIBLE'),('operations.status','CLEAR')]]),rules=[dict(rule_id='bounded-risk',priority=1,action='APPROVE',reason_code='RISK_AND_DOCUMENTED_GATES',when=[dict(field='risk.pd',op='le',value=.11)])])
    policy=cr.compile_credit_decision_policy(policy_spec).value
    timing=dict(as_of=origin,known_at=origin);gate=j['origination_gate']
    decision=dict(policy=_ref('artifact','credit_policy'),risk=dict(timing,pd=_ref('step','transaction','value',*pdpath),target=target,freshness='CURRENT'),affordability=dict(timing,status='AFFORDABLE'if gate['available_cashflow']>=gate['debt_service']else'NOT_AFFORDABLE'),eligibility=dict(timing,status='ELIGIBLE'if gate['legal_verified']else'INELIGIBLE'),operations=dict(timing,status=gate['operations']),**timing)
    components=[dict(component_id='pd',quantity='PD',value=_ref('step','transaction','value',*pdpath),subject_id=sid,subject_type='obligor',variant='cash_pool',definition=target),dict(component_id='el',quantity='EL',value=_ref('step','transaction','value',*elpath),subject_id=sid,subject_type='obligor',variant='before_payment_guarantee',definition=target,currency='EUR')]
    transitions=[dict(from_state='DRAFT',to_state='REVIEWED',event_type='review',required_gates=['checked']),dict(from_state='REVIEWED',to_state='FINAL',event_type='finalize',required_gates=['approved']),dict(from_state='FINAL',to_state='SUPERSEDED',event_type='exit',required_gates=['settled'])]
    recipe=dict(method='assessment_pipeline',version='2.0',parameters=dict(routing_rules=rules,applicability_contract=applicability,steps=[step],components=components,effects=[],transitions=transitions,event_dependencies=[dict(assessment_id=aid,dependency_type='subject',dependency_id=sid)],decision=decision))
    artifacts={k:dict(kind='innovations'if k=='entity_innovations'else'parameters',content=v,sha256=cr.content_fingerprint(v).value,known_at=origin)for k,v in j['frozen_arguments'].items()}
    artifacts['credit_policy']=dict(kind='parameters',content=policy,sha256=cr.content_fingerprint(policy).value,known_at=origin)
    bundle=dict(model=model,dependencies=dict(artifacts=artifacts),configuration=dict(assessment_recipe=recipe))
    registration=dict(model_id=model['model_id'],model_version=model['model_version'],registered_at=origin)
    registry=cr.register_rating_model_version(None,bundle,registration).value;digest=registry['entries'][0]['model_hash']
    assignment=dict(assignment_id=sid+'-champion',profile_id=sid,purpose='CASHFLOW_SIMULATION',model_id=model['model_id'],model_version='1',model_hash=digest,effective_from=origin,effective_to=None,known_at=origin)
    selected=cr.select_effective_model(registry,[assignment],sid,'CASHFLOW_SIMULATION',origin,origin).value
    attributes={k:dict(value=v,observed_at=origin,known_at=origin)for k,v in facts.items()}
    subjects=cr.build_rating_subjects({'obligor':[dict(subject_id=sid,version='1',attributes=attributes,effective_from=origin,effective_to=None,known_at=origin)]},[],origin,origin).value
    request=dict(row_id='origination',known_at='2023-01-02',model_id=model['model_id'],model_version='1',model_hash=digest,recipe=recipe,input=dict(snapshot=dict(subject_id=sid,subject_type='obligor',as_of=origin,known_at=origin,attributes=attributes),datasets={},assessment_id=aid,assessment_known_at=origin,calculated_at='2023-01-02',lifecycle_events=[],events=[]))
    original=cr.rate_subject_batch([request],registry,sid+'-origination','2023-01-02').value['rows'][0]
    assert original['status']=='OK',original
    observation=j['observed'];history=deepcopy(request);history['row_id']='closed-history';history['known_at']=observation['closure_known_at']
    history['input']['lifecycle_events']=[dict(event_id='review',type='review',occurred_at='2023-01-03',known_at='2023-01-03',reason='Cashflow, rights and legal pooling reviewed',gates={'checked':True}),dict(event_id='final',type='finalize',occurred_at='2023-01-04',known_at='2023-01-04',reason='Local approval; separate funding instruction required',gates={'approved':True}),dict(event_id='exit',type='exit',occurred_at=observation['closure_at'],known_at=observation['closure_known_at'],reason=observation['exit_reason'],gates={'settled':True})]
    def event(identity,occurred,known,payload):return dict(event_id=identity,revision='1',occurred_at=occurred,known_at=known,dependencies=[dict(type='subject',id=sid)],payload=payload)
    missed=event('missed-coupon',observation['scheduled_due_date'],observation['early_warning_at'],dict(scheduled_due=observation['scheduled_due'],received=observation['received'],source=observation['source']))
    history['input']['events']=[missed,missed,event('delinquency',observation['delinquency_at'],observation['delinquency_at'],dict(days_past_due=observation['delinquency_days'],not_yet_default=True)),event('restructuring-proposal',observation['proposal_at'],observation['proposal_at'],dict(proposal_only=True,contract_changed=False)),event('observed-default',observation['default_at'],observation['default_known_at'],dict(material_days_past_due=observation['default_days_past_due'],default_observed=True))]
    closed=cr.rate_subject_batch([history],registry,sid+'-closed',history['known_at']).value['rows'][0];assert closed['status']=='OK',closed
    replay=cr.replay_assessment_history([request,history],registry,[original,closed],'2025-02-01').value
    later_bundle=deepcopy(bundle);later_bundle['model']=later_model
    later_registry=cr.register_rating_model_version(registry,later_bundle,dict(model_id=model['model_id'],model_version='2',registered_at='2024-01-03')).value
    later_replay=cr.replay_assessment_history([request,history],later_registry,[original,closed],'2025-02-01').value
    tampered=deepcopy(closed);tampered['event_state_hash']='tampered'
    rejected=cr.replay_assessment_history([history],registry,[tampered],'2025-02-01').value
    assessment=original['value']['assessment'];by_id={c['component_id']:c for c in assessment['components']};freshness=dict(maximum_age_days=90,maximum_information_age_days=90,trigger_types=['missed_payment'])
    current=cr.check_assessment_freshness(assessment,freshness,[],'2023-02-01','2023-02-01').value
    trigger=dict(event_id='missed',subject_id=sid,type='missed_payment',occurred_at='2024-01-01',known_at='2024-01-02')
    stale=cr.check_assessment_freshness(assessment,freshness,[trigger],'2024-01-02','2024-01-02').value
    if verify:
        assert replay['status']==later_replay['status']=='REPRODUCED'and rejected['status']=='NOT_REPRODUCIBLE'
        assert original['value']['decision']['action']=='APPROVE'and closed['value']['assessment']['state']=='SUPERSEDED'
        assert [x['status']for x in closed['value']['event_trace']]==['APPLIED','DUPLICATE','APPLIED','APPLIED','APPLIED']
        assert math.isclose(by_id['pd']['value'],j['reference_pd'],abs_tol=1e-12)
        assert math.isclose(by_id['el']['value'],j['reference_el'],abs_tol=1e-12)
        assert current['status']=='CURRENT'and stale['status']=='STALE'
    report=dict(product=j['product'],variant=j['variant'],subject_count=len(subjects['subjects']),selection=selected['status'],decision=original['value']['decision']['action'],initial_state=assessment['state'],final_state=closed['value']['assessment']['state'],current=current['status'],stale=stale['status'],event_statuses=[x['status']for x in closed['value']['event_trace']],pd=by_id['pd']['value'],expected_loss=by_id['el']['value'],original_result_hash=original['result_hash'],historical_result_hash=closed['result_hash'],replay=replay['status'],replay_after_later_fit=later_replay['status'],tampered_replay=rejected['status'],observed_servicing=observation)

    return dict(report=report,registry=registry,requests=[request,history],results=[original,closed])if return_audit_artifacts else report

def run(data=None,verify_reference=True,model_path=None,return_audit_artifacts=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_governance.json').read_text())
    if model_path is None:model_path=files('creditratingsystem.datasets').joinpath(data['model_resource'])
    model=cr.load_fitted_model(model_path).value
    history=deepcopy(data['macro_history']);history['dates'].append('2024-01-01');history['release_dates'].append(['2024-01-01']);history['values'].append([.5])
    training=dict(data['macro_training'],end='2024-01-01',known_at='2024-01-01',split_id='later_macro_2016_2024')
    later=cr.fit_var(history,lags=1,training=training,target=data['macro_target'],model_id=model['model_id'],model_version='2',covariance_estimator='mle').model
    return dict(data_origin='SYNTHETIC',frozen_generation_model_hash=cr.content_fingerprint(model).value,scope='Local transaction generation provenance, frozen simulation and observed servicing; no validation labels from simulated paths.',journeys={j['case_id']:_journey(j,model,later,verify_reference,return_audit_artifacts)for j in data['journeys']})


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
