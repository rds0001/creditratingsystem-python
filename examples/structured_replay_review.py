"""Actual historical recomputation after source corrections and model replacement."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def review_case(product, case):
    origin, revised = '2024-01-01', '2024-01-16'
    model=case['model'];subject=dict(subject_id=case['subject_id'],subject_type=case['subject_type'])
    bundle=dict(model=model,dependencies={},configuration=dict(source_recipe_reference=case['source_recipe_reference']))
    registry=cr.register_rating_model_version(None,bundle,dict(model_id=model['model_id'],model_version='v1',registered_at='2023-12-31')).value
    mh=registry['entries'][0]['model_hash'];policy=cr.compile_credit_decision_policy(case['policy']).value
    def request(x,known):
        return dict(row_id=product+'-'+known,known_at=known,model_id=model['model_id'],model_version='v1',model_hash=mh,recipe=dict(method='binary_glm',version='1.0',parameters=dict(expected_target=model['target'])),input=dict(X=[[x]],feature_order=['x']))
    requests=[request(0,origin),request(math.log(3),revised)]
    events=[dict(event_id=product+'-source',revision='1',occurred_at=origin,known_at=origin,dependencies=[dict(type=subject['subject_type'],id=subject['subject_id'])],payload=dict(source_input=requests[0]['input'])),dict(event_id=product+'-source',revision='2',corrects_revision='1',occurred_at=origin,known_at=revised,dependencies=[dict(type=subject['subject_type'],id=subject['subject_id'])],payload=dict(source_input=requests[1]['input']))]
    deliveries=[events[0],events[0],events[1]]
    validation_log=[]
    for event in deliveries:
        eid=event['event_id']+':'+event['revision']
        validation_log.append(dict(event_id=eid,idempotency_key=eid,occurred_at=event['occurred_at'],known_at=event['known_at'],sequence=int(event['revision']),payload=event['payload'],payload_hash=cr.content_fingerprint(event['payload']).value,revision_of=None if event['revision']=='1'else event['event_id']+':1'))
    snapshots=[]
    def snapshot(identity,kind,content,known):
        row=dict(snapshot_id=product+'-'+identity,kind=kind,content_hash=cr.content_fingerprint(content).value,known_at=known,effective_from=origin,effective_until=None)
        snapshots.append(row);return row
    common=[snapshot('model','MODEL',bundle,'2023-12-31'),snapshot('calibration','CALIBRATION',dict(method='NO_POST_CALIBRATION',version='1',target=model['target']),'2023-12-31'),snapshot('policy','POLICY',policy,'2023-12-31')]
    sources=[snapshot('source-'+str(i),'SOURCE',r['input'],r['known_at'])for i,r in enumerate(requests)]
    log_hash=cr.content_fingerprint(validation_log).value
    dependencies=[dict(assessment_id=product+'-source-review',dependency_type=subject['subject_type'],dependency_id=subject['subject_id'])]
    def execute(index, active_registry, mode):
        req=requests[index];known=req['known_at'];row=cr.rate_subject_batch([req],active_registry,product+'-single',known).value['rows'][0]
        assert row['status']=='OK',row
        pd=row['value'][0]
        assessment=cr.compose_risk_assessment(product+'-source-review',subject,[dict(component_id='pd',quantity='PD',value=pd,**subject,variant='corrected_historical_input',definition=model['target'])],dict(profile_id=product,profile_version='1',model_id=model['model_id'],model_version='v1',model_hash=mh,input_hash=row['input_hash']),origin,known,known).value
        timing=dict(as_of=origin,known_at=known);approval=case['policy']['approval_contract']
        decision=cr.apply_credit_decision_policy(policy,risk=dict(timing,pd=pd,target=model['target'],freshness='CURRENT'),affordability=dict(timing,status='NOT_APPLICABLE',domain=approval['domain'],applicability_reference=approval['affordability']['reference'],valid_until='2024-02-01'),eligibility=dict(timing,status='ELIGIBLE'),operations=dict(timing,status='CLEAR',legal_status='VERIFIED'),as_of=origin,known_at=known).value
        state=None;applied=[];statuses=[]
        for event in deliveries:
            result=cr.apply_assessment_event(state,event,dependencies,known).value;state=result['state'];statuses.append(result['status'])
            if result['status']=='APPLIED':applied.append(event['event_id']+':'+event['revision'])
        checkpoint=dict(checkpoint_id=product+'-'+str(index),as_of=origin,known_at=known,snapshot_hashes={s['snapshot_id']:s['content_hash']for s in common+[sources[index]]},applied_event_ids=applied,event_log_hash=log_hash,state=dict(assessment=assessment,decision=decision,event_state=state,event_statuses=statuses),mode=mode)
        return checkpoint,row
    original=[execute(i,registry,'ORIGINAL')for i in range(2)]
    newer=deepcopy(bundle);newer['model']['model_version']='v2';newer['model']['parameters']['coefficients'][0]=10
    later_registry=cr.register_rating_model_version(registry,newer,dict(model_id=model['model_id'],model_version='v2',registered_at='2024-02-01')).value
    replayed=[execute(i,later_registry,'REPLAY')for i in range(2)]
    contract=dict(contract_id='actual-structured-replay',version='1',reference='same-pinned-model-policy-source-and-no-post-calibration',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=[x[0]['checkpoint_id']for x in original],absolute_tolerance=0,relative_tolerance=0)
    audit=cr.replay_consistency(validation_log,[x[0]for x in original],[x[0]for x in replayed],snapshot_manifest=snapshots,replay_contract=contract).value
    historical=cr.replay_assessment_history(requests,later_registry,[x[1]for x in original],'2024-03-01').value
    bad=deepcopy(requests[0]);bad['row_id']='invalid-'+product;bad['model_hash']='0'*64
    batch=cr.rate_subject_batch(requests+[bad],registry,product+'-batch',revised).value
    permuted=cr.rate_subject_batch([bad]+list(reversed(requests)),registry,product+'-batch',revised).value
    early=cr.rate_subject_batch([requests[1]],registry,product+'-early',origin).value
    mutated=deepcopy(replayed[0][0]);mutated['state']['assessment']['components'][0]['value']=.9
    negative=cr.replay_consistency(validation_log,[x[0]for x in original],[mutated,replayed[1][0]],snapshot_manifest=snapshots,replay_contract=contract).value
    return dict(original_checkpoints=[x[0]for x in original],actual_replay_checkpoints=[x[0]for x in replayed],audit=audit,negative_audit=negative,historical_replay=historical,batch=batch,batch_permutation_exact=batch==permuted,early_revision=early,event_log=validation_log,snapshots=snapshots,interpretation='The January estimate is restated using a later-known source correction, not overwritten. No post-calibration is applied; its explicit versioned identity contract is retained. Model replacement does not change pinned historical execution.')


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('structured_governance.json').read_text())
    out={product:review_case(product,case)for product,case in data['variants'].items()}
    if verify_reference:
        for value in out.values():
            assert [c['state']['assessment']['components'][0]['value']for c in value['original_checkpoints']]==[.5,.75]
            assert value['audit']['decision']['status']=='PASS'and value['negative_audit']['decision']['status']=='BLOCKED'
            assert value['historical_replay']['status']=='REPRODUCED'and value['batch_permutation_exact']
            assert value['early_revision']['rows'][0]['error_code']=='RATING_REQUEST_UNAVAILABLE'
            assert [c['state']['event_statuses']for c in value['original_checkpoints']]==[['APPLIED','DUPLICATE','DEFERRED'],['APPLIED','DUPLICATE','APPLIED']]
    return dict(products=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
