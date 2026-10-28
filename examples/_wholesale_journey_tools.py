"""Shared actual historical replay and observed cohort comparisons for reference journeys."""
from copy import deepcopy
import creditratingsystem as cr

def _replay(audit,source,policy=None):
    registry=audit['registry'];request=deepcopy(audit['requests'][1]);cutoff=request['known_at'];sid=request['input']['snapshot']['subject_id'];seen={};log=[]
    for e in request['input']['events']:
        seen.setdefault(e['event_id'],len(seen));log.append(dict(event_id=e['event_id'],idempotency_key=e['event_id'],occurred_at=e['occurred_at'],known_at=e['known_at'],sequence=seen[e['event_id']],payload=e['payload'],payload_hash=cr.content_fingerprint(e['payload']).value,revision_of=None))
    request['input']['events']=sorted(request['input']['events'],key=lambda e:(e['known_at'],seen[e['event_id']],e['event_id']))
    original=cr.rate_subject_batch([request],registry,sid+'-ordered-original',cutoff).value['rows'][0]
    replayed=cr.rate_subject_batch([deepcopy(request)],registry,sid+'-ordered-replay',cutoff).value['rows'][0]
    assert original['status']==replayed['status']=='OK'
    entry=registry['entries'][0];bundle=entry['bundle'];known=entry['registered_at'];snapshots=[]
    artifacts=bundle.get('dependencies',{}).get('artifacts',{});calibration=artifacts.get('calibrator',{}).get('content',dict(applied=False,reference='This frozen recipe has no separate PD calibration step'));policy=artifacts.get('credit_policy',{}).get('content',policy)
    if policy is None:raise ValueError('Actual decision policy required for replay snapshot')
    for kind,content in [('MODEL',bundle['model']),('CALIBRATION',calibration),('POLICY',policy),('SOURCE',source)]:
        available=source.get('known_at',known)if kind=='SOURCE'else known
        snapshots.append(dict(snapshot_id=kind,kind=kind,content_hash=cr.content_fingerprint(content).value,known_at=available,effective_from=available,effective_until=None))
    base=dict(checkpoint_id=sid+'-closed',as_of=cutoff,known_at=cutoff,event_log_hash=cr.content_fingerprint(log).value,snapshot_hashes={s['snapshot_id']:s['content_hash']for s in snapshots})
    def checkpoint(row):return dict(base,state=row['value'],applied_event_ids=[e['event_id']for e in row['value']['event_trace']if e['status']=='APPLIED'])
    contract=dict(contract_id=sid+'-exact-replay',version='1',reference='Two actual executions of the frozen historical registry after stable knowledge-ordering; identical duplicate deliveries retained',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=[base['checkpoint_id']],absolute_tolerance=1e-12,relative_tolerance=0)
    args=dict(event_log=log,original_checkpoints=[checkpoint(original)],replayed_checkpoints=[dict(checkpoint(replayed),mode='REPLAY')],snapshot_manifest=snapshots,replay_contract=contract)
    result=cr.replay_consistency(**args).value;changed=deepcopy(args);changed['replayed_checkpoints'][0]['state']['assessment']['components'][0]['value']+=.01;negative=cr.replay_consistency(**changed).value
    assert result['decision']['status']=='PASS'and negative['decision']['status']=='BLOCKED'
    return dict(result=result,negative=negative,inputs=args)


def _prediction_review(case,model,observed,reference_id='calibrated'):
    source=deepcopy(observed['monitoring']);pd=cr.predict_pd(model,case['features'],feature_order=case['feature_order'],expected_target=case['target']).value[0]
    entries={};known=model.value['training']['known_at'];mh=cr.content_fingerprint(model.value).value
    for identity,value in [('raw',pd),(reference_id,case['reference_pd']),('reviewed',case['reference_pd'])]:
        args=deepcopy(source)
        for row in args['predictions']:row.update(model_id=identity,forecast_id=row['case_id']+'-'+identity,pd=value,prediction_origin='EXTERNAL')
        cohort=cr.build_monitoring_cohorts(**args).value
        entries[identity]=dict(cohort=cohort,model_version='1',parameter_version='1',model_fingerprint=mh if identity=='raw'else cr.content_fingerprint(dict(model=mh,calibrator=case['calibrator'])).value,training_known_at=known)
    horizon=source['target_contract']['horizon_days'];design=dict(model_id='raw',model_version='1',parameter_version='1',data_fingerprint=cr.content_fingerprint(source).value,population_id=case['case_id']+'-actual-observed-cohort',split_id='out_of_time_2024',prediction_origin='EXTERNAL',evaluation_cutoff=source['evaluation_cutoff'],target=dict(case['target'],horizon=horizon,time_unit='days',conditioning='performing'))
    benchmark=cr.benchmark_predictions({k:v for k,v in entries.items()if k!='reviewed'},reference_model_id='raw',metrics=['brier'],matching_policy='strict',multiple_comparisons='none',design=design,uncertainty='none').value
    override_design=dict(design,model_id=reference_id);override=cr.override_monitoring(entries[reference_id],entries['reviewed'],[],policy=dict(approved_authorities=['credit_committee'],reference='No actual PD override recorded in this servicing cohort; unchanged review is explicitly audited',concentration_threshold=.6),design=override_design,metrics=['brier'],uncertainty='none').value
    assert override['metrics']['overridden_cases']==0
    return dict(benchmark=benchmark,override=override,source_model_hash=mh,raw_pd=pd,reference_pd=case['reference_pd'],reference_pipeline=reference_id,original_model_target=case['target'],interpretation='Frozen raw and declared reference prediction pipelines compared on identical genuinely observed cases; no override invented')

