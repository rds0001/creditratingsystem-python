"""Actual source-to-claim bindings, dated replay and honest unavailable benchmarks."""
from importlib.resources import files
from pathlib import Path
from copy import deepcopy
import json,runpy,math
import creditratingsystem as cr


def _replay(audit,source):
    registry=audit['registry'];request=deepcopy(audit['requests'][1]);cutoff=request['known_at'];sid=request['input']['snapshot']['subject_id'];seen={};log=[]
    for e in request['input']['events']:
        seen.setdefault(e['event_id'],len(seen));log.append(dict(event_id=e['event_id'],idempotency_key=e['event_id'],occurred_at=e['occurred_at'],known_at=e['known_at'],sequence=seen[e['event_id']],payload=e['payload'],payload_hash=cr.content_fingerprint(e['payload']).value,revision_of=None))
    request['input']['events']=sorted(request['input']['events'],key=lambda e:(e['known_at'],seen[e['event_id']],e['event_id']))
    original=cr.rate_subject_batch([request],registry,sid+'-ordered-original',cutoff).value['rows'][0]
    replayed=cr.rate_subject_batch([deepcopy(request)],registry,sid+'-ordered-replay',cutoff).value['rows'][0]
    assert original['status']==replayed['status']=='OK'
    entry=registry['entries'][0];bundle=entry['bundle'];known=entry['registered_at'];snapshots=[]
    for kind,content in [('MODEL',bundle['model']),('CALIBRATION',bundle['dependencies']['artifacts']['calibrator']['content']),('POLICY',bundle['dependencies']['artifacts']['credit_policy']['content']),('SOURCE',source)]:snapshots.append(dict(snapshot_id=kind,kind=kind,content_hash=cr.content_fingerprint(content).value,known_at=known,effective_from=known,effective_until=None))
    base=dict(checkpoint_id=sid+'-closed',as_of=cutoff,known_at=cutoff,event_log_hash=cr.content_fingerprint(log).value,snapshot_hashes={s['snapshot_id']:s['content_hash']for s in snapshots})
    def checkpoint(row):return dict(base,state=row['value'],applied_event_ids=[e['event_id']for e in row['value']['event_trace']if e['status']=='APPLIED'])
    contract=dict(contract_id=sid+'-exact-replay',version='1',reference='Two actual executions of the frozen historical registry after stable knowledge-ordering; identical duplicate deliveries retained',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=[base['checkpoint_id']],absolute_tolerance=1e-12,relative_tolerance=0)
    args=dict(event_log=log,original_checkpoints=[checkpoint(original)],replayed_checkpoints=[dict(checkpoint(replayed),mode='REPLAY')],snapshot_manifest=snapshots,replay_contract=contract)
    result=cr.replay_consistency(**args).value;changed=deepcopy(args);changed['replayed_checkpoints'][0]['state']['assessment']['components'][0]['value']+=.01;negative=cr.replay_consistency(**changed).value
    assert result['decision']['status']=='PASS'and negative['decision']['status']=='BLOCKED'
    return dict(result=result,negative=negative,inputs=args)


def _prediction_review(case,model,observed):
    source=deepcopy(observed['monitoring']);pd=cr.predict_pd(model,case['features'],feature_order=case['feature_order'],expected_target=case['target']).value[0]
    entries={};known=model.value['training']['known_at'];mh=cr.content_fingerprint(model.value).value
    for identity,value in [('raw',pd),('calibrated',case['reference_pd']),('reviewed',case['reference_pd'])]:
        args=deepcopy(source)
        for row in args['predictions']:row.update(model_id=identity,forecast_id=row['case_id']+'-'+identity,pd=value,prediction_origin='EXTERNAL')
        cohort=cr.build_monitoring_cohorts(**args).value
        entries[identity]=dict(cohort=cohort,model_version='1',parameter_version='1',model_fingerprint=mh if identity=='raw'else cr.content_fingerprint(dict(model=mh,calibrator=case['calibrator'])).value,training_known_at=known)
    horizon=source['target_contract']['horizon_days'];design=dict(model_id='raw',model_version='1',parameter_version='1',data_fingerprint=cr.content_fingerprint(source).value,population_id=case['case_id']+'-actual-observed-cohort',split_id='out_of_time_2024',prediction_origin='EXTERNAL',evaluation_cutoff=source['evaluation_cutoff'],target=dict(case['target'],horizon=horizon,time_unit='days',conditioning='performing'))
    benchmark=cr.benchmark_predictions({k:v for k,v in entries.items()if k!='reviewed'},reference_model_id='raw',metrics=['brier'],matching_policy='strict',multiple_comparisons='none',design=design,uncertainty='none').value
    override_design=dict(design,model_id='calibrated');override=cr.override_monitoring(entries['calibrated'],entries['reviewed'],[],policy=dict(approved_authorities=['credit_committee'],reference='No actual PD override recorded in this servicing cohort; unchanged review is explicitly audited',concentration_threshold=.6),design=override_design,metrics=['brier'],uncertainty='none').value
    assert override['metrics']['overridden_cases']==0
    return dict(benchmark=benchmark,override=override,source_model_hash=mh,raw_pd=pd,calibrated_pd=case['reference_pd'],interpretation='Raw and calibrated frozen prediction pipelines compared on identical genuinely observed cases; no override invented')


def run(data=None,verify_reference=True,accounts_path=None,releases_path=None,artifact_paths=None,return_artifacts=False):
    resource=files('creditratingsystem.datasets')
    if data is None:data=json.loads(resource.joinpath('wholesale_sector_journey_review.json').read_text())
    data=deepcopy(data);lineage=runpy.run_path(str(Path(__file__).with_name('input_lineage.py')))['prepare_snapshot'](accounts_path or resource.joinpath('wholesale_sector_positions.csv'),releases_path or resource.joinpath('wholesale_sector_releases.csv'),data['lineage_config'],data['lineage_config']['cutoffs'][0]);rows={r['entity_id']:r for r in lineage['rows']}
    for case in data['governance']['cases']:
        claim=next(x for x in data['claims']if x['case_id']==case['case_id']);source=rows[case['case_id']]
        if source['product_id']!=case['product']or source['balance']!=claim['source_balance']or source['x']!=case['features'][0][0]:raise ValueError('Source must bind the actual claim balance, product and frozen financial design coordinate')
        case['features']=[[source['x']]]
    governance=runpy.run_path(str(Path(__file__).with_name('wholesale_sector_governance.py')))['run'](data['governance'],artifact_paths=artifact_paths,return_audit_artifacts=True);out=[];artifacts=[]
    for case,summary,audit in zip(data['governance']['cases'],governance['report']['cases'],governance['audits']):
        adapter=case['validation_event_adapter']
        if adapter['source_target']!=case['target']or adapter['validator_target']!=dict(case['target'],event='default')or adapter['population_id']!=case['case_id']or adapter['evidence']['event_at']!=case['observed']['default_at']or (case['product']=='SOVEREIGN_DEBT'and not adapter['evidence']['available_unrestricted_cashflow']<adapter['evidence']['debt_service_due']):raise ValueError('Validation adapter must preserve target/population and bind observed insufficient payment capacity')
        sid=case['case_id'];claim=next(x for x in data['claims']if x['case_id']==sid);pd=summary['origination_pd'];decision=audit['results'][0]['value']['decision'];reason={r['reason_code']:'Declared policy rule '+r['rule_id']for r in decision['rule_trace']};explanation=cr.decompose_policy_decision(decision,reason).value
        benchmark=deepcopy(data['benchmark_template']);support=deepcopy(data['support_template']);horizon_end='2025-03-01';evaluation='2025-03-02';model_id=audit['registry']['entries'][0]['model_id']
        design=dict(model_id=model_id,model_version='1',parameter_version='1',data_fingerprint=cr.content_fingerprint(case['observed']).value,population_id=sid,split_id='out_of_time_2024',prediction_origin='EXTERNAL',evaluation_cutoff=evaluation,target=adapter['validator_target'],currency='EUR')
        common=dict(prediction_id=sid+'-origin',entity_id=sid,cluster_id=sid,segment=case['product'],as_of='2024-03-01',forecast_known_at='2024-03-01',horizon_end=horizon_end,model_id=model_id,model_version='1',parameter_version='1')
        internal=benchmark['internal_ratings'][0];internal.update(common,rating_unit_id=sid,grade='LOW'if pd<.25 else'MID'if pd<.75 else'HIGH');benchmark['design']=design
        record=support['records'][0];record.update(common,stage_predictions=dict(entity=pd,transfer=0,joint=0,final=pd,unsupported_counterfactual=pd));event=next(e for e in audit['requests'][1]['input']['events']if e['event_id']=='credit-default')
        outcome=dict(value=1,known_at=evaluation,complete_through=horizon_end,event_ids=[sid+':'+event['event_id']],event_times=[event['occurred_at']]);record['stage_outcomes']=dict(entity=outcome,final=deepcopy(outcome));support['design']=design
        for name,contract in support['stage_contract'].items():contract['reference']='Actual standalone claim ledger: no executed support or observed transfer restriction. Unobserved transfer/intersection/counterfactual labels remain unavailable.';contract['event_definition']=case['target']['event']if name in ['entity','final']else'unobserved-'+name
        b=cr.benchmark_ratings(**benchmark).value;s=cr.support_transfer_backtest(**support).value;replay=_replay(audit,rows[sid]);prediction_review=None
        if case['product']=='PUBLIC_SERVICE_FINANCE':
            name=case['model_resource'];model=cr.load_fitted_model(artifact_paths[name]if artifact_paths else resource.joinpath(name));prediction_review=_prediction_review(case,model,claim['observed_contract'])
        if verify_reference:
            assert lineage['audit']['decision']['status']=='PASS'and b['applicability']=='NOT_APPLICABLE'
            assert math.isclose(s['metrics']['by_stage']['entity']['brier'],(1-pd)**2,abs_tol=1e-12)
            assert s['metrics']['by_stage']['transfer']['applicability']=='NOT_APPLICABLE'
        out.append(dict(product=case['product'],case_id=sid,claim_id=claim['claim_id'],source_balance=rows[sid]['balance'],validation_event_adapter=adapter,original_target=case['target'],benchmark_event_definition=case['target']['event'],source_feature=rows[sid]['x'],source_snapshot_hash=lineage['selected_snapshot_hash'],governance=summary,policy_explanation=explanation,rating_benchmark=b,support_transfer=s,replay=replay['result'],negative_replay=replay['negative'],prediction_review=prediction_review,scope='Actual single-claim stage diagnostic; no external rating or causal support label fabricated'))
        artifacts.append(dict(case_id=sid,governance=audit,benchmark_inputs=benchmark,support_inputs=support,replay_inputs=replay['inputs']))
    report=dict(data_origin='SYNTHETIC',lineage=lineage['audit'],archive_sha256=lineage['archive_sha256'],cases=out)
    return dict(report=report,artifacts=artifacts,lineage_inputs=lineage['audit_inputs'])if return_artifacts else report

if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
