"""Trace new 2024 claims through the existing validated lifecycle components."""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import runpy,json,math
import creditratingsystem as cr

def run(data=None,verify_reference=True):
    resource=files('creditratingsystem.datasets')
    load=lambda n:json.loads(resource.joinpath(n+'.json').read_text())
    if data is None:data=load('structured_journey')
    helpers={n:runpy.run_path(str(Path(__file__).with_name(n+'.py')))for n in ['input_lineage','structured_governance','structured_lifecycle','structured_workout','structured_pool_review']}
    lineage=helpers['input_lineage']['prepare_snapshot'](resource.joinpath('structured_positions.csv'),resource.joinpath('structured_releases.csv'),data['lineage_config'],data['lineage_config']['cutoffs'][0])
    assert lineage['audit']['decision']['status']=='PASS'
    gd=load('structured_governance');ld=load('structured_lifecycle');wd=load('structured_workout')
    for dataset in (gd,ld,wd):dataset['variants']={p:dataset['variants'][p]for p in data['products']}
    for row in lineage['rows']:
        c=gd['variants'][row['product_id']];assert c['subject_id']==row['entity_id'] and row['x']==0 and row['balance']==data['pools']['cases'][row['product_id']]['position_binding']['position_ead']
        c['snapshot']['attributes']['x']['value']=row['x']
        # The old economic example is only a supplemental contract reference.
        c['source_recipe_reference']=dict(kind='OWN_2024_COLLATERAL_LEDGER',position_id=c['subject_id'],source_snapshot_hash=lineage['selected_snapshot_hash'],pool_contract_hash=cr.content_fingerprint(data['pools']['cases'][row['product_id']]['pool_input']).value)
    governance=helpers['structured_governance']['run'](gd)['variants']
    for p in data['products']:ld['variants'][p]['governance_assessment_hash']=governance[p]['assessment']['assessment_hash']
    lifecycle=helpers['structured_lifecycle']['run'](ld)['variants']
    workout=helpers['structured_workout']['run'](wd)['variants']
    pool_review=helpers['structured_pool_review']['run'](data['pools'],verify_reference=False)['products']
    records={};events=[];snapshots=[];checkpoints=[]
    for p in data['products']:
        c=data['pools']['cases'][p];binding=c['position_binding'];g=governance[p];w=workout[p];sid=binding['position_id']
        assert sid==gd['variants'][p]['subject_id']==ld['variants'][p]['subject_id']==wd['variants'][p]['subject_id']
        # These are claim-layer oracles, separately from household collateral LGD.
        observed=pool_review[p]['asset_pool_validation']['metrics']['observed_owned_flows']['recovery_gross']
        if p=='SECURITIZATION_TRANCHE':
            senior=min(observed,binding['priority_contract']['senior_principal']);junior=max(0,observed-senior)
            actual_claim_recovery=senior;claim_control=dict(senior_recovery=senior,junior_recovery=junior,cash_conservation=senior+junior-observed)
        elif p=='COVERED_BOND':
            residual=binding['position_ead']-observed;estate=min(residual,residual*binding['estate_recovery_fraction'])
            actual_claim_recovery=observed+estate;claim_control=dict(cover_recovery=observed,residual_claim=residual,estate_recovery=estate,aggregate_recovery=actual_claim_recovery)
        else:
            actual_claim_recovery=observed;claim_control=dict(debtor_recovery=observed,seller_recourse=0,dilution=0)
        assert math.isclose(actual_claim_recovery,binding['actual_position_recovery'],abs_tol=1e-12)
        assert math.isclose(binding['position_ead']-actual_claim_recovery,w['closed_workout']['economic_loss'],abs_tol=1e-12)
        assert not pool_review[p]['asset_pool_validation']['violations']
        # Event inputs and artifact identities preserve the same position throughout.
        raw=[('origin','2024-01-01',dict(assessment_hash=g['assessment']['assessment_hash'])),('performing','2024-01-10',dict(position_id=sid,status='PERFORMING')),('delay','2024-01-15',dict(unpaid_collateral_principal=20)),('proposal','2024-01-16',dict(status='PROPOSAL_ONLY',contract_changed=False)),('default','2024-02-01',dict(episode_id=wd['variants'][p]['episode']['episode_id'],ead=binding['position_ead'])),('settlement','2024-05-01',dict(position_recovery=actual_claim_recovery,closed_loss=w['closed_workout']['economic_loss']))]
        dependencies=[dict(assessment_id=g['assessment']['assessment_id'],dependency_type=gd['variants'][p]['subject_type'],dependency_id=sid)]
        state=None;history=[]
        for i,(kind,date,payload)in enumerate(raw):
            event=dict(event_id=p+'-'+kind,revision='1',occurred_at=date,known_at=date,dependencies=[dict(type=gd['variants'][p]['subject_type'],id=sid)],payload=payload)
            execution=cr.apply_assessment_event(state,event,dependencies,'2024-05-02').value;state=execution['state'];history.append(event)
            events.append(dict(event_id=event['event_id'],idempotency_key=event['event_id'],occurred_at=date,known_at=date,sequence=i,payload=payload,payload_hash=cr.content_fingerprint(payload).value,revision_of=None))
        snapshots.append(dict(snapshot_id=p+'-contracts',kind='SOURCE',content_hash=cr.content_fingerprint(dict(governance=gd['variants'][p],collateral=c,lifecycle=ld['variants'][p],workout=wd['variants'][p])).value,known_at='2024-01-01',effective_from='2024-01-01',effective_until=None))
        cohort_input=deepcopy(gd['variants'][p]['cohort_input'])
        forecast=cohort_input['predictions'][0]
        forecast.update(forecast_id=p+'-main-performing',case_id=p+'-main',entity_id=sid,cluster_id=sid,forecast_at='2024-01-15',known_at='2024-01-15',pd=g['performing_assessment']['components'][0]['value'])
        cohort_input.update(predictions=[forecast],outcome_events=[dict(event_id=p+'-actual-default',entity_id=sid,event_at='2024-02-01',known_at='2024-02-02',event='default',default_definition=cohort_input['target_contract']['default_definition'])],coverage=[dict(entity_id=sid,complete_through='2024-05-01',known_at='2024-05-02')],evaluation_cutoff='2024-05-02')
        cohort=cr.build_monitoring_cohorts(**cohort_input).value
        main_monitoring=cr.monitor_rating_cohort(cohort,group_by=['cohort'],reference_window=dict(start='2024-01-01',end='2024-01-31'),minimum_count=1,uncertainty='none').value
        assert main_monitoring[0]['metrics']['observed_rate']==1 and main_monitoring[0]['metrics']['brier']==.25
        records[p]=dict(position_binding=binding,source_row=next(r for r in lineage['rows']if r['product_id']==p),main_cohort_input=cohort_input,main_cohort=cohort,main_monitoring=main_monitoring,governance=g,lifecycle=lifecycle[p],workout=w,pool_validation=pool_review[p],claim_control=claim_control,event_state=state,events=history)
    # Actually recalculate all component outputs before validating their replay.
    # Observed outcomes are not forecast ancestors: separate artifact snapshots below.
    for snap,p in zip(snapshots,data['products']):
        snap['known_at']='2024-05-02';snap['effective_from']='2024-05-02'
    replay_g=helpers['structured_governance']['run'](gd)['variants']
    replay_l=helpers['structured_lifecycle']['run'](ld)['variants']
    replay_w=helpers['structured_workout']['run'](wd)['variants']
    replay_p=helpers['structured_pool_review']['run'](data['pools'],verify_reference=False)['products']
    assert governance==replay_g and lifecycle==replay_l and workout==replay_w and pool_review==replay_p
    # Confidence interval diagnostics remain in the full reports; they are not
    # operational decision state. Full within-language reexecution is exact above.
    decision_state=lambda g:{k:v for k,v in g.items()if k not in ('feature_drift','calibration_drift')}
    original=[];replayed=[];audits={};all_snapshots=[]
    for p,snap in zip(data['products'],snapshots):
        state=None;deps=[dict(assessment_id=governance[p]['assessment']['assessment_id'],dependency_type=gd['variants'][p]['subject_type'],dependency_id=gd['variants'][p]['subject_id'])]
        for event in records[p]['events']:state=cr.apply_assessment_event(state,event,deps,'2024-05-02').value['state']
        replay_cohort=cr.build_monitoring_cohorts(**records[p]['main_cohort_input']).value
        replay_monitoring=cr.monitor_rating_cohort(replay_cohort,group_by=['cohort'],reference_window=dict(start='2024-01-01',end='2024-01-31'),minimum_count=1,uncertainty='none').value
        assert replay_cohort==records[p]['main_cohort'] and replay_monitoring==records[p]['main_monitoring']
        event_log=[e for e in events if e['event_id'].startswith(p+'-')]
        local_snapshots=[snap]
        for kind,content in [('MODEL',gd['variants'][p]['model']),('CALIBRATION',dict(method='NO_POST_CALIBRATION',version='1',target=gd['variants'][p]['model']['target'])),('POLICY',governance[p]['policy'])]:
            local_snapshots.append(dict(snapshot_id=p+'-'+kind,kind=kind,content_hash=cr.content_fingerprint(content).value,known_at='2023-12-31',effective_from='2024-01-01',effective_until=None))
        all_snapshots.extend(local_snapshots)
        base=dict(checkpoint_id=p+'-settled',as_of='2024-05-02',known_at='2024-05-02',snapshot_hashes={x['snapshot_id']:x['content_hash']for x in local_snapshots},applied_event_ids=[e['event_id']for e in records[p]['events']],event_log_hash=cr.content_fingerprint(event_log).value)
        original.append(dict(base,mode='ORIGINAL',state=dict(governance=decision_state(governance[p]),lifecycle=lifecycle[p],workout=workout[p],pool=pool_review[p],main_monitoring=records[p]['main_monitoring'],events=records[p]['event_state'])))
        replayed.append(dict(base,mode='REPLAY',state=dict(governance=decision_state(replay_g[p]),lifecycle=replay_l[p],workout=replay_w[p],pool=replay_p[p],main_monitoring=replay_monitoring,events=state)))
        contract=dict(contract_id=p+'-actual-2024-position-history',version='1',reference='Actual component reexecution with separate forecast and realized servicing',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=[base['checkpoint_id']],absolute_tolerance=0,relative_tolerance=0)
        audits[p]=cr.replay_consistency(event_log,original[-1:],replayed[-1:],snapshot_manifest=local_snapshots,replay_contract=contract).value
    if verify_reference:
        assert all(a['decision']['status']=='PASS'for a in audits.values()),audits
        for p,r in records.items():
            assert r['governance']['decision']['action']=='APPROVE'
            assert r['lifecycle']['observation_windows'][0]['target']==1
            assert r['workout']['incomplete_workout']['economic_lgd']is None
            assert r['governance']['retired']['state']=='SUPERSEDED'
            assert r['pool_validation']['asset_pool_validation']['metrics']['default_probability']['brier']>.63
        assert records['SECURITIZATION_TRANCHE']['claim_control']['senior_recovery']==25
        assert records['COVERED_BOND']['claim_control']['estate_recovery']==32.5
        assert records['TRADE_RECEIVABLES']['claim_control']['debtor_recovery']==70
    return dict(full_component_reexecution_exact=True,products=records,lineage=lineage,replay_audits=audits,snapshots=all_snapshots,original_checkpoints=original,replayed_checkpoints=replayed,interpretation='New2024position and collateral ledgers bind governance, observed lifecycle and workouts. Historical2023product examples remain separate economic-contract references. The analytic30day position monitor and121day collateral default forecast are distinct targets; neither is treated as the other.')

if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
