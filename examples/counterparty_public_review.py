"""Same actual counterparty claim, explicit absent benchmark and disabled transfer overlay."""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json,runpy
import creditratingsystem as cr

def run(data=None,verify_reference=True,return_artifacts=False,accounts_path=None,releases_path=None):
 resource=files('creditratingsystem.datasets');load=lambda n:json.loads(resource.joinpath(n+'.json').read_text())
 if data is None:data=load('counterparty_public_review')
 data=deepcopy(data);p=data['product'];sid=data['subject_id'];helpers={n:runpy.run_path(str(Path(__file__).with_name(n+'.py')))for n in ['input_lineage','structured_governance','structured_lifecycle','structured_workout','structured_replay_review','counterparty_credit_products']}
 lineage=helpers['input_lineage']['prepare_snapshot'](accounts_path or resource.joinpath('counterparty_positions.csv'),releases_path or resource.joinpath('counterparty_releases.csv'),data['lineage_config'],data['lineage_config']['cutoffs'][0])
 gd,ld,wd=[load(n)for n in ['structured_governance','structured_lifecycle','structured_workout']]
 for d in [gd,ld,wd]:d['variants']={p:d['variants'][p]}
 c=gd['variants'][p];source=lineage['rows'][0]
 if not(source['entity_id']==sid==c['subject_id']==ld['variants'][p]['subject_id']==wd['variants'][p]['subject_id']and source['balance']==wd['variants'][p]['ead']and source['x']==0):raise ValueError('Counterparty source must bind actual unchanged claim and opening model input')
 c['snapshot']['attributes']['x']['value']=source['x'];c['source_recipe_reference']=dict(kind='ACTUAL_2024_CLAIM',source_snapshot_hash=lineage['selected_snapshot_hash'],subject_id=sid,contract_method_reference='counterparty_credit_products:collateralized; separate economic scenario, not the 2024 realized ledger')
 g=helpers['structured_governance']['run'](gd)['variants'][p];ld['variants'][p]['governance_assessment_hash']=g['assessment']['assessment_hash'];life=helpers['structured_lifecycle']['run'](ld)['variants'][p];workout=helpers['structured_workout']['run'](wd)['variants'][p];replay=helpers['structured_replay_review']['run'](gd)['products'][p]
 economic=helpers['counterparty_credit_products']['run']()['variants']['collateralized']
 ci=deepcopy(c['cohort_input']);f=ci['predictions'][0];f.update(forecast_id=sid+'-actual',case_id=sid,entity_id=sid,cluster_id=sid,forecast_at='2024-01-15',known_at='2024-01-15',pd=g['performing_assessment']['components'][0]['value']);ci.update(predictions=[f],outcome_events=[dict(event_id=sid+'-default',entity_id=sid,event_at=wd['variants'][p]['default_date'],known_at='2024-02-02',event='default',default_definition=ci['target_contract']['default_definition'])],coverage=[dict(entity_id=sid,complete_through='2024-05-01',known_at='2024-05-02')],evaluation_cutoff='2024-05-02')
 cohort=cr.build_monitoring_cohorts(**ci).value;monitor=cr.monitor_rating_cohort(cohort,group_by=['cohort'],reference_window=dict(start='2024-01-01',end='2024-01-31'),minimum_count=1,uncertainty='none').value
 pd=f['pd'];grade='LOW'if pd<.25 else'MID'if pd<.75 else'HIGH';data['benchmark']['internal_ratings'][0]['grade']=grade
 for k in ['entity','final','unsupported_counterfactual']:data['support']['records'][0]['stage_predictions'][k]=pd
 benchmark=cr.benchmark_ratings(**data['benchmark']).value;support=cr.support_transfer_backtest(**data['support']).value
 reasons={r['reason_code']:'Recorded policy rule '+r['rule_id']for r in g['decision']['rule_trace']};explanation=cr.decompose_policy_decision(g['decision'],reasons).value
 if verify_reference:
  assert lineage['audit']['decision']['status']=='PASS'and benchmark['applicability']=='NOT_APPLICABLE'
  assert monitor[0]['metrics']['observed_rate']==1 and monitor[0]['metrics']['brier']==.25
  assert support['metrics']['by_stage']['entity']['brier']==.25 and support['metrics']['by_stage']['transfer']['applicability']=='NOT_APPLICABLE'
  assert workout['closed_workout']['economic_loss']==48 and replay['audit']['decision']['status']=='PASS'
 out=dict(data_origin='SYNTHETIC',product=p,subject_id=sid,source_lineage_status=lineage['audit']['decision']['status'],opening_balance=source['balance'],opening_pd=pd,decision=g['decision']['action'],policy_explanation=explanation,benchmark=benchmark,support_transfer=support,actual_monitoring=monitor,workout=workout,replay_status=replay['audit']['decision']['status'],negative_replay_status=replay['negative_audit']['decision']['status'],historical_replay_status=replay['historical_replay']['status'],lifecycle=life,contract_reference=dict(closeout_exposure=economic['closeout']['closeouts'][0]['closeout_exposure'],expected_loss=economic['loss']['expected_loss'],scope='Separate signed-netting/CSA scenario; not asserted to be observed 2024 claim history'),phase_status=dict(ORIGINATION='APPROVED',PERFORMING='SAME_CLAIM_MONITORED',REVIEW_EARLY_WARNING='SOURCE_CORRECTION_AND_DELAY_TRIGGER',DELINQUENCY='OBSERVED_PAYMENT_DELAY',RESTRUCTURING='UNACCEPTED_PROPOSAL',DEFAULT_WORKOUT='FEBRUARY1_DEFAULT',EXIT='MAY1_SETTLEMENT',PORTFOLIO_MONITORING='SINGLE_CLAIM_DIAGNOSTIC_NOT_POPULATION_VALIDATION'),interpretation=data['interpretation'],raw_artifacts_available=True)
 if return_artifacts:out['raw_artifacts']=dict(lineage=lineage,governance=g,cohort=cohort,cohort_inputs=ci,lifecycle=life,workout=workout,replay=replay,benchmark_inputs=data['benchmark'],support_inputs=data['support'],economic_reference=economic)
 return out
if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
