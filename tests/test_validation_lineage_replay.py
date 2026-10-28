from copy import deepcopy
import pytest
from creditratingsystem._validation_lineage import validate_source_lineage
from creditratingsystem._validation_replay import replay_consistency
from creditratingsystem.core import _fingerprint, CreditRiskError


def lineage_fixture():
 def record(identity,weight,amount):return dict(record_id=identity,join_key='household',weight=weight,amounts={'balance':amount})
 def node(identity,kind,records,letter):
  n=dict(node_id=identity,kind=kind,schema_version='1',vintage='2024-01',revision='0',transformation_version='identity'if kind=='SOURCE'else'join-v1',content_hash=letter*64,observed_at='2024-01-01',available_at='2024-01-02',records=records)
  if kind=='SOURCE':n.update(source_id=identity,semantic_role='FEATURE')
  return n
 nodes=[node('left','SOURCE',[record('l',1,100)],'a'),node('right','SOURCE',[record('r0',1,0),record('r1',1,0)],'b'),node('out','FEATURE',[record('o0',.5,50),record('o1',.5,50)],'c')]
 joins=[dict(join_id='j',left_node_id='left',right_node_id='right',output_node_id='out',cardinality='one_to_many',join_type='inner',weight_contract='LEFT_WEIGHT_ALLOCATION',amount_origin='LEFT',mapping=[dict(output_id='o'+str(i),left_id='l',right_id='r'+str(i),allocation_weight=.5)for i in range(2)],excluded_left_ids=[],unmatched_right_ids=[])]
 recs=[dict(reconciliation_id='rec-'+n['node_id'],node_id=n['node_id'],reference='independent-input-ledger',expected_record_ids=[r['record_id']for r in n['records']],expected_total_weight=2 if n['node_id']=='right'else 1,expected_amounts={'balance':0 if n['node_id']=='right'else 100})for n in nodes]
 uses=[dict(use_id='u',node_id='out',purpose='FEATURE',as_of='2024-01-02',known_at='2024-01-02',source_snapshot_hashes={'left':'a'*64,'right':'b'*64})]
 contract=dict(contract_id='lineage',version='1',reference='declared-left-allocation',amount_fields={'balance':'EUR'},absolute_tolerance=1e-12,relative_tolerance=1e-12,required_node_ids=['left','right','out'],required_use_ids=['u'],required_join_ids=['j'])
 return dict(nodes=nodes,edges=[dict(parent_id='left',child_id='out'),dict(parent_id='right',child_id='out')],joins=joins,reconciliations=recs,uses=uses,contract=contract)


def replay_fixture():
 e0=dict(event_id='e0',idempotency_key='k0',occurred_at='2024-01-01',known_at='2024-01-01',sequence=0,payload={'balance':100},revision_of=None);e0['payload_hash']=_fingerprint(e0['payload'])
 e1=dict(event_id='e1',idempotency_key='k1',occurred_at='2024-01-01',known_at='2024-01-03',sequence=1,payload={'balance':80},revision_of='e0');e1['payload_hash']=_fingerprint(e1['payload']);events=[e0,deepcopy(e0),e1]
 snapshots=[dict(snapshot_id=kind,kind=kind,content_hash=letter*64,known_at='2023-12-01',effective_from='2023-12-01',effective_until=None)for kind,letter in zip(['MODEL','CALIBRATION','POLICY','SOURCE'],'abcd')]
 original=[dict(checkpoint_id='c'+str(i),as_of=date,known_at=date,event_log_hash=_fingerprint(events),snapshot_hashes={s['snapshot_id']:s['content_hash']for s in snapshots},applied_event_ids=['e0']if i==0 else['e0','e1'],state=dict(balance=100 if i==0 else 80,approved=True,path=[1.,2.]))for i,date in enumerate(['2024-01-02','2024-01-04'])]
 replay=deepcopy(original)
 for cp in replay:cp['mode']='REPLAY'
 contract=dict(contract_id='replay',version='1',reference='append-only-unit-case',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=['c0','c1'],absolute_tolerance=1e-10,relative_tolerance=0)
 return dict(event_log=events,original_checkpoints=original,replayed_checkpoints=replay,snapshot_manifest=snapshots,replay_contract=contract)


def test_join_actual_rows_conserve_weight_amounts_and_detect_duplication():
 kw=lineage_fixture();r=validate_source_lineage(**kw).value
 assert r['decision']['status']=='PASS'and r['node_totals']['out']['total_weight']==1 and r['node_totals']['out']['amounts']['balance']==100
 assert r['join_findings'][0]['allocated_left_weights']=={'l':1}and r['topological_order']==['left','right','out']
 for record in kw['nodes'][2]['records']:record['weight']=1;record['amounts']['balance']=100
 kw['joins'][0]['cardinality']='one_to_one';r=validate_source_lineage(**kw).value
 assert {'JOIN_CARDINALITY','JOIN_WEIGHT_MULTIPLICATION','JOIN_AMOUNT_MULTIPLICATION','RECONCILIATION_WEIGHT','RECONCILIATION_AMOUNT'}<=set(r['violation_counts'])


def test_source_late_revision_target_leakage_and_cycle():
 kw=lineage_fixture();kw['nodes'][1].update(available_at='2024-01-03',semantic_role='OUTCOME');r=validate_source_lineage(**kw).value
 assert {'TRANSFORMATION_LOOKAHEAD','SOURCE_USE_LOOKAHEAD','OUTCOME_IN_PREDICTIVE_PATH'}<=set(r['violation_counts'])
 kw=lineage_fixture();kw['uses']=[];assert validate_source_lineage(**kw).value['violation_counts']['MISSING_REQUIRED_USE']==1
 kw=lineage_fixture();kw['edges'].append(dict(parent_id='out',child_id='left'))
 with pytest.raises(CreditRiskError):validate_source_lineage(**kw)


def test_replay_late_correction_duplicate_idempotency_and_tolerance():
 kw=replay_fixture();r=replay_consistency(**kw).value
 assert r['decision']['status']=='PASS'and r['metrics']['duplicate_identical_deliveries']==1
 assert [f['known_revision_count']for f in r['checkpoint_findings']]==[0,1]
 assert not r['event_execution_performed']and not r['original_decisions_modified']
 kw['replayed_checkpoints'][1]['state']['balance']+=1e-11;assert replay_consistency(**kw).value['decision']['status']=='PASS'
 kw['replayed_checkpoints'][0]['applied_event_ids']=['e0','e0','e1'];kw['replayed_checkpoints'][0]['state']['balance']=80;kw['replayed_checkpoints'][1]['mode']='COUNTERFACTUAL';r=replay_consistency(**kw).value
 assert {'EVENT_ORDER_OR_IDEMPOTENCY_MISMATCH','CHECKPOINT_STATE_MISMATCH','COUNTERFACTUAL_NOT_REPLAY'}<=set(r['violation_counts'])


def test_replay_snapshot_changes_and_conflicting_duplicate_payload():
 kw=replay_fixture();kw['replayed_checkpoints'][0]['snapshot_hashes']['MODEL']='f'*64;kw['snapshot_manifest'][2]['effective_until']='2024-01-04';r=replay_consistency(**kw).value
 assert {'REPLAY_SNAPSHOT_CHANGED','SNAPSHOT_NOT_HISTORICALLY_APPLICABLE'}<=set(r['violation_counts'])
 kw=replay_fixture();kw['event_log'][1]['payload']['balance']=0
 with pytest.raises(CreditRiskError):replay_consistency(**kw)
