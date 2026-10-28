"""Read bank-owned local tables, freeze a historical view and audit its ledger."""
from copy import deepcopy
from importlib.resources import files
import json
import creditratingsystem as cr


def prepare_snapshot(accounts_path, releases_path, config, cutoff):
    accounts=cr.read_local_dataset(accounts_path,config['left_schema'])
    releases=cr.read_local_dataset(releases_path,config['right_schema'])
    left=deepcopy(accounts.value)
    for row in left:row['known_at']=cutoff['known_at']
    joined=cr.join_as_of(left,releases.value).value
    if any(r['release_id'] is None for r in joined):
        raise ValueError('Missing historical release: provide coverage or handle referral before scoring')
    ids={r['release_id'] for r in joined}
    selected=[r for r in releases.value if r['release_id'] in ids]
    # The archive may contain later releases. Only the genuinely selected source
    # snapshot is a predictive ancestor; the full file hash remains in provenance.
    def ledger(rows,side):
        return [dict(record_id=r['release_id'] if side=='right' else r['record_id'],join_key=r['entity_id'],weight=1,amounts={'balance':0 if side=='right' else r['balance']})for r in rows]
    def node(identity,kind,rows,side,observed,available):
        n=dict(node_id=identity,kind=kind,schema_version='1',vintage=config.get('vintage','2025-01'),revision=cutoff['revision'],transformation_version='selected-snapshot-v1' if kind=='SOURCE' else 'join-as-of-v1',content_hash=cr.content_fingerprint(rows).value,observed_at=observed,available_at=available,records=ledger(rows,side))
        if kind=='SOURCE':n.update(source_id=identity,semantic_role='FEATURE')
        return n
    nodes=[node('accounts','SOURCE',left,'left',max(r['as_of'] for r in left),max(r['known_at'] for r in left)),node('releases','SOURCE',selected,'right',max(r['observed_at'] for r in selected),max(r['available_at'] for r in selected)),node('joined','FEATURE',joined,'left',max(r['as_of'] for r in left),cutoff['known_at'])]
    control=config['controls'];right_ids=[p+'-'+cutoff['revision'] for p in control['product_ids']]
    recs=[dict(reconciliation_id='control-'+n['node_id'],node_id=n['node_id'],reference='separately-supplied-account-control-ledger',expected_record_ids=right_ids if n['node_id']=='releases' else control['record_ids'],expected_total_weight=control['total_weight'],expected_amounts={'balance':0 if n['node_id']=='releases' else control['total_balance']})for n in nodes]
    join=dict(join_id='historical-snapshot-join',left_node_id='accounts',right_node_id='releases',output_node_id='joined',cardinality='one_to_one',join_type='left',weight_contract='LEFT_WEIGHT_ALLOCATION',amount_origin='LEFT',mapping=[dict(output_id=r['record_id'],left_id=r['record_id'],right_id=r['release_id'],allocation_weight=1)for r in joined],excluded_left_ids=[],unmatched_right_ids=[])
    contract=dict(contract_id='local-input-control',version='1',reference='independent-balance-and-selected-release-control',amount_fields={'balance':'EUR'},absolute_tolerance=0,relative_tolerance=0,required_node_ids=['accounts','releases','joined'],required_use_ids=['assessment-input'],required_join_ids=['historical-snapshot-join'])
    arguments=dict(nodes=nodes,edges=[dict(parent_id=n,child_id='joined')for n in ['accounts','releases']],joins=[join],reconciliations=recs,uses=[dict(use_id='assessment-input',node_id='joined',purpose='FEATURE',as_of=max(r['as_of'] for r in left),known_at=cutoff['known_at'],source_snapshot_hashes={n['node_id']:n['content_hash']for n in nodes[:2]})],contract=contract)
    audit=cr.validate_source_lineage(**arguments).value
    return dict(rows=joined,audit=audit,audit_inputs=arguments,archive_sha256={'accounts':accounts.provenance['source_sha256'],'releases':releases.provenance['source_sha256']},selected_snapshot_hash=nodes[1]['content_hash'])


def run(accounts_path=None,releases_path=None,config=None,verify_reference=True):
    resource=files('creditratingsystem.datasets')
    if config is None:config=json.loads(resource.joinpath('input_lineage.json').read_text())
    accounts_path=accounts_path or resource.joinpath('lineage_accounts.csv')
    releases_path=releases_path or resource.joinpath('lineage_releases.csv')
    outputs={}
    for cutoff in config['cutoffs']:
        result=prepare_snapshot(accounts_path,releases_path,config,cutoff)
        if verify_reference:
            assert result['audit']['decision']['status']=='PASS'
            assert [r['income']for r in result['rows']]==cutoff['expected_income']
            assert result['audit']['node_totals']['joined']['amounts']['balance']==config['controls']['total_balance']
        outputs[cutoff['name']]=result
    # Re-execute the original view after processing the revision; no mutation.
    replay=prepare_snapshot(accounts_path,releases_path,config,config['cutoffs'][0])
    assert replay==outputs[config['cutoffs'][0]['name']]
    return dict(snapshots=outputs,original_replay_exact=True,interpretation='Historical source selection and independently supplied ledger controls; content hashes identify bytes/values and do not authenticate the source author.')


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
