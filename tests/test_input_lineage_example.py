from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json
import runpy
import pytest
import creditratingsystem as cr


def example():
    return runpy.run_path(str(Path(__file__).parents[1]/'examples/input_lineage.py'))


def test_historical_revisions_conserve_actual_account_ledger():
    result=example()['run']()
    old=result['snapshots']['original'];new=result['snapshots']['corrected']
    assert len(old['rows'])==31
    assert old['audit']['node_totals']['joined']['amounts']['balance']==49600
    assert old['audit']['node_totals']['joined']['total_weight']==31
    assert [old['rows'][0]['income'],new['rows'][0]['income']]==[1010,1110]
    assert old['archive_sha256']==new['archive_sha256']
    assert old['selected_snapshot_hash']!=new['selected_snapshot_hash']
    assert result['original_replay_exact']


def test_actual_join_duplication_and_future_or_outcome_source_are_detected():
    original=example()['run']()['snapshots']['original']['audit_inputs']
    bad=deepcopy(original);bad['nodes'][2]['records'][0]['weight']=2
    bad['nodes'][2]['records'][0]['amounts']['balance']=200
    codes=cr.validate_source_lineage(**bad).value['violation_counts']
    assert {'JOIN_WEIGHT_MULTIPLICATION','JOIN_AMOUNT_MULTIPLICATION','RECONCILIATION_WEIGHT','RECONCILIATION_AMOUNT'}<=set(codes)
    bad=deepcopy(original);bad['nodes'][1].update(available_at='2025-02-01',semantic_role='OUTCOME')
    codes=cr.validate_source_lineage(**bad).value['violation_counts']
    assert {'SOURCE_USE_LOOKAHEAD','OUTCOME_IN_PREDICTIVE_PATH'}<=set(codes)


def test_ambiguous_release_is_not_silently_selected():
    resource=files('creditratingsystem.datasets');cfg=json.loads(resource.joinpath('input_lineage.json').read_text())
    left=cr.read_local_dataset(resource.joinpath('lineage_accounts.csv'),cfg['left_schema']).value
    right=cr.read_local_dataset(resource.joinpath('lineage_releases.csv'),cfg['right_schema']).value
    other=dict(right[0],release_id='other-release',income=9999)
    with pytest.raises(cr.CreditRiskError,match='Multiple releases'):
        cr.join_as_of(left,right+[other])
