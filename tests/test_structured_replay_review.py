"""Actual execution is required before checkpoints can be validated."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('structured_replay_review',Path(__file__).parents[1]/'examples/structured_replay_review.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)

def test_restatement_and_actual_replay_keep_original_pd():
    result=example.run()
    assert len(result['products'])==4
    for row in result['products'].values():
        before,after=row['original_checkpoints']
        replay_before,replay_after=row['actual_replay_checkpoints']
        assert before['state']==replay_before['state']and after['state']==replay_after['state']
        assert before['state']['assessment']['components'][0]['value']==.5
        assert after['state']['assessment']['components'][0]['value']==.75
        assert before['snapshot_hashes']!=after['snapshot_hashes']
        assert row['audit']['metrics']['duplicate_identical_deliveries']==1
        assert row['negative_audit']['decision']['status']=='BLOCKED'
        errors=[r for r in row['batch']['rows']if r['status']=='ERROR']
        assert len(errors)==1 and errors[0]['error_code']=='RATING_INTEGRITY'
        assert row['batch_permutation_exact']
