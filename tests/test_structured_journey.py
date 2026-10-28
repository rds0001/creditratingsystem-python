"""Position recovery must reconcile to its own collateral and legal claim."""
import importlib.util
from pathlib import Path
import json
import pytest
spec=importlib.util.spec_from_file_location('structured_journey',Path(__file__).parents[1]/'examples/structured_journey.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)

def test_three_actual_claims_and_independent_settlements():
    out=example.run()
    assert out['full_component_reexecution_exact']
    assert out['lineage']['audit']['node_totals']['joined']['amounts']['balance']==280
    expected={'SECURITIZATION_TRANCHE':55,'COVERED_BOND':32.5,'TRADE_RECEIVABLES':30}
    for product,loss in expected.items():
        row=out['products'][product]
        assert row['workout']['closed_workout']['economic_loss']==loss
        assert row['position_binding']['position_id']==row['source_row']['entity_id']
        assert row['pool_validation']['asset_pool_validation']['decision']['status']=='NOT_ASSESSED'
        assert out['replay_audits'][product]['decision']['status']=='PASS'
    assert out['products']['COVERED_BOND']['claim_control']['aggregate_recovery']==67.5
    assert out['products']['TRADE_RECEIVABLES']['claim_control']['seller_recourse']==0

def test_fictitious_position_recovery_does_not_reconcile():
    data=json.loads((Path(__file__).parents[1]/'src/creditratingsystem/datasets/structured_journey.json').read_text())
    data['pools']['cases']['COVERED_BOND']['position_binding']['actual_position_recovery']=100
    with pytest.raises(AssertionError):example.run(data)
