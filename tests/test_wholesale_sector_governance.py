"""Pinned calibrated sector history; exhaustive seven histories are opt-in."""
from pathlib import Path
import runpy,json,os


def test_wholesale_sector_governance():
    root=Path(__file__).parents[1];run=runpy.run_path(str(root/'examples/wholesale_sector_governance.py'))['run']
    data=json.loads((root/'src/creditratingsystem/datasets/wholesale_sector_governance.json').read_text())
    if os.getenv('CREDITRATINGSYSTEM_EXTENDED_TESTS','').lower()!='true':data['cases']=[data['cases'][1]]
    result=run(data)['cases']
    for row in result:
        assert row['origination_pd']==row['historical_pd']
        assert row['replay']==row['replay_after_new_champion']=='REPRODUCED'
        assert row['tampered_replay']=='NOT_REPRODUCIBLE'
        assert row['event_statuses']==['APPLIED','DUPLICATE','APPLIED','APPLIED','APPLIED']
        assert row['final_state']=='SUPERSEDED'
        expected='APPROVE'if row['credit_gate_inputs']['available_unrestricted_cashflow']>=row['credit_gate_inputs']['debt_service_due']else'REFER'
        assert row['origination_decision']==expected
