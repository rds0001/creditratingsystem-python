"""Contract dates, calibration groups and replay; all seven histories opt-in."""
from pathlib import Path
from copy import deepcopy
import json,os,runpy
import pytest


def test_contractual_histories_and_date_rejection():
    root=Path(__file__).parents[1]
    run=runpy.run_path(str(root/'examples/wholesale_commitment_governance.py'))['run']
    data=json.loads((root/'src/creditratingsystem/datasets/wholesale_commitment_governance.json').read_text())
    if os.getenv('CREDITRATINGSYSTEM_EXTENDED_TESTS','').lower()!='true':data['cases']=[data['cases'][0],data['cases'][-1]]
    result=run(data,return_audit_artifacts=True)
    for case,row,audit in zip(data['cases'],result['report']['cases'],result['audits']):
        assert row['origination_decision']==case['reference_action']
        assert row['replay']==row['replay_after_new_champion']=='REPRODUCED'
        assert row['origination_pd']==row['historical_pd']
        assert row['tampered_replay']=='NOT_REPRODUCIBLE'
        events=audit['requests'][1]['input']['events'];default=next(e for e in events if e['event_id']=='credit-default')
        assert default['occurred_at']==case['observed']['default_at']
        if case['product']=='LEVERAGED_LOAN':assert default['occurred_at']=='2026-01-01'
        else:assert case['calibrator_arguments']['groups'] and default['occurred_at']=='2024-01-31'
    bad=deepcopy(data);bad['cases']=bad['cases'][:1];bad['cases'][0]['timeline']['default_known']='2023-01-01'
    with pytest.raises(ValueError,match='chronology'):run(bad,verify_reference=False)
