from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json
import runpy


def fixture():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples/retail_transitions.py'))['run']
    data=json.loads(files('creditratingsystem.datasets').joinpath('retail_transitions.json').read_text())
    return run,data


def test_dated_product_cash_claims_and_censored_exit():
    run,data=fixture();v=run(data)['products']
    assert len(v)==12
    assert v['RETAIL_CREDIT_CARD']['delinquency']['past_due_amount']==25
    assert v['RETAIL_CHARGE_CARD']['delinquency']['past_due_amount']==950
    assert v['RETAIL_AUTO_LOAN']['arrangement']['candidate']['remaining_claim']==0
    assert v['RETAIL_MICRO_LOAN']['treatment']['status']=='INFEASIBLE'
    assert v['RETAIL_AUTO_LOAN']['treatment']['status']=='FEASIBLE'


def test_user_cash_capacity_changes_gate_without_changing_proposal_value():
    run,data=fixture();name='RETAIL_AUTO_LOAN'
    data['cases']={name:deepcopy(data['cases'][name])}
    before=run(data)['products'][name]
    data['cases'][name]['arrangement']['affordability_paths'][0]['available_cash']=[0,0,0]
    after=run(data,verify_reference=False)['products'][name]
    assert after['treatment']['status']=='INFEASIBLE'
    assert before['treatment']['candidates'][0]['expected_present_value']==after['treatment']['candidates'][0]['expected_present_value']
