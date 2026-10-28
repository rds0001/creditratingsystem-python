"""Fast contract oracles; full twelve-product execution is the installed example."""
from importlib.resources import files
from pathlib import Path
import json
import runpy
import pytest
import creditratingsystem as cr


def example():return runpy.run_path(str(Path(__file__).parents[1]/'examples'/'retail_governance.py'))
def data():return json.loads(files('creditratingsystem.datasets').joinpath('retail_governance.json').read_text())

def test_twelve_distinct_payment_contracts_and_signed_liquidity():
    expected={'MORTGAGE':[6000],'HELOC':[1600,1600,-38400],'CONSOLIDATION':[250],'EDUCATION':[500,500,600],'AUTO_BALLOON':[7000,-2750],'OPERATING_LEASE':[200],'CREDIT_CARD':[350],'CHARGE_CARD':[-600],'OVERDRAFT':[-300],'BNPL':[-50,-50,0],'MICRO':[-200,-200,700,-200],'OWNER_BUSINESS':[15]}
    for case in data()['cases'].values():
        service,aux=example()['contractual_service'](case)
        result=cr.residual_income(**case['budget'],new_debt_service=service).value['residual']
        assert (result if isinstance(result,list)else[result])==pytest.approx(expected[case['kind']])
        if case['kind']=='OWNER_BUSINESS':assert aux['business_after_draw']==40
        if case['kind']=='OPERATING_LEASE':assert aux['lessor_residual_excluded']==12000


def test_card_vs_charge_same_pd_distinct_capacity_and_frozen_history():
    result=example()['run'](product_ids=['RETAIL_CREDIT_CARD','RETAIL_CHARGE_CARD'])['variants']
    credit,charge=result.values()
    assert credit['application_risk']['pd']==pytest.approx(charge['application_risk']['pd'])
    assert credit['decision']['action']=='APPROVE'and charge['decision']['action']=='DECLINE'
    for case in result.values():
        assert case['behavioral_replay']['status']=='REPRODUCED'
        assert case['missing_evidence_decision']['action']!='APPROVE'
        assert case['monitoring'][0]['status']=='INSUFFICIENT_INFORMATION'
        assert case['exit_assessment']['state']=='SUPERSEDED'
        assert case['restructuring_proposal_event']['state']['processed'][-1]['event']['payload']['contract_changed']is False


def test_future_capacity_evidence_and_double_counted_owner_income_rejected():
    d=data();d['cases']={'RETAIL_CREDIT_CARD':d['cases']['RETAIL_CREDIT_CARD']};d['cases']['RETAIL_CREDIT_CARD']['income_evidence']['known_at']='2025-01-02'
    with pytest.raises(ValueError,match='not known'):example()['run'](d)
    c=data()['cases']['RETAIL_SMALL_BUSINESS_FINANCE'];c['budget']['net_income']=[300]
    with pytest.raises(ValueError,match='once'):example()['contractual_service'](c)
