"""Twelve actual retail claims, prior-known cure forecasts and negative evidence."""
from importlib.resources import files
from pathlib import Path
import json
import runpy
import pytest
from creditratingsystem import CreditRiskError


def fixture():
    data=json.loads(files('creditratingsystem.datasets').joinpath('retail_workout_cases.json').read_text())
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples/retail_workout_cases.py'))['run']
    return data,run


def test_twelve_distinct_retail_claims_and_prior_known_forecasts():
    data,run=fixture();result=run(data)
    assert len(result['cases'])==12 and len({c['ead']for c in data['cases'].values()})==12
    h=result['cases']['RETAIL_HOME_EQUITY_LINE']['claim_allocation']['claim_results']
    assert {x['claim_id']:x['gross_recovery']for x in h}=={'prior-mortgage':40000,'owned':30000}
    b=result['cases']['RETAIL_BNPL']['claim_allocation']['claim_results']
    assert sum(x['gross_recovery']for x in b)==150 and len(b)==2
    assert result['cure_training']['known_at']<'2024-02-01'


def test_unknown_resolution_does_not_fabricate_closed_loss():
    data,run=fixture();data['cases']={'RETAIL_AUTO_LEASE':data['cases']['RETAIL_AUTO_LEASE']}
    data['cases']['RETAIL_AUTO_LEASE']['episode']['resolution_known_at']='2024-06-01'
    result=run(data,verify_reference=False)['cases']['RETAIL_AUTO_LEASE']
    assert result['closed_workout']['economic_lgd']is None and result['event_conditioned_loss']is None


def test_future_model_wrong_currency_and_duplicate_claim_rejected():
    data,run=fixture();data['cases']={'RETAIL_HOME_EQUITY_LINE':data['cases']['RETAIL_HOME_EQUITY_LINE']};data['cure_training']['training']['known_at']='2024-03-01'
    with pytest.raises(CreditRiskError,match='COLLECTIONS_LEAKAGE'):run(data)
    data,run=fixture();data['cases']={'RETAIL_MICRO_LOAN':data['cases']['RETAIL_MICRO_LOAN']};data['cases']['RETAIL_MICRO_LOAN']['cashflows'][0]['currency']='USD'
    with pytest.raises(CreditRiskError):run(data)
    data,run=fixture();data['cases']={'RETAIL_BNPL':data['cases']['RETAIL_BNPL']};c=data['cases']['RETAIL_BNPL'];c['claim_allocation']['claims'].append(c['claim_allocation']['claims'][0])
    with pytest.raises(CreditRiskError):run(data)
