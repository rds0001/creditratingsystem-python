"""Independent servicing cash must not be replaced with forecast recoveries."""
import importlib.util
from pathlib import Path
import json
import pytest

spec=importlib.util.spec_from_file_location('structured_pool_review',Path(__file__).parents[1]/'examples/structured_pool_review.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)

def test_forecast_and_observed_cash_remain_separate():
    report=example.run()['products']
    abs_case=report['SECURITIZATION_TRANCHE']
    metrics=abs_case['asset_pool_validation']['metrics']
    assert metrics['pool_cash_errors']['recovery_gross']['mean_prediction']==8
    assert metrics['observed_owned_flows']['recovery_gross']==30
    assert metrics['pool_cash_errors']['recovery_gross']['bias']==-22
    pool=abs_case['retail_pool_validation']['metrics']['pools'][0]
    assert pool['lgd_error']['mean_outcome']==pytest.approx(.75)
    assert pool['lgd_error']['mean_prediction']==pytest.approx(.65)
    assert report['COVERED_BOND']['asset_pool_validation']['metrics']['observed_owned_flows']['prepayment_principal']==100
    assert report['TRADE_RECEIVABLES']['retail_pool_validation']['applicability']=='NOT_APPLICABLE'

def test_cash_edit_and_incomplete_coverage_are_visible():
    data=json.loads((Path(__file__).parents[1]/'src/creditratingsystem/datasets/structured_pool_review.json').read_text())
    case=data['cases']['SECURITIZATION_TRANCHE']
    case['pool_accounts'][-1]['closing_cash']+=1
    result=example.run(data,verify_reference=False)['products']['SECURITIZATION_TRANCHE']
    assert abs(result['asset_pool_validation']['cash_residuals'][-1]['residual'])==1
    assert result['asset_pool_validation']['violations']
    assert result['incomplete_coverage_validation']['metrics']['default_probability']['brier']is None
    assert result['duplicate_allocation_validation']['violation_counts']['CASH_OR_LOSS_DOUBLE_ALLOCATION']>=1
