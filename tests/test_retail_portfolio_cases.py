"""Same retail cohorts, frozen portfolio risk and exact two-name tail oracle."""
from importlib.resources import files
from pathlib import Path
from copy import deepcopy
import json
import math
import runpy
import pytest
from creditratingsystem import CreditRiskError


def fixture():
    data=json.loads(files('creditratingsystem.datasets').joinpath('retail_portfolio_cases.json').read_text())
    ns=runpy.run_path(str(Path(__file__).parents[1]/'examples/retail_portfolio_cases.py'))
    return data,ns


def test_twelve_retail_cohorts_have_mature_unknown_and_grouped_risk_evidence():
    data,ns=fixture();r=ns['run'](data)
    assert len(r['cases'])==12
    for case in r['cases'].values():
        assert case['default_count_backtest']['overall']['observed_count']==6
        assert case['pool_diagnostics']['coverage']['mature_eligible_count']==6
        assert case['lgd_backtest']['coverage']['included_count']==2
        assert case['portfolio_risk']['stress']['stressed_distribution']['tails'][0]['expected_shortfall']>case['portfolio_risk']['stress']['base_distribution']['tails'][0]['expected_shortfall']
        assert all('independent_episode_standard_error'not in value for row in case['observed_pool_parameters']for value in row.values()if isinstance(value,dict))


def test_exact_two_actual_facilities_fractional_tail_mass_and_removal():
    data,ns=fixture();case=deepcopy(data['cases']['RETAIL_BNPL']);case['positions']=[case['positions'][0],case['positions'][-1]]
    r=ns['_portfolio'](case,'RETAIL_BNPL_TWO_HELD_FACILITIES');loss=.4*case['positions'][0]['ead']
    # p=.1 and .5; equal loss L: masses .45 at0, .5 atL, .05 at2L.
    assert sorted(r['simulation']['portfolio_loss_paths'])==[0,loss,loss,2*loss]
    assert math.isclose(r['distribution']['expected_loss'],.6*loss,abs_tol=1e-10)
    assert math.isclose(r['distribution']['tails'][0]['expected_shortfall'],1.25*loss,abs_tol=1e-10)
    assert math.isclose(r['paired_facility_removal']['expected_loss_change'],-.1*loss,abs_tol=1e-10)
    assert math.isclose(r['paired_facility_removal']['tail_changes'][0]['expected_shortfall_change'],-.25*loss,abs_tol=1e-10)


def test_training_leakage_and_duplicated_facility_exposure_fail():
    data,ns=fixture();data['cases']={'RETAIL_CREDIT_CARD':data['cases']['RETAIL_CREDIT_CARD']};data['cases']['RETAIL_CREDIT_CARD']['pool_fit']['observations'][0]['role']='VALIDATION'
    with pytest.raises(CreditRiskError):ns['run'](data)
    data,ns=fixture();case=data['cases']['RETAIL_CREDIT_CARD'];duplicate=deepcopy(case['positions'][0]);duplicate['position_id']='duplicate-lot';duplicate['claim_id']='duplicate-claim';case['positions'].append(duplicate)
    with pytest.raises(CreditRiskError,match='PORTFOLIO_ALLOCATION'):ns['_portfolio'](case,'duplicate')

def test_connected_household_cannot_be_split_for_uncertainty():
    data,ns=fixture();data['cases']={'RETAIL_MICRO_LOAN':data['cases']['RETAIL_MICRO_LOAN']}
    data['cases']['RETAIL_MICRO_LOAN']['predictions'][0]['cluster_id']='split-household'
    with pytest.raises(CreditRiskError):ns['run'](data)

def test_two_facilities_share_one_borrower_event_without_losing_marked_exposure():
    data,ns=fixture();case=data['cases']['RETAIL_OVERDRAFT'];case['positions'][1]['entity_id']=case['positions'][0]['entity_id']
    r=ns['_portfolio'](case,'shared-borrower')
    assert len(r['prepared']['default_group_ids'])==5
    assert all(row[0]==row[1]for row in r['simulation']['position_defaults'])
    assert all(math.isclose(x,y,abs_tol=1e-8)for x,y in zip(r['simulation']['portfolio_loss_paths'],r['migration']['marked_values']['portfolio_marked_loss_paths']))
