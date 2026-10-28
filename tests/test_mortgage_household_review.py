"""Observed mortgage validation without invented reject labels or payments."""
import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest
import creditratingsystem as cr
spec=importlib.util.spec_from_file_location('mortgage_household_review',Path(__file__).parents[1]/'examples/mortgage_household_review.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)

@pytest.fixture(scope='module')
def raw():return example.run(return_artifacts=True)

def test_joint_household_and_actual_default(raw):
    assert raw['joint_application']['monthly_income']==5000
    assert raw['joint_application']['monthly_debt_service']==1000
    assert raw['collateral']['cltv']==2/3
    assert raw['governance']['negative_applicability']['status']=='NOT_APPLICABLE'
    assert raw['household']['households'][0]['total_budget_debt_service']==900
    assert raw['closed_workout']['economic_loss']==32000
    assert raw['pool']['metrics']['overall_count']['brier']==.5625
    assert raw['pool']['coverage']['input_count']==1
    assert raw['affordability']['metrics']['factual_scheduled_minus_paid']['n']==0
    assert raw['affordability']['missing_payment_periods']
    assert raw['replay']['decision']['status']=='PASS'
    assert raw['lifecycle']['application_and_monitoring']==raw['retail']['summary']

def test_pseudo_outcome_double_income_and_replay_tampering(raw):
    c=deepcopy(raw['cohort_inputs']);c['application_targets'][2].update(target=0,label_observed=True,label_origin='PSEUDO')
    assert cr.validate_retail_cohort(**c).value['decision']['status']=='FAIL'
    a=deepcopy(raw['affordability_inputs']);a['economic_evidence'][0]['terms']['monthly_amount']*=2
    assert cr.affordability_stress_backtest(**a).value['decision']['status']=='BLOCKED'
    r=deepcopy(raw['replay_inputs']);r['replayed_checkpoints'][0]['state']['ratings'][0]['value']['pd']+=.01
    assert cr.replay_consistency(**r).value['decision']['status']=='BLOCKED'
