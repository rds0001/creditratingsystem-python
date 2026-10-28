"""Residual hurdle and open-workout tests using closed-form terminal-value oracles."""
from copy import deepcopy
from functools import lru_cache
from datetime import date,timedelta
import math
import numpy as np
import pytest
from creditratingsystem.lgd import fit_recovery_hurdle,predict_lgd,estimate_open_workout_lgd
from creditratingsystem.core import CreditRiskError
TARGET=dict(event='residual_terminal_loss_fraction',horizon_role='TIMING_REPORTING_REFERENCE',horizon=365,time_unit='days',probability_measure='PHYSICAL',conditioning='given_default',currency='EUR')
TRAIN=dict(data_fingerprint='hand',split_id='train',known_at='2024-12-31',data_origin='SYNTHETIC',workout_population='open_and_closed_defaulted_landmarks')
STAGE=dict(payout_definition='RESIDUAL_TERMINAL_GROSS_RECOVERY_OR_CURE_VALUE',cashflow_pattern='SINGLE_TERMINAL_PAYOUT_AFTER_REFERENCE',timing_outcome_independence='CONDITIONAL_ON_REFERENCE_COVARIATES',positive_distribution='lognormal',cashflow_completeness='COMPLETE_THROUGH_OBSERVATION_END',payout_horizon='ULTIMATE',timing_distribution='weibull',cure_residual_value_ratio=1.,landmark_definition='Account enters residual terminal settlement stage')
def episode(eid='a',closed=False,days=10,cured=0):
    resolution=(date(2024,1,1)+timedelta(days=days)).isoformat()
    return dict(episode_id=eid,default_date='2023-12-01',default_known_at='2023-12-01',reference_date='2024-01-01',reference_known_at='2024-01-01',features_observed_at='2024-01-01',features_known_at='2024-01-01',currency='EUR',ead_at_default=100.,residual_exposure_at_reference=80.,observation_end='2024-04-01',resolution_date=resolution if closed else None,resolution_known_at=resolution if closed else None,cured=cured,cure_value=80. if cured else 0.)
def flow(eid,amount,when,kind='RECOVERY',id=None):return dict(episode_id=eid,cashflow_id=id or eid,cashflow_date=when,known_at=when,currency='EUR',kind=kind,amount=amount)
@lru_cache(maxsize=1)
def fit():
    rows=[episode(str(i),i<6,days=10+5*i,cured=int(i<2))for i in range(8)]
    flows=[flow('4',80*math.exp(-1),rows[4]['resolution_date']),flow('5',80*math.exp(1),rows[5]['resolution_date'])]
    return fit_recovery_hurdle(rows,flows,np.empty((8,0)),feature_order=[],stage_spec=STAGE,target=TARGET,training=TRAIN,model_id='h',model_version='1',currency='EUR')
def frozen():
    m=deepcopy(fit().model)
    m['training']['known_at']='2023-12-01'
    for key in ('cure_gate','positive_gate','timing_model'):m['parameters'][key]['training']['known_at']='2023-12-01'
    # Exponential resolution rate 1/10: independent hand-defined conditional law.
    m['parameters']['timing_model']['parameters'].update(coefficients=[math.log(10)],log_time_scale=1.,support_end=1000.)
    return m

def valuation(tail=None):
    dates=['2023-12-15','2024-01-11','2024-01-21','2024-01-26','2024-02-01']
    return dict(episode_id='a',discount_curve=dict(anchor_date='2023-12-01',known_at='2024-12-31',discount_basis='user economic',points=[dict(cashflow_date=d,discount_factor=f)for d,f in zip(dates,[1,.9,.8,.7,.6])]),fx_contract=dict(reporting_currency='EUR',known_at='2024-12-31',conversion_basis='cashflow_date',rates=[dict(currency='EUR',date=d,rate=1.)for d in dates]),future_resolution_dates=['2024-01-21','2024-02-01'],resolution_time_convention='RIGHT_ENDPOINT',tail_discount_factor=tail,tail_assumption_reference='independent flat average'if tail is not None else None,unmodeled_future_costs_assumption='NONE_BEYOND_SUPPLIED')
def value(model=None,rows=None,flows=None,tail=None,costs=None,as_of='2024-01-11'):
    m=frozen()if model is None else model;m=deepcopy(m);m['training']['known_at']='2023-12-01';v=valuation(tail);v['discount_curve']['known_at']=as_of;v['fx_contract']['known_at']=as_of
    return estimate_open_workout_lgd([episode()]if rows is None else rows,[flow('a',10,'2023-12-15')]if flows is None else flows,m,np.empty((1,0)),[v],[]if costs is None else costs,feature_order=[],expected_target=TARGET,stage_spec=STAGE,as_of=as_of,currency='EUR',extrapolation='parametric')

def test_hurdle_independent_gate_and_lognormal_moments():
    f=fit();p=predict_lgd(f,[[]],feature_order=[],expected_target=TARGET,default_condition='given_default',quantile_levels=[.1,.4,.9]).value
    # 2/6 cure, 2/4 positive among noncured; mean log ratio0, MLvariance1.
    assert p['cure_probability']==pytest.approx([1/3]);assert p['positive_noncure_probability']==pytest.approx([1/3])
    assert p['conditional_positive_log_mean']==pytest.approx([0]);assert p['conditional_positive_log_variance']==pytest.approx(1)
    assert p['mean_lgd']==pytest.approx([1-(1+math.exp(.5))/3]);assert p['quantiles'][0][1:]==pytest.approx([0,1])
    assert f.model['diagnostics']['censored_count']==2

def test_conditional_remaining_time_tail_and_no_double_count():
    r=value().value['episodes'][0];mean=80*(1+math.exp(.5))/3
    expected=mean*((1-math.exp(-1))*.8+(math.exp(-1)-math.exp(-2.1))*.6)
    assert r['expected_discounted_future_terminal_value_within_grid']==pytest.approx(expected)
    assert r['known_discounted_net_cashflows']==10
    assert r['unresolved_probability']==pytest.approx(math.exp(-2.1))
    assert r['expected_final_lgd']is None
    assert r['expected_loss_within_grid']==pytest.approx(90-expected)
    full=value(tail=.4).value['episodes'][0]
    assert full['expected_final_lgd']==pytest.approx((90-expected-mean*math.exp(-2.1)*.4)/100)

def test_future_costs_once_and_closed_cure_value():
    cost=flow('a',5,'2024-01-26',kind='DIRECT_COST',id='cost');cost['known_at']='2024-01-10'
    baseline=value(tail=.4).value['episodes'][0];r=value(tail=.4,costs=[cost]).value['episodes'][0]
    assert r['expected_final_loss']-baseline['expected_final_loss']==pytest.approx(3.5)
    closed=episode(closed=True,cured=1);r=value(rows=[closed]).value['episodes'][0]
    assert r['status']=='OBSERVED_CLOSED';assert r['expected_final_lgd']==pytest.approx(.18)

def test_future_resolution_not_leaked_and_interim_recovery_rejected():
    future=episode(closed=True,days=20,cured=1)
    assert value(rows=[future]).value['episodes'][0]['status']=='UNRESOLVED_TAIL_VALUE'
    with pytest.raises(CreditRiskError,match='RECOVERY_STAGE'):value(flows=[flow('a',1,'2024-01-05')])

def test_stage_target_and_zero_ead_contracts():
    row=episode();row['ead_at_default']=0
    r=value(rows=[row],tail=.4).value['episodes'][0];assert r['expected_final_lgd']is None;assert r['expected_final_loss']is not None
    model=frozen();model['parameters']['stage_spec']['cashflow_pattern']='TOTAL_WORKOUT'
    with pytest.raises(CreditRiskError,match='RECOVERY_STAGE'):value(model=model)

def test_cdf_quantile_oracle_and_explicit_ultimate_horizon_role():
    from scipy.stats import norm
    f=fit();v=predict_lgd(f,[[]],feature_order=[],expected_target=TARGET,default_condition='given_default',quantile_levels=[.1,.4,.6,.9]).value
    assert v['payout_horizon']=='ULTIMATE'
    assert v['quantiles'][0]==pytest.approx([1-math.exp(norm.isf(.3)),0,1-math.exp(norm.isf(.8)),1])
    bad=deepcopy(f.model);del bad['target']['horizon_role']
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):predict_lgd(bad,[[]],feature_order=[],expected_target=bad['target'],default_condition='given_default')

def test_chronology_completeness_and_malformed_component_guards():
    row=episode();row['features_known_at']='2024-01-02'
    with pytest.raises(CreditRiskError,match='RECOVERY_DATE'):value(rows=[row])
    bad=frozen();bad['parameters']['amount_regression']=[]
    with pytest.raises(CreditRiskError,match='INVALID_MODEL'):value(model=bad)
    bad=frozen();del bad['parameters']['stage_spec']['cashflow_completeness']
    with pytest.raises(CreditRiskError,match='RECOVERY_STAGE'):value(model=bad)
    future=flow('a',999,'2024-02-15',id='unknown');future['known_at']='2024-02-16'
    r=value(flows=[flow('a',10,'2023-12-15'),future]).value['episodes'][0]
    assert r['known_discounted_net_cashflows']==10;assert r['excluded_cashflows']==['unknown']
    duplicate=flow('a',5,'2024-01-26',kind='DIRECT_COST');duplicate['known_at']='2024-01-10'
    with pytest.raises(CreditRiskError,match='RECOVERY_CONTRACT'):value(costs=[duplicate])
