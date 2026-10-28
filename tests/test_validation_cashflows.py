from copy import deepcopy
import pytest
from creditratingsystem.collections import build_collections_episodes
from creditratingsystem._validation_cashflows import collections_backtest
from creditratingsystem.core import CreditRiskError


def fixture():
 actions=[dict(action_id='action'+str(i),account_id='a'+str(i),treatment_id='t'+str(i),action_type='CONTACT',stage='ASSIGNED',occurred_at='2024-01-01',known_at='2024-01-01')for i in range(2)];states=[dict(account_id='a'+str(i),state_event_id='s'+str(i),state='DEFAULT',occurred_at='2024-01-01',known_at='2024-01-01')for i in range(2)];rule=dict(rule_id='explicit',version='1',problem_states=['DEFAULT'],clean_states=['CLEAN'],terminal_states=['CLOSED'],default_states=['DEFAULT'],probation_days=10,separation_days=15);coverage=[dict(account_id='a'+str(i),observed_from='2024-01-01',observed_through='2024-02-15',known_at='2024-02-15')for i in range(2)];episodes=build_collections_episodes(states,actions,coverage,rule,evaluation_cutoff='2024-02-15').value;forecasts=[];cash=[];capture=[]
 for i,ep in enumerate(episodes['episodes']):
  forecasts.append(dict(prediction_id='p'+str(i),entity_id='a'+str(i),account_id='a'+str(i),episode_id=ep['episode_id'],index_action_id='action'+str(i),cluster_id='g'+str(i),segment='collections',as_of='2024-01-01',forecast_known_at='2024-01-01',horizon_end='2024-01-31',model_id='collections',model_version='1',parameter_version='1',predicted_cashflows=[dict(cashflow_id='pred-r',kind='RECOVERY',amount=50 if i==0 else 40,currency='EUR',occurred_at='2024-01-15'),dict(cashflow_id='pred-c',kind='COST',amount=5,currency='EUR',occurred_at='2024-01-10')]))
  cash.append(dict(cashflow_id='cost'+str(i),episode_id=ep['episode_id'],kind='COST',amount=5,currency='EUR',occurred_at='2024-01-10',known_at='2024-01-10'));capture.append(dict(episode_id=ep['episode_id'],observed_from='2024-01-01',observed_through='2024-01-31',known_at='2024-02-01'))
 cash.extend([dict(cashflow_id='recovery',episode_id=episodes['episodes'][0]['episode_id'],kind='RECOVERY',amount=40,currency='EUR',occurred_at='2024-01-20',known_at='2024-01-20'),dict(cashflow_id='late-recovery',episode_id=episodes['episodes'][1]['episode_id'],kind='RECOVERY',amount=100,currency='EUR',occurred_at='2024-01-25',known_at='2024-03-01')]);d=dict(model_id='collections',model_version='1',parameter_version='1',data_fingerprint='two-episodes',population_id='collections',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2024-02-15',currency='EUR',target=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='default'));discount=dict(method='flat_effective_annual',day_count='ACT_365F',annual_rate=0,currency='EUR',known_at='2023-12-01',reference='fixed-zero-curve')
 return dict(forecasts=forecasts,episodes=episodes,cashflows=cash,actions=actions,cashflow_coverage=capture,discount_contract=discount,design=d,uncertainty='none')


def test_factual_amount_timing_and_late_known_cash_with_open_episodes():
 kw=fixture();r=collections_backtest(**kw).value;m=r['metrics']['overall']
 assert m['discounted_recovery']['bias']==25 and m['discounted_cost']['bias']==0 and m['discounted_net']['bias']==25
 assert m['recovery_weighted_days']['bias']==-5
 assert r['diagnostics']['episode_status_counts']=={'OPEN':2}
 assert 'late-recovery'not in r['diagnostics']['forecast_traces'][1]['observed_cashflow_ids']
 assert not r['diagnostics']['causal_action_effect_identified'] and not r['diagnostics']['open_episodes_treated_as_closed']
 kw.update(uncertainty='cluster_bootstrap',cluster_resamples=[['g0','g0'],['g1','g1']]);r=collections_backtest(**kw).value
 assert r['uncertainty']['intervals']['discounted_recovery/bias']['lower']==pytest.approx(10.75)
 assert r['uncertainty']['intervals']['recovery_weighted_days/bias']['reason']=='NON_ESTIMABLE_RESAMPLE'


def test_cash_capture_and_frozen_action_contracts():
 kw=fixture();kw['cashflow_coverage'][1]['observed_through']='2024-01-15';r=collections_backtest(**kw).value
 assert r['coverage']['included_count']==1 and r['coverage']['exclusion_counts']=={'INCOMPLETE_CASHFLOW_CAPTURE':1}
 kw['actions'][0]['action_type']='CHANGED_AFTER_OUTCOME'
 with pytest.raises(CreditRiskError):collections_backtest(**kw)


def test_discount_and_currency_contract():
 kw=fixture();kw['discount_contract']['annual_rate']=.1;r=collections_backtest(**kw).value
 expected=(50*1.1**(-14/365)-40*1.1**(-19/365)+40*1.1**(-14/365))/2
 assert r['metrics']['overall']['discounted_recovery']['bias']==pytest.approx(expected)
 kw['context']={'currency':'USD'}
 with pytest.raises(CreditRiskError):collections_backtest(**kw)
