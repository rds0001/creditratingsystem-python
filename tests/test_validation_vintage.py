from copy import deepcopy
import pytest
from creditratingsystem.behavioral import build_vintage_performance
from creditratingsystem._validation_vintage import vintage_backtest
from creditratingsystem.core import CreditRiskError


def fixture(incomplete=False):
 origins=[dict(account_id='a'+str(i),opened_at='2024-01-01',known_at='2024-01-01',original_balance=100,currency='EUR',cluster_id='g'+str(i))for i in range(100)]
 events=[dict(event_id='exit'+str(i),account_id='a'+str(i),event_type='PREPAID',occurred_at='2024-01-20',known_at='2024-01-20')for i in range(20)]+[dict(event_id='default'+str(i),account_id='a'+str(i),event_type='DEFAULT',occurred_at='2024-02-15',known_at='2024-02-15')for i in range(20,30)]
 coverage=[dict(account_id='a'+str(i),observed_through='2024-01-31'if incomplete and i==99 else'2024-03-31',known_at='2024-04-01')for i in range(100)]
 table=build_vintage_performance(origins,[],events,months_on_book=[0,1],evaluation_cutoff='2024-04-01',coverage=coverage,default_event='DEFAULT',delinquency_states=['DPD30'],exit_events=['PREPAID']).value;predictions=[];outcomes=[]
 for i in range(100):
  for mob,end,pc,pi in [(0,'2024-01-31',.05,.05),(1,'2024-02-29',.15,.1)]:
   predictions.append(dict(account_id='a'+str(i),months_on_book=mob,model_id='vintage',model_version='1',parameter_version='1',forecast_at='2024-01-01',known_at='2024-01-01',cumulative_pd=pc,interval_pd=pi))
   default=20<=i<30 and mob==1;exited=i<20;missing=incomplete and i==99 and mob==1
   out=dict(account_id='a'+str(i),months_on_book=mob,period_end=end,known_at='2024-04-01',default_definition='first-default-before-exit',complete=not missing,exited=exited,cumulative_default=None if missing else int(default),interval_default=None if missing else int(default),complete_through=end)
   if default or exited:out.update(event_id=('default'if default else'exit')+str(i),first_event_at='2024-02-15'if default else'2024-01-20')
   outcomes.append(out)
 d=dict(model_id='vintage',model_version='1',parameter_version='1',data_fingerprint='fixed-100',population_id='origin-cohort',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2024-04-01',target=dict(event='default',horizon=1,time_unit='months',probability_measure='PHYSICAL',conditioning='performing'));contract=dict(method='fixed_origination',weighting='count',exit_treatment='first_competing_exit',reference='original-cohort',default_definition='first-default-before-exit',horizon_convention='origination_month_end_plus_mob')
 return table,predictions,outcomes,dict(origin_cohorts=origins,denominator_contract=contract,design=d,uncertainty='none')


def test_fixed100_keeps20_exits_and_matches_operational_table_hand_values():
 table,predictions,outcomes,kw=fixture();r=vintage_backtest(table,predictions,outcomes,**kw).value;last=r['metrics']['cells'][1]
 assert last['original_accounts']==100 and last['counts']['exits']==20
 assert last['cumulative']['mean_outcome']==pytest.approx(.1)
 assert last['cumulative']['bias']==pytest.approx(.05) and last['cumulative']['brier']==pytest.approx(.0925)
 assert last['remaining_first_event_risk_count']==70
 kw.update(uncertainty='cluster_bootstrap',cluster_resamples=[['g0']*100,['g20']*100]);r=vintage_backtest(table,predictions,outcomes,**kw).value
 assert r['uncertainty']['intervals']['2024-01/EUR/1/cumulative/brier']['lower']==pytest.approx(.04)
 table[1]['accounts']=80
 with pytest.raises(CreditRiskError):vintage_backtest(table,predictions,outcomes,**kw)


def test_incomplete_vintage_is_not_reweighted_or_reported_zero_default():
 table,predictions,outcomes,kw=fixture(True);r=vintage_backtest(table,predictions,outcomes,**kw).value;last=r['metrics']['cells'][1]
 assert last['original_accounts']==100 and not last['mature']
 assert last['cumulative']['brier']is None and last['cumulative']['mean_outcome']is None
 assert last['missing_followup_accounts']==['a99']
