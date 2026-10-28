from copy import deepcopy
import pytest
from creditratingsystem.ead import aggregate_netting_set_exposure, simulate_collateral_margin
from creditratingsystem._validation_counterparty import counterparty_exposure_backtest
from creditratingsystem.core import CreditRiskError

DATES=['2024-01-01','2024-01-02','2024-01-03']

def netting(values):
 agreements=[dict(netting_set_id='n',counterparty_id='c1',trade_ids=['a','b'],legally_enforceable=True,evidence_reference='frozen-agreement',known_at=DATES[0],effective_from=DATES[0],effective_until=None),dict(netting_set_id='m',counterparty_id='c2',trade_ids=['negative'],legally_enforceable=False,evidence_reference='singleton',known_at=DATES[0],effective_from=DATES[0],effective_until=None)]
 trades=[];fx=[]
 for path,series in values.items():
  for date,net in zip(DATES,series):
   for trade,value in [('a',net+60),('b',-60),('negative',-100)]:trades.append(dict(path_id=path,date=date,trade_id=trade,currency='EUR',value=value))
   fx.append(dict(path_id=path,date=date,currency='EUR',rate=1))
 return aggregate_netting_set_exposure(trades,agreements,fx,as_of=DATES[0],reporting_currency='EUR',valuation_convention='EX_DIVIDEND_BEFORE_MARGIN').value


def design():return dict(model_id='ccr',model_version='1',parameter_version='1',data_fingerprint='frozen-exposures',population_id='counterparties',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2024-01-04',currency='EUR',target=dict(event='default',horizon=2,time_unit='days',probability_measure='PHYSICAL',conditioning='counterparty_exposure'))


def fixture():
 forecast=netting({'p'+str(i):[40,60,x]for i,x in enumerate([0,20,40,80])});actual=netting({'observed':[40,60,30]});rows=[];obs=[]
 for i,(sid,cp)in enumerate([('n','c1'),('m','c2')]):
  rows.append(dict(prediction_id='f'+str(i),entity_id=cp,cluster_id=cp,segment='ccr',as_of=DATES[0],forecast_known_at=DATES[0],horizon_end=DATES[-1],model_id='ccr',model_version='1',parameter_version='1',counterparty_id=cp,netting_set_id=sid,legal_version='1',exposure_target='UNCOLLATERALIZED',netting_paths=forecast))
  obs.append(dict(prediction_id='f'+str(i),observation_status='OBSERVED',observed_at=DATES[-1],known_at=DATES[-1],observed_path_id='observed',legal_version='1',netting_paths=actual,reported_exposure=30 if sid=='n'else 0))
 return dict(forecasts=rows,observations=obs,quantile_levels=[.5,.95],design=design(),uncertainty='none')


def closeout_fixture():
 net=netting({'p':[40,60,90]});policies=[];market=[];calls=[]
 for sid in ['m','n']:
  policies.append(dict(netting_set_id=sid,known_at=DATES[0],collateral_currency='EUR',initial_units=0,threshold_received=0,threshold_posted=0,minimum_transfer_amount=0,ownership='TITLE_TRANSFER',received_enforceable=True,posted_returnability='UNSECURED_CLAIM',posted_claim_netting_enforceable=True,reuse='ALLOWED'))
  for i,date in enumerate(DATES):
   market.append(dict(path_id='p',date=date,netting_set_id=sid,unit_price=.5 if i==2 else 1,fx_rate=1,received_haircut=0,posted_haircut=0,currency='EUR'))
   calls.append(dict(call_id=sid+str(i),path_id='p',netting_set_id=sid,call_date=date,valuation_date=date,settlement_date=date,dispute_delay_days=0))
 coll=simulate_collateral_margin(net,policies,calls,market).value;kw=fixture();row=kw['forecasts'][0];obs=kw['observations'][0];event=dict(path_id='p',netting_set_id='n',default_date=DATES[0],closeout_date=DATES[-1],margin_observation_date=DATES[0]);flows=[dict(cashflow_id='unpaid',path_id='p',netting_set_id='n',date=DATES[1],amount=10,included_in_closeout_value=False,currency='EUR')]
 row.update(exposure_target='COLLATERALIZED',netting_paths=net,collateral_paths=coll,csa_version='1',default_events=[event],intermediate_cashflows=flows,freeze_timing='AFTER_DATE_EVENTS',cashflow_convention='UNSETTLED_EXCLUDED_FROM_CLOSEOUT_VALUE',default_sampling_contract=dict(mode='JOINT_SIMULATED_PATHS',known_at=DATES[0],reference='tiny-joint-path'))
 obs.update(netting_paths=net,collateral_paths=coll,observed_path_id='p',csa_version='1',reported_exposure=0,closeout_event=event,intermediate_cashflows=flows,reported_closeout_exposure=80,closeout_known_at='2024-01-04');kw.update(forecasts=[row],observations=[obs]);return kw


def test_exact_empirical_quantiles_per_set_and_real_cluster_intervals():
 kw=fixture();r=counterparty_exposure_backtest(**kw).value
 assert r['metrics']['market_exposure']['bias']==2.5
 assert [q['value']for q in r['diagnostics']['forecast_traces'][0]['quantiles']]==[20,80]
 assert r['metrics']['quantile_exceptions'][0]['exception_rate']==.5
 assert r['metrics']['quantile_exceptions'][0]['pinball_loss']==2.5
 assert r['metrics']['conditional_default_closeout']['applicability']=='NOT_APPLICABLE'
 kw.update(uncertainty='cluster_bootstrap',cluster_resamples=[['c1','c1'],['c2','c2']]);r=counterparty_exposure_backtest(**kw).value
 assert r['uncertainty']['intervals']['market_exposure/bias']['lower']==pytest.approx(.125)
 assert r['uncertainty']['intervals']['conditional_default_closeout/bias']['reason']=='NON_ESTIMABLE_RESAMPLE'


def test_illegal_cross_set_offset_and_late_known_market_observation():
 kw=fixture();kw['observations'][0]['reported_exposure']=0;r=counterparty_exposure_backtest(**kw).value
 assert r['decision']['status']=='FAIL'and r['coverage']['included_count']==1
 assert r['diagnostics']['reconciliation_violations'][0]['code']=='ILLEGAL_OFFSET_OR_EXPOSURE_RECONCILIATION'
 kw=fixture();kw['observations'][0]['known_at']='2024-02-01';r=counterparty_exposure_backtest(**kw).value
 assert r['coverage']['exclusion_counts']=={'EXPOSURE_NOT_YET_KNOWN':1}


def test_true_frozen_unit_closeout_is_separate_from_market_margin():
 kw=closeout_fixture();r=counterparty_exposure_backtest(**kw).value
 assert r['metrics']['market_exposure']['mean_outcome']==0
 m=r['metrics']['conditional_default_closeout'];assert m['mean_prediction']==80 and m['mean_outcome']==80 and m['bias']==0
 assert r['diagnostics']['closeout_traces'][0]['closeout_details']['frozen_collateral_units']==40
 kw['observations'][0]['observation_status']='CENSORED';r=counterparty_exposure_backtest(**kw).value
 assert r['coverage']['included_count']==0 and r['diagnostics']['closeout_included_count']==1 and r['applicability']=='APPLICABLE'
 kw['observations'][0]['closeout_known_at']='2024-02-01';r=counterparty_exposure_backtest(**kw).value
 assert r['metrics']['conditional_default_closeout']['applicability']=='NOT_APPLICABLE'


def test_versions_and_missing_joint_default_contract_do_not_pass():
 kw=closeout_fixture();kw['observations'][0]['csa_version']='changed-after-outcome';r=counterparty_exposure_backtest(**kw).value
 assert r['decision']['status']=='FAIL'and r['diagnostics']['closeout_included_count']==0
 kw=closeout_fixture();kw['forecasts'][0].pop('default_sampling_contract')
 with pytest.raises(CreditRiskError):counterparty_exposure_backtest(**kw)
