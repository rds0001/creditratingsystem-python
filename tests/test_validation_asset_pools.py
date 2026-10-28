from copy import deepcopy
import pytest
from creditratingsystem._validation_asset_pools import asset_pool_backtest,_FLOW
from creditratingsystem._structured_pool import build_credit_asset_pool
from creditratingsystem._structured_paths import simulate_asset_pool
from creditratingsystem.core import CreditRiskError
from test_structured_paths import fixture as path_fixture,asset,assignment

def fixture(prepay=False):
 base=path_fixture();pool=base['asset_pool'];pool_inputs=dict(asset_contracts=[asset()],pool_assignments=[assignment()],eligibility_profile=pool['profile'],pool_id='pool',as_of='2023-01-01',information_cutoff='2023-01-01')
 base['scenario_contract'].update(scenario_ids=['d','n'],weights=[.5,.5])
 for field in ['obligor_paths','asset_paths','shared_innovations']:
  original=base[field];base[field]=[dict(row,scenario_id=s)for s in ['d','n']for row in original]
 for row in base['obligor_paths']:row['conditional_pd']=.5 if row['date']=='2023-07-01'else 0
 for row in base['shared_innovations']:row['uniform']=.25 if row['scenario_id']=='d'else .75
 paths=simulate_asset_pool(**base).value;target=dict(event='default',horizon=365,time_unit='days',probability_measure='PHYSICAL',conditioning='PIT');d=dict(model_id='pool-model',model_version='1',parameter_version='1',data_fingerprint='actual-servicer-file',population_id='pool',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2024-01-02',forecast_known_at='2023-01-01',currency='EUR',target=target)
 def event(id,date,kind,opening,closing,**flows):return dict(event_id=id,asset_id='a',date=date,known_at=date,sequence=1,currency='EUR',event_type=kind,opening_principal=opening,closing_principal=closing,**{k:flows.get(k,0)for k in _FLOW},default_event_id=flows.get('default_event_id'))
 events=[event('prepay','2023-07-01','PREPAYMENT',100,0,prepayment_principal=100,interest=5)]if prepay else[event('default','2023-07-01','DEFAULT',100,0,default_ead=100),event('recovery','2023-07-31','RECOVERY',0,0,recovery_gross=30,workout_cost=5,default_event_id='default')]
 allocations=[dict(allocation_id='alloc-'+e['event_id'],event_id=e['event_id'],pool_id='pool',share=1,known_at=e['known_at'],amounts={f:e[f]for f in _FLOW})for e in events];accounts=[];payments=[]
 for e in events:
  principal=80 if prepay else 25 if e['event_type']=='RECOVERY'else 0;interest=5 if prepay else 0
  accounts.append(dict(date=e['date'],known_at=e['date'],currency='EUR',opening_cash=0,closing_cash=0,purchase_cash=0,sale_cash=0,external_injection=0,external_distribution=0,fees_paid=0,tranche_interest_paid=interest,tranche_principal_paid=principal,equity_paid=20 if prepay else 0))
  if principal:payments.append(dict(payment_id='pay-'+e['event_id'],tranche_id='senior',date=e['date'],known_at=e['date'],currency='EUR',principal=principal,interest=interest))
 return dict(asset_pool=pool,pool_inputs=pool_inputs,forecast_paths=paths,forecast_inputs=base,servicing=events,allocations=allocations,pool_accounts=accounts,coverage=[dict(asset_id='a',complete=True,complete_through='2024-01-01',known_at='2024-01-02',currency='EUR',target=target)],design=d,cash_contract=dict(as_of='2023-01-01',known_at='2023-01-01',currency='EUR',opening_cash=0),tranche_contracts=[dict(tranche_id='senior',notional=80,currency='EUR',known_at='2023-01-01')],tranche_payments=payments)

def test_actual_recovery_and_forecast_are_distinct_and_cash_conserves():
 r=asset_pool_backtest(**fixture()).value
 assert r['decision']['status']=='NOT_ASSESSED'and not r['violations']
 assert r['metrics']['default_probability']['brier']==.25
 assert r['metrics']['pool_cash_errors']['recovery_gross']['bias']==-10
 assert r['metrics']['tranche_payments']['senior']['principal']==25
 assert all(x['residual']==0 for x in r['cash_residuals'])

def test_prepayment_not_default_and_double_allocation_detected():
 r=asset_pool_backtest(**fixture(True)).value
 assert r['metrics']['default_probability']['observed_rate']==0 and r['metrics']['prepayment_probability']['observed_rate']==1
 assert r['metrics']['tranche_payments']['senior']['principal']==80
 kw=fixture();a=deepcopy(kw['allocations'][1]);a.update(allocation_id='duplicate-claim',pool_id='other',share=1);kw['allocations'].append(a)
 assert asset_pool_backtest(**kw).value['violation_counts']['CASH_OR_LOSS_DOUBLE_ALLOCATION']==1

def test_incomplete_coverage_and_late_cash_not_zero_complete_outcome():
 kw=fixture();kw['servicing'][1]['known_at']='2024-02-01';kw['coverage'][0]['complete']=False;kw['pool_accounts']=kw['pool_accounts'][:1];kw['tranche_payments']=[]
 r=asset_pool_backtest(**kw).value
 assert r['coverage']['eligible_assets']==0 and r['metrics']['pool_cash_errors']['recovery_gross']['bias']is None
 assert r['metrics']['observed_owned_flows']['recovery_gross']==0
 kw=fixture();kw['coverage'][0]['known_at']='2023-01-01'
 with pytest.raises(CreditRiskError):asset_pool_backtest(**kw)

def test_principal_cash_and_tranche_reconciliations():
 kw=fixture();kw['servicing'][0]['closing_principal']=1;r=asset_pool_backtest(**kw).value
 assert 'ASSET_PRINCIPAL_CONSERVATION'in r['violation_counts']
 kw=fixture();kw['pool_accounts'][1]['tranche_principal_paid']=24;r=asset_pool_backtest(**kw).value
 assert r['violation_counts']['POOL_CASH_CONSERVATION']==1 and r['violation_counts']['TRANCHE_CASH_RECONCILIATION']==1

def test_changed_ownership_audits_cash_but_never_scores_original_risk_set():
 kw=fixture();h=deepcopy(kw['pool_inputs']['pool_assignments'][0]);h['share']=.5;kw['ownership_history']=[h]
 for a in kw['allocations']:
  a['share']=.5;a['amounts']={k:v*.5 for k,v in a['amounts'].items()}
 kw['pool_accounts'][1]['tranche_principal_paid']=12.5;kw['tranche_payments'][0]['principal']=12.5
 r=asset_pool_backtest(**kw).value
 assert not r['violations'] and r['metrics']['observed_owned_flows']['recovery_gross']==15
 assert r['coverage']['forecast_comparison_risk_set']=='NOT_COMPARABLE'
 assert r['metrics']['default_probability']['brier']is None and r['metrics']['pool_cash_errors']['recovery_gross']['bias']is None


def test_full_prepayment_prevents_later_performing_interest():
 kw=fixture(True);e=deepcopy(kw['servicing'][0]);e.update(event_id='after-exit',date='2023-07-31',known_at='2023-07-31',event_type='PAYMENT',opening_principal=0,prepayment_principal=0,interest=1);kw['servicing'].append(e)
 assert asset_pool_backtest(**kw).value['violation_counts']['POST_PREPAYMENT_PERFORMING_CASH']==1
