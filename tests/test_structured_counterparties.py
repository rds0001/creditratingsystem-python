from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._structured_counterparties import stress_pool_counterparties
D='2023-01-01';DATES=['2023-02-01','2023-03-01','2023-04-01']
def fixture_cp():
 contract=dict(contract_id='cp',version='1',scenario_ids=['s'],weights=[1],dates=DATES,currency='EUR',probability_measure='PHYSICAL',sampling_design='FIXED_WEIGHTED_SCENARIOS',conditioning='PIT')
 return dict(baseline_cashflows=[dict(scenario_id='s',cashflow_id='cash',date=DATES[1],amount=100)],exposure_contracts=[dict(exposure_id='account',counterparty_id='bank',netting_set_id='net',role='ACCOUNT_BANK',basis='CASH_HELD',cashflow_ids=['cash'])],netting_sets=[dict(netting_set_id='net',counterparty_id='bank',enforceable=True,collateral_contract_id='collateral',collateral_convention='RECEIVED_TRANSFERABLE',default_timing='AFTER_COLLATERAL_SNAPSHOT_BEFORE_CASHFLOWS')],exposure_paths=[dict(scenario_id='s',exposure_id='account',date=d,signed_replacement_mark=0)for d in DATES],collateral_paths=[dict(scenario_id='s',netting_set_id='net',date=d,known_at=d,collateral_value=20,haircut=.5)for d in DATES],settlement_terms=[dict(scenario_id='s',counterparty_id='bank',estate_recovery_fraction=.5,estate_capacity=1000,estate_delay_days=30,collateral_delay_days=0,replacement_cost=5,replacement_delay_days=0)],scenario_contract=contract,party_mapping=[dict(party_id='bank',default_group_id='bank')],party_default_paths=[dict(scenario_id='s',default_group_id='bank',date=d,conditional_pd=1 if d==DATES[0]else 0,default_channel='bank')for d in DATES],shared_innovations=[dict(scenario_id='s',date=d,channel_id='bank',uniform=.5)for d in DATES],as_of=D)
def run(f):return stress_pool_counterparties(**f).value

def add_swap(f,mark):
 f['exposure_contracts'].append(dict(exposure_id='swap',counterparty_id='bank',netting_set_id='net',role='SWAP',basis='REPLACEMENT',cashflow_ids=[]))
 f['exposure_paths'] += [dict(scenario_id='s',exposure_id='swap',date=d,signed_replacement_mark=mark)for d in DATES]
 return f

def test_dated_collateral_haircut_recovery_and_removed_cash_once():
 r=run(fixture_cp());n=r['netting_results'][0];e=r['ending_states'][0]
 assert n['positive_exposure']==100 and n['collateral_recovery']==10 and n['estate_recovery']==45 and n['credit_loss']==45
 assert e['adjusted_net_cash']==50 and e['additional_replacement_cost']==5 and e['conservation_residual']==0
 assert len(r['removed_baseline_cashflows'])==1
 assert next(x for x in r['adjusted_cashflows']if x['source']=='ESTATE_RECOVERY')['date']=='2023-03-03'

def test_signed_replacement_netting_and_negative_closeout_reconcile():
 f=add_swap(fixture_cp(),-60);r=run(f);n=r['netting_results'][0]
 assert n['net_closeout']==40 and n['credit_loss']==15 and r['ending_states'][0]['adjusted_net_cash']==80
 f=add_swap(fixture_cp(),-150);r=run(f);n=r['netting_results'][0]
 assert n['net_closeout']==-50 and n['credit_loss']==0 and r['ending_states'][0]['adjusted_net_cash']==95
 assert sum(x['amount']for x in r['adjusted_cashflows']if x['source']=='NEGATIVE_NET_CLOSEOUT')==-50

def test_swap_only_replacement_loss_is_not_counted_twice():
 f=fixture_cp();f['baseline_cashflows']=[];f['exposure_contracts']=[dict(exposure_id='account',counterparty_id='bank',netting_set_id='net',role='SWAP',basis='REPLACEMENT',cashflow_ids=[])]
 for row in f['exposure_paths']:row['signed_replacement_mark']=100
 r=run(f);e=r['ending_states'][0]
 assert e['baseline_cash']==0 and e['adjusted_net_cash']==-50 and e['credit_loss']==45 and e['conservation_residual']==0

def test_estate_capacity_and_replacement_cost_once_across_sets():
 f=fixture_cp();f['baseline_cashflows'].append(dict(scenario_id='s',cashflow_id='other',date=DATES[1],amount=100));f['exposure_contracts'].append(dict(exposure_id='servicer',counterparty_id='bank',netting_set_id='second',role='SERVICER',basis='CASH_HELD',cashflow_ids=['other']));n=deepcopy(f['netting_sets'][0]);n.update(netting_set_id='second',collateral_contract_id='othercoll');f['netting_sets'].append(n)
 for row in deepcopy(f['exposure_paths']):row['exposure_id']='servicer';f['exposure_paths'].append(row)
 for row in deepcopy(f['collateral_paths']):row['netting_set_id']='second';f['collateral_paths'].append(row)
 f['settlement_terms'][0]['estate_capacity']=30;r=run(f)
 assert [x['estate_recovery']for x in r['netting_results']]==[15,15]
 assert r['counterparty_results'][0]['estate_recovery']==30 and r['ending_states'][0]['additional_replacement_cost']==5
 assert r['ending_states'][0]['credit_loss']==150 and r['ending_states'][0]['adjusted_net_cash']==45

def test_same_day_cash_is_exposed_but_already_paid_cash_is_not():
 f=fixture_cp();f['baseline_cashflows'][0]['date']=DATES[0];assert run(f)['ending_states'][0]['credit_loss']==45
 f['baseline_cashflows'][0]['date']=D;r=run(f)
 assert not r['removed_baseline_cashflows'] and r['ending_states'][0]['credit_loss']==0 and r['ending_states'][0]['adjusted_net_cash']==95
 f=fixture_cp();f['party_default_paths'][0]['conditional_pd']=0;r=run(f)
 assert r['ending_states'][0]['adjusted_net_cash']==100 and r['ending_states'][0]['additional_replacement_cost']==0

def test_double_cash_netting_and_tail_guards():
 f=fixture_cp();e=deepcopy(f['exposure_contracts'][0]);e['exposure_id']='duplicate';f['exposure_contracts'].append(e)
 with pytest.raises(CreditRiskError,match='STRUCTURED_DOUBLE_COUNT'):run(f)
 f=add_swap(fixture_cp(),-50);f['netting_sets'][0]['enforceable']=False
 with pytest.raises(CreditRiskError,match='STRUCTURED_NETTING'):run(f)
 f=fixture_cp();f['collateral_paths'][0]['known_at']=DATES[1]
 with pytest.raises(CreditRiskError,match='STRUCTURED_COLLATERAL'):run(f)
 f=fixture_cp();f['settlement_terms'][0]['estate_delay_days']=1000
 with pytest.raises(CreditRiskError,match='STRUCTURED_UNRESOLVED'):run(f)

def test_existing_guarantee_receivable_and_overcollateralization_cap():
 f=fixture_cp();f['exposure_contracts'][0].update(role='GUARANTOR',basis='GUARANTEE_RECEIVABLE')
 for row in f['collateral_paths']:row['collateral_value']=1000
 r=run(f);n=r['netting_results'][0]
 assert n['collateral_recovery']==100 and n['estate_recovery']==0 and n['credit_loss']==0
 assert r['ending_states'][0]['adjusted_net_cash']==95
 f=fixture_cp();r=run(f)
 assert r['netting_results'][0]['proportional_loss_by_role']['ACCOUNT_BANK']==45
