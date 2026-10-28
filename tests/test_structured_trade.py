from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._structured_pool import build_credit_asset_pool
from creditratingsystem._structured_trade import simulate_trade_receivables
from test_structured_paths import asset,assignment,D
DATES=['2023-02-01','2023-03-01','2023-04-01']
def fixture_trade(n=1,recourse='SELLER'):
 assets=[];assign=[]
 for i in range(n):
  a=asset('a'+str(i));a.update(asset_class='INVOICE',invoice_id='i'+str(i),recourse=recourse,schedule=[dict(date=DATES[0],principal_due=100,interest_due=0)]);assets.append(a);assign.append(assignment('m'+str(i),a['asset_id']))
 profile=dict(profile_id='invoice',version='1',known_at=D,currency='EUR',product_kind='TRADE_RECEIVABLES',pool_mode='STATIC',allowed_asset_classes=['INVOICE'],maximum_dpd=30,exclude_disputed=True,eligible_recourse=['NONE','SELLER'],revolving_until=None)
 pool=build_credit_asset_pool(assets,assign,profile,pool_id='pool',as_of=D,information_cutoff=D).value
 contract=dict(contract_id='invoice-scenarios',version='1',scenario_ids=['s'],weights=[1],dates=DATES,currency='EUR',probability_measure='PHYSICAL',sampling_design='FIXED_WEIGHTED_SCENARIOS',conditioning='PIT')
 mapping=[dict(party_id='g',default_group_id='g'),dict(party_id='seller',default_group_id='seller'),dict(party_id='service',default_group_id='service')]
 paths=[dict(scenario_id='s',default_group_id=g,date=d,conditional_pd=0,default_channel=g)for g in ['g','seller','service']for d in DATES]
 innovations=[dict(scenario_id='s',date=d,channel_id=g,uniform=.5)for g in ['g','seller','service']for d in DATES]
 invoices=[dict(scenario_id='s',asset_id=a['asset_id'],payment_date=DATES[0],dilution_date=D,dilution_fraction=0,debtor_recovery_fraction=.4,debtor_recovery_date=DATES[1],debtor_recovery_cost=0)for a in assets]
 sellers=[dict(scenario_id='s',seller_id='seller',settlement_date=DATES[2],performing_capacity=1000,default_estate_capacity=20,default_recovery_fraction=.5,recourse_scope='CREDIT_ONLY')]
 services=[dict(scenario_id='s',servicer_id='service',remittance_date=DATES[2],default_commingling_recovery_fraction=.5,default_commingling_capacity=30,replacement_cost=5)]
 return dict(invoice_pool=pool,scenario_contract=contract,party_mapping=mapping,party_default_paths=paths,invoice_scenarios=invoices,seller_terms=sellers,servicer_terms=services,shared_innovations=innovations)
def run(f):return simulate_trade_receivables(**f).value

def test_performing_and_shared_credit_event_with_capped_seller_capacity():
 assert run(fixture_trade())['ending_states'][0]['net_investor_cash']==100
 f=fixture_trade(2);f['party_default_paths'][0]['conditional_pd']=1;f['seller_terms'][0]['performing_capacity']=50;r=run(f)
 assert all(x['debtor_credit_event']for x in r['invoice_results'])
 assert [x['seller_credit_recourse']for x in r['invoice_results']]==[25,25]
 assert r['seller_results'][0]['aggregate_payment']==50
 assert r['ending_states'][0]['net_credit_loss']==70 and r['ending_states'][0]['net_investor_cash']==130
 assert r['ending_states'][0]['conservation_residual']==0

def test_joint_debtor_seller_default_keeps_residual_risk_and_nonrecourse():
 f=fixture_trade(2);f['party_default_paths'][0]['conditional_pd']=1;f['party_default_paths'][3]['conditional_pd']=1;r=run(f)
 assert r['seller_results'][0]['in_default'] and r['seller_results'][0]['aggregate_payment']==20
 assert r['ending_states'][0]['net_credit_loss']==100
 f=fixture_trade(recourse='NONE');f['party_default_paths'][0]['conditional_pd']=1;r=run(f)
 assert r['seller_results'][0]['aggregate_payment']==0 and r['ending_states'][0]['net_credit_loss']==60

def test_full_dilution_separate_from_default_and_optional_commercial_recourse():
 f=fixture_trade();f['invoice_scenarios'][0]['dilution_fraction']=1;r=run(f)
 assert not r['invoice_results'][0]['debtor_credit_event'] and r['ending_states'][0]['net_dilution_loss']==100 and r['ending_states'][0]['net_credit_loss']==0
 f['seller_terms'][0]['recourse_scope']='CREDIT_AND_DILUTION';r=run(f)
 assert r['invoice_results'][0]['seller_dilution_recourse']==100 and r['ending_states'][0]['net_investor_cash']==100

def test_actual_payment_delay_exposed_to_later_debtor_default():
 f=fixture_trade(recourse='NONE');f['party_default_paths'][1]['conditional_pd']=1
 assert run(f)['ending_states'][0]['net_credit_loss']==0
 f['invoice_scenarios'][0]['payment_date']=DATES[1];r=run(f)
 assert r['invoice_results'][0]['debtor_credit_event'] and r['ending_states'][0]['net_credit_loss']==60

def test_commingled_cash_recovered_once_and_default_day_redirect():
 f=fixture_trade(2);f['party_default_paths'][7]['conditional_pd']=1;r=run(f);v=r['servicer_results'][0]
 assert v['held_cash']==200 and v['remittance']==30 and v['commingling_loss']==170 and v['replacement_cost']==5
 assert r['ending_states'][0]['net_investor_cash']==25 and r['ending_states'][0]['conservation_residual']==0
 f=fixture_trade();f['party_default_paths'][6]['conditional_pd']=1;r=run(f);v=r['servicer_results'][0]
 assert v['held_cash']==0 and v['redirected_cash']==100 and v['commingling_loss']==0 and r['ending_states'][0]['net_investor_cash']==95

def test_same_legal_entity_mapping_has_one_default_and_cost_accounting():
 f=fixture_trade();f['party_mapping'][1]['default_group_id']='g';f['party_default_paths']=[p for p in f['party_default_paths']if p['default_group_id']!='seller'];f['shared_innovations']=[p for p in f['shared_innovations']if p['channel_id']!='seller'];f['party_default_paths'][0]['conditional_pd']=1;f['invoice_scenarios'][0]['debtor_recovery_cost']=5;r=run(f)
 assert r['seller_results'][0]['in_default'] and r['seller_results'][0]['aggregate_payment']==20
 assert r['ending_states'][0]['net_investor_cash']==55 and r['ending_states'][0]['net_credit_loss']==40

def test_schema_dates_party_groups_and_horizon_guards():
 f=fixture_trade();f['party_mapping'].append(deepcopy(f['party_mapping'][0]))
 with pytest.raises(CreditRiskError,match='STRUCTURED_DEFAULT_GROUP'):run(f)
 f=fixture_trade();f['invoice_scenarios'][0]['payment_date']='2024-01-01'
 with pytest.raises(CreditRiskError,match='STRUCTURED_CALENDAR'):run(f)
 f=fixture_trade();f['servicer_terms'][0]['remittance_date']=D
 with pytest.raises(CreditRiskError,match='STRUCTURED_CALENDAR'):run(f)
 with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):simulate_trade_receivables(**fixture_trade(),context={'horizon':1})
