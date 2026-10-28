from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._structured_pool import build_credit_asset_pool
from creditratingsystem._structured_paths import simulate_asset_pool
D='2023-01-01';DATES=['2023-07-01','2024-01-01']
def asset(id='a',group='g',cls='AUTO_LOAN'):
 return dict(asset_id=id,asset_class=cls,invoice_id=None,obligor_id=group,default_group_id=group,seller_id='seller',servicer_id='service',currency='EUR',observed_at=D,known_at=D,origination_date='2022-01-01',balance_date=D,principal_balance=100,schedule=[dict(date=DATES[0],principal_due=40,interest_due=5),dict(date=DATES[1],principal_due=60,interest_due=3)],overdue_days=0,disputed=False,dilution_reserve=0,recourse='NONE',collateral_allocations=[])
def assignment(id='m',aid='a',start=D,price=100):return dict(assignment_id=id,pool_id='pool',asset_id=aid,known_at=D,effective_from=start,effective_until=None,share=1,purchase_price=price,purchase_priority=1)
def fixture(assets=None,assignments=None,revolving=False):
 assets=assets or [asset()];assignments=assignments or [assignment()]
 p=dict(profile_id='p',version='1',known_at=D,currency='EUR',product_kind='ABS',pool_mode='REVOLVING'if revolving else'STATIC',allowed_asset_classes=['AUTO_LOAN'],maximum_dpd=30,exclude_disputed=True,eligible_recourse=['NONE'],revolving_until=DATES[1]if revolving else None)
 pool=build_credit_asset_pool(assets,assignments,p,pool_id='pool',as_of=D,information_cutoff=D).value
 contract=dict(contract_id='c',version='1',scenario_ids=['s'],weights=[1],dates=DATES,currency='EUR',probability_measure='PHYSICAL',sampling_design='FIXED_WEIGHTED_SCENARIOS',conditioning='PIT',default_timing='BEFORE_DUE',prepayment_timing='AFTER_COUPON',dilution_treatment='PROPORTIONAL_REMAINING_SCHEDULE',reinvestment_funding='PRINCIPAL_CASH_ONLY',initial_reinvestment_cash=0)
 gps=[dict(scenario_id='s',default_group_id=g,date=d,conditional_pd=0,default_channel='default-'+g)for g in sorted({a['default_group_id']for a in assets})for d in DATES]
 aps=[dict(scenario_id='s',asset_id=a['asset_id'],date=d,conditional_prepayment_probability=0,prepayment_channel='prepay-'+a['asset_id'],dilution_fraction=0,recovery_fraction=.4,recovery_delay_days=30,recovery_cost=5,servicing_cost=0,interest_multiplier=1,overdue_days=0,disputed=False)for a in assets for d in DATES]
 innovations=[dict(scenario_id='s',date=p['date'],channel_id=p['default_channel'],uniform=.5)for p in gps]+[dict(scenario_id='s',date=p['date'],channel_id=p['prepayment_channel'],uniform=.5)for p in aps]
 return dict(asset_pool=pool,scenario_contract=contract,obligor_paths=gps,asset_paths=aps,shared_innovations=innovations)
def run(f):return simulate_asset_pool(**f).value

def test_performing_contract_and_principal_conservation():
 r=run(fixture());cash=[c for c in r['cashflows']if c['date']in DATES]
 assert [c['principal_available']for c in cash]==[40,60]
 assert [c['interest_available']for c in cash]==[5,3]
 assert all(x['principal_conservation_residual']==0 for x in r['asset_events'])
 assert r['ending_states'][0]['terminal_complete']

def test_shared_obligor_default_and_dated_recovery_once():
 f=fixture([asset(),asset('b')],[assignment(),assignment('b','b')]);f['obligor_paths'][0]['conditional_pd']=1
 r=run(f);defaults=[e for e in r['asset_events']if e['event']=='DEFAULT']
 assert len(defaults)==2 and sum(x['default_ead']for x in defaults)==200
 assert sum(c['principal_available']for c in r['cashflows'])==0
 assert sum(c['interest_available']for c in r['cashflows'])==0
 assert sum(c['recovery_gross']for c in r['cashflows'])==80
 assert sum(c['workout_cost']for c in r['cashflows'])==10
 assert r['cashflows'][-1]['cumulative_modeled_credit_loss']==130
 rec=next(c for c in r['cashflows']if c['recovery_gross'])
 assert rec['date']=='2023-07-31' and r['ending_states'][0]['terminal_complete']
 assert sum(e['new_default']for e in r['group_events'])==1

def test_competing_default_before_prepayment_and_coupon():
 f=fixture();f['asset_paths'][0]['conditional_prepayment_probability']=1
 r=run(f);assert r['asset_events'][0]['event']=='PREPAYMENT'
 assert sum(c['principal_available']for c in r['cashflows'])==100 and sum(c['interest_available']for c in r['cashflows'])==5
 f['obligor_paths'][0]['conditional_pd']=1;r=run(f)
 assert r['asset_events'][0]['event']=='DEFAULT' and sum(c['interest_available']for c in r['cashflows'])==0
 assert sum(e['prepayment_principal']for e in r['asset_events'])==0

def test_dilution_scales_remaining_schedule_without_default():
 f=fixture();f['asset_paths'][0]['dilution_fraction']=.25;r=run(f)
 assert sum(c['principal_available']for c in r['cashflows'])==75
 assert sum(c['interest_available']for c in r['cashflows'])==6
 assert r['cashflows'][-1]['cumulative_dilution_loss']==25
 assert r['cashflows'][-1]['cumulative_modeled_credit_loss']==0
 assert not any(e['new_default']for e in r['group_events'])
 f['asset_paths'][0]['dilution_fraction']=1;r=run(f)
 assert r['asset_events'][0]['event']=='DILUTED' and sum(c['recovery_gross']for c in r['cashflows'])==0
 assert r['ending_states'][0]['terminal_complete']

def test_reinvestment_is_financed_and_starts_after_purchase_date_payments():
 a,b=asset(),asset('b','h');a['schedule']=[dict(date=DATES[0],principal_due=100,interest_due=10)];b['schedule']=[dict(date=DATES[0],principal_due=50,interest_due=5),dict(date=DATES[1],principal_due=50,interest_due=2)]
 f=fixture([a,b],[assignment(),assignment('next','b',DATES[0],50)],True)
 r=run(f);first=next(c for c in r['cashflows']if c['date']==DATES[0]);last=r['cashflows'][-1]
 assert first['principal_collections']==100 and first['purchase_cash']==50
 assert first['closing_reinvestment_cash']==50 and first['principal_available']==0
 assert first['interest_available']==10 # warehouse's earlier coupon belongs to prior owner
 assert last['principal_available']==100 and last['interest_available']==2
 assert r['acquisitions'][0]['acquired_principal']==50 and r['ending_states'][0]['terminal_complete']
 assert all(c['reinvestment_conservation_residual']==0 for c in r['cashflows'])

def test_reinvestment_dispute_and_cash_failure_no_retry():
 a,b=asset(),asset('b','h');f=fixture([a,b],[assignment(),assignment('next','b',DATES[0],50)],True)
 r=run(f);assert r['acquisitions'][0]['reason']=='INSUFFICIENT_PRINCIPAL_CASH'
 assert sum(c['principal_available']for c in r['cashflows'])==100
 f['asset_paths'][2]['disputed']=True;r=run(f)
 assert r['acquisitions'][0]['reason']=='ACQUISITION_ELIGIBILITY_FAILED'

def test_pending_recovery_not_terminal_and_interest_scenario():
 f=fixture();f['obligor_paths'][0]['conditional_pd']=1;f['asset_paths'][0]['recovery_delay_days']=1000;r=run(f)
 assert not r['ending_states'][0]['terminal_complete'] and r['ending_states'][0]['pending_recoveries'][0]['gross']==40
 assert sum(c['recovery_gross']for c in r['cashflows'])==0
 f=fixture();f['asset_paths'][0]['interest_multiplier']=2
 assert sum(c['interest_available']for c in run(f)['cashflows'])==13

def test_innovation_completeness_and_temporal_contracts():
 f=fixture();f['shared_innovations'].append(dict(scenario_id='s',date=DATES[0],channel_id='unused',uniform=.5))
 with pytest.raises(CreditRiskError,match='STRUCTURED_ASSET_PATH'):run(f)
 f=fixture();f['scenario_contract']['dates']=[DATES[1]]
 with pytest.raises(CreditRiskError,match='STRUCTURED_CALENDAR'):run(f)
 f=fixture();f['asset_paths'][0]['conditional_prepayment_probability']=1.1
 with pytest.raises(CreditRiskError,match='STRUCTURED_INPUT'):run(f)

def test_context_calendar_and_conditioning_guard():
 for context in [dict(conditioning='TTC'),dict(horizon=1),dict(time_unit='years')]:
  with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):simulate_asset_pool(**fixture(),context=context)
 assert simulate_asset_pool(**fixture(),context=dict(conditioning='PIT',horizon=365,time_unit='days')).value['origin']==D

def test_malformed_bound_pool_controlled_error():
 f=fixture();f['asset_pool']={'sha256':'not-a-binding'}
 with pytest.raises(CreditRiskError,match='STRUCTURED_ARTIFACT'):run(f)
