from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._structured_paths import simulate_asset_pool
from creditratingsystem._structured_waterfall import run_tranche_waterfall
from test_structured_paths import fixture, DATES

def pool(f=None):return simulate_asset_pool(**(f or fixture())).value

def profile():
 return dict(structure_id='senior-junior',version='1',currency='EUR',payment_dates=DATES,tranches=[dict(tranche_id='senior',seniority=1,notional=80,legal_maturity=DATES[-1],coupon_fractions={DATES[0]:.05,DATES[1]:.05}),dict(tranche_id='junior',seniority=2,notional=20,legal_maturity=DATES[-1],coupon_fractions={DATES[0]:.05,DATES[1]:.05})],fees={d:0 for d in DATES},initial_reserve=0,reserve_target=0,liquidity_limit=0,principal_mode='SEQUENTIAL',principal_for_interest='TRIGGER_ONLY',excess_interest_to_principal=True,trigger_seniority=1,oc_threshold=1,ic_threshold=1,terminal_policy='REQUIRE_COMPLETE')
def run(p=None,paths=None):return run_tranche_waterfall(paths or pool(),p or profile()).value

def test_performing_sequential_actual_opening_principal_coupons():
 r=run();a=r['tranche_cashflows'];assert [(x['interest_paid'],x['principal_paid'])for x in a]==[(4,40),(1,0),(2,40),(1,20)]
 assert all(x['ultimate_principal_loss']==0 and not x['legal_maturity_failure'] for x in r['ending_states'])
 assert all(abs(c['cash_conservation_residual'])<1e-12 for c in r['cash_accounts'])

def test_pro_rata_principal_allocation():
 p=profile();p['principal_mode']='PRO_RATA';r=run(p);rows=r['tranche_cashflows']
 assert rows[0]['principal_paid']==32 and rows[1]['principal_paid']==8
 assert rows[2]['interest_paid']==pytest.approx(2.4) and rows[3]['interest_paid']==pytest.approx(.6)
 assert rows[2]['principal_paid']==48 and rows[3]['principal_paid']==12

def test_default_recovery_fees_and_terminal_loss_priority():
 f=fixture();f['obligor_paths'][0]['conditional_pd']=1
 p=profile();p['tranches'][0]['coupon_fractions']={d:0 for d in DATES};p['tranches'][1]['coupon_fractions']={d:0 for d in DATES};r=run(p,pool(f))
 assert sum(x['fees_paid']for x in r['cash_accounts'])==5
 end={x['tranche_id']:x for x in r['ending_states']}
 assert end['senior']['principal_paid']==35 and end['senior']['ultimate_principal_loss']==45
 assert end['junior']['ultimate_principal_loss']==20
 assert sum(x['principal_write_down']for x in r['tranche_cashflows'])==65

def test_trigger_switches_pro_rata_to_sequential_and_diverts_principal():
 p=profile();p['principal_mode']='PRO_RATA';p['oc_threshold']=2
 f=fixture();f['asset_paths'][0]['interest_multiplier']=0;r=run(p,pool(f));first=r['cash_accounts'][0]
 assert first['trigger_breached'] and first['interest_paid']==5 and first['principal_paid']==35
 assert r['tranche_cashflows'][0]['principal_paid']==35 and r['tranche_cashflows'][1]['principal_paid']==0

def test_reserve_is_cash_support_and_liquidity_is_repayable():
 f=fixture();f['asset_paths'][0]['interest_multiplier']=0;p=profile();p['principal_for_interest']='NEVER';p['oc_threshold']=0;p['ic_threshold']=0;p['initial_reserve']=5
 r=run(p,pool(f));first=r['cash_accounts'][0]
 assert first['interest_paid']==5 and first['closing_reserve']==0 and first['principal_paid']==40
 p['initial_reserve']=0;p['liquidity_limit']=5;r=run(p,pool(f));first=r['cash_accounts'][0]
 assert first['liquidity_draw']==5 and first['liquidity_repayment']==5 and first['principal_paid']==35
 assert all(abs(x['cash_conservation_residual'])<1e-12 for x in r['cash_accounts'])

def test_late_interest_cure_does_not_erase_timely_failure():
 f=fixture();f['asset_paths'][0]['interest_multiplier']=0;p=profile();p['principal_for_interest']='NEVER';p['oc_threshold']=0;p['ic_threshold']=0;r=run(p,pool(f))
 assert r['tranche_cashflows'][0]['closing_interest_arrears']==4
 assert all(x['timely_interest_failure']for x in r['ending_states'])
 assert all(x['interest_arrears']==0 for x in r['ending_states'])
 assert sum(x['ultimate_principal_loss']for x in r['ending_states'])==7

def test_open_tail_no_false_ultimate_and_incomplete_terminal_rejected():
 f=fixture();f['obligor_paths'][0]['conditional_pd']=1;f['asset_paths'][0]['recovery_delay_days']=1000;paths=pool(f)
 with pytest.raises(CreditRiskError,match='STRUCTURED_UNRESOLVED'):run(paths=paths)
 p=profile();p['terminal_policy']='KEEP_OPEN';r=run(p,paths)
 assert all(x['ultimate_principal_loss']is None for x in r['ending_states'])
 assert all(x['principal_write_down']==0 for x in r['tranche_cashflows'])

def test_legal_maturity_failure_persists_after_cure():
 p=profile();p['tranches'][0]['legal_maturity']=DATES[0];p['tranches'][0]['coupon_fractions'][DATES[1]]=0;r=run(p)
 senior=r['ending_states'][0];assert senior['legal_maturity_failure'] and senior['ultimate_principal_loss']==0

def test_structure_identity_calendar_and_context_rejections():
 p=profile();p['tranches'][1]['seniority']=1
 with pytest.raises(CreditRiskError,match='STRUCTURED_TRANCHE'):run(p)
 p=profile();p['payment_dates']=[DATES[0]]
 with pytest.raises(CreditRiskError,match='STRUCTURED_CALENDAR'):run(p)
 with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):run_tranche_waterfall(pool(),profile(),context={'conditioning':'TTC'})

def test_calendar_horizon_and_numerical_ratio_guard():
 with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):run_tranche_waterfall(pool(),profile(),context={'horizon':1,'time_unit':'years'})
 p=profile();p['tranches'][0]['notional']=1e-310
 with pytest.raises(CreditRiskError,match='NUMERICAL_FAILURE'):run(p)
