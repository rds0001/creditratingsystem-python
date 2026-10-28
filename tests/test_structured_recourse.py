from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._structured_pool import build_credit_asset_pool
from creditratingsystem._structured_paths import simulate_asset_pool
from creditratingsystem._structured_recourse import covered_bond_dual_recourse
from test_structured_paths import fixture,asset,assignment,D,DATES

def fixture_cover(kind='MORTGAGE_COVER',extension=False,default=D):
 cls='RESIDENTIAL_MORTGAGE'if kind=='MORTGAGE_COVER'else'PUBLIC_SECTOR_CLAIM'
 a=asset();a['asset_class']=cls
 p=dict(profile_id='cover',version='1',known_at=D,currency='EUR',product_kind=kind,pool_mode='STATIC',allowed_asset_classes=[cls],maximum_dpd=30,exclude_disputed=True,eligible_recourse=['NONE'],revolving_until=None)
 cover=build_credit_asset_pool([a],[assignment()],p,pool_id='pool',as_of=D,information_cutoff=D).value
 f=fixture();f['asset_pool']=cover;paths=simulate_asset_pool(**f).value
 bond=dict(bond_id='covered',version='1',currency='EUR',legal_profile_id='caller-law-1',principal=100,original_maturity=DATES[-1],extended_maturity='2024-07-01'if extension else None,interest_schedule={},extension_interest_schedule={'2024-07-01':5}if extension else {},extension_rule='ISSUER_DEFAULT_AND_DECLARED_TRIGGER'if extension else'NONE',cover_cash_convention='SEGREGATED_UNTIL_ISSUER_DEFAULT',estate_rule='RESIDUAL_AFTER_COMPLETE_COVER',innovation_hash=paths['innovation_hash'])
 issuer=[dict(scenario_id='s',issuer_default_date=default,issuer_capacity={DATES[-1]:100},extension_triggered=extension,estate_payment_date='2024-07-01'if extension else DATES[-1],estate_recovery_fraction=.5,estate_capacity=1000)]
 return dict(cover_pool=cover,cover_pool_paths=paths,bond_contract=bond,issuer_paths=issuer),f

def run(f):return covered_bond_dual_recourse(**f).value

def test_full_cover_capped_and_nondefault_issuer_only():
 f,_=fixture_cover();r=run(f);e=r['ending_states'][0]
 assert e['cover_payment']==100 and e['estate_payment']==0 and e['ultimate_shortfall']==0 and e['unused_cover_cash']==8
 f['issuer_paths'][0]['issuer_default_date']=None;r=run(f);e=r['ending_states'][0]
 assert e['issuer_payment']==100 and e['cover_payment']==0 and e['estate_payment']==0 and e['unused_cover_cash']==108

def test_joint_issuer_cover_default_capped_residual_estate():
 f,poolf=fixture_cover();poolf['obligor_paths'][0]['conditional_pd']=1;f['cover_pool_paths']=simulate_asset_pool(**poolf).value
 r=run(f);e=r['ending_states'][0]
 assert e['cover_payment']==35 and e['estate_payment']==32.5 and e['ultimate_shortfall']==32.5
 assert sum(x['issuer_payment']+x['cover_payment']+x['estate_payment']for x in r['cashflows'])==67.5
 assert all(x['cover_conservation_residual']==0 for x in r['cashflows'])
 f['issuer_paths'][0]['estate_capacity']=10;assert run(f)['ending_states'][0]['ultimate_shortfall']==55

def test_public_cover_route_and_extension_cash_timing():
 f,_=fixture_cover('PUBLIC_SECTOR_COVER',True);r=run(f);e=r['ending_states'][0]
 assert e['extension_activated'] and e['contractual_claim']==105 and e['cover_payment']==105
 assert e['original_date_principal_deferral_or_shortfall']==100 and e['effective_maturity_principal_shortfall']==0
 assert all(row['principal_paid']==0 for row in r['cashflows']if row['date']<'2024-07-01')
 assert e['unused_cover_cash']==3 and e['ultimate_shortfall']==0

def test_extension_not_activated_does_not_charge_extra_coupon():
 f,_=fixture_cover(extension=True);f['issuer_paths'][0]['extension_triggered']=False;r=run(f);e=r['ending_states'][0]
 assert e['contractual_claim']==100 and e['effective_maturity']==DATES[-1] and e['cover_payment']==100

def test_delayed_liquidity_distinct_from_ultimate_and_no_extra_gain():
 f,_=fixture_cover(default=None);f['issuer_paths'][0]['issuer_capacity']={'2024-07-01':120};f['issuer_paths'][0]['estate_payment_date']='2024-07-01';r=run(f);e=r['ending_states'][0]
 assert e['effective_maturity_principal_shortfall']==100 and e['ultimate_shortfall']==0 and e['issuer_payment']==100

def test_route_extension_settlement_and_binding_guards():
 f,_=fixture_cover(extension=True);f['issuer_paths'][0]['issuer_default_date']=None
 with pytest.raises(CreditRiskError,match='STRUCTURED_EXTENSION'):run(f)
 f,_=fixture_cover();f['issuer_paths'][0]['estate_payment_date']=DATES[0]
 with pytest.raises(CreditRiskError,match='STRUCTURED_CALENDAR'):run(f)
 f,_=fixture_cover();f['bond_contract']['innovation_hash']='wrong'
 with pytest.raises(CreditRiskError,match='STRUCTURED_SHARED_PATHS'):run(f)
 f,poolf=fixture_cover();poolf['obligor_paths'][0]['conditional_pd']=1;poolf['asset_paths'][0]['recovery_delay_days']=1000;f['cover_pool_paths']=simulate_asset_pool(**poolf).value
 with pytest.raises(CreditRiskError,match='STRUCTURED_UNRESOLVED'):run(f)

def test_settlement_horizon_context_guard():
 f,_=fixture_cover()
 with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):covered_bond_dual_recourse(**f,context=dict(horizon=1,time_unit='years'))
 assert covered_bond_dual_recourse(**f,context=dict(horizon=365,time_unit='days')).value['horizon_end']==DATES[-1]
