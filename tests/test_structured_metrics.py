from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._structured_metrics import estimate_tranche_credit_metrics, stress_tranche_structure
from creditratingsystem._structured_waterfall import run_tranche_waterfall
from test_structured_waterfall import pool,profile,fixture,DATES

def target(event='ULTIMATE_PRINCIPAL'):return dict(event=event,horizon=365,time_unit='days',probability_measure='PHYSICAL',conditioning='PIT')
def paths(p=None,f=None):return run_tranche_waterfall(pool(f),p or profile()).value
DF={d:1 for d in DATES}
def metric(w=None,event='ULTIMATE_PRINCIPAL',df=None):return estimate_tranche_credit_metrics(w or paths(),target(event),df or DF).value

def test_ultimate_zero_and_iid_endpoint_uncertainty():
 f=fixture();f['scenario_contract']['sampling_design']='IID_UNWEIGHTED';r=metric(paths(f=f));m=r['metrics'][0]
 assert m['event_probability']==0 and m['expected_ultimate_principal_loss']==0
 assert m['event_probability_interval']==pytest.approx([0,.975])
 assert m['discounted_cash_shortfall_mcse']is None

def test_liquidity_delay_without_ultimate_loss_different_targets():
 p=profile();p['principal_for_interest']='NEVER';p['oc_threshold']=p['ic_threshold']=0
 f=fixture();f['asset_paths'][0]['interest_multiplier']=0;f['asset_paths'][1]['interest_multiplier']=4
 w=paths(p,f);timely=metric(w,'TIMELY_INTEREST');ultimate=metric(w)
 assert all(m['event_probability']==1 for m in timely['metrics'])
 assert all(m['event_probability']==0 and m['expected_ultimate_principal_loss']==0 for m in ultimate['metrics'])
 assert all(m['expected_discounted_contractual_cash_shortfall']==0 for m in timely['metrics'])


def test_static_senior_junior_closed_form_and_loss_outside_interest_event():
 f=fixture();f['obligor_paths'][0]['conditional_pd']=1;p=profile()
 for t in p['tranches']:t['coupon_fractions']=dict.fromkeys(DATES,0)
 r=metric(paths(p,f),'TIMELY_INTEREST');a,b=r['metrics']
 assert a['event_probability']==0 and a['expected_ultimate_principal_loss']==45 and a['expected_principal_loss_outside_selected_event']==45
 assert a['conditional_principal_loss_given_selected_event']is None and b['expected_ultimate_principal_loss']==20
 assert a['expected_discounted_contractual_cash_shortfall']==45


def test_discounted_cured_coupon_delay_includes_timing_and_early_principal_benefit():
 p=profile();p['tranches']=[dict(tranche_id='senior',seniority=1,notional=100,legal_maturity=DATES[-1],coupon_fractions={DATES[0]:.05,DATES[1]:0})];p['principal_for_interest']='NEVER';p['oc_threshold']=p['ic_threshold']=0
 f=fixture();f['asset_paths'][0]['interest_multiplier']=0;f['asset_paths'][1]['interest_multiplier']=2
 w=paths(p,f);r=metric(w,'TIMELY_INTEREST',{DATES[0]:1,DATES[1]:.9});m=r['metrics'][0]
 assert m['expected_ultimate_principal_loss']==0 and m['event_probability']==1
 # Principal 100 and the delayed first coupon 5 are both received at final date.
 assert m['expected_discounted_contractual_cash_shortfall']==pytest.approx(.5)


def test_weighted_two_path_loss_and_no_iid_interval():
 f=fixture();f['scenario_contract']['scenario_ids']=['s','t'];f['scenario_contract']['weights']=[.75,.25]
 for name in ['obligor_paths','asset_paths','shared_innovations']:
  extra=deepcopy(f[name])
  for row in extra:row['scenario_id']='t'
  f[name]+=extra
 f['obligor_paths'][2]['conditional_pd']=1;p=profile()
 for t in p['tranches']:t['coupon_fractions']=dict.fromkeys(DATES,0)
 r=metric(paths(p,f));m=r['metrics'][0]
 assert m['event_probability']==.25 and m['expected_ultimate_principal_loss']==11.25
 assert m['conditional_principal_loss_given_selected_event']==45 and m['event_probability_interval']is None


def test_paired_structure_stress_and_wrong_innovation_rejected():
 base=paths();p=profile();p['initial_reserve']=10;alt=paths(p)
 r=stress_tranche_structure(base,[dict(alternative_id='reserve',tranche_paths=alt,assumption='External funded support ten')],target(),DF).value
 assert r['comparisons'][0]['deltas'][0]['changes']['event_probability']==0
 f=fixture();f['shared_innovations'][0]['uniform']=.6;other=paths(f=f)
 with pytest.raises(CreditRiskError,match='STRUCTURED_SHARED_PATHS'):stress_tranche_structure(base,[dict(alternative_id='bad',tranche_paths=other,assumption='Unpaired')],target(),DF)


def test_incomplete_and_target_discount_rejections():
 f=fixture();f['obligor_paths'][0]['conditional_pd']=1;f['asset_paths'][0]['recovery_delay_days']=1000;p=profile();p['terminal_policy']='KEEP_OPEN';w=paths(p,f)
 with pytest.raises(CreditRiskError,match='STRUCTURED_UNRESOLVED'):metric(w)
 assert metric(w,'TIMELY_INTEREST')['metrics'][0]['expected_ultimate_principal_loss']is None
 bad=target();bad['horizon']=1
 with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):estimate_tranche_credit_metrics(paths(),bad,DF)
 with pytest.raises(CreditRiskError,match='STRUCTURED_DISCOUNT'):estimate_tranche_credit_metrics(paths(),target(),{DATES[0]:1})

def test_joint_default_stress_same_marginal_pool_loss_changes_tranche_risk():
 from test_structured_paths import asset,assignment
 f=fixture([asset(),asset('b','h')],[assignment(),assignment('b','b')])
 f['scenario_contract']['scenario_ids']=['s','t'];f['scenario_contract']['weights']=[.5,.5]
 for name in ['obligor_paths','asset_paths','shared_innovations']:
  extra=deepcopy(f[name])
  for row in extra:row['scenario_id']='t'
  f[name]+=extra
 for row in f['obligor_paths']:row['conditional_pd']=.5 if row['date']==DATES[0] else 0
 for row in f['asset_paths']:
  row.update(recovery_fraction=0,recovery_cost=0,interest_multiplier=0)
  if row['asset_id']=='b':row['prepayment_channel']='default-h'
 f['shared_innovations']=[r for r in f['shared_innovations']if r['channel_id']!='prepay-b']
 for row in f['shared_innovations']:
  if row['channel_id']=='default-g':row['uniform']=.25 if row['scenario_id']=='s'else .75
  if row['channel_id']=='default-h':row['uniform']=.75 if row['scenario_id']=='s'else .25
 p=profile();p['tranches'][0]['notional']=180
 for t in p['tranches']:t['coupon_fractions']=dict.fromkeys(DATES,0)
 anti=paths(p,f);positive=deepcopy(f)
 for row in positive['obligor_paths']:
  if row['default_group_id']=='h':row['default_channel']='default-g'
 joint=paths(p,positive)
 r=stress_tranche_structure(anti,[dict(alternative_id='joint',tranche_paths=joint,assumption='Same marginal defaults; perfectly shared group innovation replaces opposite group innovation')],target(),DF).value
 a=r['baseline']['metrics'];b=r['comparisons'][0]['metrics']['metrics']
 assert sum(x['expected_ultimate_principal_loss']for x in a)==100==sum(x['expected_ultimate_principal_loss']for x in b)
 assert [x['expected_ultimate_principal_loss']for x in a]==[80,20]
 assert [x['expected_ultimate_principal_loss']for x in b]==[90,10]
 assert [x['event_probability']for x in a]==[1,1] and [x['event_probability']for x in b]==[.5,.5]
