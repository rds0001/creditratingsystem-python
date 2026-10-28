from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._decisioning_economics import compute_limit_frontier, compute_risk_based_price_grid, evaluate_treatment_candidates
D='2023-01-01'
def target():return dict(event='default',horizon=365,time_unit='days',probability_measure='PHYSICAL',conditioning='initially_performing')
def profile():return dict(profile_id='p',version='1',known_at=D,currency='EUR')
def offer(id='a',amount=100):return dict(offer_id=id,version='1',known_at=D,currency='EUR',subject_id='s',amount=amount,tenor_days=365,assumption_reference='approved-scenario')
def limit_profile():return dict(profile(),target=target(),maximum_amount=300,maximum_ead=300,maximum_expected_loss=20,maximum_concentration=.5,concentration_denominator=1000,existing_group_exposure=100,tolerance=1e-12)
def limit_offer(id,amount,pd,cap=150):return dict(offer(id,amount),target=target(),pd=pd,lgd=.5,ead=amount,risk_conditioning='CONDITIONAL_ON_THIS_OFFER',risk_reference='model-frozen',affordability_status='AFFORDABLE',affordability_maximum_amount=cap,eligibility_status='ELIGIBLE')
def price_profile():return dict(profile(),default_timing='BEFORE_DUE',prepayment_timing='AFTER_COUPON',funding_treatment='DISCOUNT_CURVE_ONLY',selection_assumption='QUOTE_CONDITIONED_INPUTS_NOT_CAUSAL')
def quote():return dict(offer(),annual_rate=.1,target=target(),risk_conditioning='CONDITIONAL_ON_THIS_ACCEPTED_QUOTE',risk_reference='quote-conditioned',takeup_probability=.5,takeup_reference='declared-demand',upfront_fee=0,origination_cost=0,offer_cost=1,periods=[dict(payment_date='2024-01-01',scheduled_principal=100,conditional_default_probability=0,conditional_prepayment_probability=0,lgd=.4,recovery_date='2024-01-01',payment_discount_factor=1/1.1,recovery_discount_factor=1/1.1,servicing_cost=0,prepayment_fee=0)])
def treatment_profile():return dict(profile(),subject_id='s',baseline_id='no-action',baseline_present_value=20,identification='CONDITIONAL_SCENARIO_NOT_CAUSAL',allowed_actions=['ARRANGEMENT'])
def treatment(id='a',df=1):return dict(treatment_id=id,version='1',known_at=D,subject_id='s',currency='EUR',action='ARRANGEMENT',prediction_reference='conditional-model',support_status='SUPPORTED',affordability_status='AFFORDABLE',eligibility_status='ELIGIBLE',uptake_probability=.5,offer_cost=1,accepted_upfront_cost=2,rejected_present_value=10,cashflows=[dict(cashflow_id='f',date='2024-01-01',expected_recovery=100,expected_cost=0,discount_factor=df)])

def test_limit_affordability_and_nonmonotone_enumeration():
 r=compute_limit_frontier([limit_offer('a',100,.5),limit_offer('b',150,.1),limit_offer('c',200,.05)],limit_profile(),as_of=D).value
 assert [x['feasible']for x in r['candidates']]==[False,True,False]
 assert r['candidates'][0]['expected_loss']==25 and r['candidates'][1]['expected_loss']==7.5
 assert r['candidates'][2]['failures']==['AFFORDABILITY_AMOUNT']
 assert r['frontier_offer_ids']==['b'] and r['maximum_feasible_amount']==150
 assert r['candidates'][1]['binding_constraints']==['AFFORDABILITY_AMOUNT']

def test_limit_frontier_dominance_and_unknown_not_zero_risk():
 rows=[limit_offer('a',100,.2,300),limit_offer('b',150,.1,300),limit_offer('c',200,.12,300)]
 r=compute_limit_frontier(rows,limit_profile(),as_of=D).value
 assert r['frontier_offer_ids']==['b','c']
 rows[0]['affordability_maximum_amount']=None
 assert not compute_limit_frontier(rows,limit_profile(),as_of=D).value['candidates'][0]['feasible']
 rows[0]['risk_conditioning']='BASELINE'
 with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):compute_limit_frontier(rows,limit_profile(),as_of=D)

def test_price_funding_break_even_without_risk_or_other_cost():
 q=quote();q['offer_cost']=0;r=compute_risk_based_price_grid([q],price_profile(),as_of=D).value['candidates'][0]
 assert r['conditional_booked_npv']==pytest.approx(0,abs=1e-12)
 assert r['expected_npv_per_offer']==pytest.approx(0,abs=1e-12)
 assert r['cashflows'][0]['expected_interest']==10

def test_price_competing_hazards_no_double_count_and_dated_recovery():
 q=quote();p=q['periods'][0];p.update(conditional_default_probability=.2,conditional_prepayment_probability=.3,prepayment_fee=2)
 r=compute_risk_based_price_grid([q],price_profile(),as_of=D).value['candidates'][0]
 # default:12 recovery; prepaid:30 principal+.6 fee; survivors:50 principal;
 # all nondefaults pay8 interest. PV100.6/1.1 -100 = -8.54545.
 assert r['conditional_booked_npv']==pytest.approx(100.6/1.1-100)
 assert r['expected_npv_per_offer']==pytest.approx(.5*(100.6/1.1-100)-1)
 assert r['conditional_discounted_expected_loss']==pytest.approx(8/1.1)
 assert r['lifetime_pd']==.2 and r['prepayment_probability']==.3
 assert r['terminal_performing_probability']==.5
 p['recovery_date']='2025-01-01';p['recovery_discount_factor']=1/1.1**2
 later=compute_risk_based_price_grid([q],price_profile(),as_of=D).value['candidates'][0]
 assert later['conditional_booked_npv']==pytest.approx(88.6/1.1+12/1.1**2-100)

def test_price_quote_selection_different_risk_and_unsupported_contracts():
 a=quote();b=deepcopy(a);b['offer_id']='b';b['annual_rate']=.2;b['periods'][0]['conditional_default_probability']=.8
 r=compute_risk_based_price_grid([a,b],price_profile(),as_of=D).value['candidates']
 assert r[1]['annual_rate']>r[0]['annual_rate'] and r[1]['expected_npv_per_offer']<r[0]['expected_npv_per_offer']
 for field,value in [('conditional_default_probability',1.1),('scheduled_principal',50)]:
  q=quote();q['periods'][0][field]=value
  with pytest.raises(CreditRiskError):compute_risk_based_price_grid([q],price_profile(),as_of=D)
 q=quote();q['target']['horizon']=366
 with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):compute_risk_based_price_grid([q],price_profile(),as_of=D)
 q=quote();q['periods'][0]['recovery_discount_factor']=.5
 with pytest.raises(CreditRiskError,match='DECISION_PRICING_CONTRACT'):compute_risk_based_price_grid([q],price_profile(),as_of=D)

def test_treatment_present_value_uptake_and_affordability():
 a,b=treatment(),treatment('b',.5)
 r=evaluate_treatment_candidates([a,b],treatment_profile(),as_of=D).value
 assert r['candidates'][0]['accepted_present_value']==98
 assert r['candidates'][0]['expected_present_value']==53
 assert r['candidates'][1]['expected_present_value']==28
 assert r['feasible_ranking']==['a','b']
 a['affordability_status']='UNAFFORDABLE'
 r=evaluate_treatment_candidates([a,b],treatment_profile(),as_of=D).value
 assert r['feasible_ranking']==['b'] and r['candidates'][0]['expected_present_value']==53
 b['support_status']='UNSUPPORTED'
 r=evaluate_treatment_candidates([a,b],treatment_profile(),as_of=D).value
 assert r['status']=='INFEASIBLE' and r['candidates'][1]['expected_present_value']is None

def test_dates_currency_and_identity():
 q=quote();q['known_at']='2023-01-02'
 with pytest.raises(CreditRiskError,match='DECISION_FUTURE_INPUT'):compute_risk_based_price_grid([q],price_profile(),as_of=D)
 with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):evaluate_treatment_candidates([treatment()],treatment_profile(),as_of=D,context={'currency':'USD'})
 with pytest.raises(CreditRiskError,match='DECISION_DUPLICATE'):compute_limit_frontier([limit_offer('a',100,.1)]*2,limit_profile(),as_of=D)

def test_two_period_competing_risk_survival_and_amortization():
 q=quote();q['tenor_days']=730;q['target']['horizon']=730
 p=q['periods'][0];p.update(scheduled_principal=40,conditional_default_probability=.1,conditional_prepayment_probability=.2,payment_discount_factor=1,recovery_discount_factor=1)
 # 2024 is leap year; choose next endpoint365 days later for equal coupons.
 p2=deepcopy(p);p2.update(payment_date='2024-12-31',recovery_date='2024-12-31',scheduled_principal=60,conditional_default_probability=.2,conditional_prepayment_probability=.1)
 q['periods'].append(p2)
 r=compute_risk_based_price_grid([q],price_profile(),as_of=D).value['candidates'][0]
 assert r['lifetime_pd']==pytest.approx(.24) and r['prepayment_probability']==pytest.approx(.27)
 assert r['terminal_performing_probability']==pytest.approx(.49)
 assert r['cashflows'][0]['discounted_net_cashflow']==pytest.approx(63)
 assert r['cashflows'][1]['discounted_net_cashflow']==pytest.approx(42)
 assert r['conditional_booked_npv']==pytest.approx(5)


def test_context_horizon_and_conditioning_match_quote_target():
 for context in ({'horizon':366},{'time_unit':'years'},{'conditioning':'other'}):
  with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):compute_risk_based_price_grid([quote()],price_profile(),as_of=D,context=context)
  with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):compute_limit_frontier([limit_offer('a',100,.1)],limit_profile(),as_of=D,context=context)
