from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._simulation_cashflows import build_debt_schedule,apply_cash_waterfall


def terms(**kw):
    d=dict(facility_id='f',currency='EUR',principal=100,rate_type='fixed',annual_rate=.1,rate_convention='nominal_simple',day_count='ACT/365F',amortization='bullet',maturity_date='2023-01-01');d.update(kw);return d


def calendar():
    return [dict(start_date='2021-01-01',end_date='2022-01-01',payment_date='2022-01-01'),dict(start_date='2022-01-01',end_date='2023-01-01',payment_date='2023-01-01')]


def test_bullet_equal_principal_and_annuity_against_independent_present_value():
    b=build_debt_schedule(terms(),calendar()).value['schedule']
    assert [r['net_payment_due']for r in b]==pytest.approx([10,110])
    a=build_debt_schedule(terms(amortization='annuity'),calendar()).value
    assert a['constant_payment']==pytest.approx(121/2.1)
    assert [r['net_payment_due']for r in a['schedule']]==pytest.approx([121/2.1]*2)
    assert a['schedule'][-1]['closing_principal']==0
    e=build_debt_schedule(terms(amortization='equal_principal'),calendar()).value['schedule']
    assert [r['net_payment_due']for r in e]==pytest.approx([60,55])
    z=build_debt_schedule(terms(amortization='annuity',annual_rate=0),calendar()).value['schedule']
    assert [r['net_payment_due']for r in z]==pytest.approx([50,50])


def test_floating_fixings_negative_interest_and_no_implicit_refinancing():
    c=calendar()
    for r in c:r['fixing_date']=r['start_date']
    b=build_debt_schedule(terms(rate_type='floating',spread=.01),c,base_rate_fixings=[dict(fixing_date='2021-01-01',base_rate=-.02),dict(fixing_date='2022-01-01',base_rate=.03)]).value['schedule']
    assert b[0]['debt_service_due']==0 and b[0]['lender_refund_due']==pytest.approx(1)
    assert b[1]['net_payment_due']==pytest.approx(104)
    c=calendar()
    for r in c:r['principal_due']=20
    with pytest.raises(CreditRiskError,match='SIMULATION_AMORTIZATION'):build_debt_schedule(terms(amortization='custom'),c)
    c[1]['start_date']='2022-01-02'
    with pytest.raises(CreditRiskError,match='SIMULATION_CALENDAR'):build_debt_schedule(terms(),c)


def policy():
    return dict(policy_id='wf',version='1',initial_reserve=20,reserve_priority=2,excess_reserve_policy='retain',arrears_policy='carry_without_interest',claims=[dict(claim_id='senior',priority=1,reserve_eligible=True),dict(claim_id='junior',priority=3,reserve_eligible=False)])


def periods():
    return [dict(scenario_id='s',date='2022-01-01',currency='EUR',cash_available=100,reserve_target=20,amounts_due=dict(senior=110,junior=10)),dict(scenario_id='s',date='2023-01-01',currency='EUR',cash_available=130,reserve_target=20,amounts_due=dict(senior=100,junior=10))]


def test_reserve_priority_carry_once_and_cash_conservation():
    r=apply_cash_waterfall(periods(),policy(),currency='EUR').value['periods']
    assert r[0]['payments'][0]['payment_from_reserve']==10
    assert r[0]['payments'][1]['closing_arrears']==10
    assert r[0]['closing_reserve']==10
    assert r[1]['reserve_topup']==10
    assert r[1]['payments'][1]['total_due']==20
    assert r[1]['payments'][1]['closing_arrears']==0
    assert all(x['cash_identity_residual']==0 and x['residual_equity_cash']==0 for x in r)
    assert apply_cash_waterfall(periods()[::-1],policy(),currency='EUR').value['periods']==r


def test_restricted_reserve_and_explicit_release():
    p=policy();p['claims'][0]['reserve_eligible']=False
    one=periods()[:1];one[0].update(cash_available=0,reserve_target=0)
    r=apply_cash_waterfall(one,p,currency='EUR').value['periods'][0]
    assert r['closing_reserve']==20 and r['payments'][0]['total_paid']==0
    p['excess_reserve_policy']='release_before_claims'
    r=apply_cash_waterfall(one,p,currency='EUR').value['periods'][0]
    assert r['reserve_release']==20 and r['payments'][0]['total_paid']==20
    p['reserve_priority']=1
    with pytest.raises(CreditRiskError,match='SIMULATION_WATERFALL'):apply_cash_waterfall(one,p,currency='EUR')


def test_reserve_cures_coupon_shortfall_but_unfunded_maturity_defaults():
    from creditratingsystem import detect_cashflow_default
    debt=build_debt_schedule(terms(),calendar()).value['schedule']
    p=dict(policy_id='single',version='1',initial_reserve=1,reserve_priority=2,excess_reserve_policy='retain',arrears_policy='carry_without_interest',claims=[dict(claim_id='debt',priority=1,reserve_eligible=True)])
    cash=[dict(scenario_id='s',date=r['payment_date'],currency='EUR',cash_available=c,reserve_target=0,amounts_due=dict(debt=r['debt_service_due']))for r,c in zip(debt,[9,10])]
    liquidity=apply_cash_waterfall(cash,p,currency='EUR').value['periods']
    assert liquidity[0]['payments'][0]['closing_arrears']==pytest.approx(0,abs=1e-12)
    assert liquidity[1]['payments'][0]['closing_arrears']==pytest.approx(100)
    default_policy=dict(policy_id='reference',version='1',basis='BASEL_CRE36_REFERENCE',assessment_level='obligor',dscr_threshold=1,ltv_threshold=1,dscr_comparison='lt',ltv_comparison='gt',past_due_days_threshold=90,past_due_comparison='gt',past_due_rule_reference='CRE36.68',materiality_rule_reference='declared',utp_rule_reference='declared-economic-assessment',dual_test_as_utp=True,economic_joint_timing='SAME_DATE',default_state='ABSORBING_FIRST_DEFAULT')
    initial=dict(scenario_id='s',entity_id='o',facility_id='f',date='2021-01-01',currency='EUR',cfads=0,debt_service_due=0,debt_outstanding=100,asset_value=80,liquidity_buffer_available=1,materiality_met=False,material_days_past_due=0,utp_indicators=[])
    rows=[initial]
    for d,c,l in zip(debt,cash,liquidity):
        rows.append(dict(initial,date=d['payment_date'],cfads=c['cash_available'],debt_service_due=d['debt_service_due'],liquidity_buffer_available=l['opening_reserve']))
    result=detect_cashflow_default(rows,default_policy,currency='EUR').value
    assert result['first_default_events'][0]['first_default_date']=='2023-01-01'
    assert result['first_default_events'][0]['last_nondefault_date']=='2022-01-01'
