from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.simulation import detect_cashflow_default


def policy(**changes):
    return dict(policy_id='default-reference',version='1',basis='BASEL_CRE36_REFERENCE',assessment_level='obligor',dscr_threshold=1,ltv_threshold=1,dscr_comparison='lt',ltv_comparison='gt',past_due_days_threshold=90,past_due_comparison='gt',past_due_rule_reference='CRE36.68(2)',materiality_rule_reference='declared-national-policy',utp_rule_reference='approved-financial-assessment',dual_test_as_utp=True,economic_joint_timing='SAME_DATE',default_state='ABSORBING_FIRST_DEFAULT',**changes)


def row(**changes):
    result=dict(scenario_id='s',entity_id='o',facility_id='f',date='2024-01-01',currency='EUR',cfads=110,debt_service_due=100,debt_outstanding=100,asset_value=120,liquidity_buffer_available=0,materiality_met=False,material_days_past_due=0,utp_indicators=[])
    result.update(changes);return result


def test_dual_test_all_four_states_and_boundary_conventions():
    rows=[row(scenario_id=str(i),cfads=cf,asset_value=asset)for i,(cf,asset)in enumerate([(110,120),(90,120),(110,80),(90,80)])]
    result=detect_cashflow_default(rows,policy(),currency='EUR').value
    assert [r['dscr_breach']for r in result['facility_tests']]==[False,True,False,True]
    assert [r['ltv_breach']for r in result['facility_tests']]==[False,False,True,True]
    assert [r['defaulted']for r in result['first_default_events']]==[False,False,False,True]
    boundary=detect_cashflow_default([row(cfads=100,asset_value=100)],policy(),currency='EUR').value
    assert not boundary['first_default_events'][0]['defaulted']
    inclusive=policy();inclusive.update(dscr_comparison='le',ltv_comparison='ge')
    assert detect_cashflow_default([row(cfads=100,asset_value=100)],inclusive,currency='EUR').value['facility_tests'][0]['economic_dual_breach']
    assert not detect_cashflow_default([row(cfads=100,asset_value=100)],inclusive,currency='EUR').value['first_default_events'][0]['defaulted']


def test_adequate_value_does_not_cancel_utp_or_past_due_90_91_boundary():
    rows=[row(scenario_id='utp',asset_value=1000,utp_indicators=['distressed-restructuring']),row(scenario_id='days',asset_value=1000,materiality_met=True,material_days_past_due=90),row(scenario_id='days',date='2024-01-02',asset_value=1000,materiality_met=True,material_days_past_due=91),row(scenario_id='days',date='2024-01-03',asset_value=1000)]
    value=detect_cashflow_default(rows,policy(),currency='EUR').value
    assert value['first_default_events'][0]['first_default_date']=='2024-01-02'
    assert value['first_default_events'][1]['reasons']==['UNLIKELY_TO_PAY']
    assert value['assessments'][2]['in_default']and not value['assessments'][2]['regulatory_trigger']


def test_obligor_propagation_retail_scope_and_no_cross_date_dual_combination():
    rows=[row(utp_indicators=['bankruptcy']),row(facility_id='f2')]
    r=detect_cashflow_default(rows,policy(),currency='EUR').value
    assert len(r['first_default_events'])==1 and r['first_default_events'][0]['defaulted']
    p=policy();p.update(assessment_level='facility',facility_scope_eligibility='RETAIL_APPROVED')
    r=detect_cashflow_default(rows,p,currency='EUR').value
    assert [e['defaulted']for e in r['first_default_events']]==[True,False]
    separated=[row(cfads=90),row(date='2024-02-01',asset_value=80)]
    assert not detect_cashflow_default(separated,policy(),currency='EUR').value['first_default_events'][0]['defaulted']
    assert detect_cashflow_default(rows[::-1],p,currency='EUR').value==r


def test_zero_denominators_and_declared_dual_role():
    rows=[row(scenario_id='zero_due',debt_service_due=0,cfads=-10,asset_value=0),row(scenario_id='zero_asset',cfads=90,asset_value=0),row(scenario_id='no_debt',debt_service_due=0,debt_outstanding=0,asset_value=0)]
    r=detect_cashflow_default(rows,policy(),currency='EUR').value
    assert [e['defaulted']for e in r['first_default_events']]==[False,True,False]
    p=policy();p['dual_test_as_utp']=False
    result=detect_cashflow_default([row(cfads=90,asset_value=80)],p,currency='EUR').value
    assert result['facility_tests'][0]['economic_dual_breach']and not result['first_default_events'][0]['defaulted']


def test_explicit_materiality_clock_policy_and_duplicate_guards():
    with pytest.raises(CreditRiskError,match='SIMULATION_ARREARS'):detect_cashflow_default([row(material_days_past_due=91)],policy(),currency='EUR')
    with pytest.raises(CreditRiskError,match='SIMULATION_ARREARS'):detect_cashflow_default([row(materiality_met=True,material_days_past_due=90),row(date='2024-01-02',materiality_met=True,material_days_past_due=100)],policy(),currency='EUR')
    with pytest.raises(CreditRiskError,match='SIMULATION_ARREARS'):detect_cashflow_default([row(materiality_met=True,material_days_past_due=90),row(date='2024-01-02',materiality_met=True,material_days_past_due=80)],policy(),currency='EUR')
    with pytest.raises(CreditRiskError,match='SIMULATION_DUPLICATE'):detect_cashflow_default([row(),row()],policy(),currency='EUR')
    p=policy();p['past_due_days_threshold']=180
    with pytest.raises(CreditRiskError,match='SIMULATION_POLICY'):detect_cashflow_default([row()],p,currency='EUR')
    p['supervisory_exception_eligibility']='RETAIL_OR_PSE_APPROVED'
    assert not detect_cashflow_default([row(materiality_met=True,material_days_past_due=180)],p,currency='EUR').value['first_default_events'][0]['defaulted']
    p['assessment_level']='facility'
    with pytest.raises(CreditRiskError,match='SIMULATION_POLICY'):detect_cashflow_default([row()],p,currency='EUR')


def test_default_time_censoring_and_initial_risk_set():
    records=[row(scenario_id="initial",utp_indicators=["bankruptcy"]),row(scenario_id="later"),row(scenario_id="later",date="2024-02-01",utp_indicators=["bankruptcy"]),row(scenario_id="none")]
    events=detect_cashflow_default(records,policy(),currency="EUR").value["first_default_events"]
    assert events[0]["initial_default"] and events[0]["last_nondefault_date"] is None
    assert events[0]["default_time_status"]=="LEFT_CENSORED_AT_INITIAL_STATE"
    assert not events[1]["initial_default"] and events[1]["last_nondefault_date"]=="2024-01-01"
    assert events[1]["default_time_status"]=="DETECTED_AT_GRID_ENDPOINT"
    assert events[1]["observed_through"]=="2024-02-01"
    assert events[2]["default_time_status"]=="NO_DEFAULT_OBSERVED"
    assert events[2]["last_nondefault_date"]=="2024-01-01"


def test_available_liquidity_remedies_economic_signal_but_never_other_utp():
    r=detect_cashflow_default([row(cfads=90,asset_value=80,liquidity_buffer_available=10)],policy(),currency='EUR').value
    assert r['facility_tests'][0]['economic_dual_breach']
    assert not r['facility_tests'][0]['utp_from_dual_test']
    assert not r['first_default_events'][0]['defaulted']
    assert detect_cashflow_default([row(cfads=90,asset_value=80,liquidity_buffer_available=10,utp_indicators=['bankruptcy'])],policy(),currency='EUR').value['first_default_events'][0]['defaulted']
