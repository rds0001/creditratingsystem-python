import math
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._simulation_drivers import simulate_operating_cashflows,simulate_asset_values


def operating_contract():
    return dict(contract_id='operations',version='1',currency='EUR',period_unit='year',quantity_unit='units',initial_working_capital=10,include_asset_disposals_in_cfads=False)


def operating_rows():
    r=dict(scenario_id='s',period_start='2021-01-01',period_end='2022-01-01',currency='EUR',base_quantity=10,base_unit_price=20,base_unit_variable_cost=6,base_fixed_cost=20,volume_multiplier=1,price_multiplier=1,variable_cost_multiplier=1,fixed_cost_multiplier=1,working_capital_revenue_ratio=.1,cash_taxes=10,capex=15,asset_disposals=7,other_operating_cash=0)
    return [r,dict(r,period_start='2022-01-01',period_end='2023-01-01',volume_multiplier=.5,cash_taxes=5,capex=0)]


def asset_parameters(**kw):
    r=dict(asset_id='a',currency='EUR',valuation_date='2021-01-01',initial_value=100,salvage_value=20,initial_age_years=2,economic_life_years=10,depreciation_model='straight_line',idiosyncratic_volatility=0,probability_measure='PHYSICAL',year_basis_days=365);r.update(kw);return r


def asset_rows():
    return [dict(scenario_id='s',date='2022-01-01',market_index=1.2,innovation=1),dict(scenario_id='s',date='2023-01-01',market_index=1,innovation=-1)]


def test_operating_bridge_and_working_capital_release():
    v=simulate_operating_cashflows(operating_rows(),operating_contract()).value['periods']
    assert [r['cfads']for r in v]==[85,55]
    assert [r['change_working_capital']for r in v]==[10,-10]
    assert v[0]['revenue']==200 and v[0]['variable_cost']==60 and v[0]['ebitda']==120
    p=operating_contract();p['include_asset_disposals_in_cfads']=True
    assert [r['cfads']for r in simulate_operating_cashflows(operating_rows(),p).value['periods']]==[92,62]
    assert simulate_operating_cashflows(operating_rows()[::-1],p).value==simulate_operating_cashflows(operating_rows(),p).value


def test_physical_ageing_and_shared_innovation_path_oracles():
    v=simulate_asset_values(asset_parameters(),asset_rows()).value['paths']
    assert [r['ageing_baseline']for r in v]==[90,80]
    assert [r['asset_value']for r in v]==[108,80]
    v=simulate_asset_values(asset_parameters(idiosyncratic_volatility=.2),asset_rows()).value['paths']
    assert v[0]['asset_value']==pytest.approx(108*math.exp(.18))
    assert v[1]['asset_value']==pytest.approx(80*math.exp(-.04))
    for r in v:assert 100+r['ageing_contribution']+r['market_contribution']+r['idiosyncratic_contribution']==pytest.approx(r['asset_value'])
    assert simulate_asset_values(asset_parameters(),asset_rows()[::-1]).value==simulate_asset_values(asset_parameters(),asset_rows()).value


def test_geometric_ageing_and_target_calendar_guards():
    v=simulate_asset_values(asset_parameters(depreciation_model='geometric_to_salvage',annual_depreciation_rate=math.log(2)),asset_rows()).value['paths']
    assert [r['ageing_baseline']for r in v]==pytest.approx([60,40])
    with pytest.raises(CreditRiskError,match='SIMULATION_ASSET'):simulate_asset_values(asset_parameters(probability_measure='Q'),asset_rows())
    with pytest.raises(CreditRiskError,match='SIMULATION_CALENDAR'):simulate_asset_values(asset_parameters(),asset_rows()*2)
    rows=operating_rows();rows[1]['period_start']='2022-02-01'
    with pytest.raises(CreditRiskError,match='SIMULATION_CALENDAR'):simulate_operating_cashflows(rows,operating_contract())


def test_physical_calculation_context_matches_public_measure_vocabulary():
    from creditratingsystem import build_calculation_context
    ctx=build_calculation_context(currency='EUR',probability_measure='PHYSICAL')
    assert simulate_asset_values(asset_parameters(),asset_rows(),context=ctx).value['paths'][0]['probability_measure']=='PHYSICAL'
    q=build_calculation_context(currency='EUR',probability_measure='RISK_NEUTRAL')
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):simulate_asset_values(asset_parameters(),asset_rows(),context=q)
