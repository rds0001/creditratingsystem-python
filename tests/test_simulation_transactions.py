from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.core import _fingerprint
from creditratingsystem._simulation_transactions import simulate_transactions


def fixture():
    cube=dict(schema_version='1.0',kind='ScenarioCube',axes=['scenario','time','factor'],scenario_ids=['s'],time_grid=[1.,2.],factor_ids=['market'],values=[[[1.],[1.]]],weights=[1.],units=['relative_index'],origin='2021-01-01',frequency='annual',probability_measure='PHYSICAL',sampling_contract=dict(law='FORECAST_CONDITIONAL',conditioning_scope='PIT'))
    calendar=[dict(start_date='2021-01-01',end_date='2022-01-01',payment_date='2022-01-01'),dict(start_date='2022-01-01',end_date='2023-01-01',payment_date='2023-01-01')]
    for i,row in enumerate(calendar):row['scenario_time']=i+1.
    base=[]
    for cash in (9,10):
     base.append(dict(base_quantity=1,base_unit_price=cash,base_unit_variable_cost=0,base_fixed_cost=0,volume_multiplier=1,price_multiplier=1,variable_cost_multiplier=1,fixed_cost_multiplier=1,working_capital_revenue_ratio=0,cash_taxes=0,capex=0,asset_disposals=0,other_operating_cash=0))
    asset=dict(asset_id='a',currency='EUR',valuation_date='2021-01-01',initial_value=80,salvage_value=80,initial_age_years=2,economic_life_years=10,depreciation_model='straight_line',idiosyncratic_volatility=0,probability_measure='PHYSICAL',year_basis_days=365)
    params=dict(currency='EUR',reserve_id='reserve',initial_reserve=1,reserve_targets=[0,0],loans=[dict(terms=dict(facility_id='f',currency='EUR',principal=100,rate_type='fixed',annual_rate=.1,rate_convention='nominal_simple',day_count='ACT/365F',amortization='bullet',maturity_date='2023-01-01'),priority=1,deficit_cost_share=1)],operating_contract=dict(contract_id='operations',version='1',currency='EUR',period_unit='year',quantity_unit='units',initial_working_capital=0,include_asset_disposals_in_cfads=False),operating_base=base,operating_bindings={},assets=[dict(parameters=asset,market_binding=dict(factor_id='market',unit='relative_index',transform='level'),innovation_channel='asset-z',recovery=dict(delay_days=0,realization_fraction=1,cost=0,annual_discount_rate=0))],default_policy=dict(policy_id='default-reference',version='1',basis='BASEL_CRE36_REFERENCE',assessment_level='obligor',dscr_threshold=1,ltv_threshold=1,dscr_comparison='lt',ltv_comparison='gt',past_due_days_threshold=90,past_due_comparison='gt',past_due_rule_reference='CRE36.68(2)',materiality_rule_reference='declared',utp_rule_reference='declared',dual_test_as_utp=True,economic_joint_timing='SAME_DATE',default_state='ABSORBING_FIRST_DEFAULT'),materiality=dict(absolute_threshold=1,relative_threshold=.01,comparison='gt_both',continuous_between_grid_dates=True),external_evidence=[dict(scenario_id='s',date=date,facility_id='external',materiality_met=False,material_days_past_due=0,utp_indicators=[])for date in ['2021-01-01','2022-01-01','2023-01-01']],recovery=dict(policy=dict(policy_id='recovery',version='1',currency='EUR',allocation='LEGAL_CAP',cost_rule='PROCEEDS_FIRST_EXPLICIT_DEFICIT_SHARES',completion='ULTIMATE_COMPLETE'),reserve_recovery='REMAINING_RESERVE_AT_DEFAULT'),sampling_design='IID_UNWEIGHTED',legal_pooling='FULL_OBLIGOR_CROSS_COLLATERALIZED',arrears_interest='NO_ADDITIONAL_ARREARS_INTEREST',payment_allocation='INTEREST_BEFORE_PRINCIPAL')
    snapshot=dict(snapshot_id='p',version='1',parameters=dict(p=params))
    recipe=dict(transaction_id='t',version='1',subject_id='o',parameter_id='p',calendar=calendar,parameter_hash=_fingerprint(params),cube_hash=_fingerprint(cube))
    innovations=[dict(scenario_id='s',date=date,channel_id='asset-z',value=0.)for date in ['2022-01-01','2023-01-01']]
    return [recipe],cube,innovations,snapshot


def run(args):
    recipes,cube,innovations,snapshot=args
    for recipe in recipes:
        recipe["parameter_hash"]=_fingerprint(snapshot["parameters"][recipe["parameter_id"]])
        recipe["cube_hash"]=_fingerprint(cube)
    return simulate_transactions(*args).value


def test_reserve_cures_coupon_but_does_not_refinance_maturity():
    r=run(fixture())['transactions'][0];path=r['paths'][0]
    assert path['default_event']['first_default_date']=='2023-01-01'
    assert path['going_concern_ledger'][0]['liquidity']['closing_reserve']==0
    assert path['ead_at_default']==100
    assert r['risk']['expected_loss']==20 and r['risk']['conditional_mean_lgd']==.2
    assert len(r['recovery']['claim_results'])==1


def test_partial_payment_uses_actual_principal_not_contractual_zero_balance():
    args=fixture();p=args[3]['parameters']['p'];p['operating_base'][1]['base_unit_price']=60
    p['assets'][0]['parameters'].update(initial_value=40,salvage_value=40)
    r=run(args)['transactions'][0]
    assert r['paths'][0]['ead_at_default']==50
    assert r['risk']['expected_loss']==10
    assert r['paths'][0]['going_concern_ledger'][1]['facilities'][0]['principal_paid']==50


def test_good_collateral_never_vetoes_material_arrears():
    args=fixture();p=args[3]['parameters']['p'];p['initial_reserve']=0
    for row in p['operating_base']:row['base_unit_price']=0
    p['assets'][0]['parameters'].update(initial_value=1000,salvage_value=1000)
    r=run(args)['transactions'][0]
    assert r['paths'][0]['default_event']['first_default_date']=='2023-01-01'
    assert r['paths'][0]['default_event']['reasons']==['MATERIAL_PAST_DUE']
    assert r['paths'][0]['ead_at_default']==120
    assert r['risk']['pd']==1 and r['risk']['expected_loss']==0


def test_initial_default_excluded_and_unused_reserve_recovered_once():
    args=fixture();p=args[3]['parameters']['p'];p['external_evidence'][0]['utp_indicators']=['bankruptcy']
    r=run(args)['transactions'][0]
    assert r['risk']['status']=='EMPTY_RISK_SET'
    assert r['paths'][0]['going_concern_ledger']==[]
    assert r['recovery']['claim_results'][0]['economic_loss']==19
    assert r['recovery']['pool_results'][0]['realized_proceeds']==81


def test_shared_liquidity_not_multiplied_by_facility_count():
    args=fixture();p=args[3]['parameters']['p'];second=deepcopy(p['loans'][0]);second['terms']['facility_id']='f2';second['priority']=2
    second['deficit_cost_share']=.5;p['loans'][0]['deficit_cost_share']=.5;p['loans'].append(second)
    r=run(args)['transactions'][0]
    assert r['paths'][0]['default_event']['first_default_date']=='2022-01-01'
    assert r['paths'][0]['ead_at_default']==210
    assert r['risk']['expected_loss']==130
    assert sum(x['payment_from_reserve']for x in r['paths'][0]['going_concern_ledger'][0]['liquidity']['payments'])==1


def test_no_default_and_frozen_recipe_integrity():
    args=fixture();args[3]['parameters']['p']['operating_base'][1]['base_unit_price']=120
    r=run(args)['transactions'][0]
    assert r['risk']['pd']==0 and r['risk']['expected_loss']==0
    assert r['recovery']['claim_results']==[]
    args=fixture();args[3]['parameters']['p']['initial_reserve']=2
    with pytest.raises(CreditRiskError,match='TRANSACTION_SEAL'):simulate_transactions(*args)


def test_transaction_order_invariance_and_duplicate_resource_rejection():
    args=fixture();recipe=deepcopy(args[0][0]);recipe.update(transaction_id='t2',subject_id='o2',parameter_id='p2');p=deepcopy(args[3]['parameters']['p'])
    p['reserve_id']='reserve2';p['loans'][0]['terms']['facility_id']='f2';p['assets'][0]['parameters']['asset_id']='a2';p['operating_contract']['contract_id']='operations2'
    args[0].append(recipe);args[3]['parameters']['p2']=p
    a=run(args);args[0].reverse();b=run(args)
    assert a==b
    p['assets'][0]['parameters']['asset_id']='a'
    with pytest.raises(CreditRiskError,match='TRANSACTION_DUPLICATE'):run(args)


def test_counterparty_interrupts_receipts_once_and_borrower_default_is_separate():
    args=fixture();p=args[3]['parameters']['p'];dates=['2022-01-01','2023-01-01']
    paths=[dict(scenario_id='s',counterparty_id='tenant',period_start=start,period_end=end,conditional_pd=q,innovation_channel='cp-u',probability_measure='PHYSICAL')for start,end,q in [('2021-01-01',dates[0],1),(dates[0],dates[1],0)]]
    contract=dict(contract_id='rent',counterparty_id='tenant',currency='EUR',default_payment_timing='BEFORE_DUE',cure_probability=0,cure_delay_days=0,replacement_enabled=False,replacement_delay_days=0,replacement_payment_fraction=0,cashflows=[dict(date=dates[0],amount=9),dict(date=dates[1],amount=10)])
    p['counterparties']=dict(paths=paths,contracts=[contract],cure_draws=[dict(scenario_id='s',contract_id='rent',innovation_channel='cure-u',innovation_date='2021-01-01')],baseline_receipts='INCLUDED_ONCE_IN_OPERATING_CFADS')
    args[2].extend([dict(scenario_id='s',date=d,channel_id='cp-u',value=.5)for d in dates]+[dict(scenario_id='s',date='2021-01-01',channel_id='cure-u',value=.5)])
    r=run(args)['transactions'][0]
    assert r['paths'][0]['default_event']['first_default_date']==dates[0]
    assert r['paths'][0]['going_concern_ledger'][0]['cfads']==0
    assert r['paths'][0]['ead_at_default']==109
    assert r['risk']['expected_loss']==29 and r['risk']['pd']==1


def test_calendar_horizon_and_sampling_law_are_not_relabelled():
    args=fixture();args[0][0]['calendar'][1].update(end_date='2022-01-02',payment_date='2022-01-02')
    with pytest.raises(CreditRiskError,match='TRANSACTION_CALENDAR'):run(args)
    args=fixture();args[1]['sampling_contract']=dict(law='STATIONARY_GAUSSIAN',conditioning_scope='TTC',cycle_law_id='declared-equilibrium')
    r=run(args)
    assert r['sampling_contract']==args[1]['sampling_contract']
    assert r['transactions'][0]['risk']['target']['scenario_sampling_contract']['conditioning_scope']=='TTC'
