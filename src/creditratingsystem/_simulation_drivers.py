"""Explicit operating-ledger and physical asset-value path models."""
import math
from copy import deepcopy
from collections import defaultdict
from .core import _fail,_result,_fingerprint,_context_value
from .data import _rows,_required,_date
from .simulation import _number,_text


def simulate_operating_cashflows(driver_paths, operating_contract, *, context=None):
    """Translate supplied volume, price and cost scenarios into reconciled CFADS."""
    rows=_rows(driver_paths,'driver_paths');contract=deepcopy(operating_contract)
    if not isinstance(contract,dict):_fail('SIMULATION_OPERATING','Explicit operating contract required')
    _required(contract,['contract_id','version','currency','period_unit','quantity_unit','initial_working_capital','include_asset_disposals_in_cfads'])
    for key in ('contract_id','version','currency','period_unit','quantity_unit'):_text(contract[key],key)
    if contract['period_unit']not in ('day','month','quarter','year')or not isinstance(contract['include_asset_disposals_in_cfads'],bool):_fail('SIMULATION_OPERATING','Declare period unit and Boolean disposal treatment')
    initial=_number(contract['initial_working_capital'],'initial_working_capital',0);ctx=_context_value(context)
    if ctx.get('currency')is not None and ctx['currency']!=contract['currency']:_fail('TARGET_MISMATCH','Operating currency conflicts with context')
    groups=defaultdict(list);seen=set()
    for row in rows:
        _required(row,['scenario_id','period_start','period_end','currency','base_quantity','base_unit_price','base_unit_variable_cost','base_fixed_cost','volume_multiplier','price_multiplier','variable_cost_multiplier','fixed_cost_multiplier','working_capital_revenue_ratio','cash_taxes','capex','asset_disposals','other_operating_cash'])
        start=_date(row['period_start'],'period_start');end=_date(row['period_end'],'period_end');key=(row['scenario_id'],row['period_end'])
        if end<=start or key in seen:_fail('SIMULATION_CALENDAR','Positive unique operating periods required')
        if row['currency']!=contract['currency']:_fail('TARGET_MISMATCH','Operating records must share reporting currency')
        seen.add(key);groups[row['scenario_id']].append(row)
    outputs=[]
    for scenario,periods in sorted(groups.items()):
        nwc=initial;last_end=None
        for row in sorted(periods,key=lambda r:r['period_end']):
            if last_end is not None and row['period_start']!=last_end:_fail('SIMULATION_CALENDAR','Operating periods must be contiguous within each scenario')
            last_end=row['period_end'];v={k:_number(row[k],k,0)for k in ('base_quantity','base_unit_price','base_unit_variable_cost','base_fixed_cost','volume_multiplier','price_multiplier','variable_cost_multiplier','fixed_cost_multiplier','working_capital_revenue_ratio','capex','asset_disposals')}
            quantity=v['base_quantity']*v['volume_multiplier'];price=v['base_unit_price']*v['price_multiplier'];revenue=quantity*price;variable_cost=quantity*v['base_unit_variable_cost']*v['variable_cost_multiplier'];fixed_cost=v['base_fixed_cost']*v['fixed_cost_multiplier'];tax=_number(row['cash_taxes'],'cash_taxes');other=_number(row['other_operating_cash'],'other_operating_cash')
            closing_nwc=revenue*v['working_capital_revenue_ratio'];delta_nwc=closing_nwc-nwc;ebitda=revenue-variable_cost-fixed_cost
            disposal=v['asset_disposals']if contract['include_asset_disposals_in_cfads']else 0.0
            cfads=ebitda-tax-v['capex']-delta_nwc+other+disposal
            if not all(math.isfinite(x)for x in (quantity,price,revenue,variable_cost,fixed_cost,closing_nwc,delta_nwc,ebitda,cfads)):_fail('NUMERICAL_FAILURE','Operating bridge exceeds finite representation')
            outputs.append(dict(scenario_id=scenario,period_start=row['period_start'],period_end=row['period_end'],currency=contract['currency'],quantity=quantity,unit_price=price,revenue=revenue,variable_cost=variable_cost,fixed_cost=fixed_cost,ebitda=ebitda,cash_taxes=tax,capex=v['capex'],opening_working_capital=nwc,closing_working_capital=closing_nwc,change_working_capital=delta_nwc,other_operating_cash=other,asset_disposals=v['asset_disposals'],disposals_in_cfads=disposal,cfads=cfads))
            nwc=closing_nwc
    return _result(dict(periods=outputs,contract=contract,contract_hash=_fingerprint(contract)),'simulation.simulate_operating_cashflows','operating_cashflows',ctx,diagnostics=dict(input_fingerprint=_fingerprint(rows)),limitations=('An explicit quantity-price, variable/fixed-cost and working-capital-to-period-revenue model, not a universal sector model. Inputs are totals for declared periods and consistent quantity units. Taxes are supplied cash amounts including refunds; no tax-loss or interest-tax shield is inferred. Financing, reserves and debt service are excluded from operating CFADS. Driver scenarios may be dependent but no independent shocks are silently generated.',))


def simulate_asset_values(asset_parameters, market_paths, *, context=None):
    """Combine declared ageing, market indices and shared idiosyncratic innovations."""
    rows=_rows(market_paths,'market_paths');params=deepcopy(asset_parameters)
    if not isinstance(params,dict):_fail('SIMULATION_ASSET','Explicit physical asset parameters required')
    _required(params,['asset_id','currency','valuation_date','initial_value','salvage_value','initial_age_years','economic_life_years','depreciation_model','idiosyncratic_volatility','probability_measure','year_basis_days'])
    
    for key in ('currency','probability_measure','depreciation_model'):_text(params[key],key)
    valuation=_date(params['valuation_date'],'valuation_date')
    if params['probability_measure']!='PHYSICAL' or params['depreciation_model']not in ('straight_line','geometric_to_salvage'):_fail('SIMULATION_ASSET','Physical P-measure and explicit supported ageing model required')
    initial=_number(params['initial_value'],'initial_value',0);salvage=_number(params['salvage_value'],'salvage_value',0);age=_number(params['initial_age_years'],'initial_age_years',0);life=_number(params['economic_life_years'],'economic_life_years',0);sigma=_number(params['idiosyncratic_volatility'],'idiosyncratic_volatility',0);basis=_number(params['year_basis_days'],'year_basis_days',0)
    if initial<salvage or life<=age or basis<=0:_fail('SIMULATION_ASSET','Initial value >= salvage, remaining economic life and positive day basis required')
    depreciation=None
    if params['depreciation_model']=='geometric_to_salvage':
        _required(params,['annual_depreciation_rate']);depreciation=_number(params['annual_depreciation_rate'],'annual_depreciation_rate',0)
    ctx=_context_value(context)
    if ctx.get('probability_measure')is not None and ctx['probability_measure']!='PHYSICAL':_fail('TARGET_MISMATCH','Physical asset paths cannot be labelled risk-neutral')
    if ctx.get('currency')is not None and ctx['currency']!=params['currency']:_fail('TARGET_MISMATCH','Asset currency conflicts with context')
    groups=defaultdict(list);seen=set()
    for row in rows:
        _required(row,['scenario_id','date','market_index','innovation']);date=_date(row['date'],'date');key=(row['scenario_id'],row['date'])
        if date<=valuation or key in seen:_fail('SIMULATION_CALENDAR','Unique asset dates strictly after valuation required')
        seen.add(key);groups[row['scenario_id']].append(row)
    outputs=[]
    for scenario,periods in sorted(groups.items()):
        last=valuation;log_idio=0.0
        for row in sorted(periods,key=lambda r:r['date']):
            date=_date(row['date'],'date');dt=(date-last).days/basis;elapsed=(date-valuation).days/basis;market=_number(row['market_index'],'market_index',0);innovation=_number(row['innovation'],'innovation')
            increment=-.5*sigma*sigma*dt+sigma*math.sqrt(dt)*innovation;log_idio+=increment
            if not math.isfinite(log_idio)or not -709<log_idio<709:_fail('NUMERICAL_FAILURE','Asset innovation path exceeds supported finite log range')
            factor=math.exp(log_idio);remaining=max(life-age-elapsed,0)
            ageing=remaining/(life-age)if params['depreciation_model']=='straight_line'else math.exp(-depreciation*elapsed)
            baseline=salvage+(initial-salvage)*ageing;market_value=baseline*market;value=market_value*factor
            if not math.isfinite(value):_fail('NUMERICAL_FAILURE','Asset value exceeds finite representation')
            outputs.append(dict(scenario_id=scenario,asset_id=params['asset_id'],date=row['date'],currency=params['currency'],probability_measure='PHYSICAL',age_years=age+elapsed,remaining_life_years=remaining,ageing_baseline=baseline,ageing_contribution=baseline-initial,market_index=market,market_contribution=market_value-baseline,idiosyncratic_factor=factor,idiosyncratic_contribution=value-market_value,asset_value=value,year_fraction=dt,innovation=innovation))
            last=date
    return _result(dict(paths=outputs,parameters=params,parameters_hash=_fingerprint(params)),'simulation.simulate_asset_values','asset_values',ctx,diagnostics=dict(input_fingerprint=_fingerprint(rows)),limitations=('Physical market values are not collateral realisation proceeds. Salvage is an ageing baseline, not a guaranteed price floor: market and idiosyncratic factors also affect it. Idiosyncratic innovations must be standard normal with the declared joint dependence; the exp(-sigma^2 dt/2) adjustment gives unit mean under that law, not for an arbitrary stress sample. Market indices are levels relative to valuation, not incremental returns. Geometric ageing approaches salvage asymptotically; declared life remains a diagnostic.',))
