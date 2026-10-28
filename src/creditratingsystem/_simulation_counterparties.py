"""Shared first-default paths and dated contract cashflow interruption."""
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
import math
from .core import _fail,_result,_fingerprint,_context_value
from .data import _rows,_required,_date
from .simulation import _number,_text


def _prob(value,name,uniform=False):
    v=_number(value,name,0)
    if v>1 or (uniform and v==1):_fail('SIMULATION_PROBABILITY',name+' is outside probability or [0,1) innovation domain')
    return v


def simulate_contract_counterparties(counterparty_paths, contracts, cure_uniforms, *, currency, context=None):
    """Map shared counterparty defaults to interrupted, cured or replaced payments."""
    paths=_rows(counterparty_paths,'counterparty_paths');contracts=_rows(contracts,'contracts');draws=_rows(cure_uniforms,'cure_uniforms');_text(currency,'currency')
    ctx=_context_value(context)
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Counterparty cashflow currency conflicts with context')
    if ctx.get('probability_measure')is not None and ctx['probability_measure']!='PHYSICAL':_fail('TARGET_MISMATCH','Counterparty paths require physical probabilities')
    groups=defaultdict(list);seen=set()
    for row in paths:
        _required(row,['scenario_id','counterparty_id','period_start','period_end','conditional_pd','uniform','probability_measure'])
        start=_date(row['period_start'],'period_start');end=_date(row['period_end'],'period_end');key=(row['scenario_id'],row['counterparty_id'],row['period_end'])
        if row['probability_measure']!='PHYSICAL':_fail('TARGET_MISMATCH','Counterparty conditional PD must be physical P')
        if end<=start or key in seen:_fail('SIMULATION_CALENDAR','Positive unique counterparty periods required')
        _prob(row['conditional_pd'],'conditional_pd');_prob(row['uniform'],'uniform',True);seen.add(key);groups[(row['scenario_id'],row['counterparty_id'])].append(row)
    if not groups or not contracts:_fail('SIMULATION_COUNTERPARTY','Nonempty paths and contracts required')
    states={};events=[]
    for key,periods in sorted(groups.items()):
        periods=sorted(periods,key=lambda r:r['period_end']);last=None;first=None;grid={periods[0]['period_start']}
        for row in periods:
            if last is not None and row['period_start']!=last:_fail('SIMULATION_CALENDAR','Counterparty periods must be contiguous')
            last=row['period_end'];grid.add(last)
            if first is None and row['uniform']<row['conditional_pd']:first=last
        states[key]=dict(default=first,grid=grid)
        events.append(dict(scenario_id=key[0],counterparty_id=key[1],first_default_date=first,defaulted=first is not None,initially_performing=True,observed_through=last,default_time_convention='PERIOD_END_BEFORE_PAYMENT'))
    innovations={}
    for row in draws:
        _required(row,['scenario_id','contract_id','uniform']);key=(row['scenario_id'],row['contract_id'])
        if key in innovations:_fail('SIMULATION_DUPLICATE','Duplicate contract cure innovation')
        innovations[key]=_prob(row['uniform'],'cure_uniform',True)
    ids=set();definitions=[]
    for c in contracts:
        _required(c,['contract_id','counterparty_id','currency','default_payment_timing','cure_probability','cure_delay_days','replacement_enabled','replacement_delay_days','replacement_payment_fraction','cashflows'])
        if c['contract_id']in ids:_fail('SIMULATION_DUPLICATE','Contract IDs must be unique')
        ids.add(c['contract_id']);_text(c['default_payment_timing'],'default_payment_timing')
        if c['currency']!=currency or c['default_payment_timing']!='BEFORE_DUE':_fail('SIMULATION_COUNTERPARTY','One reporting currency and explicit BEFORE_DUE default timing required')
        cp=_prob(c['cure_probability'],'cure_probability');cd=_number(c['cure_delay_days'],'cure_delay_days',0);rd=_number(c['replacement_delay_days'],'replacement_delay_days',0);rf=_number(c['replacement_payment_fraction'],'replacement_payment_fraction',0)
        if cd!=int(cd)or rd!=int(rd)or not isinstance(c['replacement_enabled'],bool):_fail('SIMULATION_COUNTERPARTY','Integer cure/replacement delays and Boolean replacement flag required')
        if not c['replacement_enabled']and(rd!=0 or rf!=0):_fail('SIMULATION_COUNTERPARTY','Disabled replacement requires zero delay and zero fraction')
        cash=_rows(c['cashflows'],'cashflows');dates=set()
        if not cash:_fail('SIMULATION_COUNTERPARTY','Contract requires explicit cashflow dates')
        for flow in cash:
            _required(flow,['date','amount']);_date(flow['date'],'date');_number(flow['amount'],'amount',0)
            if flow['date']in dates:_fail('SIMULATION_DUPLICATE','Duplicate contract payment date')
            dates.add(flow['date'])
        definitions.append((c,cp,int(cd),int(rd),rf,sorted(cash,key=lambda r:r['date'])))
    scenarios=sorted({k[0]for k in states});used=set();outputs=[];contract_events=[]
    for scenario in scenarios:
        for c,cp,cd,rd,rf,cash in sorted(definitions,key=lambda d:d[0]['contract_id']):
            key=(scenario,c['counterparty_id']);drawkey=(scenario,c['contract_id'])
            if key not in states or drawkey not in innovations:_fail('SIMULATION_COUNTERPARTY','Every contract requires counterparty path and cure innovation in every scenario')
            state=states[key];used.add(drawkey);first=state['default'];cured=False;cure_date=None;replacement_date=None
            if first is not None:
                default_date=_date(first,'first_default_date');cured=innovations[drawkey]<cp
                try:
                    if cured:cure_date=(default_date+timedelta(days=cd)).isoformat()
                    elif c['replacement_enabled']:replacement_date=(default_date+timedelta(days=cd+rd)).isoformat()
                except OverflowError:_fail('SIMULATION_CALENDAR','Cure or replacement date outside supported calendar')
            contract_events.append(dict(scenario_id=scenario,contract_id=c['contract_id'],counterparty_id=c['counterparty_id'],counterparty_default_date=first,contract_cured=cured,cure_date=cure_date,replacement_date=replacement_date,cure_uniform=innovations[drawkey]))
            for flow in cash:
                date=flow['date']
                if date not in state['grid']:_fail('SIMULATION_CALENDAR','Payment dates must lie on the supplied counterparty assessment grid')
                if first is None or date<first:factor=1.0;status='ORIGINAL_PERFORMING'
                elif cure_date is not None and date>=cure_date:factor=1.0;status='CONTRACT_CURED'
                elif replacement_date is not None and date>=replacement_date:factor=rf;status='REPLACED'
                else:factor=0.0;status='INTERRUPTED'
                amount=float(flow['amount']);received=amount*factor
                if not math.isfinite(received):_fail('NUMERICAL_FAILURE','Replacement cashflow exceeds finite representation')
                outputs.append(dict(scenario_id=scenario,contract_id=c['contract_id'],counterparty_id=c['counterparty_id'],date=date,currency=currency,scheduled_cash=amount,received_cash=received,cash_change=received-amount,payment_fraction=factor,status=status))
    if used!=set(innovations):_fail('SIMULATION_COUNTERPARTY','Unused cure innovations rejected')
    return _result(dict(counterparty_events=events,contract_events=contract_events,cashflows=outputs,currency=currency),'simulation.simulate_contract_counterparties','contract_cashflows',ctx,diagnostics=dict(paths_fingerprint=_fingerprint(paths),contracts_fingerprint=_fingerprint(contracts),innovations_fingerprint=_fingerprint(draws)),limitations=('All counterparties are initially performing. Conditional default draws share a counterparty event across its contracts; dependence is supplied, not generated independently per contract. Events occur at assessment-grid endpoints before payments. Cure waits interrupt payments; successful cure restores future payments, without inventing arrears recoveries. Failed cure precedes the replacement delay. Replacement is assumed performing for this horizon, with explicit payment fraction. Counterparty default affects operating receipts and must not also be counted as a borrower default; the borrower has its own debt-service and UTP assessment.',))
