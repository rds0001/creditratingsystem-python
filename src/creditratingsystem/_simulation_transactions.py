"""Frozen shared-scenario cash-pool transaction execution."""
from copy import deepcopy
from datetime import timedelta, date as _calendar_date
from calendar import monthrange
import math
import numpy as np
from .core import _fail,_result,_context_value,_fingerprint,_numeric,CalcResult
from .data import _rows,_required,_date
from .simulation import _number,_text,detect_cashflow_default
from ._simulation_cashflows import build_debt_schedule,apply_cash_waterfall
from ._simulation_drivers import simulate_operating_cashflows,simulate_asset_values
from ._simulation_counterparties import simulate_contract_counterparties
from ._simulation_recovery import simulate_recovery_waterfall
from ._simulation_risk import estimate_simulation_risk


def _transaction_cube(value):
    cube=deepcopy(value.value if isinstance(value,CalcResult)else value)
    if not isinstance(cube,dict):_fail('TRANSACTION_CUBE','ScenarioCube required')
    _required(cube,['schema_version','kind','axes','scenario_ids','time_grid','factor_ids','values','weights','units','origin','frequency','probability_measure','sampling_contract'])
    if cube['schema_version']!='1.0'or cube['kind']!='ScenarioCube'or cube['axes']!=['scenario','time','factor']or cube['probability_measure']!='PHYSICAL':_fail('TRANSACTION_CUBE','Physical ScenarioCube with declared scenario/time/factor axes required')
    for key in ('scenario_ids','factor_ids'):
        ids=cube[key]
        if not isinstance(ids,list)or not ids or any(not isinstance(x,str)or not x for x in ids)or len(set(ids))!=len(ids):_fail('TRANSACTION_CUBE','Nonempty distinct axis IDs required')
    times=_numeric(cube['time_grid'],'time_grid',vector_only=True);values=_numeric(cube['values'],'values');weights=_numeric(cube['weights'],'weights',vector_only=True)
    if len(times)==0 or np.any(times<=0)or np.any(np.diff(times)<=0)or values.shape!=(len(cube['scenario_ids']),len(times),len(cube['factor_ids']))or weights.shape!=(len(cube['scenario_ids']),)or np.any(weights<0)or not math.isclose(math.fsum(weights),1,abs_tol=1e-12,rel_tol=0):_fail('TRANSACTION_CUBE','Aligned finite axes and normalized nonnegative scenario weights required')
    if not isinstance(cube['units'],list)or len(cube['units'])!=len(cube['factor_ids'])or any(not isinstance(u,str)or not u for u in cube['units']):_fail('TRANSACTION_CUBE','Explicit factor units required')
    origin=_date(cube['origin'],'origin')
    periods={'monthly':12,'quarterly':4,'annual':1}.get(cube['frequency'])
    if periods is None or not np.allclose(times,np.arange(1,len(times)+1)/periods,rtol=0,atol=1e-12):_fail('TRANSACTION_CALENDAR','Cube times must match declared monthly/quarterly/annual frequency')
    sampling=cube['sampling_contract']
    if not isinstance(sampling,dict)or sampling.get('law')not in ('FORECAST_CONDITIONAL','STATIONARY_GAUSSIAN','DECLARED_CYCLE_MIXTURE')or not sampling.get('conditioning_scope'):_fail('TRANSACTION_CUBE','Preserve explicit forecast/stationary/cycle-mixture law and conditioning scope')
    return cube,values,weights


def _bound_factor(binding,cube,values,s,t):
    if not isinstance(binding,dict):_fail('TRANSACTION_BINDING','Explicit factor binding required')
    _required(binding,['factor_id','unit','transform'])
    if binding['factor_id']not in cube['factor_ids']:_fail('TRANSACTION_BINDING','Unknown scenario factor')
    k=cube['factor_ids'].index(binding['factor_id'])
    if cube['units'][k]!=binding['unit']or binding['transform']not in ('level','exp'):_fail('TRANSACTION_BINDING','Factor units and supported transformation must match')
    v=float(values[s,t,k])
    if binding['transform']=='exp':
        if not -709<v<709:_fail('NUMERICAL_FAILURE','Factor exponential exceeds finite range')
        v=math.exp(v)
    return v


def simulate_transactions(transaction_recipes,scenario_cube,entity_innovations,parameter_snapshot,*,context=None):
    """Execute disjoint complete obligor cash pools on a common frozen scenario cube."""
    cube,values,weights=_transaction_cube(scenario_cube);recipes=_rows(transaction_recipes,'transaction_recipes');innovations=_rows(entity_innovations,'entity_innovations');snapshot=deepcopy(parameter_snapshot);ctx=_context_value(context)
    if not isinstance(snapshot,dict):_fail('TRANSACTION_SNAPSHOT','Versioned parameter snapshot required')
    _required(snapshot,['snapshot_id','version','parameters'])
    if not isinstance(snapshot['parameters'],dict):_fail('TRANSACTION_SNAPSHOT','Parameter map required')
    for k in ('snapshot_id','version'):_text(snapshot[k],k)
    if ctx.get('probability_measure')not in (None,'PHYSICAL'):_fail('TARGET_MISMATCH','Transaction simulation is physical')
    imap={}
    for row in innovations:
        _required(row,['scenario_id','date','channel_id','value']);_date(row['date'],'date')
        for k in ('scenario_id','channel_id'):_text(row[k],k)
        key=(row['scenario_id'],row['date'],row['channel_id'])
        if key in imap or row['scenario_id']not in cube['scenario_ids']:_fail('TRANSACTION_INNOVATION','Unique identified innovations on the shared scenario set required')
        imap[key]=_number(row['value'],'innovation')
    seen={k:set()for k in ('transaction','subject','asset','facility','reserve','contract','operation')};used=set();bundles=[];counterparty_registry={}
    for recipe in sorted(recipes,key=lambda r:r.get('transaction_id','')):
        _required(recipe,['transaction_id','version','subject_id','parameter_id','calendar','parameter_hash','cube_hash'])
        for k in ('transaction_id','version','subject_id','parameter_id'):_text(recipe[k],k)
        if recipe['parameter_id']not in snapshot['parameters']:_fail('TRANSACTION_SNAPSHOT','Missing parameter ID')
        params=deepcopy(snapshot['parameters'][recipe['parameter_id']])
        if recipe['parameter_hash']!=_fingerprint(params)or recipe['cube_hash']!=_fingerprint(cube):_fail('TRANSACTION_SEAL','Recipe frozen parameter/cube hashes must match')
        for k,value in [('transaction',recipe['transaction_id']),('subject',recipe['subject_id'])]:
            if value in seen[k]:_fail('TRANSACTION_DUPLICATE','Transactions must represent disjoint complete obligor cash pools')
            seen[k].add(value)
        _required(params,['currency','reserve_id','initial_reserve','reserve_targets','loans','operating_contract','operating_base','operating_bindings','assets','default_policy','materiality','external_evidence','recovery','sampling_design','legal_pooling','arrears_interest','payment_allocation'])
        currency=params['currency'];_text(currency,'currency')
        operation_id=params['operating_contract'].get('contract_id')
        _text(operation_id,'operating contract_id')
        if operation_id in seen['operation']:_fail('TRANSACTION_EFFECT','Operating cash source cannot be counted in several transactions')
        seen['operation'].add(operation_id)
        if ctx.get('currency')not in (None,currency):_fail('TARGET_MISMATCH','Transaction currency mismatch')
        if params['legal_pooling']!='FULL_OBLIGOR_CROSS_COLLATERALIZED'or params['arrears_interest']!='NO_ADDITIONAL_ARREARS_INTEREST'or params['payment_allocation']!='INTEREST_BEFORE_PRINCIPAL'or params['default_policy'].get('assessment_level')!='obligor':_fail('TRANSACTION_CONTRACT','Explicit whole-obligor pooling, scheduled-only interest, payment allocation and obligor default scope required')
        _text(params['reserve_id'],'reserve_id')
        if params['reserve_id']in seen['reserve']or params['reserve_id']in seen['asset']:_fail('TRANSACTION_DUPLICATE','Cash reserve cannot be reused across transactions')
        seen['reserve'].add(params['reserve_id']);initial_reserve=_number(params['initial_reserve'],'initial_reserve',0)
        calendar=_rows(recipe['calendar'],'calendar');h=len(cube['time_grid'])
        if len(calendar)!=h or len(params['reserve_targets'])!=h or len(params['operating_base'])!=h:_fail('TRANSACTION_CALENDAR','All transaction paths must cover the shared scenario horizon')
        dates=[];previous=cube['origin']
        for t,period in enumerate(calendar):
            _required(period,['start_date','end_date','payment_date','scenario_time'])
            if period['start_date']!=previous or period['payment_date']!=period['end_date']or period['scenario_time']!=cube['time_grid'][t]:_fail('TRANSACTION_CALENDAR','Contiguous payment-at-period-end calendar explicitly mapped to cube times required')
            if _date(period['end_date'],'end_date')<=_date(previous,'previous'):_fail('TRANSACTION_CALENDAR','Positive periods required')
            origin=_date(cube['origin'],'origin');months=(t+1)*{'monthly':1,'quarterly':3,'annual':12}[cube['frequency']];index=origin.year*12+origin.month-1+months;year,month0=divmod(index,12);month=month0+1
            day=monthrange(year,month)[1]if origin.day==monthrange(origin.year,origin.month)[1]else min(origin.day,monthrange(year,month)[1])
            if period['end_date']!=_calendar_date(year,month,day).isoformat():_fail('TRANSACTION_CALENDAR','Calendar dates must match cube frequency from origin with end-of-month preservation')
            previous=period['end_date'];dates.append(previous)
        loans=_rows(params['loans'],'loans');assets=_rows(params['assets'],'assets')
        if not loans or not assets:_fail('TRANSACTION_CONTRACT','At least one debt facility and explicitly modeled asset required')
        schedules={};priorities=set()
        for loan in loans:
            _required(loan,['terms','priority','deficit_cost_share']);fid=loan['terms']['facility_id'];priority=_number(loan['priority'],'priority')
            if fid=='__operating_deficit__'or fid in seen['facility']or priority in priorities:_fail('TRANSACTION_DUPLICATE','Unique facilities and explicit unique liquidity payment ranks required')
            seen['facility'].add(fid);priorities.add(priority)
            if loan['terms']['currency']!=currency:_fail('TARGET_MISMATCH','Loan currency mismatch')
            schedules[fid]=build_debt_schedule(loan['terms'],calendar,base_rate_fixings=loan.get('base_rate_fixings'),context=ctx).value['schedule']
        asset_paths={}
        for asset in assets:
            _required(asset,['parameters','market_binding','innovation_channel','recovery']);ap=asset['parameters'];aid=ap['asset_id']
            if aid in seen['asset']or aid in seen['reserve']:_fail('TRANSACTION_DUPLICATE','Collateral asset cannot be reused across transaction pools')
            seen['asset'].add(aid)
            if ap['currency']!=currency or ap['valuation_date']!=cube['origin']:_fail('TRANSACTION_CALENDAR','Asset currency and initial valuation date must match cube')
            market=[]
            for s,sid in enumerate(cube['scenario_ids']):
                for t,date in enumerate(dates):
                    key=(sid,date,asset['innovation_channel'])
                    if key not in imap:_fail('TRANSACTION_INNOVATION','Missing shared asset innovation')
                    used.add(key);market.append(dict(scenario_id=sid,date=date,market_index=_bound_factor(asset['market_binding'],cube,values,s,t),innovation=imap[key]))
            result=simulate_asset_values(ap,market,context=ctx).value['paths'];asset_paths[aid]={(r['scenario_id'],r['date']):r['asset_value']for r in result}
        op_rows=[]
        for s,sid in enumerate(cube['scenario_ids']):
            for t,date in enumerate(dates):
                row=deepcopy(params['operating_base'][t]);row.update(scenario_id=sid,period_start=calendar[t]['start_date'],period_end=date,currency=currency)
                if row.get('asset_disposals',0)!=0:_fail('TRANSACTION_ASSET','Going-concern asset disposals require explicit asset-state removal; unsupported disposal reuse rejected')
                for field,binding in params['operating_bindings'].items():
                    if field not in ('volume_multiplier','price_multiplier','variable_cost_multiplier','fixed_cost_multiplier'):_fail('TRANSACTION_BINDING','Operating bindings require declared multiplier fields')
                    row[field]=_bound_factor(binding,cube,values,s,t)
                op_rows.append(row)
        operating=simulate_operating_cashflows(op_rows,params['operating_contract'],context=ctx).value['periods'];op={(r['scenario_id'],r['period_end']):r['cfads']for r in operating}
        # The subsequent ledger keeps actual balances separate from contractual schedules.
        counterparty=None
        if params.get('counterparties')is not None:
            cp=params['counterparties'];_required(cp,['paths','contracts','cure_draws','baseline_receipts'])
            if cp['baseline_receipts']!='INCLUDED_ONCE_IN_OPERATING_CFADS':_fail('TRANSACTION_EFFECT','Declare counterparty receipts already included once in baseline CFADS')
            cp_paths=deepcopy(_rows(cp['paths'],'counterparty_paths'));cure=deepcopy(_rows(cp['cure_draws'],'cure_draws'))
            for row in cp_paths:
                _required(row,['innovation_channel']);key=(row['scenario_id'],row['period_end'],row.pop('innovation_channel'))
                if key not in imap:_fail('TRANSACTION_INNOVATION','Missing counterparty default innovation')
                used.add(key);row['uniform']=imap[key]
            for row in cure:
                _required(row,['innovation_channel','innovation_date']);key=(row['scenario_id'],row.pop('innovation_date'),row.pop('innovation_channel'))
                if key not in imap:_fail('TRANSACTION_INNOVATION','Missing cure innovation')
                used.add(key);row['uniform']=imap[key]
            for contract_row in cp['contracts']:
                cid=contract_row['contract_id']
                if cid in seen['contract']:_fail('TRANSACTION_EFFECT','A monetary operating contract cannot be counted in several transactions')
                seen['contract'].add(cid)
            for cp_id in {r['counterparty_id']for r in cp_paths}:
                signature=_fingerprint(sorted([r for r in cp_paths if r['counterparty_id']==cp_id],key=lambda r:(r['scenario_id'],r['period_end'])))
                if cp_id in counterparty_registry and counterparty_registry[cp_id]!=signature:_fail('TRANSACTION_COUNTERPARTY','Shared counterparty must have identical default probability and innovation paths')
                counterparty_registry[cp_id]=signature
            counterparty=simulate_contract_counterparties(cp_paths,cp['contracts'],cure,currency=currency,context=ctx).value
            for flow in counterparty['cashflows']:
                key=(flow['scenario_id'],flow['date'])
                if key not in op:_fail('TRANSACTION_CALENDAR','Counterparty receipts must align with operating payment grid')
                op[key]+=flow['cash_change']
        material=params['materiality'];_required(material,['absolute_threshold','relative_threshold','comparison','continuous_between_grid_dates'])
        absolute=_number(material['absolute_threshold'],'absolute_threshold',0);relative=_number(material['relative_threshold'],'relative_threshold',0)
        if relative>1 or material['comparison']!='gt_both'or material['continuous_between_grid_dates']is not True:_fail('TRANSACTION_MATERIALITY','Explicit strict absolute and relative thresholds and no unrecorded between-grid payments required')
        policy=dict(policy_id=recipe['transaction_id']+'::liquidity',version=recipe['version'],initial_reserve=initial_reserve,reserve_priority=max(priorities)+1,excess_reserve_policy='retain',arrears_policy='carry_without_interest',claims=[dict(claim_id='__operating_deficit__',priority=min(priorities)-1,reserve_eligible=True)]+[dict(claim_id=l['terms']['facility_id'],priority=l['priority'],reserve_eligible=True)for l in loans])
        cash=[]
        for sid in cube['scenario_ids']:
            for t,date in enumerate(dates):
                if any(schedules[l['terms']['facility_id']][t]['interest_due']<0 for l in loans):_fail('TRANSACTION_INTEREST','Negative coupon netting needs explicit lender-refund state; use granular schedules outside this cash-pool recipe')
                cfads=op[(sid,date)];amounts={l['terms']['facility_id']:schedules[l['terms']['facility_id']][t]['debt_service_due']for l in loans};amounts['__operating_deficit__']=max(-cfads,0)
                cash.append(dict(scenario_id=sid,date=date,currency=currency,cash_available=max(cfads,0),reserve_target=params['reserve_targets'][t],amounts_due=amounts))
        liquidity=apply_cash_waterfall(cash,policy,currency=currency,context=ctx).value['periods'];liq={(r['scenario_id'],r['date']):r for r in liquidity}
        evidence=_rows(params['external_evidence'],'external_evidence');external_ids=set();external_keys=set();default_rows=[];ledger={};state_at={}
        for row in evidence:
            _required(row,['scenario_id','date','facility_id','materiality_met','material_days_past_due','utp_indicators'])
            key=(row['scenario_id'],row['date'],row['facility_id'])
            if key in external_keys or row['scenario_id']not in cube['scenario_ids']or row['date']not in [cube['origin']]+dates or row['facility_id']==recipe['transaction_id']+'::pool':_fail('TRANSACTION_EVIDENCE','Distinct outside-evidence facilities on the full shared grid required')
            external_keys.add(key);external_ids.add(row['facility_id'])
            default_rows.append(dict(row,entity_id=recipe['subject_id'],currency=currency,cfads=0,debt_service_due=0,debt_outstanding=0,asset_value=0,liquidity_buffer_available=0))
        expected={(sid,date,fid)for sid in cube['scenario_ids']for date in [cube['origin']]+dates for fid in external_ids}
        if not external_ids or external_keys!=expected:_fail('TRANSACTION_EVIDENCE','Explicit complete outside-obligation assessment, including assessed no-indicator rows, required')
        for sid in cube['scenario_ids']:
            principal={l['terms']['facility_id']:float(l['terms']['principal'])for l in loans};interest={fid:0. for fid in principal};prev_material=False;clock=0;last_date=cube['origin'];ledger[sid]=[]
            initial_asset=sum(a['parameters']['initial_value']for a in assets)
            state_at[(sid,cube['origin'])]=dict(principal=principal.copy(),interest=interest.copy(),assets={a['parameters']['asset_id']:a['parameters']['initial_value']for a in assets},reserve=initial_reserve)
            default_rows.append(dict(scenario_id=sid,entity_id=recipe['subject_id'],facility_id=recipe['transaction_id']+'::pool',date=cube['origin'],currency=currency,cfads=0,debt_service_due=0,debt_outstanding=sum(principal.values()),asset_value=initial_asset,liquidity_buffer_available=initial_reserve,materiality_met=False,material_days_past_due=0,utp_indicators=[]))
            for t,date in enumerate(dates):
                cashrow=liq[(sid,date)];payments={p['claim_id']:p for p in cashrow['payments']};total_due=arrears=0.;facility_states=[]
                for loan in loans:
                    fid=loan['terms']['facility_id'];schedule=schedules[fid][t];payment=payments[fid];interest[fid]+=schedule['interest_due'];paid_interest=min(interest[fid],payment['total_paid']);interest[fid]-=paid_interest;paid_principal=payment['total_paid']-paid_interest
                    if paid_principal>principal[fid]+1e-10*max(1,principal[fid]):_fail('TRANSACTION_ACCOUNTING','Actual principal payment exceeds outstanding balance')
                    principal[fid]=max(principal[fid]-paid_principal,0);total_due+=payment['total_due'];arrears+=payment['closing_arrears']
                    facility_states.append(dict(facility_id=fid,principal_paid=paid_principal,interest_paid=paid_interest,remaining_principal=principal[fid],unpaid_interest=interest[fid],ead=principal[fid]+interest[fid],closing_arrears=payment['closing_arrears']))
                ead=sum(principal.values())+sum(interest.values());ratio=0 if ead==0 else arrears/ead;flag=arrears>absolute and ratio>relative
                clock=clock+(_date(date,'date')-_date(last_date,'last_date')).days if flag and prev_material else 0;prev_material=flag;last_date=date
                asset_values={a['parameters']['asset_id']:asset_paths[a['parameters']['asset_id']][(sid,date)]for a in assets}
                buffer=max(cashrow['opening_reserve']-payments['__operating_deficit__']['opening_arrears'],0)
                default_rows.append(dict(scenario_id=sid,entity_id=recipe['subject_id'],facility_id=recipe['transaction_id']+'::pool',date=date,currency=currency,cfads=op[(sid,date)],debt_service_due=total_due,debt_outstanding=ead,asset_value=sum(asset_values.values()),liquidity_buffer_available=buffer,materiality_met=flag,material_days_past_due=clock,utp_indicators=[]))
                ledger[sid].append(dict(date=date,cfads=op[(sid,date)],liquidity=cashrow,facilities=facility_states,material_arrears=arrears,material_days_past_due=clock,asset_values=asset_values))
                state_at[(sid,date)]=dict(principal=principal.copy(),interest=interest.copy(),assets=asset_values,reserve=cashrow['closing_reserve'])
        detected=detect_cashflow_default(default_rows,params['default_policy'],currency=currency,context=ctx).value
        recovery=params['recovery'];_required(recovery,['policy','reserve_recovery'])
        if recovery['reserve_recovery']!='REMAINING_RESERVE_AT_DEFAULT':_fail('TRANSACTION_RECOVERY','Remaining pool reserve must be recovered once at the same default event')
        # Reserve cash joins collateral recovery once; it is no longer available to going-concern payments.
        events=[];recovery_claims=[];realizations=[];trials=[];path_results=[]
        for event in detected['first_default_events']:
            sid=event['scenario_id'];date=event['first_default_date'];event_id=recipe['transaction_id']+'::'+sid+'::first_default';ead=0
            if event['defaulted']:
                state=state_at[(sid,date)];events.append(dict(scenario_id=sid,default_event_id=event_id,subject_id=recipe['subject_id'],default_date=date))
                for loan in loans:
                    fid=loan['terms']['facility_id'];balance=state['principal'][fid]+state['interest'][fid];ead+=balance
                    recovery_claims.append(dict(scenario_id=sid,default_event_id=event_id,subject_id=recipe['subject_id'],claim_id=fid,pool_id=recipe['transaction_id'],currency=currency,ead_at_default=balance,legal_balance=balance,priority=loan.get('recovery_priority',loan['priority']),deficit_cost_share=loan['deficit_cost_share']))
                for asset in assets:
                    aid=asset['parameters']['asset_id'];terms=asset['recovery'];_required(terms,['delay_days','realization_fraction','cost','annual_discount_rate'])
                    delay=_number(terms['delay_days'],'delay_days',0);rate=_number(terms['annual_discount_rate'],'annual_discount_rate')
                    if delay!=int(delay)or rate<=-1:_fail('TRANSACTION_RECOVERY','Integer nonnegative recovery delay and effective annual discount rate > -1 required')
                    try:sale_date=(_date(date,'default_date')+timedelta(days=int(delay))).isoformat();df=math.exp(-math.log1p(rate)*delay/365)
                    except (OverflowError,ValueError):_fail('TRANSACTION_RECOVERY','Recovery date/discount outside supported range')
                    realizations.append(dict(scenario_id=sid,default_event_id=event_id,pool_id=recipe['transaction_id'],currency=currency,realization_id=event_id+'::'+aid,collateral_id=aid,date=sale_date,gross_value=state['assets'][aid],realization_fraction=terms['realization_fraction'],cost=terms['cost'],discount_factor=df,discount_anchor_date=date))
                realizations.append(dict(scenario_id=sid,default_event_id=event_id,pool_id=recipe['transaction_id'],currency=currency,realization_id=event_id+'::reserve',collateral_id=params['reserve_id'],date=date,gross_value=state['reserve'],realization_fraction=1,cost=0,discount_factor=1,discount_anchor_date=date))
            retained=[row for row in ledger[sid]if date is None or row['date']<=date]
            path_results.append(dict(scenario_id=sid,default_event=event,going_concern_ledger=retained,ead_at_default=ead))
        recovered=simulate_recovery_waterfall(events,recovery_claims,realizations,recovery['policy'],context=ctx).value
        for s,sid in enumerate(cube['scenario_ids']):
            path=next(p for p in path_results if p['scenario_id']==sid);event=path['default_event'];loss=sum(c['economic_loss']for c in recovered['claim_results']if c['scenario_id']==sid)
            trial_weight=1 if params['sampling_design']=='IID_UNWEIGHTED'else float(weights[s])
            trials.append(dict(scenario_id=sid,subject_id=recipe['subject_id'],initial_default=event['initial_default'],defaulted=event['defaulted'],horizon_complete=True,weight=trial_weight,ead_at_default=path['ead_at_default'],economic_loss=loss))
        if params['sampling_design']=='IID_UNWEIGHTED'and not np.allclose(weights,1/len(weights),rtol=0,atol=1e-14):_fail('TRANSACTION_SAMPLING','IID trial declaration requires uniform scenario weights')
        contract=dict(contract_id=recipe['transaction_id']+'::risk',version=recipe['version'],subject_id=recipe['subject_id'],currency=currency,probability_measure='PHYSICAL',horizon=cube['time_grid'][-1],time_unit='years',conditioning='initially_performing',sampling_design=params['sampling_design'],event='first_default',scenario_sampling_contract=deepcopy(cube['sampling_contract']))
        risk=estimate_simulation_risk(trials,contract,context=ctx).value
        bundles.append(dict(transaction_id=recipe['transaction_id'],subject_id=recipe['subject_id'],recipe_hash=_fingerprint(recipe),parameter_hash=recipe['parameter_hash'],paths=path_results,default_assessment=detected,recovery=recovered,risk=risk,counterparty=counterparty))
    if used!=set(imap):_fail('TRANSACTION_INNOVATION','Unused innovations rejected; shared channels may be consumed by several declared drivers')
    return _result(dict(transactions=bundles,scenario_cube_hash=_fingerprint(cube),parameter_snapshot_hash=_fingerprint(snapshot),innovation_hash=_fingerprint(innovations),sampling_contract=deepcopy(cube['sampling_contract'])),'simulation.simulate_transactions','transaction_paths',ctx,limitations=('One complete legally pooled obligor per recipe, no implicit pooling of contractual compartments. Contractually scheduled nonnegative coupons; no extra interest on arrears, interest paid before principal. Cash dates include all payments; materiality is assumed continuous between dates without unrecorded cash. Independent outside-obligation evidence is caller supplied. Baseline contract receipts are included once; counterparty interruptions enter as cash changes, not an added borrower PD. Going-concern paths stop at first default; recovery uses that event, remaining actual exposure, collateral and reserve once. Recovery law here uses supplied fixed delays/costs/fractions and common default-date asset scenarios. Sampling law and PIT/TTC/stress conditioning are preserved, not inferred from an average.',))
