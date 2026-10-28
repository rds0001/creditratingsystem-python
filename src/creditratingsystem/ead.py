"""Dated drawn exposure, conversion factors and counterparty exposure paths."""
from copy import deepcopy
import math
import numpy as np
from .core import CalcResult, FitResult, _result, _fail, _numeric, _probability, _fingerprint, _model_artifact, _fit_result
from .data import _rows, _required, _unique, _date, _days
from .lgd import _scalar, _currency
from ._loss_paths import _multiply, _sum


def exposure_schedule(contract,events,annual_interest_rates,*,period_end_dates,day_count,
                      default_timing,allow_overlimit,currency,context=None):
    """Accrue simple annual interest between dated contractual events.

    Snapshot amounts already include all activity through as_of. Future events
    must be known at as_of. Per-date sequence fixes same-day event ordering.
    Default timing refers to all scheduled events on a period-end date.
    """
    ctx=_currency(currency,context)
    if not isinstance(contract,dict):_fail('EXPOSURE_CONTRACT','Explicit facility snapshot required')
    _required(contract,['facility_id','currency','as_of','drawn_principal','credit_limit','accrued_interest'])
    if contract['currency']!=currency:_fail('TARGET_MISMATCH','Facility currency differs')
    start=_date(contract['as_of'],'as_of');principal=_scalar(contract['drawn_principal'],'drawn_principal',0);limit=_scalar(contract['credit_limit'],'credit_limit',0);interest=_scalar(contract['accrued_interest'],'accrued_interest',0)
    if day_count not in ('ACT/365F','ACT/360'):_fail('EXPOSURE_CONVENTION','Choose ACT/365F or ACT/360 for simple annual interest')
    if default_timing not in ('BEFORE_PERIOD_END_EVENTS','AFTER_PERIOD_END_EVENTS') or type(allow_overlimit) is not bool:_fail('EXPOSURE_CONVENTION','Declare default timing and Boolean overlimit permission')
    if not isinstance(period_end_dates,list) or not period_end_dates:_fail('EXPOSURE_DATE','Explicit increasing period-end dates required')
    dates=[_date(d,'period_end_date')for d in period_end_dates]
    if dates[0]<=start or any(a>=b for a,b in zip(dates,dates[1:])):_fail('EXPOSURE_DATE','Period ends must increase after snapshot')
    rates=_numeric(annual_interest_rates,'annual_interest_rates',vector_only=True)
    if len(rates)!=len(dates) or np.any(rates<0):_fail('EXPOSURE_RATE','One finite nonnegative simple annual rate per period required')
    entries=_rows(events,'events');_unique(entries,'event_id','events');seen=set()
    for row in entries:
        _required(row,['event_date','known_at','sequence','kind','amount'])
        date,known=(_date(row[k],k)for k in ('event_date','known_at'));sequence=_days(row['sequence'],'sequence');_scalar(row['amount'],'event amount',0)
        if date<=start or date>dates[-1] or known>start:_fail('EXPOSURE_DATE','Scheduled events must follow snapshot, fit horizon and already be known at snapshot')
        if row['kind'] not in ('DRAW','PRINCIPAL_REPAYMENT','INTEREST_PAYMENT','LIMIT_CHANGE'):_fail('EXPOSURE_EVENT','Unknown exposure event kind')
        if (row['event_date'],sequence) in seen:_fail('EXPOSURE_AMBIGUITY','Resolve same-day event sequence')
        seen.add((row['event_date'],sequence))
    entries.sort(key=lambda r:(r['event_date'],r['sequence']));rows=[];ledger=[];clock=start;pointer=0;denominator=365 if day_count=='ACT/365F' else 360
    for j,end in enumerate(dates):
        opening=principal;opening_interest=interest;new_interest=0.;before_end=None
        while pointer<len(entries) and _date(entries[pointer]['event_date'],'event_date')<=end:
            row=entries[pointer];when=_date(row['event_date'],'event_date')
            accrued=float(_multiply(principal,rates[j],(when-clock).days/denominator));interest=_sum((interest,accrued));new_interest=_sum((new_interest,accrued));clock=when
            if when==end and before_end is None:before_end=_sum((principal,interest))
            amount=float(row['amount']);kind=row['kind'];old_principal=principal;old_interest=interest;old_limit=limit
            if kind=='DRAW':
                principal=_sum((principal,amount))
                if principal>limit and not allow_overlimit:_fail('EXPOSURE_LIMIT','Draw would exceed declared available commitment')
            elif kind=='PRINCIPAL_REPAYMENT':
                if amount>principal:_fail('EXPOSURE_PAYMENT','Principal repayment exceeds outstanding; allocate refunds separately')
                principal-=amount
            elif kind=='INTEREST_PAYMENT':
                if amount>interest:_fail('EXPOSURE_PAYMENT','Interest payment exceeds accrued interest')
                interest-=amount
            else:limit=amount
            ledger.append(dict(event_id=row['event_id'],event_date=row['event_date'],sequence=row['sequence'],kind=kind,amount=amount,
                principal_before=old_principal,principal_after=principal,interest_before=old_interest,interest_after=interest,limit_before=old_limit,limit_after=limit))
            pointer+=1
        accrued=float(_multiply(principal,rates[j],(end-clock).days/denominator));interest=_sum((interest,accrued));new_interest=_sum((new_interest,accrued));clock=end;after=_sum((principal,interest))
        if before_end is None:before_end=after
        rows.append(dict(period_end=period_end_dates[j],opening_principal=opening,opening_accrued_interest=opening_interest,
            interest_accrued=new_interest,closing_principal=principal,accrued_interest=interest,credit_limit=limit,undrawn_commitment=max(limit-principal,0.),overlimit_amount=max(principal-limit,0.),
            exposure_before_period_end_events=before_end,exposure_after_period_end_events=after,conditional_ead=before_end if default_timing=='BEFORE_PERIOD_END_EVENTS' else after))
    value=dict(facility_id=contract['facility_id'],as_of=contract['as_of'],currency=currency,day_count=day_count,interest_convention='simple_annual_noncompounding_on_drawn_principal',default_timing=default_timing,allow_overlimit=allow_overlimit,schedule=rows,event_ledger=ledger)
    value['schedule_hash']=_fingerprint(value)
    return _result(value,'ead.exposure_schedule',currency,ctx,
        diagnostics=dict(input_fingerprint=_fingerprint(dict(contract=contract,events=entries,annual_interest_rates=rates.tolist(),period_end_dates=period_end_dates))),
        limitations=('Known scheduled draws and payments are explicit assumptions, not forecasts inferred from future behavior. Interest is not automatically capitalized; fees, FX and collateral require separately declared inputs.',))


def realized_ccf(reference_drawn,reference_limit,ead_at_default,*,reference_date,default_date,
                 reference_known_at,default_known_at,evaluation_cutoff,currency,minimum_undrawn=0.,context=None):
    """Compute raw conversion without clipping negative or above-one outcomes."""
    ctx=_currency(currency,context);drawn=_scalar(reference_drawn,'reference_drawn',0);limit=_scalar(reference_limit,'reference_limit',0);ead=_scalar(ead_at_default,'ead_at_default',0);threshold=_scalar(minimum_undrawn,'minimum_undrawn',0)
    ref,default,rknown,dknown,cutoff=(_date(d,n)for d,n in ((reference_date,'reference_date'),(default_date,'default_date'),(reference_known_at,'reference_known_at'),(default_known_at,'default_known_at'),(evaluation_cutoff,'evaluation_cutoff')))
    if not rknown<=ref<default<=dknown<=cutoff:_fail('EXPOSURE_DATE','Require frozen reference before default, default known by evaluation')
    denominator=_sum((limit,-drawn));increase=_sum((ead,-drawn));ratio=None
    if denominator<=0:status='NO_POSITIVE_UNDRAWN'
    elif denominator<=threshold:status='BELOW_MINIMUM_UNDRAWN'
    else:
        ratio=increase/denominator
        if not math.isfinite(ratio):_fail('NUMERICAL_FAILURE','CCF outside finite range')
        status='DEFINED'
    return _result(dict(ccf=ratio,denominator_status=status,undrawn_at_reference=denominator,exposure_increase=increase,reference_drawn=drawn,reference_limit=limit,ead_at_default=ead,
        reconstructed_ead=None if ratio is None else _sum((drawn,float(_multiply(ratio,denominator))))),'ead.realized_ccf','conversion_factor',ctx,
        diagnostics=dict(currency=currency,reference_date=reference_date,default_date=default_date,reference_known_at=reference_known_at,default_known_at=default_known_at,evaluation_cutoff=evaluation_cutoff,minimum_undrawn=threshold),
        limitations=('Reference commitment and exposure must share the same currency and exposure definition. Raw CCF may be negative or exceed one; no floor, ceiling or automatic limit-change adjustment is imposed.',))


def _path_artifact(value,kind):
    if isinstance(value,CalcResult):value=value.value
    if not isinstance(value,dict) or value.get('artifact_type')!=kind:_fail('EXPOSURE_ARTIFACT','Expected '+kind+' artifact')
    frozen=deepcopy(value);digest=frozen.pop('artifact_hash',None)
    if digest!=_fingerprint(frozen):_fail('INTEGRITY_ERROR','Exposure artifact content differs from its binding')
    return value


def _seal(value):
    value['artifact_hash']=_fingerprint(value);return value


def aggregate_netting_set_exposure(trade_value_paths,netting_agreements,fx_paths,*,
                                   as_of,reporting_currency,valuation_convention,context=None):
    """Sum signed aligned trade values only inside evidenced effective netting sets."""
    ctx=_currency(reporting_currency,context);start=_date(as_of,'as_of')
    if valuation_convention!='EX_DIVIDEND_BEFORE_MARGIN':_fail('EXPOSURE_CONVENTION','Declare EX_DIVIDEND_BEFORE_MARGIN trade valuation')
    rows=_rows(trade_value_paths,'trade_value_paths');agreements=_rows(netting_agreements,'netting_agreements');fx=_rows(fx_paths,'fx_paths')
    if not rows or not agreements:_fail('EMPTY_INPUT','Trade paths and netting agreements required')
    _unique(rows,['path_id','date','trade_id'],'trade_value_paths');_unique(agreements,'netting_set_id','netting_agreements');_unique(fx,['path_id','date','currency'],'fx_paths')
    paths=sorted(set(r['path_id']for r in rows));dates=sorted(set(r['date']for r in rows));trades=sorted(set(r['trade_id']for r in rows));membership={};metadata=[]
    for date in dates:
        if _date(date,'date')<start:_fail('EXPOSURE_DATE','Valuation dates cannot precede as_of')
    for agreement in agreements:
        _required(agreement,['counterparty_id','trade_ids','legally_enforceable','evidence_reference','known_at','effective_from'])
        if 'effective_until' not in agreement:_fail('NETTING_CONTRACT','Explicit effective_until date or null required')
        if any(not isinstance(agreement[k],str) or not agreement[k].strip()for k in ('counterparty_id','evidence_reference')):_fail('NETTING_CONTRACT','Counterparty and legal evidence identifiers required')
        members=agreement['trade_ids']
        if not isinstance(members,list) or not members or any(not isinstance(t,str) or not t for t in members) or len(set(members))!=len(members):_fail('NETTING_CONTRACT','Distinct nonempty trade membership required')
        if type(agreement['legally_enforceable'])is not bool or (not agreement['legally_enforceable'] and len(members)>1):_fail('NETTING_ELIGIBILITY','Multiple-trade netting requires enforceability evidence; false is allowed only for singleton sets')
        first=_date(agreement['effective_from'],'effective_from');last=None if agreement['effective_until'] is None else _date(agreement['effective_until'],'effective_until')
        if _date(agreement['known_at'],'known_at')>start or first>_date(dates[0],'date') or (last is not None and last<=_date(dates[-1],'date')):_fail('NETTING_ELIGIBILITY','Agreement must be known at snapshot and cover every valuation date [from,until)')
        for trade in members:
            if trade in membership:_fail('NETTING_OVERLAP','A trade cannot belong to multiple sets')
            membership[trade]=agreement['netting_set_id']
        metadata.append(deepcopy(agreement))
    if set(membership)!=set(trades):_fail('NETTING_MEMBERSHIP','Every modeled trade must have exactly one declared membership')
    lookup={};currencies={}
    for row in rows:
        _required(row,['currency','value']);value=_scalar(row['value'],'trade value')
        if not isinstance(row['currency'],str) or not row['currency']:_fail('INVALID_CURRENCY','Trade currency required')
        if row['trade_id']in currencies and currencies[row['trade_id']]!=row['currency']:_fail('EXPOSURE_CONVENTION','Trade currency changes require explicit redenomination preprocessing')
        currencies[row['trade_id']]=row['currency'];lookup[(row['path_id'],row['date'],row['trade_id'])]=value
    if len(rows)!=len(paths)*len(dates)*len(trades):_fail('PATH_ALIGNMENT','Complete common path/date/trade grid required')
    rates={}
    for row in fx:
        _required(row,['rate']);rate=_scalar(row['rate'],'FX rate',0)
        if rate<=0 or (row['currency']==reporting_currency and rate!=1):_fail('INVALID_FX','Positive FX and exact reporting-currency identity required')
        rates[(row['path_id'],row['date'],row['currency'])]=rate
    output=[]
    for path in paths:
        for date in dates:
            for agreement in sorted(metadata,key=lambda a:a['netting_set_id']):
                values=[]
                for trade in agreement['trade_ids']:
                    key=(path,date,currencies[trade])
                    if key not in rates:_fail('MISSING_FX','Exact scenario/date/currency FX required')
                    values.append(float(_multiply(lookup[(path,date,trade)],rates[key])))
                net=_sum(values);output.append(dict(path_id=path,date=date,netting_set_id=agreement['netting_set_id'],signed_net_value=net,positive_exposure=max(net,0.),gross_positive=_sum(v for v in values if v>0),gross_negative=_sum(v for v in values if v<0)))
    value=_seal(dict(artifact_type='NETTING_PATHS',provenance=_result(None,'ead.context',context=ctx).provenance,as_of=as_of,currency=reporting_currency,valuation_convention=valuation_convention,path_ids=paths,dates=dates,netting_agreements=sorted(metadata,key=lambda a:a['netting_set_id']),paths=output))
    return _result(value,'ead.aggregate_netting_set_exposure',reporting_currency,ctx,diagnostics=dict(input_fingerprint=_fingerprint(dict(trade_value_paths=rows,netting_agreements=agreements,fx_paths=fx))),limitations=('Enforceability and membership are supplied evidence, not legal determinations. Signed values are aggregated before flooring and never offset across netting sets.',))


def _collateral_market(rows,artifact,policies):
    rows=_rows(rows,'collateral_market_paths');_unique(rows,['path_id','date','netting_set_id'],'collateral_market_paths');lookup={}
    expected={(p,d,s)for p in artifact['path_ids']for d in artifact['dates']for s in policies}
    for row in rows:
        _required(row,['unit_price','fx_rate','received_haircut','posted_haircut','currency'])
        key=(row['path_id'],row['date'],row['netting_set_id'])
        if key not in expected:_fail('PATH_ALIGNMENT','Collateral market row outside netting grid')
        policy=policies[key[2]];price=_scalar(row['unit_price'],'collateral unit price',0);fx=_scalar(row['fx_rate'],'collateral FX',0);hr=_scalar(row['received_haircut'],'received_haircut',0);hp=_scalar(row['posted_haircut'],'posted_haircut',0)
        if price<=0 or fx<=0 or hr>=1 or hp>=1 or row['currency']!=policy['collateral_currency'] or (row['currency']==artifact['currency'] and fx!=1):_fail('COLLATERAL_MARKET','Positive collateral prices/FX and haircuts in [0,1), with matching currency, required')
        unit_value=float(_multiply(price,fx))
        if unit_value<=0:_fail('NUMERICAL_FAILURE','Collateral unit value underflow')
        lookup[key]=dict(row,unit_reporting_value=unit_value)
    if set(lookup)!=expected:_fail('PATH_ALIGNMENT','Complete collateral market grid required')
    return lookup


def _collateral_exposure(value,units,market,policy):
    nominal=float(_multiply(units,market['unit_reporting_value']))
    received=float(_multiply(max(nominal,0.),1-market['received_haircut']))
    posted=0. if policy['posted_returnability']=='BANKRUPTCY_REMOTE' else max(-nominal,0.)
    raw=_sum((value,-received,posted)) if policy['posted_claim_netting_enforceable'] else _sum((max(_sum((value,-received)),0.),posted))
    return dict(collateral_nominal=nominal,recognized_received=received,posted_unsecured_claim=posted,residual_exposure=max(raw,0.))


def simulate_collateral_margin(netting_paths,margin_policies,call_calendar,collateral_market_paths,*,context=None):
    """Simulate unit-denominated margin calls with explicit information and settlement dates."""
    artifact=_path_artifact(netting_paths,'NETTING_PATHS');ctx=_currency(artifact['currency'],context);start=_date(artifact['as_of'],'as_of');policies=_rows(margin_policies,'margin_policies');_unique(policies,'netting_set_id','margin_policies');policy_map={r['netting_set_id']:r for r in policies}
    if set(policy_map)!={a['netting_set_id']for a in artifact['netting_agreements']}:_fail('NETTING_MEMBERSHIP','Exactly one margin policy per netting set required')
    for policy in policies:
        _required(policy,['known_at','collateral_currency','initial_units','threshold_received','threshold_posted','minimum_transfer_amount','ownership','received_enforceable','posted_returnability','posted_claim_netting_enforceable','reuse'])
        if _date(policy['known_at'],'known_at')>start:_fail('COLLATERAL_DATE','Margin policy must be known at as_of')
        if not isinstance(policy['collateral_currency'],str) or not policy['collateral_currency']:_fail('INVALID_CURRENCY','Collateral currency required')
        _scalar(policy['initial_units'],'initial_units')
        for k in ('threshold_received','threshold_posted','minimum_transfer_amount'):_scalar(policy[k],k,0)
        if policy['ownership']not in ('TITLE_TRANSFER','SECURITY_INTEREST') or policy['received_enforceable']is not True or policy['posted_returnability']not in ('UNSECURED_CLAIM','BANKRUPTCY_REMOTE') or policy['reuse']not in ('ALLOWED','PROHIBITED'):_fail('COLLATERAL_OWNERSHIP','Explicit ownership, enforceable received collateral, returnability and reuse terms required')
        if type(policy['posted_claim_netting_enforceable'])is not bool:_fail('COLLATERAL_OWNERSHIP','Explicit Boolean posted-claim netting enforceability required')
        if policy['ownership']=='TITLE_TRANSFER' and policy['posted_returnability']=='BANKRUPTCY_REMOTE':_fail('COLLATERAL_OWNERSHIP','Title-transferred posted collateral cannot be assumed bankruptcy-remote')
    market=_collateral_market(collateral_market_paths,artifact,policy_map);calls=_rows(call_calendar,'call_calendar');_unique(calls,'call_id','call_calendar');_unique(calls,['path_id','netting_set_id','call_date'],'call_calendar');call_map={}
    from datetime import timedelta
    for row in calls:
        _required(row,['valuation_date','settlement_date','dispute_delay_days'])
        key=(row['path_id'],row['call_date'],row['netting_set_id']);valuation=_date(row['valuation_date'],'valuation_date');call=_date(row['call_date'],'call_date');settlement=_date(row['settlement_date'],'settlement_date');delay=_days(row['dispute_delay_days'],'dispute_delay_days')
        try:effective=settlement+timedelta(days=delay)
        except (OverflowError,ValueError):_fail('COLLATERAL_DATE','Effective settlement exceeds the calendar date domain')
        if key not in market or row['valuation_date']not in artifact['dates'] or not valuation<=call<=settlement:_fail('COLLATERAL_DATE','Calls require known grid valuation no later than call and settlement no earlier than call')
        if effective<=_date(artifact['dates'][-1],'last date') and effective.isoformat()not in artifact['dates']:_fail('PATH_ALIGNMENT','Effective settlement must occur on the explicit grid or after its end')
        call_map[key]=dict(row,effective_settlement_date=effective.isoformat())
    net={(r['path_id'],r['date'],r['netting_set_id']):r['signed_net_value']for r in artifact['paths']};ledger=[];output=[]
    for path in artifact['path_ids']:
        for set_id in sorted(policy_map):
            policy=policy_map[set_id];units=float(policy['initial_units']);pending=[]
            for date in artifact['dates']:
                key=(path,date,set_id);opening=units;settled=[c for c in pending if c['effective_settlement_date']==date];units=_sum([units]+[c['transfer_units']for c in settled]);pending=[c for c in pending if c['effective_settlement_date']!=date];before_calls=units;call=call_map.get(key)
                if call is not None:
                    vk=(path,call['valuation_date'],set_id);v=net[vk];m=market[vk];desired=v-policy['threshold_received'] if v>policy['threshold_received'] else v+policy['threshold_posted'] if v< -policy['threshold_posted'] else 0.
                    divisor=float(_multiply(m['unit_reporting_value'],1-(m['received_haircut']if desired>=0 else m['posted_haircut'])))
                    if divisor<=0:_fail('NUMERICAL_FAILURE','Haircut-adjusted collateral unit value underflow')
                    target_units=desired/divisor
                    if not math.isfinite(target_units):_fail('NUMERICAL_FAILURE','Margin target units outside finite range')
                    committed=_sum([units]+[c['transfer_units']for c in pending]);transfer=_sum((target_units,-committed));transfer_value=abs(float(_multiply(transfer,m['unit_reporting_value'])));execute=transfer!=0 and transfer_value>=policy['minimum_transfer_amount']
                    entry=dict(call,valuation_net_value=v,target_units=target_units,committed_units=committed,transfer_units=transfer if execute else 0.,transfer_reporting_value=transfer_value if execute else 0.,status='CALLED' if execute else 'BELOW_MTA_OR_ZERO')
                    ledger.append(entry)
                    if execute:
                        if call['effective_settlement_date']==date:units=_sum((units,transfer))
                        else:pending.append(entry)
                metrics=_collateral_exposure(net[key],units,market[key],policy)
                output.append(dict(path_id=path,date=date,netting_set_id=set_id,signed_net_value=net[key],opening_units=opening,units_before_calls=before_calls,closing_units=units,pending_units=_sum(c['transfer_units']for c in pending),**metrics))
    value=_seal(dict(artifact_type='COLLATERAL_PATHS',provenance=deepcopy(artifact['provenance']),netting_artifact_hash=artifact['artifact_hash'],as_of=artifact['as_of'],currency=artifact['currency'],path_ids=artifact['path_ids'],dates=artifact['dates'],margin_policies=sorted(policies,key=lambda p:p['netting_set_id']),collateral_market_paths=[market[k]for k in sorted(market)],paths=output,call_ledger=ledger,mta_convention='GROSS_REPORTING_MARKET_VALUE',event_order='PRIOR_SETTLEMENTS_THEN_CALL_THEN_IMMEDIATE_SETTLEMENT'))
    return _result(value,'ead.simulate_collateral_margin',artifact['currency'],ctx,provenance=artifact['provenance'],limitations=('Calls use declared valuation dates, with outstanding units included to avoid duplicate calls. Received haircuts reduce recognized collateral; unsegregated posted collateral remains a full nominal return claim. Reuse permissions are recorded but no reinvestment or funding benefit is inferred.',))


def closeout_exposure_at_default(netting_paths,collateral_paths,default_events,intermediate_cashflows,*,
                                 freeze_timing,cashflow_convention,context=None):
    """Freeze collateral units at default; revalue them at explicit MPOR closeout."""
    net=_path_artifact(netting_paths,'NETTING_PATHS');coll=_path_artifact(collateral_paths,'COLLATERAL_PATHS');ctx=_currency(net['currency'],context)
    if coll['netting_artifact_hash']!=net['artifact_hash']:_fail('TARGET_MISMATCH','Collateral and netting path bindings differ')
    if freeze_timing not in ('BEFORE_DATE_EVENTS','AFTER_DATE_EVENTS') or cashflow_convention!='UNSETTLED_EXCLUDED_FROM_CLOSEOUT_VALUE':_fail('CLOSEOUT_CONVENTION','Declare collateral freeze timing and excluded unsettled cashflows')
    events=_rows(default_events,'default_events');_unique(events,['path_id','netting_set_id'],'default_events');flows=_rows(intermediate_cashflows,'intermediate_cashflows');_unique(flows,'cashflow_id','intermediate_cashflows');keys={(r['path_id'],r['netting_set_id'])for r in events}
    for f in flows:
        _required(f,['path_id','netting_set_id','date','amount','included_in_closeout_value','currency']);_date(f['date'],'cashflow date');_scalar(f['amount'],'cashflow amount')
        if (f['path_id'],f['netting_set_id'])not in keys or f['currency']!=net['currency'] or f['included_in_closeout_value']is not False:_fail('CLOSEOUT_CASHFLOW','Cashflows must refer to a modeled default in reporting currency and be explicitly excluded from closeout values')
    nlookup={(r['path_id'],r['date'],r['netting_set_id']):r for r in net['paths']};clookup={(r['path_id'],r['date'],r['netting_set_id']):r for r in coll['paths']};policies={p['netting_set_id']:p for p in coll['margin_policies']};market={(r['path_id'],r['date'],r['netting_set_id']):r for r in coll['collateral_market_paths']};output=[]
    for event in events:
        _required(event,['default_date','closeout_date','margin_observation_date']);observation=_date(event['margin_observation_date'],'margin_observation_date');default=_date(event['default_date'],'default_date');closeout=_date(event['closeout_date'],'closeout_date');dk=(event['path_id'],event['default_date'],event['netting_set_id']);ck=(event['path_id'],event['closeout_date'],event['netting_set_id'])
        if closeout<default or observation>default:_fail('CLOSEOUT_DATE','Require margin observation no later than default and closeout no earlier than default')
        if dk not in nlookup or ck not in nlookup:_fail('INSUFFICIENT_PATH_EXTENSION','Default and closeout require exact dates on the extended market/collateral grid')
        units=clookup[dk]['opening_units'if freeze_timing=='BEFORE_DATE_EVENTS'else'closing_units'];selected=[f for f in flows if f['path_id']==event['path_id']and f['netting_set_id']==event['netting_set_id']]
        if any(not default<_date(f['date'],'date')<=closeout for f in selected):_fail('CLOSEOUT_CASHFLOW','Unsettled flows must lie in (default,closeout]')
        cash=_sum(f['amount']for f in selected);value=_sum((nlookup[ck]['signed_net_value'],cash));metrics=_collateral_exposure(value,units,market[ck],policies[event['netting_set_id']])
        output.append(dict(path_id=event['path_id'],netting_set_id=event['netting_set_id'],default_date=event['default_date'],closeout_date=event['closeout_date'],margin_observation_date=event['margin_observation_date'],margin_period_days=(closeout-observation).days,default_to_closeout_days=(closeout-default).days,closeout_net_value=nlookup[ck]['signed_net_value'],unsettled_contractual_cashflows=cash,frozen_collateral_units=units,closeout_exposure=metrics.pop('residual_exposure'),**metrics))
    return _result(dict(currency=net['currency'],freeze_timing=freeze_timing,cashflow_convention=cashflow_convention,closeouts=output),'ead.closeout_exposure_at_default',net['currency'],ctx,provenance=net['provenance'],diagnostics=dict(netting_artifact_hash=net['artifact_hash'],collateral_artifact_hash=coll['artifact_hash']),limitations=('Margin after the freeze cannot reduce closeout exposure. Frozen units are revalued at closeout market/FX/haircut; no path extrapolation or accrued funding interest is invented. Signed intervening unsettled claims are included exactly once.',))


def _ead_regression(d,y,w):
    """Frequency least squares with independent-observation HC0 covariance."""
    total=float(w.sum());weighted=d*np.sqrt(w)[:,None];singular=np.linalg.svd(weighted,compute_uv=False)
    if len(singular)<d.shape[1] or singular[-1]<=singular[0]*1e-10:_fail('RANK_DEFICIENT','Exposure regression requires full weighted design rank')
    coefficient=np.linalg.lstsq(weighted,y*np.sqrt(w),rcond=1e-10)[0];residual=y-d@coefficient
    with np.errstate(over='ignore',invalid='ignore'):
        bread=np.linalg.inv(d.T@(w[:,None]*d));covariance=bread@(d.T@((w*residual**2)[:,None]*d))@bread;variance=float(np.dot(w,residual**2)/total)
    if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(covariance)) or not math.isfinite(variance):_fail('NUMERICAL_FAILURE','Exposure regression outside finite range')
    return dict(coefficients=coefficient.tolist(),covariance=covariance.tolist(),covariance_type='HC0_FREQUENCY_IID',residual_variance=variance,rank=d.shape[1],frequency_total=total)


def fit_ead_model(exposure_episodes,X_train,*,feature_order,target_mode,model_spec,target,training,
                  model_id,model_version,currency,weights=None,minimum_undrawn=0.,fit_intercept=True,context=None):
    """Fit a chronologically frozen CCF mean or nonnegative direct-EAD hurdle distribution."""
    from .classification import _design,fit_binary_glm
    from .lgd import _lgd_target_context
    ctx=_currency(currency,context);x,d=_design(X_train,feature_order,fit_intercept);rows=_rows(exposure_episodes,'exposure_episodes');_unique(rows,'episode_id','exposure_episodes')
    if len(rows)!=len(x):_fail('INCOMPATIBLE_SHAPE','One exposure episode per design row required')
    if target_mode not in ('ccf','direct_ead'):_fail('EAD_MODEL_SPEC','Declare ccf or direct_ead target mode')
    if not isinstance(target,dict) or target.get('event')!='ead' or target.get('probability_measure')!='PHYSICAL' or target.get('conditioning')!='given_default' or target.get('time_unit')!='days' or target.get('currency')!=currency:_fail('TARGET_MISMATCH','Exposure target requires PHYSICAL, given_default, integer-day horizon and currency')
    ctx=_lgd_target_context(target,ctx,model_version)
    horizon=_days(target.get('horizon'),'target.horizon',1);cutoff=_date(training.get('known_at')if isinstance(training,dict)else None,'training.known_at');threshold=_scalar(minimum_undrawn,'minimum_undrawn',0)
    w=np.ones(len(x))if weights is None else _numeric(weights,'weights',vector_only=True)
    if len(w)!=len(x) or np.any(w<0) or np.any(w!=np.floor(w)) or not np.any(w>0) or not math.isfinite(float(w.sum())) or w.sum()>2**53-1:_fail('INVALID_WEIGHTS','Positive total exact integer frequency weights required')
    targets=[];valid=[];excluded=[]
    for i,row in enumerate(rows):
        _required(row,['reference_date','reference_known_at','default_date','default_known_at','features_observed_at','features_known_at','reference_drawn','reference_limit','ead_at_default','currency'])
        ref,known,default,dk,fo,fk=(_date(row[k],k)for k in ('reference_date','reference_known_at','default_date','default_known_at','features_observed_at','features_known_at'))
        if row['currency']!=currency or not known<=ref<default<=dk<=cutoff or fo>ref or fk>ref or fk<fo or (default-ref).days!=horizon:_fail('EXPOSURE_DATE','Reference/feature dates must be frozen before default at the exact target lookback and known by training cutoff')
        drawn=_scalar(row['reference_drawn'],'reference_drawn',0);limit=_scalar(row['reference_limit'],'reference_limit',0);ead=_scalar(row['ead_at_default'],'ead_at_default',0)
        if target_mode=='ccf':
            denominator=_sum((limit,-drawn))
            if denominator<=threshold:
                excluded.append(dict(episode_id=row['episode_id'],reason='NO_POSITIVE_UNDRAWN'if denominator<=0 else'BELOW_MINIMUM_UNDRAWN'));continue
            value=_sum((ead,-drawn))/denominator
            if not math.isfinite(value):_fail('NUMERICAL_FAILURE','Realized CCF outside finite range')
        else:value=ead
        if w[i]>0:valid.append(i);targets.append(value)
    if not valid:_fail('INSUFFICIENT_EXPOSURE_SUPPORT','No positive-weight eligible exposure rows remain')
    da=d[valid];xa=x[valid];ya=np.array(targets);wa=w[valid];params=dict(target_mode=target_mode,model_spec=deepcopy(model_spec),fit_intercept=fit_intercept,currency=currency,minimum_undrawn=threshold,feature_min=xa.min(axis=0).tolist(),feature_max=xa.max(axis=0).tolist(),weight_type='frequency')
    if target_mode=='ccf':
        if model_spec!={'mean_model':'identity','covariance':'HC0'}:_fail('EAD_MODEL_SPEC','CCF requires identity conditional mean and HC0 frequency covariance')
        params['regression']=_ead_regression(da,ya,wa);params['distribution_identified']=False
    else:
        if not isinstance(model_spec,dict) or set(model_spec)!={'positive_distribution','zero_model'} or model_spec['positive_distribution']!='lognormal' or model_spec['zero_model']not in ('none','logit','all_zero'):_fail('EAD_MODEL_SPEC','Direct EAD requires lognormal amount and zero_model none/logit/all_zero')
        positive=ya>0;zero_model=model_spec['zero_model'];params['distribution_identified']=True
        if zero_model=='all_zero':
            if np.any(positive):_fail('EAD_COMPONENT_SUPPORT','all_zero requires all active EAD targets equal zero')
            params.update(positive_probability=0.,positive_regression=None,positive_gate=None)
        else:
            if not np.any(positive) or (zero_model=='none'and not np.all(positive)) or (zero_model=='logit'and np.all(positive)):_fail('EAD_COMPONENT_SUPPORT','Declared zero/positive stages must have the required positive-weight support')
            params['positive_feature_min']=xa[positive].min(axis=0).tolist();params['positive_feature_max']=xa[positive].max(axis=0).tolist()
            params['positive_regression']=_ead_regression(da[positive],np.log(ya[positive]),wa[positive])
            if params['positive_regression']['residual_variance']<=1e-24:_fail('NON_IDENTIFIABLE','Lognormal positive amount requires nonzero residual dispersion')
            params['positive_probability']=1. if zero_model=='none'else None;params['positive_gate']=None
            if zero_model=='logit':
                gate_target=dict(target,event='positive_ead_given_default')
                params['positive_gate']=fit_binary_glm(xa,positive.astype(int),link='logit',feature_order=feature_order,fit_intercept=fit_intercept,target=gate_target,training=training,model_id=model_id+':positive',model_version=model_version,weights=wa).model
    diagnostics=dict(eligible_rows=len(valid),excluded_rows=excluded,frequency_total=float(wa.sum()),weight_type='frequency',target_mode=target_mode,feature_cutoff='observed_and_known_at_or_before_reference',target_lookback_days=horizon)
    model=_model_artifact(dict(schema_version='1.0',model_id=model_id,model_version=model_version,method_id='ead.fit_ead_model',feature_order=list(feature_order),parameters=params,preprocessing=dict(transformation='identity'),target=target,training=training,diagnostics=diagnostics,provenance=dict(data_origin=training.get('data_origin','USER_SUPPLIED'),parameter_origin='ESTIMATED'),input_fingerprint=_fingerprint(dict(episodes=rows,X=x.tolist(),weights=w.tolist()))))
    return _fit_result(model,'ead.fit_ead_model',diagnostics=diagnostics,context=ctx)


def predict_ead(fitted_model_or_schedule,facility_covariates,*,feature_order,expected_target,
                reference_dates,default_dates,currency,reference_drawn=None,reference_limit=None,
                quantile_levels=None,context=None):
    """Replay exact schedule dates or frozen conditional exposure models."""
    from .classification import _design,predict_pd
    from .lgd import _lgd_target_context
    from scipy.special import ndtri
    ctx=_currency(currency,context)
    source=fitted_model_or_schedule.value if isinstance(fitted_model_or_schedule,CalcResult)and not isinstance(fitted_model_or_schedule,FitResult)else fitted_model_or_schedule
    if isinstance(source,dict)and 'schedule_hash'in source:
        frozen=deepcopy(source);digest=frozen.pop('schedule_hash')
        if digest!=_fingerprint(frozen):_fail('INTEGRITY_ERROR','Exposure schedule content differs from its binding')
        if source.get('currency')!=currency or facility_covariates is not None or feature_order!=[] or expected_target is not None or reference_drawn is not None or reference_limit is not None or quantile_levels is not None:_fail('EXPOSURE_CONVENTION','Schedule prediction takes exact dates and currency, without regression inputs')
        if not isinstance(default_dates,list)or not default_dates or reference_dates!=[source['as_of']]*len(default_dates):_fail('EXPOSURE_DATE','Schedule reference dates must equal its frozen as_of')
        lookup={r['period_end']:r['conditional_ead']for r in source['schedule']}
        if any(date not in lookup for date in default_dates):_fail('EXPOSURE_DATE','Schedule default dates must match explicit period endpoints')
        return _result(dict(conditional_ead=[lookup[d]for d in default_dates],default_dates=default_dates,reference_dates=reference_dates,currency=currency,source='EXPOSURE_SCHEDULE',quantile_levels=None,quantiles=None),'ead.predict_ead',currency,ctx,diagnostics=dict(schedule_hash=digest))
    model=_model_artifact(fitted_model_or_schedule);p=model['parameters'];ctx=_lgd_target_context(model['target'],ctx,model['model_version'])
    if model['method_id']!='ead.fit_ead_model' or list(feature_order)!=model['feature_order'] or expected_target!=model['target'] or p.get('currency')!=currency:_fail('TARGET_MISMATCH','Frozen EAD method, target, currency and feature order must match')
    x,d=_design(facility_covariates,feature_order,p.get('fit_intercept'));n=len(x)
    if not isinstance(reference_dates,list)or not isinstance(default_dates,list)or len(reference_dates)!=n or len(default_dates)!=n:_fail('EXPOSURE_DATE','One reference/default date per prediction required')
    for ref,date in zip(reference_dates,default_dates):
        if _date(ref,'reference_date')<_date(model['training']['known_at'],'training.known_at') or (_date(date,'default_date')-_date(ref,'reference_date')).days!=model['target']['horizon']:_fail('EXPOSURE_DATE','Prediction reference-to-default horizon must match training target')
    levels=None if quantile_levels is None else _numeric(quantile_levels,'quantile_levels',vector_only=True)
    if levels is not None and (np.any((levels<=0)|(levels>=1))or np.any(np.diff(levels)<=0)):_fail('INVALID_QUANTILE','Strictly increasing interior probabilities required')
    lo=_numeric(p.get('feature_min'),'feature_min',allow_empty=True,vector_only=True);hi=_numeric(p.get('feature_max'),'feature_max',allow_empty=True,vector_only=True)
    if lo.shape!=(x.shape[1],)or hi.shape!=lo.shape:_fail('INVALID_MODEL','Frozen EAD support dimension mismatch')
    diagnostics=dict(model_id=model['model_id'],model_fingerprint=_fingerprint(model),outside_marginal_training_range=np.any((x<lo)|(x>hi),axis=1).tolist())
    value=dict(default_dates=default_dates,reference_dates=reference_dates,currency=currency,source='FITTED_MODEL',quantile_levels=None if levels is None else levels.tolist(),quantiles=None)
    if p.get('target_mode')=='ccf':
        if levels is not None:_fail('DISTRIBUTION_NOT_IDENTIFIED','CCF conditional mean does not identify exposure quantiles')
        drawn=_numeric(reference_drawn,'reference_drawn',vector_only=True);limit=_numeric(reference_limit,'reference_limit',vector_only=True)
        if drawn.shape!=(n,)or limit.shape!=(n,)or np.any(drawn<0)or np.any(limit<0):_fail('EXPOSURE_INPUT','Aligned nonnegative reference drawn and limits required')
        denominator=limit-drawn
        if np.any(denominator<=p['minimum_undrawn']):_fail('INSUFFICIENT_EXPOSURE_SUPPORT','Prediction undrawn must exceed the frozen CCF threshold')
        beta=_numeric(p['regression'].get('coefficients'),'coefficients',vector_only=True)
        if beta.shape!=(d.shape[1],):_fail('INVALID_MODEL','CCF coefficient dimension mismatch')
        ccf=d@beta;ead=np.array([_sum((a,float(_multiply(b,c))))for a,b,c in zip(drawn,denominator,ccf)])
        if np.any(ead<0):_fail('NEGATIVE_EAD_PREDICTION','Raw CCF conditional mean implies negative exposure; revise model/support rather than clipping')
        cov=_numeric(p['regression'].get('covariance'),'covariance')
        if cov.shape!=(d.shape[1],d.shape[1]):_fail('INVALID_MODEL','CCF covariance dimension mismatch')
        mean_variance=np.einsum('ij,jk,ik->i',d,cov,d)
        if np.any(mean_variance<0)or not np.all(np.isfinite(mean_variance)):_fail('NUMERICAL_FAILURE','CCF mean covariance outside nonnegative finite range')
        se=np.sqrt(mean_variance);ead_se=_multiply(denominator,se)
        value.update(conditional_ead=ead.tolist(),conditional_ccf=ccf.tolist(),conditional_ccf_mean_standard_error=se.tolist(),conditional_ead_mean_standard_error=ead_se.tolist(),distribution=None)
    elif p.get('target_mode')=='direct_ead':
        if reference_drawn is not None or reference_limit is not None:_fail('EXPOSURE_CONVENTION','Direct EAD regression does not consume CCF reconstruction amounts')
        if p['model_spec']['zero_model']=='all_zero':positive=np.zeros(n);mu=np.zeros(n);variance=0.;ead=np.zeros(n);quantiles=None if levels is None else np.zeros((n,len(levels)))
        else:
            pl=_numeric(p.get('positive_feature_min'),'positive_feature_min',allow_empty=True,vector_only=True);ph=_numeric(p.get('positive_feature_max'),'positive_feature_max',allow_empty=True,vector_only=True)
            if pl.shape!=lo.shape or ph.shape!=lo.shape:_fail('INVALID_MODEL','Positive-stage support dimension mismatch')
            diagnostics['outside_marginal_training_range']=(np.asarray(diagnostics['outside_marginal_training_range'])|np.any((x<pl)|(x>ph),axis=1)).tolist()
            regression=p['positive_regression'];beta=_numeric(regression.get('coefficients'),'coefficients',vector_only=True);variance=_scalar(regression.get('residual_variance'),'residual_variance',0)
            if beta.shape!=(d.shape[1],):_fail('INVALID_MODEL','Positive amount coefficient dimension mismatch')
            mu=d@beta
            if p['positive_gate']is None:positive=np.ones(n)
            else:positive=np.asarray(predict_pd(p['positive_gate'],x,feature_order=feature_order,expected_target=p['positive_gate']['target']).value)
            with np.errstate(over='ignore',under='ignore'):ead=positive*np.exp(mu+variance/2)
            quantiles=None
            if levels is not None:
                quantiles=np.zeros((n,len(levels)))
                for i in range(n):
                    for j,u in enumerate(levels):
                        if u>1-positive[i]:
                            q=(u-(1-positive[i]))/positive[i]
                            with np.errstate(over='ignore'):quantiles[i,j]=np.exp(mu[i]+math.sqrt(variance)*-ndtri((1-u)/positive[i]))
        if not np.all(np.isfinite(ead))or(quantiles is not None and not np.all(np.isfinite(quantiles))):_fail('NUMERICAL_FAILURE','Direct EAD mean or quantiles outside finite range')
        value.update(conditional_ead=ead.tolist(),positive_probability=positive.tolist(),distribution='ZERO_HURDLE_LOGNORMAL',quantiles=None if quantiles is None else quantiles.tolist())
    else:_fail('INVALID_MODEL','Unknown EAD target mode')
    return _result(value,'ead.predict_ead',currency,ctx,diagnostics=diagnostics,provenance=model['provenance'],limitations=('Conditional on the frozen default horizon. CCF means are unbounded working regressions, not loss distributions; negative implied exposure is rejected. Lognormal amount quantiles require the fitted distributional assumption and independent training observations.',))

from ._ead_authorizations import reconcile_card_authorizations
