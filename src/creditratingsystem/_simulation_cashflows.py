"""Dated contractual debt and cash-conserving liquidity waterfalls."""
import math
from copy import deepcopy
from collections import defaultdict
from scipy.special import logsumexp
from .core import _fail, _result, _fingerprint, _context_value
from .data import _rows, _required, _date
from .simulation import _number, _text


def build_debt_schedule(loan_terms, calendar, *, base_rate_fixings=None, context=None):
    """Build contractual principal/interest legs, without assuming refinancing."""
    if not isinstance(loan_terms,dict):_fail('SIMULATION_TERMS','loan_terms must be a mapping')
    terms=deepcopy(loan_terms);periods=_rows(calendar,'calendar')
    _required(terms,['facility_id','currency','principal','rate_type','rate_convention','day_count','amortization','maturity_date'])
    
    for key in ('currency','rate_type','rate_convention','day_count','amortization'):_text(terms[key],key)
    principal=_number(terms['principal'],'principal',0)
    if principal<=0 or not periods:_fail('SIMULATION_TERMS','Positive principal and nonempty calendar required')
    if terms['rate_type']not in ('fixed','floating')or terms['rate_convention']not in ('nominal_simple','effective_annual')or terms['day_count']not in ('ACT/365F','ACT/360'):_fail('SIMULATION_TERMS','Explicit supported rate type/convention and day count required')
    if terms['amortization']not in ('bullet','equal_principal','annuity','custom'):_fail('SIMULATION_TERMS','Unknown amortization')
    if terms['amortization']=='annuity'and terms['rate_type']!='fixed':_fail('SIMULATION_TERMS','Constant-payment annuity requires a fixed rate; floating reset schedules require explicit principal amounts')
    ctx=_context_value(context)
    if ctx.get('currency')is not None and ctx['currency']!=terms['currency']:_fail('TARGET_MISMATCH','Debt currency conflicts with context')
    maturity=_date(terms['maturity_date'],'maturity_date');fixings={}
    if terms['rate_type']=='floating':
        _required(terms,['spread'])
        for fix in _rows(base_rate_fixings,'base_rate_fixings'):
            _required(fix,['fixing_date','base_rate']);_date(fix['fixing_date'],'fixing_date')
            if fix['fixing_date']in fixings:_fail('SIMULATION_DUPLICATE','Duplicate rate fixing')
            fixings[fix['fixing_date']]=_number(fix['base_rate'],'base_rate')
    elif base_rate_fixings is not None:_fail('SIMULATION_TERMS','Fixed-rate schedule must not consume base-rate fixings')
    else:_required(terms,['annual_rate'])
    validated=[];last_end=None;last_pay=None;used=set()
    for period in periods:
        _required(period,['start_date','end_date','payment_date'])
        start=_date(period['start_date'],'start_date');end=_date(period['end_date'],'end_date');payment=_date(period['payment_date'],'payment_date')
        if start>=end or payment<end or(last_end is not None and start!=last_end)or(last_pay is not None and payment<=last_pay):_fail('SIMULATION_CALENDAR','Contiguous positive accrual periods and strictly increasing payments at/after period end required')
        yf=(end-start).days/(365 if terms['day_count']=='ACT/365F'else 360)
        if terms['rate_type']=='fixed':rate=_number(terms['annual_rate'],'annual_rate')
        else:
            _required(period,['fixing_date']);fixdate=_date(period['fixing_date'],'fixing_date')
            if fixdate>start or period['fixing_date']not in fixings:_fail('SIMULATION_FIXING','Each floating period needs an explicit fixing dated no later than period start')
            used.add(period['fixing_date']);rate=fixings[period['fixing_date']]+_number(terms['spread'],'spread')
        if terms['rate_convention']=='effective_annual':
            if rate<=-1:_fail('SIMULATION_RATE','Effective annual rate must exceed -1')
            exponent=math.log1p(rate)*yf
            if exponent>709 or exponent< -709:_fail('SIMULATION_RATE','Accrual exponent outside supported finite range')
            growth=math.exp(exponent)
        else:growth=1+rate*yf
        if not math.isfinite(growth)or growth<=0:_fail('SIMULATION_RATE','Accrual growth factor must be finite and positive')
        validated.append((period,yf,rate,growth));last_end=end;last_pay=payment
    if last_pay!=maturity:_fail('SIMULATION_CALENDAR','Final payment must equal contractual maturity')
    if set(fixings)!=used:_fail('SIMULATION_FIXING','Unused or missing rate fixings are not accepted')
    payment_amount=None
    if terms['amortization']=='annuity':
        discounts=[];logfactor=0
        for _,_,_,growth in validated:
            logfactor+=math.log(growth);discounts.append(-logfactor)
        logpayment=math.log(principal)-float(logsumexp(discounts))
        if not -709<logpayment<709:_fail('NUMERICAL_FAILURE','Annuity payment outside supported finite range')
        payment_amount=math.exp(logpayment)
    balance=principal;result=[];tol=1e-10*max(1,principal)
    for index,(period,yf,rate,growth)in enumerate(validated):
        opening=balance;interest=opening*(rate*yf if terms['rate_convention']=='nominal_simple'else math.expm1(math.log1p(rate)*yf));last=index==len(validated)-1
        if terms['amortization']=='bullet':repay=opening if last else 0.0
        elif terms['amortization']=='equal_principal':repay=opening if last else principal/len(validated)
        elif terms['amortization']=='annuity':repay=opening if last else payment_amount-interest
        else:
            _required(period,['principal_due']);repay=_number(period['principal_due'],'principal_due',0)
        if repay < -tol or repay>opening+tol:_fail('SIMULATION_AMORTIZATION','Principal repayment must be between zero and opening principal')
        repay=min(opening,max(0,repay));balance=opening-repay
        net=interest+repay
        if not all(math.isfinite(x)for x in (interest,repay,balance,net)):_fail('NUMERICAL_FAILURE','Debt schedule exceeds finite representation')
        result.append(dict(facility_id=terms['facility_id'],currency=terms['currency'],start_date=period['start_date'],end_date=period['end_date'],payment_date=period['payment_date'],year_fraction=yf,annual_rate=rate,opening_principal=opening,interest_due=interest,principal_due=repay,debt_service_due=max(net,0),lender_refund_due=max(-net,0),net_payment_due=net,closing_principal=balance))
    if balance>tol:_fail('SIMULATION_AMORTIZATION','Contractual maturity must repay all principal; refinancing is not assumed')
    return _result(dict(schedule=result,terms=terms,terms_hash=_fingerprint(terms),constant_payment=payment_amount),'simulation.build_debt_schedule','debt_schedule',ctx,limitations=('Contractual accrual on opening principal, no fees, capitalization, prepayment option or assumed refinancing. Negative interest is retained and net lender refunds reported separately. ACT/360 and ACT/365F only; contractual calendars are supplied explicitly.',))


def apply_cash_waterfall(cash_periods, waterfall_policy, *, currency, context=None):
    """Allocate actual liquidity, reserves and carried unpaid claims in explicit order."""
    rows=_rows(cash_periods,'cash_periods');policy=deepcopy(waterfall_policy);_text(currency,'currency')
    if not isinstance(policy,dict):_fail('SIMULATION_WATERFALL','Explicit versioned waterfall policy required')
    _required(policy,['policy_id','version','initial_reserve','reserve_priority','excess_reserve_policy','arrears_policy','claims'])
    for key in ('policy_id','version','excess_reserve_policy','arrears_policy'):_text(policy[key],key)
    reserve_initial=_number(policy['initial_reserve'],'initial_reserve',0);reserve_priority=_number(policy['reserve_priority'],'reserve_priority')
    if policy['excess_reserve_policy']not in ('retain','release_before_claims')or policy['arrears_policy']!='carry_without_interest':_fail('SIMULATION_WATERFALL','Declare reserve excess treatment and carry_without_interest arrears')
    claims=_rows(policy['claims'],'claims');byid={};priorities={reserve_priority}
    for claim in claims:
        _required(claim,['claim_id','priority','reserve_eligible']);priority=_number(claim['priority'],'priority')
        if claim['claim_id']in byid or priority in priorities or not isinstance(claim['reserve_eligible'],bool):_fail('SIMULATION_WATERFALL','Unique claim IDs/priorities and Boolean reserve eligibility required')
        byid[claim['claim_id']]=claim;priorities.add(priority)
    if not claims:_fail('SIMULATION_WATERFALL','At least one claim required')
    ordered=sorted([(float(c['priority']),c['claim_id'])for c in claims]+[(reserve_priority,None)])
    ctx=_context_value(context)
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Waterfall currency conflicts with context')
    groups=defaultdict(list);seen=set()
    for row in rows:
        _required(row,['scenario_id','date','currency','cash_available','reserve_target','amounts_due']);_date(row['date'],'date')
        if row['currency']!=currency:_fail('TARGET_MISMATCH','All waterfall amounts must use reporting currency')
        key=(row['scenario_id'],row['date'])
        if key in seen:_fail('SIMULATION_DUPLICATE','Duplicate scenario/date')
        seen.add(key)
        if not isinstance(row['amounts_due'],dict)or set(row['amounts_due'])!=set(byid):_fail('SIMULATION_WATERFALL','Every date requires each claim amount, explicitly including zeros')
        groups[row['scenario_id']].append(row)
    outputs=[]
    for scenario,periods in sorted(groups.items()):
        reserve=reserve_initial;arrears={k:0.0 for k in byid}
        for row in sorted(periods,key=lambda r:r['date']):
            cash=_number(row['cash_available'],'cash_available',0);target=_number(row['reserve_target'],'reserve_target',0);cash_open=cash;reserve_open=reserve;release=0.0;topup=0.0;payments=[]
            if policy['excess_reserve_policy']=='release_before_claims':release=max(reserve-target,0);reserve-=release;cash+=release
            for _,cid in ordered:
                if cid is None:
                    topup=min(cash,max(target-reserve,0));cash-=topup;reserve+=topup;continue
                new_due=_number(row['amounts_due'][cid],'amount_due',0);carry=arrears[cid];demand=carry+new_due
                from_cash=min(cash,demand);cash-=from_cash
                from_reserve=min(reserve,demand-from_cash)if byid[cid]['reserve_eligible']else 0.0;reserve-=from_reserve
                paid=from_cash+from_reserve;arrears[cid]=max(demand-paid,0)
                payments.append(dict(claim_id=cid,new_amount_due=new_due,opening_arrears=carry,total_due=demand,payment_from_cash=from_cash,payment_from_reserve=from_reserve,total_paid=paid,closing_arrears=arrears[cid]))
            paid=math.fsum(p['total_paid']for p in payments);residual=cash_open+reserve_open-paid-reserve-cash
            if not math.isfinite(residual)or abs(residual)>1e-10*max(1,cash_open+reserve_open):_fail('NUMERICAL_FAILURE','Waterfall cash identity failed')
            outputs.append(dict(scenario_id=scenario,date=row['date'],currency=currency,cash_available=cash_open,opening_reserve=reserve_open,reserve_release=release,reserve_topup=topup,closing_reserve=reserve,reserve_target=target,payments=payments,residual_equity_cash=cash,cash_identity_residual=residual))
    return _result(dict(periods=outputs,policy=policy,policy_hash=_fingerprint(policy),currency=currency),'simulation.apply_cash_waterfall','cash_waterfall',ctx,limitations=('No reserve amount is operating revenue. Due amounts exclude carried arrears, which this function carries once without interest. Negative operating cashflows must be represented by explicit cost claims and available liquidity. Liquidity shortage is not automatically regulatory default; materiality, duration and UTP are assessed separately.',))
