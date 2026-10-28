"""Explicit small-business cashflow and household interactions without PD transfer."""
from datetime import timedelta
from .core import CalcResult,_result,_fail,_fingerprint
from .data import _required,_date
from ._loss_paths import _sum
from .retail_features import _text,_strings,_number,_releases
from ._retail_behavior import _window,_history_coverage


def build_retail_sme_features(business_cashflows,owner_drawings,household_obligations,liability_links,coverage,*,business_id,owner_customer_ids,business_debt_facility_ids,household_debt_service_window,as_of,information_cutoff,lookback_days,currency,context=None):
    """Separate operating cash, owner transfers and explicitly linked household debt."""
    anchor,cutoff,ctx,days,start=_window(as_of,information_cutoff,lookback_days,context);_text(business_id,'business_id');owners=_strings(owner_customer_ids,'owner_customer_ids',False);debt_ids=_strings(business_debt_facility_ids,'business_debt_facility_ids');_text(currency,'currency');complete=_history_coverage(coverage,cutoff,start,anchor)
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Context currency differs from business currency')
    envelope=household_obligations.to_dict()if isinstance(household_obligations,CalcResult)else household_obligations
    if not isinstance(envelope,dict)or envelope.get('method_id')!='retail_features.aggregate_household_obligations'or not isinstance(envelope.get('value'),dict)or not isinstance(envelope.get('context'),dict):_fail('RETAIL_ARTIFACT','Supply household aggregation result with dated context')
    household=envelope['value'];hctx=envelope['context'];window=household_debt_service_window
    if not isinstance(window,dict)or window.get('from')!=(start+timedelta(days=1)).isoformat()or window.get('through')!=as_of or window.get('period')!=household.get('period'):_fail('TARGET_MISMATCH','Household debt-service period must explicitly cover the same cashflow window')
    if household.get('currency')!=currency or hctx.get('as_of')!=as_of or hctx.get('known_at')!=information_cutoff:_fail('TARGET_MISMATCH','Household obligations currency and dated context must match')
    households=household.get('households')
    if not isinstance(households,list):_fail('RETAIL_ARTIFACT','Household aggregation lacks household rows')
    selected=[];seen_owners=set();overlap=set()
    for row in households:
        _required(row,['household_id','customer_ids','facility_ids','known_budget_debt_service']);members=_strings(row['customer_ids'],'household.customer_ids');facilities=_strings(row['facility_ids'],'household.facility_ids');matched=set(members)&set(owners)
        if not matched:continue
        if seen_owners&matched:_fail('RETAIL_ENTITY','Owner has multiple economic households in same snapshot')
        seen_owners|=matched;overlap|=set(facilities)&set(debt_ids);_number(row['known_budget_debt_service'],'known_budget_debt_service',0)
        if 'total_budget_debt_service'not in row:_fail('RETAIL_ARTIFACT','Explicit nullable household total required')
        if row['total_budget_debt_service']is not None:_number(row['total_budget_debt_service'],'total_budget_debt_service',0)
        selected.append(dict(household_id=row['household_id'],owner_customer_ids=sorted(matched),known_budget_debt_service=row['known_budget_debt_service'],total_budget_debt_service=row['total_budget_debt_service'],facility_ids=facilities))
    selected.sort(key=lambda r:r['household_id']);missing=sorted(set(owners)-seen_owners);household_complete=not missing and all(r['total_budget_debt_service']is not None for r in selected);household_service=_sum(r['known_budget_debt_service']for r in selected)
    rows,flow_excluded=_releases(business_cashflows,'transaction_id',cutoff,'business cashflows');visible={};flows=[];allowed={'SALES','OPERATING_COST','TAX','CAPEX','DEBT_SERVICE','FINANCING','OWNER_DRAW','OWNER_CONTRIBUTION'}
    for row in rows:
        _required(row,['business_id','posted_at','amount','category','currency']);posted=_date(row['posted_at'],'posted_at');_number(row['amount'],'amount')
        if row['business_id']!=business_id or row['currency']!=currency:_fail('RETAIL_ENTITY','Business flows must match business identity and currency')
        if posted>_date(row['known_at'],'known_at'):_fail('RETAIL_DATE','Business posting cannot follow receipt')
        if row['category']not in allowed:_fail('RETAIL_CASHFLOW','Explicit supported business category required')
        if row['category']=='SALES':_text(row.get('counterparty_id'),'sales counterparty_id')
        if row['category']=='DEBT_SERVICE'and row.get('facility_id')not in debt_ids:_fail('RETAIL_ENTITY','Debt service must identify a declared business debt facility')
        if row['category']=='OWNER_DRAW'and row['amount']>0 or row['category']=='OWNER_CONTRIBUTION'and row['amount']<0:_fail('RETAIL_TRANSFER','Owner draws are business debits; contributions are credits')
        if posted<=anchor:visible[row['transaction_id']]=row
        if start<posted<=anchor:flows.append(row)
    draws,draw_excluded=_releases(owner_drawings,'drawing_id',cutoff,'owner drawings');matched_business=set();matched_household=set();draw_rows=[]
    for row in draws:
        _required(row,['business_transaction_id','household_transaction_id','owner_customer_id','business_posted_at','household_posted_at','business_amount','household_amount','currency']);original=visible.get(row['business_transaction_id']);bdate=_date(row['business_posted_at'],'business_posted_at');hdate=_date(row['household_posted_at'],'household_posted_at');known=_date(row['known_at'],'known_at');ba=_number(row['business_amount'],'business_amount');ha=_number(row['household_amount'],'household_amount',0);_text(row['household_transaction_id'],'household_transaction_id')
        if original is None or original['category']!='OWNER_DRAW'or row['owner_customer_id']not in owners or row['currency']!=currency or ba!=-ha or ba!=original['amount']or row['business_posted_at']!=original['posted_at']:_fail('RETAIL_TRANSFER','Owner drawing must match opposite business and household transfer legs exactly')
        if max(bdate,hdate)>known or hdate>anchor:_fail('RETAIL_DATE','Owner transfer evidence cannot anticipate household receipt')
        if row['business_transaction_id']in matched_business or row['household_transaction_id']in matched_household:_fail('DOUBLE_COUNTING','One owner-draw transfer cannot be assigned twice')
        matched_business.add(row['business_transaction_id']);matched_household.add(row['household_transaction_id'])
        if start<bdate<=anchor or start<hdate<=anchor:draw_rows.append(dict(drawing_id=row['drawing_id'],owner_customer_id=row['owner_customer_id'],business_transaction_id=row['business_transaction_id'],household_transaction_id=row['household_transaction_id'],business_posted_at=row['business_posted_at'],household_posted_at=row['household_posted_at'],business_amount=ba,household_amount=ha))
    unmatched=sorted(r['transaction_id']for r in flows if r['category']=='OWNER_DRAW'and r['transaction_id']not in matched_business)
    totals={k:_sum(r['amount']for r in flows if r['category']==k)for k in sorted(allowed)};operating=_sum(totals[k]for k in ('SALES','OPERATING_COST','TAX'));before_debt=_sum((operating,totals['CAPEX']));after_debt=_sum((before_debt,totals['DEBT_SERVICE']));customers={};months={}
    for offset in range(1,days+1):months.setdefault((start+timedelta(days=offset)).isoformat()[:7],[])
    for row in flows:
        if row['category']=='SALES':customers.setdefault(row['counterparty_id'],[]).append(row['amount']);months[row['posted_at'][:7]].append(row['amount'])
    customer_sales={k:_sum(v)for k,v in sorted(customers.items())};monthly_sales={k:_sum(v)for k,v in sorted(months.items())};sales=totals['SALES'];hhi=None if sales<=0 or any(v<0 for v in customer_sales.values())else _sum((v/sales)**2 for v in customer_sales.values());peak=None if sales<=0 or any(v<0 for v in monthly_sales.values())else max(monthly_sales.values())/sales
    rows,liability_excluded=_releases(liability_links,'liability_id',cutoff,'owner liabilities');liabilities=[]
    for row in rows:
        _required(row,['business_id','customer_id','liability_type','effective_from','evidence_reference'])
        if any(k not in row for k in ('effective_until','liability_fraction','maximum_amount')):_fail('RETAIL_CONTRACT','Liability end, fraction and cap require explicit nullable values')
        first=_date(row['effective_from'],'effective_from');last=None if row['effective_until']is None else _date(row['effective_until'],'effective_until')
        if last is not None and last<=first:_fail('RETAIL_DATE','Liability interval must be nonempty')
        if row['business_id']!=business_id or row['customer_id']not in owners:_fail('RETAIL_ENTITY','Legal links must match business and supplied owners')
        if row['liability_type']not in ('LIMITED_OWNERSHIP','PERSONAL_GUARANTEE','UNLIMITED_LIABILITY'):_fail('RETAIL_LIABILITY','Declare legal liability type; ownership is not a guarantee')
        _text(row['evidence_reference'],'evidence_reference')
        if row['liability_fraction']is not None and not 0<=_number(row['liability_fraction'],'liability_fraction')<=1:_fail('RETAIL_LIABILITY','Liability fraction must be in [0,1]')
        if row['maximum_amount']is not None:_number(row['maximum_amount'],'maximum_amount',0)
        if first<=anchor and(last is None or anchor<last):liabilities.append(row)
    combined=None if not complete or not household_complete or overlap else _sum((after_debt,-household_service))
    return _result(dict(business_id=business_id,owner_customer_ids=sorted(owners),as_of=as_of,information_cutoff=information_cutoff,currency=currency,lookback_days=days,coverage_complete=complete,business_category_net=totals,known_operating_cash_before_capex_and_debt=operating,known_cash_before_business_debt=before_debt,known_cash_after_business_debt=after_debt,cash_after_business_debt=after_debt if complete else None,owner_draw_business_outflows=-totals['OWNER_DRAW'],matched_owner_draw_household_inflows=_sum(r['household_amount']for r in draw_rows if start<_date(r['household_posted_at'],'household_posted_at')<=anchor),owner_transfers=sorted(draw_rows,key=lambda r:r['drawing_id']),unmatched_business_drawing_ids=unmatched,customer_sales=customer_sales,customer_sales_hhi=hhi,monthly_sales=monthly_sales,observed_peak_month_sales_share=peak,owner_households=selected,missing_owner_household_ids=missing,known_household_budget_debt_service=household_service,total_household_budget_debt_service=household_service if household_complete else None,overlapping_business_household_facility_ids=sorted(overlap),cash_after_business_and_household_debt=combined,legal_liability_links=liabilities),'retail_features.build_retail_sme_features',currency,ctx,diagnostics=dict(coverage=coverage,household_result_fingerprint=_fingerprint(envelope),household_debt_service_window=window,excluded_cashflow_releases=flow_excluded,excluded_drawing_releases=draw_excluded,excluded_liability_releases=liability_excluded),limitations=('Business sales are not household income. Operating cash excludes financing, owner drawings and owner contributions; matched drawings are transfers, never a second source of consolidated cash. Cash after household debt excludes household living costs and therefore is not an affordability or distributable-income measure. Shared business/household debt facility IDs suppress combined subtraction to avoid double counting. Monthly sales and customer concentration describe supplied observations, not an estimated seasonal forecast. Legal ownership, guarantees and unlimited liability remain explicit; no owner or business PD is copied.',))
