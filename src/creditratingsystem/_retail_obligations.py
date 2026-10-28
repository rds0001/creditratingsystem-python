"""Dated purchase-plan obligations and explicit household/business interaction."""
from datetime import date,timedelta
from .core import _result,_fail
from .data import _required,_date,_days
from ._loss_paths import _sum
from .retail_features import _text,_number,_releases,_coverage
from ._retail_behavior import _window


def build_bnpl_obligation_features(plans,installments,events,provider_coverage,*,customer_id,as_of,information_cutoff,lookback_days,near_term_days,currency,context=None):
    """Aggregate known purchase plans, allocated payments/refunds and overlap."""
    anchor,cutoff,ctx,days,start=_window(as_of,information_cutoff,lookback_days,context);_text(customer_id,'customer_id');_text(currency,'currency');near=_days(near_term_days,'near_term_days',1)
    if near>date.max.toordinal()-anchor.toordinal():_fail('RETAIL_DATE','Near-term horizon exceeds supported calendar')
    end=anchor+timedelta(days=near);coverage=_coverage(provider_coverage,cutoff)
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Context currency differs from BNPL currency')
    rows,excluded_plans=_releases(plans,'plan_id',cutoff,'purchase plans');visible={}
    for row in rows:
        _required(row,['customer_id','provider_id','purchased_at','currency','schedule_complete'])
        if any(k not in row for k in ('closed_at','refinances_plan_id')):_fail('RETAIL_CONTRACT','Plan closed_at and refinances_plan_id must be explicit nullable fields')
        if type(row['schedule_complete'])is not bool:_fail('RETAIL_COVERAGE','Each plan requires explicit schedule completeness')
        purchased=_date(row['purchased_at'],'purchased_at');closed=None if row['closed_at']is None else _date(row['closed_at'],'closed_at');known=_date(row['known_at'],'known_at')
        if purchased>known or (closed is not None and not purchased<=closed<=known):_fail('RETAIL_DATE','Purchase/actual closure must be known after occurrence')
        if row['customer_id']!=customer_id or row['currency']!=currency or row['provider_id']not in coverage['sources']:_fail('RETAIL_ENTITY','Plans must match customer, currency and declared providers')
        if purchased<=anchor:visible[row['plan_id']]=row
    for row in visible.values():
        if row['refinances_plan_id']is not None:
            previous=visible.get(row['refinances_plan_id'])
            if previous is None or previous['plan_id']==row['plan_id']or previous['purchased_at']>=row['purchased_at']:_fail('RETAIL_PLAN','Explicit refinanced plan must be an earlier visible plan')
    rows,excluded_installments=_releases(installments,'installment_id',cutoff,'installments');due={}
    for row in rows:
        _required(row,['plan_id','due_at','amount','currency']);plan=visible.get(row['plan_id'])
        if plan is None:_fail('RETAIL_PLAN','Visible installment lacks a visible customer plan')
        if _date(row['due_at'],'due_at')<_date(plan['purchased_at'],'purchased_at')or row['currency']!=currency:_fail('RETAIL_DATE','Installment date/currency incompatible with purchase plan')
        _number(row['amount'],'installment amount',0);due[row['installment_id']]=row
    rows,excluded_events=_releases(events,'event_id',cutoff,'plan events');visible_events={};financial={'PAYMENT','MERCHANT_REFUND','PAYMENT_REVERSAL','REFUND_REVERSAL'}
    for row in rows:
        _required(row,['plan_id','occurred_at','kind','amount','currency'])
        if 'installment_id'not in row or 'related_event_id'not in row:_fail('RETAIL_CONTRACT','Event installment and related event must be explicit nullable fields')
        plan=visible.get(row['plan_id']);occurred=_date(row['occurred_at'],'occurred_at')
        if plan is None or row['currency']!=currency:_fail('RETAIL_PLAN','Visible event must match a visible plan and currency')
        if not _date(plan['purchased_at'],'purchased_at')<=occurred<=_date(row['known_at'],'known_at'):_fail('RETAIL_DATE','Plan-event occurrence/receipt chronology invalid')
        _number(row['amount'],'event amount',0)
        if row['kind']not in financial|{'DISPUTE_OPEN','DISPUTE_CLOSE'}:_fail('RETAIL_PLAN','Unsupported plan event')
        if row['kind']in financial:
            if plan['closed_at']is not None and occurred>_date(plan['closed_at'],'closed_at'):_fail('RETAIL_PLAN','Financial event follows actual plan closure; correct closure or separate customer cashflows')
            installment=due.get(row['installment_id'])
            if installment is None or installment['plan_id']!=row['plan_id']:_fail('RETAIL_PLAN','Financial events require explicit same-plan installment allocation')
        elif row['amount']!=0:_fail('RETAIL_PLAN','Dispute events do not themselves reduce debt; use an allocated merchant refund')
        if occurred<=anchor:visible_events[row['event_id']]=row
    settlements={k:[]for k in due};refunds={k:[]for k in due};reversals={};open_disputes={};closed_disputes=set()
    for key,row in sorted(visible_events.items()):
        kind=row['kind'];related=row['related_event_id']
        if kind in ('PAYMENT','MERCHANT_REFUND'):
            if related is not None:_fail('RETAIL_PLAN','Original financial event must not identify a reversed event')
            (settlements if kind=='PAYMENT'else refunds)[row['installment_id']].append(row['amount'])
        elif kind in ('PAYMENT_REVERSAL','REFUND_REVERSAL'):
            original=visible_events.get(related);expected='PAYMENT'if kind=='PAYMENT_REVERSAL'else 'MERCHANT_REFUND'
            if original is None or original['kind']!=expected or original['installment_id']!=row['installment_id']or original['occurred_at']>row['occurred_at']:_fail('RETAIL_REVERSAL','Plan reversal must match an earlier allocated original event')
            reversals.setdefault(related,[]).append(row['amount']);(settlements if kind=='PAYMENT_REVERSAL'else refunds)[row['installment_id']].append(-row['amount'])
        elif kind=='DISPUTE_OPEN':
            if related is not None:_fail('RETAIL_PLAN','Dispute opening cannot identify a closed dispute')
            open_disputes[key]=row['plan_id']
        else:
            original=visible_events.get(related)
            if original is None or original['kind']!='DISPUTE_OPEN'or original['plan_id']!=row['plan_id']or original['occurred_at']>row['occurred_at']or related in closed_disputes:_fail('RETAIL_PLAN','Dispute closure needs one earlier open dispute on the same plan')
            closed_disputes.add(related)
    for key,amounts in reversals.items():
        if _sum(amounts)>visible_events[key]['amount']:_fail('RETAIL_REVERSAL','Plan reversal amounts exceed original event')
    outputs=[];plan_outstanding={k:[]for k in visible};near_cashflows={};overdue=[];due_today=[]
    for key,row in sorted(due.items(),key=lambda kv:(kv[1]['due_at'],kv[0])):
        paid=_sum(settlements[key]);refunded=_sum(refunds[key]);remaining=_sum((row['amount'],-paid,-refunded))
        if remaining<0:_fail('RETAIL_PLAN','Net allocated payments and refunds exceed installment; supply separate customer cash refund outside debt reduction')
        plan_outstanding[row['plan_id']].append(remaining);d=_date(row['due_at'],'due_at')
        if d<anchor:overdue.append(remaining)
        elif d==anchor:due_today.append(remaining)
        elif d<=end:near_cashflows.setdefault(row['due_at'],[]).append(remaining)
        outputs.append(dict(installment_id=key,plan_id=row['plan_id'],due_at=row['due_at'],contractual_amount=row['amount'],net_paid=paid,net_merchant_refund=refunded,outstanding=remaining))
    plan_summary=[]
    for key,row in sorted(visible.items()):
        amount=_sum(plan_outstanding[key]);closed=row['closed_at']is not None and _date(row['closed_at'],'closed_at')<=anchor
        if closed and amount!=0:_fail('RETAIL_PLAN','Closed plan retains unpaid supplied installments; resolve payments/refunds or residual liability explicitly')
        plan_summary.append(dict(plan_id=key,provider_id=row['provider_id'],purchased_at=row['purchased_at'],closed_at=row['closed_at'],active=not closed,schedule_complete=row['schedule_complete'],known_outstanding=amount,installment_count=len(plan_outstanding[key]),refinances_plan_id=row['refinances_plan_id']))
    concurrency=[]
    for offset in range(1,days+1):
        d=(start+timedelta(days=offset)).isoformat();count=sum(row['purchased_at']<=d and(row['closed_at']is None or d<row['closed_at'])for row in visible.values());providers={row['provider_id']for row in visible.values()if row['purchased_at']<=d and(row['closed_at']is None or d<row['closed_at'])};concurrency.append(dict(date=d,known_active_plan_count=count,known_active_provider_count=len(providers)))
    new=[row for row in visible.values()if start<_date(row['purchased_at'],'purchased_at')<=anchor];known_total=_sum(row['known_outstanding']for row in plan_summary);complete=coverage['status']=='COMPLETE'and all(row['schedule_complete']and row['installment_count']>0 for row in plan_summary)
    return _result(dict(customer_id=customer_id,as_of=as_of,information_cutoff=information_cutoff,currency=currency,provider_coverage=coverage,coverage_complete=complete,active_plan_count=sum(row['active']for row in plan_summary),plans_with_positive_outstanding=sum(row['known_outstanding']>0 for row in plan_summary),maximum_known_concurrent_plans=max(row['known_active_plan_count']for row in concurrency),maximum_known_concurrent_providers=max(row['known_active_provider_count']for row in concurrency),new_plan_count=len(new),repeat_purchase_count=sum(row['plan_id']!=min(visible.values(),key=lambda r:(r['purchased_at'],r['plan_id']))['plan_id']for row in new),explicit_refinancing_count=sum(row['refinances_plan_id']is not None for row in new),open_dispute_count=len(set(open_disputes)-closed_disputes),known_outstanding=known_total,total_outstanding=known_total if complete else None,known_overdue_amount=_sum(overdue),known_due_today=_sum(due_today),near_term_days=near,known_near_term_due=_sum(v for values in near_cashflows.values()for v in values),near_term_cashflows=[dict(due_at=k,known_due=_sum(v))for k,v in sorted(near_cashflows.items())],plans=plan_summary,installments=outputs,daily_known_concurrency=concurrency),'retail_features.build_bnpl_obligation_features',currency,ctx,diagnostics=dict(excluded_plan_releases=excluded_plans,excluded_installment_releases=excluded_installments,excluded_event_releases=excluded_events),limitations=('Provider completeness is supplied, never inferred from a bureau or an empty file. Financial events require explicit installment allocation, without guessed FIFO. Disputes are separate nonfinancial events and do not imply default or suspend debt automatically. Refunds reduce only explicitly allocated unpaid installments; excess customer refunds are separate cashflows. Active-plan overlap uses actual purchase/closure dates visible at this cutoff, not assumed contractual maturity; it is a retrospective known-history description. Repeated purchases are not inferred refinancing; refinancing needs an explicit earlier-plan link.',))
