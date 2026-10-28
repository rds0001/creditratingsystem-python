"""Dated retail account behavior; descriptive features never classify default."""
from datetime import timedelta
import math
from .core import _result,_fail
from .data import _rows,_required,_date,_days,_unique
from ._loss_paths import _sum
from .retail_features import _cutoffs,_text,_number,_releases
from .behavioral import compute_days_past_due


def _window(as_of,information_cutoff,lookback_days,context):
    anchor,cutoff,ctx=_cutoffs(as_of,information_cutoff,context);days=_days(lookback_days,'lookback_days',1)
    if days>=anchor.toordinal():_fail('RETAIL_DATE','Observation window precedes supported calendar')
    return anchor,cutoff,ctx,days,anchor-timedelta(days=days)


def _history_coverage(coverage,cutoff,start,anchor):
    if not isinstance(coverage,dict):_fail('RETAIL_COVERAGE','Explicit dated historical coverage required')
    _required(coverage,['status','known_at','observed_from','complete_through'])
    if coverage['status']not in ('COMPLETE','PARTIAL','UNKNOWN')or _date(coverage['known_at'],'coverage.known_at')>cutoff:_fail('RETAIL_COVERAGE','Coverage status must be known at cutoff')
    first=_date(coverage['observed_from'],'coverage.observed_from');last=_date(coverage['complete_through'],'coverage.complete_through')
    if last<first or last>_date(coverage['known_at'],'coverage.known_at'):_fail('RETAIL_COVERAGE','Coverage interval or receipt chronology invalid')
    return coverage['status']=='COMPLETE'and first<=start+timedelta(days=1)and last>=anchor


def build_payment_history_features(obligations,payments,coverage,*,account_id,as_of,information_cutoff,lookback_days,allocation_policy,currency,grace_days,context=None):
    """Replay daily historical ledgers with contemporaneous knowledge cutoffs."""
    anchor,cutoff,ctx,days,start=_window(as_of,information_cutoff,lookback_days,context);_text(account_id,'account_id');_text(currency,'currency');grace=_days(grace_days,'grace_days');complete=_history_coverage(coverage,cutoff,start,anchor)
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Context currency differs from payment currency')
    obs=_rows(obligations,'obligations');pays=_rows(payments,'payments')
    if any(r.get('account_id')!=account_id for r in obs+pays):_fail('RETAIL_ENTITY','Supply only the explicitly requested account ledger')
    # Core ledger validates identifiers, reversal lineage, currency and allocation.
    def replay(date):
        result=compute_days_past_due(obs,pays,as_of=date.isoformat(),known_at=min(cutoff,date).isoformat(),allocation_policy=allocation_policy,currency=currency).value
        for key in ('past_due_amount','unpaid_due_amount','net_payments','unallocated_payments'):
            if not math.isfinite(result[key]):_fail('NUMERICAL_FAILURE','Historical ledger total outside finite range')
        return result
    daily=[];cache={};transitions=[]
    for offset in range(1,days+1):
        date=start+timedelta(days=offset);r=replay(date);cache[date.isoformat()]=r;state='ARREARS'if r['past_due_amount']>0 else 'CURRENT'
        daily.append(dict(date=date.isoformat(),information_cutoff=min(cutoff,date).isoformat(),dpd=r['dpd'],past_due_amount=r['past_due_amount'],unpaid_due_amount=r['unpaid_due_amount'],state=state))
        if len(daily)>1 and daily[-2]['state']!=state:transitions.append(dict(date=date.isoformat(),from_state=daily[-2]['state'],to_state=state))
    assessments=[];not_assessable=[]
    for obligation in sorted(obs,key=lambda r:(r['due_at'],r['obligation_id'])):
        due=_date(obligation['due_at'],'due_at');known=_date(obligation['known_at'],'known_at')
        if not start<due<=anchor or known>cutoff:continue
        if grace>anchor.toordinal()-due.toordinal():not_assessable.append(dict(obligation_id=obligation['obligation_id'],reason='GRACE_NOT_ELAPSED'));continue
        assessment=due+timedelta(days=grace)
        if known>min(cutoff,assessment):not_assessable.append(dict(obligation_id=obligation['obligation_id'],reason='OBLIGATION_NOT_CONTEMPORANEOUSLY_KNOWN'));continue
        r=cache.get(assessment.isoformat())or replay(assessment);item=next(v for v in r['obligations']if v['obligation_id']==obligation['obligation_id']);amount=item['amount'];paid=item['applied'];state='ZERO_OBLIGATION'if amount==0 else 'MISSED'if paid==0 else 'PARTIAL'if paid<amount else 'FULL'
        assessments.append(dict(obligation_id=obligation['obligation_id'],due_at=obligation['due_at'],assessment_at=assessment.isoformat(),amount=amount,applied=paid,payment_ratio=None if amount==0 else paid/amount,state=state))
    counts={k:sum(r['state']==k for r in assessments)for k in ('MISSED','PARTIAL','FULL','ZERO_OBLIGATION')};amount=_sum(r['amount']for r in assessments);applied=_sum(r['applied']for r in assessments)
    known_features=dict(max_dpd=max(r['dpd']for r in daily),mean_dpd=_sum(r['dpd']/days for r in daily),arrears_days=sum(r['past_due_amount']>0 for r in daily),missed_payment_count=counts['MISSED'],partial_payment_count=counts['PARTIAL'],full_payment_count=counts['FULL'],assessment_payment_ratio=None if amount==0 else applied/amount)
    return _result(dict(account_id=account_id,as_of=as_of,information_cutoff=information_cutoff,currency=currency,lookback_days=days,grace_days=grace,coverage_complete=complete,known_history_features=known_features,features=known_features if complete else {k:None for k in known_features},daily_history=daily,assessments=assessments,not_assessable=not_assessable,transitions=transitions),'retail_features.build_payment_history_features','account_behavior',ctx,diagnostics=dict(allocation_policy=allocation_policy,coverage=coverage,knowledge_policy='MIN_EVALUATION_CUTOFF_AND_HISTORICAL_DATE'),limitations=('Replay requires complete historical contractual obligations and payment/reversal ledger, including pre-window balances and prepayments. Source coverage is asserted, never inferred from an empty ledger. Each historical date uses only knowledge then available; late-known reversals do not rewrite earlier days. Grace changes only the assessment date, never the contractual due date or DPD. Current/arrears are descriptive unpaid-ledger states, not regulatory default labels.',))


def build_revolving_features(balance_history,statements,coverage,*,account_id,as_of,information_cutoff,lookback_days,currency,context=None):
    """Separate balance movement, limit changes and minimum/full repayment."""
    anchor,cutoff,ctx,days,start=_window(as_of,information_cutoff,lookback_days,context);_text(account_id,'account_id');_text(currency,'currency');complete=_history_coverage(coverage,cutoff,start,anchor)
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Context currency differs from revolving currency')
    rows,excluded=_releases(balance_history,'snapshot_id',cutoff,'balance history');daily=[];seen=set()
    for row in rows:
        _required(row,['account_id','observed_at','balance','credit_limit','activity_status','currency']);observed=_date(row['observed_at'],'observed_at')
        if observed>_date(row['known_at'],'known_at'):_fail('RETAIL_DATE','Balance observation cannot follow receipt')
        if row['account_id']!=account_id or row['currency']!=currency:_fail('RETAIL_ENTITY','Revolving history must match account and currency')
        balance=_number(row['balance'],'balance',0);limit=_number(row['credit_limit'],'credit_limit',0)
        if row['activity_status']not in ('ACTIVE','INACTIVE','CLOSED'):_fail('RETAIL_INPUT','Explicit activity status required; zero draw does not imply inactivity')
        if row['activity_status']=='CLOSED'and(balance!=0 or limit!=0):_fail('RETAIL_INPUT','Closed facility requires zero residual balance and limit')
        if not start<observed<=anchor:continue
        if row['observed_at']in seen:_fail('DATA_DUPLICATE','One resolved balance snapshot per account and day required')
        seen.add(row['observed_at']);utilization=None if limit==0 else balance/limit
        if utilization is not None and not math.isfinite(utilization):_fail('NUMERICAL_FAILURE','Utilization outside finite range')
        daily.append(dict(date=row['observed_at'],balance=balance,credit_limit=limit,utilization=utilization,overlimit=balance>limit,activity_status=row['activity_status'],undrawn_active=row['activity_status']=='ACTIVE'and balance==0,snapshot_id=row['snapshot_id'],release_seq=row['release_seq']))
    daily.sort(key=lambda r:r['date']);complete=complete and len(daily)==days;changes=[]
    for previous,current in zip(daily,daily[1:]):
        db=_sum((current['balance'],-previous['balance']));dl=_sum((current['credit_limit'],-previous['credit_limit']));du=None;balance_effect=None;limit_effect=None
        if previous['credit_limit']>0 and current['credit_limit']>0:
            du=_sum((current['utilization'],-previous['utilization']));balance_effect=db/current['credit_limit'];limit_effect=previous['utilization']*(-dl/current['credit_limit'])
            if not all(math.isfinite(v)for v in (balance_effect,limit_effect)):_fail('NUMERICAL_FAILURE','Utilization change decomposition outside finite range')
        changes.append(dict(from_date=previous['date'],to_date=current['date'],elapsed_days=(_date(current['date'],'date')-_date(previous['date'],'date')).days,balance_change=db,limit_change=dl,utilization_change=du,balance_effect_at_new_limit=balance_effect,limit_effect_on_previous_balance=limit_effect))
    statement_rows,statement_excluded=_releases(statements,'statement_id',cutoff,'statements');assessments=[];pending=[]
    for row in statement_rows:
        _required(row,['account_id','issued_at','due_at','observed_at','statement_balance','minimum_due','paid_by_due','currency']);issued=_date(row['issued_at'],'issued_at');due=_date(row['due_at'],'due_at');observed=_date(row['observed_at'],'observed_at')
        if not issued<=observed<=_date(row['known_at'],'known_at') or due<issued:_fail('RETAIL_DATE','Statement dates or receipt chronology invalid')
        if row['account_id']!=account_id or row['currency']!=currency:_fail('RETAIL_ENTITY','Statement must match account and currency')
        balance=_number(row['statement_balance'],'statement_balance',0);minimum=_number(row['minimum_due'],'minimum_due',0);paid=_number(row['paid_by_due'],'paid_by_due',0)
        if minimum>balance:_fail('RETAIL_INPUT','Minimum payment cannot exceed supplied statement payable balance')
        if not start<due<=anchor:continue
        if observed<due:pending.append(row['statement_id']);continue
        state='NO_AMOUNT_DUE'if balance==0 else 'FULL'if paid>=balance else 'MINIMUM_MET'if paid>=minimum else 'BELOW_MINIMUM'
        assessments.append(dict(statement_id=row['statement_id'],due_at=row['due_at'],statement_balance=balance,minimum_due=minimum,paid_by_due=paid,full_repayment_ratio=None if balance==0 else paid/balance,minimum_repayment_ratio=None if minimum==0 else paid/minimum,repayment_state=state))
        if any(v is not None and not math.isfinite(v)for v in (assessments[-1]['full_repayment_ratio'],assessments[-1]['minimum_repayment_ratio'])):_fail('NUMERICAL_FAILURE','Repayment ratio outside finite range')
    assessments.sort(key=lambda r:(r['due_at'],r['statement_id']));utilizations=[r['utilization']for r in daily if r['utilization']is not None];latest=daily[-1]if daily else None;trend=None
    if len(daily)>1 and daily[0]['utilization']is not None and daily[-1]['utilization']is not None:trend=_sum((daily[-1]['utilization'],-daily[0]['utilization']))/(_date(daily[-1]['date'],'date')-_date(daily[0]['date'],'date')).days
    return _result(dict(account_id=account_id,as_of=as_of,information_cutoff=information_cutoff,currency=currency,lookback_days=days,coverage_complete=complete,observed_days=len(daily),missing_days=days-len(daily),known_overlimit_days=sum(r['overlimit']for r in daily),overlimit_days=sum(r['overlimit']for r in daily)if complete else None,latest_snapshot=latest,mean_observed_utilization=None if not utilizations else _sum(v/len(utilizations)for v in utilizations),utilization_endpoint_trend_per_day=trend,balance_changes=changes,daily_history=daily,statement_assessments=assessments,unmatured_statement_ids=sorted(pending),full_payer_count=sum(r['repayment_state']=='FULL'for r in assessments),minimum_without_full_count=sum(r['repayment_state']=='MINIMUM_MET'for r in assessments),below_minimum_count=sum(r['repayment_state']=='BELOW_MINIMUM'for r in assessments)),'retail_features.build_revolving_features','account_behavior',ctx,diagnostics=dict(coverage=coverage,excluded_balance_releases=excluded,excluded_statement_releases=statement_excluded),limitations=('Balances are nonnegative drawn exposure, not signed deposit credits. Missing daily observations are not carried forward or counted as zero overlimit days. Utilization changes are decomposed using new-limit balance movement plus the effect of changing the limit on the previous balance; this describes balance movement, not gross draw transactions. Supplied paid_by_due must be the allocated amount by contractual due date, not payments observed after it. Activity is supplied explicitly and is not inferred from zero balance.',))
