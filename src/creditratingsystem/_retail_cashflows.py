"""Local retail cashflow classifications with dated lineage and coverage."""
from datetime import timedelta
import math
from .core import _result,_fail,_fingerprint
from .data import _required,_date,_rows
from ._loss_paths import _sum
from .retail_features import _text,_strings,_number,_releases
from ._retail_behavior import _window,_history_coverage


def build_transaction_features(transactions,daily_balances,category_mapping,coverage,*,customer_id,own_account_ids,as_of,information_cutoff,lookback_days,currency,context=None):
    """Describe booked cashflows; own transfers, borrowing and refunds are not income."""
    anchor,cutoff,ctx,days,start=_window(as_of,information_cutoff,lookback_days,context);_text(customer_id,'customer_id');accounts=_strings(own_account_ids,'own_account_ids',False);_text(currency,'currency');complete=_history_coverage(coverage,cutoff,start,anchor)
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Context currency differs from cashflow reporting currency')
    if not isinstance(category_mapping,dict):_fail('RETAIL_MAPPING','Dated local category mapping required')
    _required(category_mapping,['mapping_id','version','known_at','categories'])
    if _date(category_mapping['known_at'],'mapping.known_at')>cutoff:_fail('RETAIL_DATE','Category mapping unavailable at cutoff')
    categories=category_mapping['categories'];allowed={'SALARY','OTHER_INCOME','BUSINESS_REVENUE','CONSUMPTION','DEBT_SERVICE','BORROWING','REFUND','OWN_TRANSFER','OWNER_DRAW','TAX','UNKNOWN'}
    if not isinstance(categories,dict)or any(not isinstance(k,str)or not k or not isinstance(v,str)or v not in allowed for k,v in categories.items()):_fail('RETAIL_MAPPING','Unsupported local category mapping')
    rows,excluded=_releases(transactions,'transaction_id',cutoff,'transactions');visible={}
    for row in rows:
        _required(row,['account_id','posted_at','value_at','amount','currency','kind','category_code']);posted=_date(row['posted_at'],'posted_at');_date(row['value_at'],'value_at');_text(row['category_code'],'category_code')
        if posted>_date(row['known_at'],'known_at'):_fail('RETAIL_DATE','Transaction receipt cannot precede posting')
        if row['account_id']not in accounts or row['currency']!=currency:_fail('RETAIL_ENTITY','Transactions must match supplied own accounts and currency')
        _number(row['amount'],'amount')
        if row['kind']not in ('ORIGINAL','REVERSAL'):_fail('RETAIL_TRANSACTION','Declare ORIGINAL or REVERSAL kind')
        if posted<=anchor:visible[row['transaction_id']]=row
    reversal_totals={};classified=[];pairs={}
    for key in sorted(visible):
        row=visible[key];original=row
        if row['kind']=='REVERSAL':
            original=visible.get(row.get('reverses_transaction_id'))
            if original is None or original['kind']!='ORIGINAL'or original['account_id']!=row['account_id']or original['posted_at']>row['posted_at']or row['amount']==0 or original['amount']==0 or (row['amount']>0)==(original['amount']>0):_fail('RETAIL_REVERSAL','Reversal needs an earlier visible same-account original of opposite sign')
            reversal_totals.setdefault(original['transaction_id'],[]).append(abs(row['amount']))
        category=categories.get(original['category_code'],'UNKNOWN')
        if category=='OWN_TRANSFER':
            _required(original,['transfer_id','counterparty_own_account_id']);_text(original['transfer_id'],'transfer_id')
            if original['counterparty_own_account_id']not in accounts or original['counterparty_own_account_id']==original['account_id']:_fail('RETAIL_TRANSFER','Own transfer requires a distinct explicitly owned counterparty account')
            if row['kind']=='ORIGINAL':pairs.setdefault(original['transfer_id'],[]).append(row)
        if start<_date(row['posted_at'],'posted_at')<=anchor:classified.append(dict(transaction_id=key,account_id=row['account_id'],posted_at=row['posted_at'],value_at=row['value_at'],amount=row['amount'],category=category,kind=row['kind'],reverses_transaction_id=row.get('reverses_transaction_id'),release_seq=row['release_seq']))
    for key,amounts in reversal_totals.items():
        if _sum(amounts)>abs(visible[key]['amount']):_fail('RETAIL_REVERSAL','Total reversals exceed original amount')
    unmatched=[]
    for transfer_id,legs in sorted(pairs.items()):
        if len(legs)==1:unmatched.append(transfer_id)
        elif len(legs)!=2 or legs[0]['account_id']!=legs[1]['counterparty_own_account_id']or legs[1]['account_id']!=legs[0]['counterparty_own_account_id']or _sum(r['amount']for r in legs)!=0:_fail('RETAIL_TRANSFER','Own-transfer legs must match two opposite same-currency account movements')
    classified.sort(key=lambda r:(r['posted_at'],r['transaction_id']));bycategory={k:[]for k in sorted(allowed)}
    for row in classified:bycategory[row['category']].append(row['amount'])
    totals={k:dict(credits=_sum(v for v in values if v>0),debits=_sum(-v for v in values if v<0),net=_sum(values),transaction_count=len(values))for k,values in bycategory.items()}
    inflows=_sum(r['amount']for r in classified if r['amount']>0);outflows=_sum(-r['amount']for r in classified if r['amount']<0);external=[r for r in classified if r['category']not in ('OWN_TRANSFER','OWNER_DRAW')];bydate={}
    for row in external:bydate.setdefault(row['posted_at'],[]).append(row['amount'])
    daily_net=[_sum(bydate.get((start+timedelta(days=i)).isoformat(),[]))for i in range(1,days+1)];mean=_sum(v/days for v in daily_net);scale=max((abs(v)for v in daily_net),default=0);sd=0.0 if scale==0 else scale*math.sqrt(_sum((v/scale-mean/scale)**2/days for v in daily_net))
    if not math.isfinite(sd):_fail('NUMERICAL_FAILURE','Daily cashflow volatility outside finite range')
    balances,balance_excluded=_releases(daily_balances,'balance_id',cutoff,'daily balances');grid={};balance_lineage=[]
    for row in balances:
        _required(row,['account_id','observed_at','closing_balance','currency']);observed=_date(row['observed_at'],'observed_at')
        if observed>_date(row['known_at'],'known_at'):_fail('RETAIL_DATE','Balance receipt precedes observation')
        if row['account_id']not in accounts or row['currency']!=currency:_fail('RETAIL_ENTITY','Daily balances must match own accounts and currency')
        _number(row['closing_balance'],'closing_balance')
        if not start<observed<=anchor:continue
        key=(row['account_id'],row['observed_at'])
        if key in grid:_fail('DATA_DUPLICATE','One resolved daily balance per own account required')
        grid[key]=row['closing_balance'];balance_lineage.append(dict(balance_id=row['balance_id'],release_seq=row['release_seq']))
    buffer=[];overdraft_account_days=0;complete_balance_days=0
    for offset in range(1,days+1):
        date=(start+timedelta(days=offset)).isoformat();values=[grid[(account,date)]for account in accounts if(account,date)in grid];overdraft_account_days+=sum(v<0 for v in values)
        if len(values)==len(accounts):buffer.append(_sum(values));complete_balance_days+=1
    known_income=_sum(totals[k]['net']for k in ('SALARY','OTHER_INCOME'));known_expense=-_sum(totals[k]['net']for k in ('CONSUMPTION','DEBT_SERVICE','TAX'));unknown_volume=_sum(abs(v)for v in bycategory['UNKNOWN']);gross_volume=_sum(abs(r['amount'])for r in classified)
    return _result(dict(customer_id=customer_id,own_account_ids=sorted(accounts),as_of=as_of,information_cutoff=information_cutoff,currency=currency,lookback_days=days,coverage_complete=complete,category_totals=totals,known_gross_inflows=inflows,known_gross_outflows=outflows,known_net_cashflow=_sum((inflows,-outflows)),known_external_net_cashflow=_sum(r['amount']for r in external),known_classified_income=known_income,classified_income=known_income if complete else None,known_classified_expense=known_expense,classified_expense=known_expense if complete else None,daily_external_net_cashflow=daily_net,known_daily_net_population_sd=sd,daily_net_population_sd=sd if complete else None,unknown_category_absolute_volume=unknown_volume,unknown_category_fraction=None if gross_volume==0 else unknown_volume/gross_volume,complete_balance_days=complete_balance_days,missing_balance_account_days=days*len(accounts)-len(grid),known_overdraft_account_days=overdraft_account_days,overdraft_account_days=overdraft_account_days if complete and len(grid)==days*len(accounts)else None,minimum_observed_complete_household_balance=min(buffer)if buffer else None,mean_observed_complete_household_balance=_sum(v/len(buffer)for v in buffer)if buffer else None,classified_transactions=classified,unmatched_own_transfer_ids=unmatched),'retail_features.build_transaction_features',currency,ctx,diagnostics=dict(mapping_id=category_mapping['mapping_id'],mapping_version=category_mapping['version'],mapping_fingerprint=_fingerprint(category_mapping),coverage=coverage,excluded_transaction_releases=excluded,excluded_balance_releases=balance_excluded,balance_lineage=sorted(balance_lineage,key=lambda r:r['balance_id'])),limitations=('Cashflows use posting dates; value dates are retained without silently changing the window. Reversals retain their own cash date and inherit original classification; both legs neutralize within a common window, while cross-window reversals remain actual cash movements. Refunds, borrowing, business sales and owner/own-account transfers are not household income. Missing categories remain UNKNOWN rather than consumer spending. Complete history is asserted by supplied coverage; cash volatility from incomplete records is only a known-record statistic. Own-account ownership and local categories are asserted, not inferred.',))
