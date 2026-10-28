"""Dated loan-tape, ownership, servicing and tranche-cash reconciliation."""
from copy import deepcopy
from collections import Counter
from .core import _fail,_result,_fingerprint
from .data import _date
from ._structured_pool import _checked,build_credit_asset_pool
from ._structured_paths import simulate_asset_pool
from ._validation_common import _vb_design,_vb_number,_vb_string,_vb_errors
from ._validation_system import _vs_index
from ._validation_affordability import _va_differences
from ._monitoring_metrics import _metrics
from .hybrid import _hy_horizon_end
from ._validation_policy import _vp_finite

_FLOW=('scheduled_principal','prepayment_principal','default_ead','dilution','interest','recovery_gross','workout_cost','servicing_cost')


def asset_pool_backtest(asset_pool,pool_inputs,forecast_paths,forecast_inputs,servicing,allocations,pool_accounts,coverage,*,design,cash_contract,tranche_contracts=(),tranche_payments=(),ownership_history=None,tolerance=1e-10,context=None):
    """Reconcile observed cash and mature competing events; no scenario becomes truth."""
    d=_vb_design(design,context);tol=_vb_number(tolerance,'tolerance',0);pool=_checked(asset_pool,'CreditAssetPool');forecast=_checked(forecast_paths,'AssetPoolPaths');origin=pool['as_of'];cut=d['evaluation_cutoff'];end=_hy_horizon_end(origin,d['target']).isoformat()
    if d.get('currency')!=pool['profile']['currency']or forecast['pool_hash']!=pool['sha256']or forecast['origin']!=origin or forecast['scenario_contract']['probability_measure']!='PHYSICAL'or forecast['scenario_contract']['conditioning']!=d['target']['conditioning']or forecast['scenario_contract']['dates'][-1]!=end:_fail('VALIDATION_ASSET_POOL','Exact physical pool, currency, conditioning and horizon required')
    if _date(d.get('forecast_known_at'),'forecast_known_at')>_date(origin,'origin'):_fail('VALIDATION_LOOKAHEAD','Frozen forecast must be available at pool origin')
    if any('context'in x for x in (pool_inputs,forecast_inputs)):_fail('VALIDATION_ASSET_POOL','Replay arguments inherit validation context')
    if _checked(forecast_inputs.get('asset_pool'),'CreditAssetPool')['sha256']!=pool['sha256']:_fail('VALIDATION_ASSET_POOL','Forecast input pool differs')
    rebuilt=build_credit_asset_pool(**pool_inputs).value;replayed=simulate_asset_pool(**forecast_inputs).value;violations=[]
    def violation(code,record_id=None,details=None):violations.append(dict(code=code,record_id=record_id,details=details))
    for name,actual,expected in [('FROZEN_POOL_RECONCILIATION',pool,rebuilt),('FROZEN_PATH_RECONCILIATION',forecast,replayed)]:
        differences=_va_differences(actual,expected,tol)
        if differences:violation(name,details=differences)
    if cash_contract.get('currency')!=d['currency']or cash_contract.get('as_of')!=origin or _date(cash_contract.get('known_at'),'cash known_at')>_date(origin,'origin'):_fail('VALIDATION_ASSET_POOL','Dated origin cash contract required')
    cash=_vb_number(cash_contract.get('opening_cash'),'opening cash',0)
    assets=pool['asset_contracts'];events,em=_vs_index(servicing,'event_id','servicing');allocation_rows,_=_vs_index(allocations,'allocation_id','allocations');cov,cv=_vs_index(coverage,'asset_id','coverage');accounts,_=_vs_index(pool_accounts,'date','pool accounts');tranches,tm=_vs_index(list(tranche_contracts),'tranche_id','tranches');payments,_=_vs_index(list(tranche_payments),'payment_id','tranche payments')
    history=deepcopy(pool_inputs['pool_assignments']if ownership_history is None else ownership_history);history,_=_vs_index(history,'assignment_id','ownership history')
    for h in history:
        if h.get('asset_id')not in assets:_fail('VALIDATION_ASSET_POOL','Ownership references unknown asset')
        for k in ('pool_id','asset_id'):_vb_string(h.get(k),k)
        share=_vb_number(h.get('share'),'share',0)
        if share>1:_fail('VALIDATION_ASSET_POOL','Ownership share exceeds whole asset')
        for k in ('known_at','effective_from'):_date(h.get(k),k)
        if h.get('effective_until')is not None and _date(h['effective_until'],'until')<=_date(h['effective_from'],'from'):_fail('VALIDATION_ASSET_POOL','Positive ownership interval required')
    def owners(aid,date):
        result={}
        for h in history:
            if h['asset_id']==aid and h['known_at']<=cut and h['effective_from']<=date and(h.get('effective_until')is None or date<h['effective_until']):result[h['pool_id']]=result.get(h['pool_id'],0)+h['share']
        if sum(result.values())>1+tol:violation('OWNERSHIP_OVERALLOCATION',aid,dict(date=date,total_share=sum(result.values())))
        return result
    visible=[];ignored=[]
    for e in events:
        aid=e.get('asset_id')
        if aid not in assets:_fail('VALIDATION_ASSET_POOL','Servicing asset absent from loan tape')
        at=_date(e.get('date'),'event date');known=_date(e.get('known_at'),'event known_at')
        if known<at or at<=_date(origin,'origin'):_fail('VALIDATION_ASSET_POOL','Servicing must follow origin and knowledge cannot precede event')
        if e.get('currency')!=d['currency']:_fail('VALIDATION_ASSET_POOL','Servicing currency differs')
        for field in _FLOW+('opening_principal','closing_principal'):_vb_number(e.get(field),field,0)
        sequence=_vb_number(e.get('sequence'),'sequence',0)
        if sequence!=int(sequence):_fail('VALIDATION_ASSET_POOL','Integer intraday sequence required')
        if e.get('event_type')not in ('PAYMENT','PREPAYMENT','DEFAULT','RECOVERY','COST','DILUTION'):_fail('VALIDATION_ASSET_POOL','Explicit servicing event type required')
        if e['date']>end or e['known_at']>cut:ignored.append(dict(event_id=e['event_id'],reason='AFTER_HORIZON_OR_NOT_YET_KNOWN'))
        else:visible.append(e)
    visible.sort(key=lambda e:(e['date'],e['sequence'],e['event_id']));ordered=set();balance={k:a['principal_balance']for k,a in assets.items()};defaulted={};prepaid={};asset_totals={k:{f:0. for f in _FLOW}for k in assets};owned_cash={};allocation_by={}
    for a in allocation_rows:
        if a.get('event_id')not in em:_fail('VALIDATION_ASSET_POOL','Allocation references unknown servicing event')
        allocation_by.setdefault(a['event_id'],[]).append(a)
    for e in visible:
        aid=e['asset_id'];key=e['event_id'];order=(aid,e['date'],e['sequence'])
        if order in ordered:_fail('VALIDATION_ASSET_POOL','Unique asset/date/sequence required')
        ordered.add(order)
        if abs(e['opening_principal']-balance[aid])>tol*max(1,balance[aid]):violation('SERVICING_BALANCE_CONTINUITY',key)
        residual=e['opening_principal']-sum(e[k]for k in ('scheduled_principal','prepayment_principal','default_ead','dilution'))-e['closing_principal']
        if abs(residual)>tol*max(1,e['opening_principal']):violation('ASSET_PRINCIPAL_CONSERVATION',key,dict(residual=residual))
        if e['event_type']=='DEFAULT':
            if aid in defaulted or aid in prepaid or e['default_ead']<=0 or any(e[k]>tol for k in ('scheduled_principal','prepayment_principal','interest','closing_principal')):violation('INVALID_ABSORBING_DEFAULT',key)
            defaulted[aid]=e
        elif e['default_ead']>0:violation('DEFAULT_AMOUNT_WITHOUT_DEFAULT_EVENT',key)
        if e['event_type']=='PREPAYMENT':
            if aid in defaulted or aid in prepaid or e['prepayment_principal']<=0 or e['closing_principal']>tol:violation('INVALID_PREPAYMENT',key)
            prepaid[aid]=e
        elif e['prepayment_principal']>0:violation('PREPAYMENT_AMOUNT_WITHOUT_EVENT',key)
        if aid in defaulted and defaulted[aid]['event_id']!=key and any(e[k]>tol for k in ('scheduled_principal','prepayment_principal','interest','opening_principal','closing_principal')):violation('POST_DEFAULT_PERFORMING_CASH',key)
        if aid in prepaid and prepaid[aid]['event_id']!=key and any(e[k]>tol for k in ('scheduled_principal','prepayment_principal','interest','opening_principal','closing_principal')):violation('POST_PREPAYMENT_PERFORMING_CASH',key)
        reference_date=e['date']
        if e['recovery_gross']>0 or e['workout_cost']>0:
            default=defaulted.get(aid)
            if default is None or e.get('default_event_id')!=default['event_id']:violation('RECOVERY_WITHOUT_PRIOR_REFERENCED_DEFAULT',key)
            else:reference_date=default['date']
        shares=owners(aid,reference_date);arows=allocation_by.get(key,[]);seen=set();actual_shares={}
        for ar in arows:
            pid=_vb_string(ar.get('pool_id'),'allocation pool');share=_vb_number(ar.get('share'),'allocation share',0)
            if pid in seen:_fail('VALIDATION_ASSET_POOL','One allocation per event/pool required')
            seen.add(pid);actual_shares[pid]=share
            if _date(ar.get('known_at'),'allocation known_at')>_date(cut,'cutoff'):violation('ALLOCATION_NOT_KNOWN',ar['allocation_id'])
            amounts=ar.get('amounts')
            if not isinstance(amounts,dict)or set(amounts)!=set(_FLOW):_fail('VALIDATION_ASSET_POOL','Complete allocated flow map required')
            for f in _FLOW:
                amount=_vb_number(amounts[f],'allocated '+f,0)
                if abs(amount-e[f]*share)>tol*max(1,amount):violation('ALLOCATED_AMOUNT_RECONCILIATION',ar['allocation_id'],dict(field=f))
        if set(actual_shares)!=set(shares)or any(abs(actual_shares.get(p,0)-s)>tol for p,s in shares.items()):violation('LEGAL_OWNERSHIP_RECONCILIATION',key)
        if sum(actual_shares.values())>1+tol:violation('CASH_OR_LOSS_DOUBLE_ALLOCATION',key)
        owned=shares.get(pool['pool_id'],0);daily=owned_cash.setdefault(e['date'],{f:0. for f in _FLOW})
        for f in _FLOW:asset_totals[aid][f]+=e[f];daily[f]+=owned*e[f]
        balance[aid]=e['closing_principal']
    visible_accounts=[];account_map={};cash_residuals=[]
    for r in sorted(accounts,key=lambda x:x['date']):
        at=_date(r['date'],'account date');known=_date(r.get('known_at'),'account known_at')
        if known<at:_fail('VALIDATION_ASSET_POOL','Account receipt precedes date')
        if r['date']>end or r['known_at']>cut:continue
        if r['date']<=origin or r.get('currency')!=d['currency']:_fail('VALIDATION_ASSET_POOL','Pool account date/currency differs')
        fields=('opening_cash','closing_cash','purchase_cash','sale_cash','external_injection','external_distribution','fees_paid','tranche_interest_paid','tranche_principal_paid','equity_paid')
        for f in fields:_vb_number(r.get(f),f,0)
        if abs(r['opening_cash']-cash)>tol*max(1,cash):violation('POOL_CASH_CONTINUITY',r['date'])
        flows=owned_cash.get(r['date'],{f:0. for f in _FLOW});incoming=sum(flows[f]for f in ('scheduled_principal','prepayment_principal','interest','recovery_gross'));outgoing=sum(flows[f]for f in ('workout_cost','servicing_cost'))+sum(r[f]for f in ('purchase_cash','external_distribution','fees_paid','tranche_interest_paid','tranche_principal_paid','equity_paid'));residual=r['opening_cash']+incoming+r['sale_cash']+r['external_injection']-outgoing-r['closing_cash']
        if abs(residual)>tol*max(1,r['opening_cash']+incoming+r['external_injection']+r['sale_cash']):violation('POOL_CASH_CONSERVATION',r['date'],dict(residual=residual))
        cash_residuals.append(dict(date=r['date'],residual=residual,owned_asset_incoming=incoming));cash=r['closing_cash'];account_map[r['date']]=r;visible_accounts.append(r)
    for date in owned_cash:
        if date not in account_map:violation('MISSING_POOL_CASH_ACCOUNT',date)
    paid={};tranche_totals={}
    for t in tranches:
        if t.get('currency')!=d['currency']or _date(t.get('known_at'),'tranche known_at')>_date(origin,'origin'):_fail('VALIDATION_ASSET_POOL','Frozen tranche currency/date required')
        _vb_number(t.get('notional'),'tranche notional',0);tranche_totals[t['tranche_id']]=dict(principal=0.,interest=0.)
    for r in payments:
        if r.get('tranche_id')not in tm:_fail('VALIDATION_ASSET_POOL','Unknown tranche payment')
        at=_date(r.get('date'),'payment date');known=_date(r.get('known_at'),'payment known_at')
        if known<at or at<=_date(origin,'origin')or r.get('currency')!=d['currency']:_fail('VALIDATION_ASSET_POOL','Tranche payment chronology/currency differs')
        principal=_vb_number(r.get('principal'),'principal',0);interest=_vb_number(r.get('interest'),'interest',0)
        if r['date']>end or r['known_at']>cut:continue
        date=r['date'];p=paid.setdefault(date,dict(principal=0.,interest=0.));p['principal']+=principal;p['interest']+=interest;v=tranche_totals[r['tranche_id']];v['principal']+=principal;v['interest']+=interest
        if v['principal']>tm[r['tranche_id']]['notional']+tol:violation('TRANCHE_PRINCIPAL_EXCEEDS_NOTIONAL',r['payment_id'])
    for date in sorted(set(paid)|set(account_map)):
        account=account_map.get(date);p=paid.get(date,dict(principal=0.,interest=0.))
        if account is None or abs(p['principal']-account['tranche_principal_paid'])>tol or abs(p['interest']-account['tranche_interest_paid'])>tol:violation('TRANCHE_CASH_RECONCILIATION',date)
    ids=sorted({r['assignment']['asset_id']for r in pool['active_assignments']+pool['reinvestment_candidates']});eligible={};exclusions=[]
    for aid in ids:
        c=cv.get(aid)
        if c is not None and(c.get('target')!=d['target']or _date(c.get('known_at'),'coverage known_at')<_date(c.get('complete_through'),'complete_through')):_fail('VALIDATION_ASSET_POOL','Coverage target and through/receipt chronology must match')
        ok=c is not None and c.get('complete')is True and _date(c.get('complete_through'),'complete_through')>=_date(end,'horizon')and _date(c.get('known_at'),'coverage known_at')<=_date(cut,'cutoff')and end<=cut
        if c is not None and c.get('currency')!=d['currency']:_fail('VALIDATION_ASSET_POOL','Coverage currency differs')
        eligible[aid]=ok
        if not ok:exclusions.append(dict(asset_id=aid,reason='IMMATURE_OR_INCOMPLETE_SERVICING_COVERAGE'))
    def ownership_key(rows):
        return sorted((h['asset_id'],h['share'],h['effective_from'],h.get('effective_until')or '')for h in rows if h['pool_id']==pool['pool_id'])
    same_risk_set=bool(ids)and pool['profile']['pool_mode']=='STATIC'and ownership_key(history)==ownership_key(pool_inputs['pool_assignments'])and all(h['known_at']<=origin and h['effective_from']<=origin and h.get('effective_until')is None for h in history if h['pool_id']==pool['pool_id'])
    weights=dict(zip(forecast['scenario_ids'],forecast['scenario_weights']));pd=[];prepay=[];forecast_events=forecast['asset_events']
    for aid in ids:
        probability=sum(w for s,w in weights.items()if any(e['scenario_id']==s and e['asset_id']==aid and e['event']=='DEFAULT'for e in forecast_events));pp=sum(w for s,w in weights.items()if any(e['scenario_id']==s and e['asset_id']==aid and e['event']=='PREPAYMENT'for e in forecast_events));row=dict(eligible=eligible[aid]and same_risk_set and not violations,weight=1.,target=int(aid in defaulted),pd=probability,cluster_id=assets[aid]['default_group_id']);pd.append(row);prepay.append(dict(row,target=int(aid in prepaid),pd=pp))
    comparable=all(eligible.values())and not violations and same_risk_set
    actual_cash={f:sum(day[f]for day in owned_cash.values())for f in _FLOW};expected={f:sum(weights[c['scenario_id']]*c[f]for c in forecast['cashflows'])for f in ('principal_collections','interest_available','recovery_gross','workout_cost','servicing_cost')};mapping={'principal_collections':actual_cash['scheduled_principal']+actual_cash['prepayment_principal'],'interest_available':actual_cash['interest'],'recovery_gross':actual_cash['recovery_gross'],'workout_cost':actual_cash['workout_cost'],'servicing_cost':actual_cash['servicing_cost']};cash_errors={k:_vb_errors([expected[k]],[v],[1. if comparable else 0.])for k,v in mapping.items()}
    counts=Counter(v['code']for v in violations);metrics=dict(default_probability=_metrics(pd),prepayment_probability=_metrics(prepay),pool_cash_errors=cash_errors,observed_owned_flows=actual_cash,tranche_payments=tranche_totals);_vp_finite(dict(metrics=metrics,cash_residuals=cash_residuals,violations=violations))
    return _result(dict(artifact_type='ValidationResult',design=d,metrics=metrics,decision=dict(status='BLOCKED'if violations else'NOT_ASSESSED',rule='LEDGER_AND_FROZEN_ARTIFACT_RECONCILIATION'),violations=violations,violation_counts=dict(sorted(counts.items())),p_value=None,coverage=dict(exclusions=exclusions,ignored_events=ignored,eligible_assets=sum(eligible.values()),asset_count=len(ids),forecast_comparison_risk_set='MATCHED'if same_risk_set else'NOT_COMPARABLE'),cash_residuals=cash_residuals,uncertainty=dict(status='NOT_ESTIMATED',reason='One realized asset-pool path; supplied scenario weights are forecast probabilities, not empirical confidence draws'),pool_cash_comparison=dict(applicability='APPLICABLE'if comparable else'NOT_APPLICABLE',reason=None if comparable else'INCOMPLETE_INVALID_OR_CHANGED_OWNERSHIP_RISK_SET'),waterfall_priority_validation='NOT_ASSESSED_CASH_RECONCILIATION_ONLY',input_fingerprint=_fingerprint(dict(servicing=servicing,allocations=allocations,pool_accounts=pool_accounts,coverage=coverage,ownership_history=history,tranche_contracts=list(tranche_contracts),tranche_payments=list(tranche_payments))),forecast_hash=forecast['sha256'],pool_hash=pool['sha256'],modeled_recovery_is_observed_truth=False),'validation.asset_pool_backtest','validation_metrics',context,limitations=('Actual servicing cash and coverage are independent supplied evidence, not scenario outcomes. Legal ownership and allocated cash/loss amounts reconcile without multiplying a whole claim into multiple pools. Recovery ownership follows its referenced default claim; subsequent claim sales require a different explicit claim ledger. Prepayment is a competing exit, not default. Dynamic or changed ownership still receives ledger audits but no same-risk-set pooled forecast cash score. Tranche cash totals and principal caps are checked; contractual waterfall priority is not inferred. No empirical confidence interval is claimed from one realized pool and forecast scenario weights.',))
