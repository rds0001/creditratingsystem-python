"""Verified household allocation and frozen contractual affordability checks."""
from copy import deepcopy
import math
import numpy as np
from .core import _fail, _fingerprint
from .data import _date
from .affordability import stress_affordability
from ._expert_common import _ex_object, _ex_ids
from ._validation_common import _vb_design, _vb_records, _vb_string, _vb_number, _vb_errors, _vb_draws, _vb_bands
from ._validation_retail import _vr_rows
from ._validation_system import _vs_index, _vs_finish


def _va_differences(a,b,tolerance,path=''):
    if type(a) in (int,float) and type(b) in (int,float):
        return [] if math.isclose(a,b,rel_tol=tolerance,abs_tol=tolerance) else [path]
    if isinstance(a,dict) and isinstance(b,dict):
        out=[] if set(a)==set(b) else[path+'/<keys>']
        for key in sorted(set(a)&set(b)):out.extend(_va_differences(a[key],b[key],tolerance,path+'/'+key))
        return out
    if isinstance(a,list) and isinstance(b,list):
        out=[] if len(a)==len(b) else[path+'/<length>']
        for i,(x,y) in enumerate(zip(a,b)):out.extend(_va_differences(x,y,tolerance,path+'/'+str(i)))
        return out
    return [] if type(a)is type(b) and a==b else[path]


def affordability_stress_backtest(cases,economic_evidence,allocations,*,design,payment_outcomes=None,
                                   tolerance=1e-10,uncertainty='none',cluster_resamples=None,confidence_level=.95,context=None):
    """Reconcile existing frozen monthly stress projections to verified sources."""
    d=_vb_design(design,context);currency=_vb_string(d.get('currency'),'design currency');tol=_vb_number(tolerance,'tolerance',0)
    rows=_vr_rows(cases,'cases')
    for row in rows:
        if hasattr(row.get('assessment'),'value'):row['assessment']=row['assessment'].value
    rows=_vb_records(rows,d);proofs,ei=_vs_index(economic_evidence,'evidence_id','economic_evidence');shares,_=_vs_index(allocations,'allocation_id','allocations')
    payments=_vr_rows([] if payment_outcomes is None else payment_outcomes,'payment_outcomes');violations=[];traces=[];payment_traces=[];groups=('incomes','expenses','contracts','collateral');cutoff=_date(d['evaluation_cutoff'],'evaluation_cutoff')
    def issue(code,unit,detail):violations.append(dict(code=code,unit_id=unit,detail=detail))
    for e in proofs:
        for k in ('economic_id','reference'):_vb_string(e.get(k),k)
        if e.get('kind')not in groups or type(e.get('verified'))is not bool:_fail('AFFORDABILITY_VALIDATION_EVIDENCE','Explicit economic kind and verified Boolean required')
        if _date(e.get('known_at'),'evidence known_at')<_date(e.get('observed_at'),'evidence observed_at'):_fail('AFFORDABILITY_VALIDATION_EVIDENCE','Evidence knowledge precedes observation')
        terms=_ex_object(e.get('terms'),'economic terms');_vb_string(terms.get('currency'),'economic currency')
        if e['kind']=='incomes':
            if terms.get('basis')!='net' or e.get('source_basis')not in ('net','gross'):_fail('AFFORDABILITY_VALIDATION_INCOME','Normalized monthly net terms and original basis required')
            amount=_vb_number(terms.get('monthly_amount'),'monthly net amount')
            if e['source_basis']=='gross':
                gross=_vb_number(e.get('monthly_gross_amount'),'monthly gross',0);deduct=_vb_number(e.get('monthly_tax_and_deductions'),'monthly deductions',0);_vb_string(e.get('normalization_reference'),'normalization_reference')
                if not math.isclose(amount,gross-deduct,rel_tol=tol,abs_tol=tol):issue('GROSS_NET_RECONCILIATION',e['evidence_id'],'Verified gross less deductions differs from normalized net income')
    ai={};sums={};used_households={}
    for a in shares:
        for k in ('economic_id','household_id'):_vb_string(a.get(k),k)
        date=_date(a.get('as_of'),'allocation as_of');known=_date(a.get('known_at'),'allocation known_at');fraction=_vb_number(a.get('share'),'allocation share',0)
        if fraction>1 or a.get('mode')not in ('PROPORTIONAL','FULL_RECOURSE'):_fail('AFFORDABILITY_VALIDATION_ALLOCATION','Explicit proportional/full-recourse share in [0,1] required')
        key=(a['economic_id'],a['household_id'],a['as_of'])
        if key in ai:_fail('AFFORDABILITY_VALIDATION_ALLOCATION','Duplicate economic/household/as-of allocation')
        ai[key]=a
        if known>date:issue('ALLOCATION_LOOKAHEAD',a['allocation_id'],'Allocation unavailable at its declared as-of')
        if a['mode']=='PROPORTIONAL':
            pool=(a['economic_id'],a['as_of']);sums[pool]=sums.get(pool,0)+fraction
        elif fraction!=1:issue('FULL_RECOURSE_SHARE',a['allocation_id'],'Full recourse requires the entire contractual obligation')
    for (economic,date),total in sorted(sums.items()):
        if total>1+tol:issue('SHARED_ECONOMIC_DOUBLE_COUNT',economic+'@'+date,'Proportional household shares exceed one')
    global_evidence_invalid=bool(violations)
    case_index={};period_index={};verified_count=0
    for row in rows:
        identity=row['prediction_id'];household=_vb_string(row.get('household_id'),'household_id');baseline=deepcopy(_ex_object(row.get('baseline'),'baseline'));scenario=deepcopy(_ex_object(row.get('scenario'),'scenario'));assessment=_ex_object(row.get('assessment'),'assessment');bindings=_ex_object(row.get('evidence_bindings'),'evidence_bindings');expected=_ex_object(row.get('expected_economic_ids'),'expected_economic_ids')
        if set(expected)!=set(groups):_fail('AFFORDABILITY_VALIDATION_EXPECTED','Explicit expected IDs for all four economic groups required')
        if baseline.get('as_of')!=row['as_of']or baseline.get('currency')!=currency:_fail('AFFORDABILITY_VALIDATION_SCOPE','Baseline as-of/currency differs from forecast')
        for field in ('scenario_known_at','profile_known_at'):
            if _date(row.get(field),field)>_date(row['forecast_known_at'],'forecast_known_at'):_fail('AFFORDABILITY_VALIDATION_LOOKAHEAD','Scenario and policy thresholds must be frozen before assessment')
        _vb_string(row.get('profile_version'),'profile_version');before=len(violations);allids=[]
        for kind in groups:
            required=_ex_ids(expected[kind],kind+' expected IDs',empty=True);items=_vr_rows(baseline.get(kind),'baseline '+kind);ids=_ex_ids([v.get('economic_id')for v in items],kind+' actual IDs',empty=True);allids.extend(ids)
            if set(required)!=set(ids):issue('MISSING_OR_EXTRA_ECONOMIC_COMPONENT',identity,kind+' differs from declared required evidence universe')
            for item in items:
                economic=item['economic_id'];e=ei.get(bindings.get(economic));a=ai.get((economic,household,row['as_of']))
                if e is None or e['economic_id']!=economic or e['kind']!=kind or not e['verified']:
                    issue('INCOMPLETE_ECONOMIC_EVIDENCE',identity,economic+' lacks verified matching source');continue
                if e['known_at']>row['forecast_known_at']or e['observed_at']>row['as_of']:
                    issue('ECONOMIC_EVIDENCE_LOOKAHEAD',identity,economic+' unavailable at original assessment');continue
                if a is None or a['known_at']>row['forecast_known_at']:
                    issue('MISSING_HOUSEHOLD_ALLOCATION',identity,economic+' has no historically known household share');continue
                if a['mode']=='FULL_RECOURSE'and kind!='contracts':issue('INVALID_FULL_RECOURSE_COMPONENT',identity,economic+' is not a contractual obligation')
                terms=deepcopy(e['terms']);factor=a['share'];amountfields=('principal','balloon_amount')if kind=='contracts'else('value',)if kind=='collateral'else('monthly_amount',)
                for field in amountfields:terms[field]=_vb_number(terms.get(field),'source '+field)*factor
                terms['economic_id']=economic
                if _va_differences(terms,item,tol):issue('ECONOMIC_TERM_RECONCILIATION',identity,economic+' differs from verified allocated source terms')
                used_households.setdefault(economic,set()).add(household)
        if len(set(allids))!=len(allids):issue('DUPLICATE_ECONOMIC_PURPOSE',identity,'Same source appears in multiple budget groups')
        if set(bindings)!=set(allids):issue('EVIDENCE_BINDING_COVERAGE',identity,'Bindings must cover exactly the frozen economic components')
        recomputed=stress_affordability(baseline,scenario).value
        if recomputed['periods'][-1]['date']!=row['horizon_end']:_fail('AFFORDABILITY_VALIDATION_HORIZON','Projection endpoint differs from declared validation horizon')
        differences=_va_differences(recomputed,assessment,tol)
        if differences:issue('FROZEN_ASSESSMENT_RECONCILIATION',identity,'Projection differs at '+','.join(differences))
        verified=not global_evidence_invalid and len(violations)==before;verified_count+=verified;case_index[identity]=row
        for period in recomputed['periods']:period_index[(identity,period['date'])]=period
        traces.append(dict(prediction_id=identity,household_id=household,verified_projection=verified,recomputed_assessment=recomputed,differing_paths=differences,positive_residual_implies_zero_pd=False))
    seen=set();actual_scheduled=[];actual_paid=[];payment_weights=[];payment_case_positions=[];period_exclusions=[]
    positions={r['prediction_id']:i for i,r in enumerate(rows)}
    for p in payments:
        identity=p.get('prediction_id');date=p.get('period_date');key=(identity,date)
        if key in seen or key not in period_index:_fail('AFFORDABILITY_VALIDATION_PAYMENT','Unique payment row must match a forecast period')
        seen.add(key);period=_date(date,'payment period');known=_date(p.get('known_at'),'payment known_at');through=_date(p.get('observed_through'),'payment observed_through')
        if known<through or type(p.get('complete'))is not bool or p.get('currency')!=currency:_fail('AFFORDABILITY_VALIDATION_PAYMENT','Explicit complete capture, currency and subsequent knowledge required')
        scheduled=_vb_number(p.get('scheduled_payment'),'actual scheduled payment',0);paid=_vb_number(p.get('paid_amount'),'actual paid amount',0);reason=None
        if period>cutoff or known>cutoff:reason='IMMATURE_OR_UNKNOWN_PAYMENT'
        elif not p['complete']or through<period:reason='INCOMPLETE_PAYMENT_CAPTURE'
        if reason:period_exclusions.append(dict(prediction_id=identity,period_date=date,reason=reason))
        else:
            actual_scheduled.append(scheduled);actual_paid.append(paid);payment_weights.append(_vb_number(case_index[identity].get('weight',1),'case weight',0));payment_case_positions.append(positions[identity])
        payment_traces.append(dict(prediction_id=identity,period_date=date,eligible=reason is None,actual_scheduled_payment=scheduled,actual_paid_amount=paid,projected_scenario_debt_service=period_index[key]['debt_service'],scenario_equals_realized_claimed=False))
    pp=np.asarray(actual_scheduled);yy=np.asarray(actual_paid);ww=np.asarray(payment_weights);ix=np.asarray(payment_case_positions,dtype=int)
    def metric(draw):return _vb_errors(pp,yy,ww*draw[ix])
    factual=metric(np.ones(len(rows)));draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);samples=None if draws is None else[{k:metric(draw)[k]for k in ('bias','rmse')}for draw in draws]
    payload=dict(design=d,applicability='APPLICABLE'if rows else'NOT_APPLICABLE',metrics=dict(cases=len(rows),verified_projection_cases=verified_count,factual_scheduled_minus_paid=factual),uncertainty=_vb_bands(samples,confidence_level),projection_traces=traces,payment_traces=payment_traces,payment_exclusions=period_exclusions,missing_payment_periods=[dict(prediction_id=key[0],period_date=key[1])for key in sorted(set(period_index)-seen)],economic_dependence={k:sorted(v)for k,v in sorted(used_households.items())},scenario_projection_is_default_probability=False,factual_payments_are_scenario_realization=False)
    result=_vs_finish('affordability_stress_backtest',payload,violations,dict(cases=rows,economic_evidence=proofs,allocations=shares,payment_outcomes=payments,design=d),context)
    if not rows:result.value['decision']['status']='NOT_APPLICABLE'
    return result
