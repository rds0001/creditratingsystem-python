"""Fixed-origination vintage validation of supplied operational performance tables."""
from copy import deepcopy
import numpy as np
from .core import _fail,_fingerprint,_result
from .data import _date
from ._validation_common import _vb_design,_vb_string,_vb_number,_vb_draws,_vb_bands,_vb_errors
from ._validation_retail import _vr_rows
from ._expert_common import _ex_object,_ex_ids


def vintage_backtest(vintage_table,account_predictions,account_outcomes,*,origin_cohorts,denominator_contract,design,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Reconcile per-account frozen paths to fixed-origin operational vintage cells."""
    from .behavioral import _month
    d=_vb_design(design,context);table=_vr_rows(vintage_table,'vintage_table');predictions=_vr_rows(account_predictions,'account_predictions');outcomes=_vr_rows(account_outcomes,'account_outcomes');origins=_vr_rows(origin_cohorts,'origin_cohorts');contract=deepcopy(_ex_object(denominator_contract,'denominator_contract'))
    if d['target']['time_unit']!='months' or int(d['target']['horizon'])!=d['target']['horizon'] or d['target']['conditioning']!='performing':_fail('VINTAGE_TARGET','Whole-month maximum MOB and performing-at-origin target required')
    if contract.get('method')!='fixed_origination'or contract.get('weighting')not in ('count','original_balance')or contract.get('exit_treatment')!='first_competing_exit':_fail('VINTAGE_DENOMINATOR','Declare fixed origination count/original-balance weights and first competing exit')
    for key in ('reference','default_definition','horizon_convention'):_vb_string(contract.get(key),key)
    if contract['horizon_convention']!='origination_month_end_plus_mob':_fail('VINTAGE_HORIZON','Operational vintage horizons are cohort month-end plus MOB')
    _ex_ids([r.get('account_id')for r in origins],'origin account IDs',empty=True);members={};origin_index={};weights=[];sampling_rows=[]
    for i,row in enumerate(origins):
        account=row['account_id'];opened=_date(row.get('opened_at'),'opened_at');known=_date(row.get('known_at'),'known_at');_vb_string(row.get('currency'),'currency');_vb_string(row.get('cluster_id'),'cluster_id');balance=_vb_number(row.get('original_balance'),'original_balance',0)
        if known<opened or known>_date(d['evaluation_cutoff'],'evaluation_cutoff'):_fail('VINTAGE_ORIGIN','Origin must be known by evaluation and not before opening')
        key=(row['opened_at'][:7],row['currency']);members.setdefault(key,[]).append(i);origin_index[account]=i;weights.append(1. if contract['weighting']=='count'else balance);sampling_rows.append(dict(cluster_id=row['cluster_id'],entity_id=account))
    pred_index={};outcome_index={}
    for rows,index,label in [(predictions,pred_index,'prediction'),(outcomes,outcome_index,'outcome')]:
        for row in rows:
            account=row.get('account_id');mob=_vb_number(row.get('months_on_book'),'months_on_book',0)
            if account not in origin_index or int(mob)!=mob or mob>d['target']['horizon']:_fail('VINTAGE_ACCOUNT','Known account and integer MOB required')
            key=(account,int(mob))
            if key in index:_fail('VINTAGE_DUPLICATE','Duplicate account/MOB '+label)
            index[key]=row
    weights=np.array(weights);cells=[];seen=set();used_predictions=set();used_outcomes=set();cutoff=_date(d['evaluation_cutoff'],'evaluation_cutoff');draws=_vb_draws(uncertainty,cluster_resamples,sampling_rows,confidence_level)
    for cell in sorted(table,key=lambda c:(c['cohort'],c['currency'],c['months_on_book'])):
        cohort=_vb_string(cell.get('cohort'),'cohort');currency=_vb_string(cell.get('currency'),'currency');mob=_vb_number(cell.get('months_on_book'),'months_on_book',0);key=(cohort,currency,int(mob))
        if mob!=int(mob)or key in seen:_fail('VINTAGE_CELL','Unique integer-MOB vintage cells required')
        seen.add(key);indices=members.get((cohort,currency),[]);end=_month(_date(cohort+'-01','cohort month'),int(mob),True)
        if not indices or cell.get('period_end')!=end.isoformat():_fail('VINTAGE_CELL','Cell must refer to known original cohort and exact MOB endpoint')
        N=len(indices);balance=sum(origins[i]['original_balance']for i in indices)
        if cell.get('accounts')!=N or abs(cell.get('original_balance',-1)-balance)>1e-10*max(1,balance):_fail('VINTAGE_DENOMINATOR','Operational denominator differs from original membership; exits cannot shrink it')
        cumulative=[];interval=[];observed_cumulative=[];observed_interval=[];complete=[];exits=[];gaps=[]
        for i in indices:
            origin=origins[i];account=origin['account_id'];pair=(account,int(mob));prediction=pred_index.get(pair);outcome=outcome_index.get(pair)
            if prediction is None:_fail('VINTAGE_PREDICTION','Every original account needs a frozen prediction, including exited accounts')
            used_predictions.add(pair)
            for field in ('model_id','model_version','parameter_version'):
                if prediction.get(field)!=d[field]:_fail('VINTAGE_VERSION','Prediction differs from frozen design')
            at=_date(prediction.get('forecast_at'),'forecast_at');known=_date(prediction.get('known_at'),'prediction known_at')
            if at!=_date(origin['opened_at'],'opened_at')or known>at:_fail('VINTAGE_LOOKAHEAD','Vintage forecasts must be frozen at original account opening')
            pc=_vb_number(prediction.get('cumulative_pd'),'cumulative_pd',0);pi=_vb_number(prediction.get('interval_pd'),'interval_pd',0)
            if pc>1 or pi>pc:_fail('VINTAGE_PD','Cumulative/interval probabilities must be ordered within [0,1]')
            cumulative.append(pc);interval.append(pi);iscomplete=False;yc=yi=None;exited=False
            if outcome is not None:
                used_outcomes.add(pair);when=_date(outcome.get('known_at'),'outcome known_at')
                if outcome.get('period_end')!=end.isoformat()or outcome.get('default_definition')!=contract['default_definition']:_fail('VINTAGE_TARGET','Observed account target/end differs from vintage contract')
                if type(outcome.get('complete'))is not bool or type(outcome.get('exited'))is not bool:_fail('VINTAGE_OUTCOME','Explicit complete/exit flags required')
                if when<_date(origin['opened_at'],'opened_at'):_fail('VINTAGE_LOOKAHEAD','Outcome cannot be known before account origin')
                if when<=cutoff:
                    iscomplete=outcome['complete'];yc=outcome.get('cumulative_default');yi=outcome.get('interval_default');exited=outcome['exited']
                    if any(v is not None and(type(v)is bool or v not in (0,1))for v in (yc,yi)):_fail('VINTAGE_OUTCOME','Observed default indicators must be binary/null')
                    if iscomplete and(yc is None or yi is None):_fail('VINTAGE_OUTCOME','Complete account paths require both observed indicators')
                    if (yc is not None and yi is not None and yi>yc)or(exited and yc==1):_fail('VINTAGE_OUTCOME','Interval/cumulative defaults and first competing exits are inconsistent')
                    if yc==1 or exited:
                        _vb_string(outcome.get('event_id'),'first event_id');event_at=_date(outcome.get('first_event_at'),'first_event_at')
                        if not iscomplete or event_at<_date(origin['opened_at'],'opened_at')or event_at>end or event_at>when:_fail('VINTAGE_EVENT','First event must be observed inside the origin-to-end window')
                        previous_end=_month(_date(cohort+'-01','cohort month'),int(mob)-1,True)
                        if yc==1 and yi!=int(event_at>previous_end):_fail('VINTAGE_EVENT','Interval default must lie in the named calendar MOB interval')
                    elif iscomplete:
                        through=_date(outcome.get('complete_through'),'complete_through')
                        if through<end or through>when:_fail('VINTAGE_FOLLOWUP','No-event complete outcomes require coverage through exact vintage endpoint')
            if not iscomplete:gaps.append(account)
            observed_cumulative.append(yc);observed_interval.append(yi);complete.append(iscomplete);exits.append(exited)
        counts=dict(defaults=sum(y==1 for y in observed_cumulative),incidence=sum(y==1 for y in observed_interval),exits=sum(exits),complete_accounts=sum(complete))
        if any(cell.get(k)!=v for k,v in counts.items()):_fail('VINTAGE_RECONCILIATION','Operational vintage counts differ from separately supplied account outcomes')
        mature=end<=cutoff and all(complete)
        if cell.get('mature')!=mature:_fail('VINTAGE_MATURITY','Cell maturity differs from calendar and original-cohort follow-up')
        for field,count in [('cumulative_default_rate',counts['defaults']),('incidence_rate',counts['incidence'])]:
            expected=count/N if mature else None
            if (cell.get(field)is None)!=(expected is None)or(expected is not None and abs(cell[field]-expected)>1e-12):_fail('VINTAGE_RECONCILIATION','Operational rate violates original denominator or maturity')
        cells.append(dict(key=key,indices=indices,period_end=end.isoformat(),original_accounts=N,original_balance=balance,mature=mature,counts=counts,missing_followup_accounts=gaps,remaining_first_event_risk_count=N-counts['defaults']-counts['exits'],cumulative=np.array(cumulative),interval=np.array(interval),observed_cumulative=np.array([0 if y is None else y for y in observed_cumulative]),observed_interval=np.array([0 if y is None else y for y in observed_interval])))
    if set(pred_index)!=used_predictions or set(outcome_index)!=used_outcomes:_fail('VINTAGE_COVERAGE','Prediction/outcome account-MOB cells must match requested operational table')
    for account in origin_index:
        ages=sorted(m for a,m in pred_index if a==account)
        for before,after in zip(ages[:-1],ages[1:]):
            a=pred_index[(account,before)];b=pred_index[(account,after)]
            if b['cumulative_pd']<a['cumulative_pd']-1e-12:_fail('VINTAGE_CURVE','Cumulative forecast decreases')
            if after==before+1 and abs(b['interval_pd']-(b['cumulative_pd']-a['cumulative_pd']))>1e-12:_fail('VINTAGE_CURVE','Consecutive interval forecast differs from cumulative increment')
    def calculate(draw):
        output=[]
        for c in cells:
            weight=weights[c['indices']]*draw[c['indices']];metrics={}
            for label in ('cumulative','interval'):
                p=c[label];y=c['observed_'+label];w=weight if c['mature']else np.zeros(len(weight));m=_vb_errors(p,y,w);m['brier']=float(np.dot(w/w.sum(),(p-y)**2))if w.sum()>0 else None;metrics[label]=m
            output.append(dict(cohort=c['key'][0],currency=c['key'][1],months_on_book=c['key'][2],period_end=c['period_end'],original_accounts=c['original_accounts'],original_balance=c['original_balance'],original_weight_sum=float(weight.sum()),mature=c['mature'],counts=c['counts'],remaining_first_event_risk_count=c['remaining_first_event_risk_count'],missing_followup_accounts=c['missing_followup_accounts'],**metrics))
        return output
    values=calculate(np.ones(len(origins)));samples=None
    if draws is not None:
        samples=[]
        for draw in draws:
            result=calculate(draw);samples.append({r['cohort']+'/'+r['currency']+'/'+str(r['months_on_book'])+'/'+label+'/'+metric:r[label][metric]for r in result for label in ('cumulative','interval')for metric in ('bias','brier')})
    value=dict(artifact_type='ValidationResult',metrics=dict(cells=values),uncertainty=_vb_bands(samples,confidence_level),design=d,denominator_contract=contract,applicability='APPLICABLE'if any(c['mature']for c in cells)else'NOT_APPLICABLE',reason=None if any(c['mature']for c in cells)else'NO_COMPLETE_MATURE_VINTAGE',decision=dict(status='NOT_ASSESSED'),original_cohort_fingerprint=_fingerprint(origins),operational_table_fingerprint=_fingerprint(table),prediction_fingerprint=_fingerprint(predictions),outcome_fingerprint=_fingerprint(outcomes),survivorship_denominator_preserved=True,censoring_adjustment='NONE_INCOMPLETE_CELLS_NOT_ESTIMABLE')
    return _result(value,'validation.vintage_backtest','vintage_validation',context,limitations=('Validates supplied operational cells and separately supplied account outcomes, not an alternative target builder. Fixed origin denominators retain exits; incomplete original cohorts are not complete-case-reweighted into performance claims. Cluster intervals condition on fixed predictions. MOB endpoints are cohort calendar month ends, not equal days since each opening.',))
