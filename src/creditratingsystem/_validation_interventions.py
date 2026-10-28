"""Dated override audit and paired descriptive prediction assessment."""
from collections import Counter
from copy import deepcopy
import numpy as np
from .core import _fail,_fingerprint
from .data import _date
from ._expert_common import _ex_object,_ex_ids
from ._monitoring_metrics import _cohort
from ._validation_common import _vb_design,_vb_string,_vb_number,_vb_result
from ._validation_benchmarks import benchmark_predictions


def override_monitoring(pre_override,post_override,override_ledger,*,policy,design,metrics,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Audit known, valid override chains and compare the same mature cases."""
    d=_vb_design(design,context);p=deepcopy(_ex_object(policy,'policy'));authorities=_ex_ids(p.get('approved_authorities'),'approved_authorities');_vb_string(p.get('reference'),'policy reference')
    threshold=_vb_number(p.get('concentration_threshold'),'concentration_threshold')
    if not 0<threshold<=1:_fail('OVERRIDE_POLICY','Concentration threshold must be in (0,1]')
    pre=deepcopy(_ex_object(pre_override,'pre_override'));post=deepcopy(_ex_object(post_override,'post_override'));a=_cohort(pre.get('cohort'));b=_cohort(post.get('cohort'))
    models=[sorted({r['model_id']for r in x['rows']})for x in (a,b)]
    if any(len(x)!=1 for x in models)or models[0]==models[1]:_fail('OVERRIDE_MODELS','Distinct pre/post model identifiers required')
    pre_id,post_id=models[0][0],models[1][0]
    compared=benchmark_predictions({pre_id:pre,post_id:post},reference_model_id=pre_id,metrics=metrics,matching_policy='strict',multiple_comparisons='none',design=d,uncertainty=uncertainty,cluster_resamples=cluster_resamples,confidence_level=confidence_level,context=context).value
    left={r['case_id']:r for r in a['rows']};right={r['case_id']:r for r in b['rows']}
    if not isinstance(override_ledger,(list,tuple))or any(not isinstance(r,dict)for r in override_ledger):_fail('OVERRIDE_LEDGER','Override ledger must be an array of objects')
    ledger=deepcopy(list(override_ledger));_ex_ids([r.get('override_id')for r in ledger],'override IDs',empty=True);index={};audit=[];reasons=Counter();author_counts=Counter();expired=[];future=[]
    for entry in ledger:
        for key in ('case_id','reason','authority'):_vb_string(entry.get(key),key)
        if entry['case_id']not in left:_fail('OVERRIDE_CASE','Ledger references an unknown paired case')
        applied=_date(entry.get('applied_at'),'applied_at');known=_date(entry.get('known_at'),'known_at');expiry=_date(entry.get('expires_at'),'expires_at')
        if known<applied or expiry<=applied:_fail('OVERRIDE_DATE','Knowledge must not precede action; expiry must follow action')
        for key in ('from_pd','to_pd'):
            v=_vb_number(entry.get(key),key)
            if not 0<=v<=1:_fail('OVERRIDE_PD','Override probabilities must be in [0,1]')
        date=_date(left[entry['case_id']]['forecast_at'],'forecast_at');status='ACTIVE'
        if known>date or applied>date:status='NOT_YET_KNOWN';future.append(entry['override_id'])
        elif expiry<=date:status='EXPIRED';expired.append(entry['override_id'])
        elif entry['authority']not in authorities:_fail('OVERRIDE_AUTHORITY','Active override lacks approved authority')
        if status=='ACTIVE':index.setdefault(entry['case_id'],[]).append(entry);reasons[entry['reason']]+=1;author_counts[entry['authority']]+=1
        audit.append(dict(override_id=entry['override_id'],case_id=entry['case_id'],status=status))
    traces=[];changed=0;up=0;down=0;multiple=[];by_group={}
    for case_id,row in sorted(left.items()):
        entries=sorted(index.get(case_id,[]),key=lambda e:(e['applied_at'],e['known_at'],e['override_id']));current=row['pd']
        if len({(e['applied_at'],e['known_at'])for e in entries})!=len(entries):_fail('OVERRIDE_AMBIGUITY','Equally dated active actions need upstream resolution')
        for entry in entries:
            if abs(entry['from_pd']-current)>1e-12:_fail('OVERRIDE_CHAIN','Active override chain does not start at pre-override PD or has a gap')
            current=entry['to_pd']
        if abs(current-right[case_id]['pd'])>1e-12:_fail('OVERRIDE_REPLAY','Post-override PD does not reproduce the active ledger chain')
        delta=right[case_id]['pd']-row['pd'];applied=bool(entries);changed+=applied;up+=int(delta>0);down+=int(delta<0)
        if len(entries)>1:multiple.append(case_id)
        traces.append(dict(case_id=case_id,pre_pd=row['pd'],post_pd=right[case_id]['pd'],pd_change=delta,active_override_ids=[e['override_id']for e in entries],active_count=len(entries),direction='PD_INCREASE'if delta>0 else'PD_DECREASE'if delta<0 else'UNCHANGED'))
        for group_type in ('segment','cohort'):
            key=group_type+':'+row[group_type];group=by_group.setdefault(key,dict(cases=0,overridden_cases=0));group['cases']+=1;group['overridden_cases']+=int(applied)
    for group in by_group.values():group['override_rate']=group['overridden_cases']/group['cases']
    N=len(left);total_actions=sum(reasons.values());concentration={}
    for label,counts in [('reason',reasons),('authority',author_counts)]:
        concentration[label]=dict(counts=dict(sorted(counts.items())),maximum_share=max(counts.values())/total_actions if total_actions else None,flag=bool(total_actions and max(counts.values())/total_actions>threshold),threshold=threshold,denominator='ACTIVE_ACTIONS')
    values=dict(cases=N,overridden_cases=changed,override_rate=changed/N if N else None,pd_increase_cases=up,pd_decrease_cases=down,unchanged_cases=N-up-down,by_group=by_group,concentration=concentration,paired_performance=compared['metrics'])
    rows=[dict(prediction_id=r['forecast_id'],entity_id=r['entity_id'],cluster_id=r['cluster_id'])for r in a['rows']]
    diagnostics=dict(policy=p,ledger_fingerprint=_fingerprint(ledger),ledger_audit=audit,case_traces=traces,expired_override_ids=sorted(expired),not_yet_known_override_ids=sorted(future),multiple_active_case_ids=multiple,performance_coverage=compared['coverage'],performance_diagnostics=compared['diagnostics'],causal_effect_identified=False,rate_population='ALL_PAIRED_FORECAST_CASES',performance_population='COMMON_MATURE_ELIGIBLE_CASES',active_interval='applied_at <= forecast_at < expires_at; known_at <= forecast_at')
    return _vb_result('override_monitoring',values,d,rows,[],compared['uncertainty'],context,diagnostics)


def support_transfer_backtest(records,*,stage_contract,union_contract,design,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Validate separately observable stage events and cash amounts, never infer labels."""
    from ._validation_common import _vb_records,_vb_draws,_vb_bands,_vb_errors
    d=_vb_design(design,context);rows=_vb_records(records,d);stages=deepcopy(_ex_object(stage_contract,'stage_contract'));union=deepcopy(_ex_object(union_contract,'union_contract'));names=sorted(stages)
    if not names:_fail('STAGE_CONTRACT','At least one identified stage required')
    for name,spec in stages.items():
        _vb_string(name,'stage name');_ex_object(spec,'stage specification');_vb_string(spec.get('event_definition'),'event_definition');_vb_string(spec.get('reference'),'reference')
        if spec.get('kind')not in ('binary','amount')or type(spec.get('observable'))is not bool:_fail('STAGE_CONTRACT','Declare binary/amount kind and genuinely observable stage')
        if spec['kind']=='amount':
            _vb_string(spec.get('currency'),'currency')
            if spec['currency']!=d.get('currency'):_fail('STAGE_CURRENCY','Amount stage currency must equal validation currency')
    for key in ('component_a','component_b','joint','union'):
        if union.get(key)not in stages or stages[union[key]]['kind']!='binary':_fail('STAGE_UNION','Four binary stage names required for A, B, intersection and union')
    if len(set(union.values()))!=4:_fail('STAGE_UNION','Distinct stage names required')
    n=len(rows);pred={s:np.zeros(n)for s in names};observed={s:np.zeros(n)for s in names};weight=np.ones(n);eligible={s:np.zeros(n,dtype=bool)for s in names};excluded={s:[]for s in names};reconciliation=[]
    for i,row in enumerate(rows):
        predictions=_ex_object(row.get('stage_predictions'),'stage_predictions');outcomes=_ex_object(row.get('stage_outcomes'),'stage_outcomes')
        if set(predictions)!=set(names)or not set(outcomes)<=set(names):_fail('STAGE_SCHEMA','Complete prediction stage keys and known outcome keys required')
        weight[i]=_vb_number(row.get('weight',1),'weight',0);event_dates={}
        for name in names:
            spec=stages[name];p=_vb_number(predictions[name],'stage prediction',0)
            if spec['kind']=='binary'and p>1:_fail('STAGE_PD','Binary probability exceeds one')
            pred[name][i]=p;outcome=outcomes.get(name);reason=None
            if not spec['observable']:
                if outcome is not None:_fail('STAGE_COUNTERFACTUAL','Unobservable stages cannot carry purported observed outcomes')
                reason='STAGE_NOT_OBSERVABLE'
            elif _date(row['horizon_end'],'horizon_end')>_date(d['evaluation_cutoff'],'evaluation_cutoff'):reason='IMMATURE_HORIZON'
            elif outcome is None:reason='STAGE_OUTCOME_MISSING'
            else:
                outcome=_ex_object(outcome,'stage outcome');known=_date(outcome.get('known_at'),'stage known_at');end=_date(row['horizon_end'],'horizon_end');complete=_date(outcome.get('complete_through'),'complete_through');y=_vb_number(outcome.get('value'),'stage outcome',0)
                if spec['kind']=='binary'and y not in (0,1):_fail('STAGE_OUTCOME','Binary observed stage must be zero or one')
                if known<_date(row['as_of'],'as_of')or complete>known:_fail('STAGE_LOOKAHEAD','Stage outcome coverage cannot exceed knowledge date')
                ids=_ex_ids(outcome.get('event_ids'),'event_ids',empty=True)
                if spec['kind']=='binary'and ((y==1 and not ids)or(y==0 and ids)):_fail('STAGE_EVENT','Positive stages need source event IDs; negative stages must have none')
                if spec['kind']=='binary':
                    times=outcome.get('event_times')
                    if not isinstance(times,(list,tuple))or len(times)!=len(ids):_fail('STAGE_EVENT','Each source event requires an aligned event date')
                    for event_id,time in zip(ids,times):
                        if event_id in event_dates and event_dates[event_id]!=time:_fail('STAGE_EVENT','Shared source event has inconsistent dates across stages')
                        event_dates[event_id]=time
                        t=_date(time,'stage event date')
                        if t<_date(row['as_of'],'as_of')or t>end or t>known:_fail('STAGE_EVENT','Observed source event lies outside forecast horizon or knowledge date')
                if known>_date(d['evaluation_cutoff'],'evaluation_cutoff'):reason='STAGE_OUTCOME_NOT_YET_KNOWN'
                elif complete<end:reason='INCOMPLETE_STAGE_FOLLOWUP'
                elif weight[i]==0:reason='ZERO_EVALUATION_WEIGHT'
                else:observed[name][i]=y;eligible[name][i]=True
            if reason:excluded[name].append(dict(prediction_id=row['prediction_id'],reason=reason))
        a,b,j,u=[union[k]for k in ('component_a','component_b','joint','union')];pa,pb,pj,pu=[pred[s][i]for s in (a,b,j,u)]
        if pj<max(0,pa+pb-1)-1e-12 or pj>min(pa,pb)+1e-12 or abs(pu-(pa+pb-pj))>1e-12:_fail('STAGE_UNION','Frozen predictions violate Frechet bounds or count the joint event twice')
        all_observed=all(eligible[s][i]for s in (a,b,j,u));ids=None
        if all_observed:
            ya,yb,yj,yu=[observed[s][i]for s in (a,b,j,u)]
            if yj!=ya*yb or yu!=max(ya,yb):_fail('STAGE_RECONCILIATION','Separately observed intersection/union labels disagree with component events')
            ids=sorted(set(outcomes[a]['event_ids'])|set(outcomes[b]['event_ids']))
            if set(outcomes[u]['event_ids'])!=set(ids):_fail('STAGE_RECONCILIATION','Union provenance must deduplicate component source event IDs')
        reconciliation.append(dict(prediction_id=row['prediction_id'],predicted_union=pu,predicted_joint=pj,observed_reconciled=all_observed,distinct_component_event_ids=ids,distinct_component_event_count=None if ids is None else len(ids)))
    def calculate(draw):
        out={}
        for name in names:
            w=weight*draw*eligible[name];m=_vb_errors(pred[name],observed[name],w);active=w>0;m.update(event_definition=stages[name]['event_definition'],kind=stages[name]['kind'],observable=stages[name]['observable'])
            if stages[name]['kind']=='binary':m['brier']=float(np.sum(w*(pred[name]-observed[name])**2)/w.sum())if active.any()else None
            out[name]=m
        return out
    values=calculate(np.ones(n));draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);samples=None
    if draws is not None:
        samples=[]
        for draw in draws:
            result=calculate(draw);samples.append({name+'/'+key:result[name][key]for name in names for key in ('bias','mae','rmse')+ (('brier',)if stages[name]['kind']=='binary'else())})
    all_excluded=[dict(prediction_id=row['prediction_id'],reason='NO_OBSERVABLE_MATURE_STAGE')for i,row in enumerate(rows)if not any(eligible[s][i]for s in names)]
    diagnostics=dict(stage_contract=stages,union_contract=union,stage_exclusions=excluded,event_reconciliation=reconciliation,stage_labels_inferred_from_final_label=False,support_counterfactual_identified=False,amounts_are_observed_contractual_realizations=True,uncertainty_population='DECLARED_SAMPLING_CLUSTERS_CONDITIONAL_ON_STAGE_OBSERVABILITY')
    return _vb_result('support_transfer_backtest',dict(by_stage=values),d,rows,all_excluded,_vb_bands(samples,confidence_level),context,diagnostics)
