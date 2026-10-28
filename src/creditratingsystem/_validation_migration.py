"""Versioned rating-transition backtests with endpoint coverage and absorbing states."""
import math
from copy import deepcopy
import numpy as np
from .core import _numeric,_fail,_fingerprint
from .data import _date
from ._expert_common import _ex_object,_ex_ids
from ._validation_common import (_vb_design,_vb_records,_vb_observed,_vb_draws,_vb_bands,_vb_result,_vb_string)


def rating_migration_backtest(records,predicted_transition_matrices,*,states,state_contract,design,uncertainty,weights=None,cluster_resamples=None,confidence_level=.95,context=None):
    """Score matched endpoint transitions without inventing censored self-transitions."""
    d=_vb_design(design,context)
    if d['target']['conditioning']!='origin_state':_fail('VALIDATION_TARGET','Migration target must condition on origin_state')
    rows=_vb_records(records,d);levels=_ex_ids(states,'states');contract=_ex_object(state_contract,'state_contract');k=len(levels);n=len(rows)
    for key in ('scale_id','scale_version','reference'):_vb_string(contract.get(key),key)
    known=_date(contract.get('known_at'),'state contract known_at')
    if any(known>_date(r['as_of'],'as_of')for r in rows):_fail('VALIDATION_LOOKAHEAD','State definition must be frozen before forecasts')
    groups={key:_ex_ids(contract.get(key),key,empty=key!='default_states')for key in ('default_states','cure_states','exit_states','absorbing_states')}
    if any(set(v)-set(levels)for v in groups.values())or any(set(groups[a])&set(groups[b])for a,b in [('default_states','cure_states'),('default_states','exit_states'),('cure_states','exit_states')]):_fail('VALIDATION_STATES','Explicit disjoint default/cure/exit roles and known absorbing states required')
    raw=_numeric(predicted_transition_matrices,'predicted_transition_matrices');matrices=np.repeat(raw[None,:,:],n,axis=0)if raw.shape==(k,k)else raw
    if matrices.shape!=(n,k,k)or np.any((matrices<0)|(matrices>1))or np.any(np.abs(matrices.sum(axis=2)-1)>1e-12):_fail('VALIDATION_TRANSITIONS','Aligned nonnegative row-stochastic transition matrices required')
    for state in groups['absorbing_states']:
        j=levels.index(state)
        if not np.array_equal(matrices[:,j,:],np.tile(np.eye(k)[j],(n,1))):_fail('VALIDATION_ABSORPTION','Declared absorbing states require exactly identity transition rows')
    w=np.ones(n)if weights is None else _numeric(weights,'weights',allow_empty=True,vector_only=True)
    if w.shape!=(n,)or np.any(w<0)or not np.isfinite(w.sum()):_fail('VALIDATION_WEIGHT','Aligned finite nonnegative migration weights required')
    origin=np.zeros(n,dtype=int);destination=np.zeros(n,dtype=int);eligible=np.ones(n);exclusions=[]
    for i,row in enumerate(rows):
        if row.get('scale_id')!=contract['scale_id']or row.get('scale_version')!=contract['scale_version']or row.get('origin_state')not in levels:_fail('VALIDATION_STATES','Forecast scale/version and known origin state required')
        origin[i]=levels.index(row['origin_state']);reason=_vb_observed(row,d);status=row.get('observation_status')
        if status not in ('OBSERVED','CENSORED'):_fail('VALIDATION_OBSERVATION','Declare observed or censored endpoint state')
        if reason is None and status=='CENSORED':reason='CENSORED_ENDPOINT'
        if reason is None:
            state=row.get('destination_state')
            if state not in levels:_fail('VALIDATION_STATES','Observed destination must belong to declared scale')
            at=_date(row.get('state_observed_at'),'state_observed_at');end=_date(row['horizon_end'],'horizon_end')
            if at<_date(row['as_of'],'as_of')or at>_date(row['outcome_known_at'],'outcome_known_at'):_fail('VALIDATION_LOOKAHEAD','Endpoint observation date incompatible with observation/knowledge dates')
            if at!=end and not(at<end and state in groups['absorbing_states']):reason='ENDPOINT_STATE_NOT_OBSERVED'
            elif row['origin_state']in groups['absorbing_states']and state!=row['origin_state']:reason='OBSERVED_ABSORBING_STATE_VIOLATION'
            else:destination[i]=levels.index(state)
        if reason is not None:eligible[i]=0;exclusions.append(dict(prediction_id=row['prediction_id'],reason=reason))
    probabilities=matrices[np.arange(n),origin]if n else np.empty((0,k));truth=np.eye(k)[destination]if n else np.empty((0,k));loss=((probabilities-truth)**2).sum(axis=1);prob_observed=probabilities[np.arange(n),destination]if n else np.empty(0);draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level)
    def calculate(draw):
        weight=w*eligible*draw;total=float(weight.sum())
        if not np.isfinite(total):_fail('VALIDATION_NUMERICAL','Resampled weight total exceeds finite range')
        counts=np.zeros((k,k));expected=np.zeros((k,k));rawcounts=np.zeros((k,k));origins=[]
        for i in range(n):
            if weight[i]>0:counts[origin[i],destination[i]]+=weight[i];expected[origin[i]]+=weight[i]*probabilities[i];rawcounts[origin[i],destination[i]]+=draw[i]
        def score(mask):
            mass=weight*mask;total=float(mass.sum());active=mass>0
            if total==0:return dict(brier=None,log_loss=None,log_loss_status='NOT_APPLICABLE',applicability='NOT_APPLICABLE',reason='EMPTY_WEIGHTED_ORIGIN')
            infinite=np.any(prob_observed[active]==0);ll=None if infinite else float(-np.dot(mass[active]/total,np.log(prob_observed[active])))
            return dict(brier=float(np.dot(mass/total,loss)),log_loss=ll,log_loss_status='INFINITE'if infinite else'FINITE',applicability='APPLICABLE',reason=None)
        for j,state in enumerate(levels):
            mass=float(counts[j].sum());origins.append(dict(origin_state=state,n=int(np.sum((origin==j)&(weight>0))),weight_sum=mass,observed_counts=counts[j].tolist(),unweighted_counts=rawcounts[j].tolist(),expected_counts=expected[j].tolist(),observed_probabilities=(counts[j]/mass).tolist()if mass>0 else None,expected_probabilities=(expected[j]/mass).tolist()if mass>0 else None,residuals=(counts[j]-expected[j]).tolist()if mass>0 else None,**score(origin==j)))
        overall=score(np.ones(n,dtype=bool));overall['weight_sum']=total;overall['destination_default_share']=float(np.sum(counts[:,[levels.index(s)for s in groups['default_states']]])/total)if total>0 else None
        return dict(overall=overall,by_origin=origins,states=levels,multiclass_brier_convention='SUM_OF_K_SQUARED_PROBABILITY_ERRORS_RANGE_0_TO_2',observed_weighted_counts=counts.tolist(),expected_weighted_counts=expected.tolist())
    metrics=calculate(np.ones(n));samples=None
    if draws is not None:
        samples=[]
        for draw in draws:
            m=calculate(draw);flat={key:m['overall'][key]for key in ('brier','log_loss','destination_default_share')}
            for origin_row in m['by_origin']:
                for j,state in enumerate(levels):flat['transition/'+origin_row['origin_state']+'/'+state]=None if origin_row['observed_probabilities']is None else origin_row['observed_probabilities'][j]
            samples.append(flat)
    diagnostics=dict(state_contract=deepcopy(contract),transition_fingerprint=_fingerprint(matrices.tolist()),weights=w.tolist(),endpoint_policy='EXACT_HORIZON_OBSERVATION_OR_EARLIER_DECLARED_ABSORBING_STATE',default_cure_exit_separate=True,no_inferred_censored_self_transition=True,log_loss_boundary_policy='EXACT_NULL_PLUS_INFINITE',observed_absorption_violations=sum(e['reason']=='OBSERVED_ABSORBING_STATE_VIOLATION'for e in exclusions))
    result=_vb_result('rating_migration_backtest',metrics,d,rows,exclusions,_vb_bands(samples,confidence_level),context,diagnostics,'migration_validation')
    if metrics['overall']['brier']is None:result.value.update(applicability='NOT_APPLICABLE',reason='NO_POSITIVE_WEIGHT_OBSERVED_TRANSITIONS')
    return result
