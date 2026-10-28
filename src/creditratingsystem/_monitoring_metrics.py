"""Outcome-mature monitoring, declared resampling and local statistical diagnostics."""
from collections.abc import Mapping
from datetime import date,timedelta
import math
import numpy as np
from scipy.stats import beta,wasserstein_distance
from .core import CalcResult,_fail,_numeric,_probability,_result,_fingerprint
from .validation import auc_gini,brier_score,log_loss


def _rows(x,allow_empty=False):
    if not isinstance(x,(list,tuple)) or (not x and not allow_empty) or any(not isinstance(r,Mapping) for r in x):_fail('INVALID_TABLE','List of records required')
    return x


def _string(x,name):
    if not isinstance(x,str) or not x.strip():_fail('INVALID_INPUT',name+' must be a nonempty string')
    return x


def _scalar(x,name,low=None,high=None):
    v=_numeric(x,name)
    if v.ndim or (low is not None and v<low) or (high is not None and v>high):_fail('INVALID_INPUT',name+' outside scalar domain')
    return float(v)


def _day(x):
    try:
        d=date.fromisoformat(x)
        if d.isoformat()!=x:raise ValueError
        return d
    except (ValueError,TypeError):_fail('INVALID_DATE','ISO calendar date required')


def _confidence(x):
    c=_scalar(x,'confidence_level',0,1)
    if c in (0,1):_fail('INVALID_INPUT','Confidence level must be interior')
    return c


def _interval(k,n,c):
    if n==0:return {'lower':None,'upper':None}
    a=(1-c)/2
    return {'lower':0. if k==0 else float(beta.ppf(a,k,n-k+1)),'upper':1. if k==n else float(beta.ppf(1-a,k+1,n-k))}


def build_monitoring_cohorts(predictions,outcome_events,coverage,*,target_contract,evaluation_cutoff,context=None):
    """Link frozen physical default forecasts to complete, mature observation windows."""
    rows=_rows(predictions);events=_rows(outcome_events,True);cover=_rows(coverage,True);cutoff=_day(evaluation_cutoff)
    if not isinstance(target_contract,Mapping) or target_contract.get('probability_measure')!='PHYSICAL' or target_contract.get('event')!='default':_fail('INCOMPATIBLE_TARGET','Explicit physical default target required')
    _string(target_contract.get('default_definition'),'default_definition');_string(target_contract.get('unit'),'unit')
    horizon=_scalar(target_contract.get('horizon_days'),'horizon_days',1)
    if horizon!=int(horizon):_fail('INCOMPATIBLE_TARGET','Positive integer horizon days required')
    known_events={};rating_events={};event_ids=set();coverage_map={};ignored=0
    for row in events:
        identifier=_string(row.get('event_id'),'event_id');entity=_string(row.get('entity_id'),'entity_id')
        if identifier in event_ids:_fail('DUPLICATE_EVENT','Event IDs must be unique')
        event_ids.add(identifier);at=_day(row.get('event_at'));known=_day(row.get('known_at'))
        if known<at:_fail('INVALID_CHRONOLOGY','Outcome cannot be known before occurrence')
        if row.get('event') not in ('default','exit','rating'):_fail('INVALID_OUTCOME','Only default, rating or censoring exit events supported')
        if row['event']=='rating':_string(row.get('grade'),'grade')
        if row.get('default_definition')!=target_contract['default_definition']:_fail('INCOMPATIBLE_TARGET','Outcome definition differs')
        if known>cutoff:ignored+=1;continue
        if row['event']=='rating':rating_events.setdefault(entity,[]).append((at,known,identifier,row['grade']))
        else:known_events.setdefault(entity,[]).append((at,known,row['event']))
    for row in cover:
        entity=_string(row.get('entity_id'),'entity_id');through=_day(row.get('complete_through'));known=_day(row.get('known_at'))
        if known<through:_fail('INVALID_CHRONOLOGY','Complete-through declaration cannot precede completion')
        if known>cutoff:ignored+=1;continue
        old=coverage_map.get(entity)
        if old is None or (through,known)>old:coverage_map[entity]=(through,known)
    output=[];ids=set()
    for row in rows:
        forecast=_string(row.get('forecast_id'),'forecast_id')
        if forecast in ids:_fail('DUPLICATE_FORECAST','Unique forecast IDs required')
        ids.add(forecast)
        for key in ('case_id','entity_id','cluster_id','cohort','segment','model_id','model_version'):_string(row.get(key),key)
        if _fingerprint(row.get('target'))!=_fingerprint(target_contract):_fail('INCOMPATIBLE_TARGET','Prediction target differs from cohort contract')
        at=_day(row.get('forecast_at'));known=_day(row.get('known_at'))
        if known>at:_fail('FORECAST_LEAKAGE','Frozen forecast must be known by forecast date')
        end=at+timedelta(days=int(horizon));pd=_scalar(row.get('pd'),'pd',0,1);weight=_scalar(row.get('weight',1),'weight',0)
        history=known_events.get(row['entity_id'],[])
        prior=any(t<=at and event=='default' for t,k,event in history)
        exits=[t for t,k,event in history if event=='exit' and t>at]
        exited_before=any(t<=at and event=='exit' for t,k,event in history)
        exit_at=min(exits) if exits else None
        defaults=[(t,k) for t,k,event in history if event=='default' and at<t<=end and (exit_at is None or t<=exit_at)]
        default=min(defaults) if defaults else None
        mature=end<=cutoff;complete=coverage_map.get(row['entity_id']);covered=complete is not None and complete[0]>=end
        censored=exit_at is not None and exit_at<end and default is None
        target=1 if default is not None else 0 if mature and covered and not censored and not prior and not exited_before else None
        reason='FUTURE_FORECAST' if at>cutoff else 'PRIOR_DEFAULT' if prior else 'EXITED_BEFORE_FORECAST' if exited_before else 'IMMATURE' if not mature else 'CENSORED' if censored else 'MISSING_OUTCOME' if target is None else 'ELIGIBLE'
        ratings=[e for e in rating_events.get(row['entity_id'],[]) if at<e[0]<=end]
        grade=(target_contract.get('default_grade') if target==1 else max(ratings)[3] if ratings else row.get('origin_grade')) if mature and covered and not censored and not prior and not exited_before else None
        out={key:row[key] for key in ('forecast_id','case_id','entity_id','cluster_id','cohort','segment','model_id','model_version','forecast_at')}
        out.update(pd=pd,weight=weight,target=target,outcome_end=end.isoformat(),outcome_known_at=default[1].isoformat() if default else complete[1].isoformat() if target==0 else None,mature=mature,complete_through=complete[0].isoformat() if complete else None,eligible=reason=='ELIGIBLE',status=reason,prior_default=prior,
            origin_grade=row.get('origin_grade'),outcome_grade=grade,override=row.get('override'),decision=row.get('decision'),prediction_origin=row.get('prediction_origin'))
        if out['override'] is not None and type(out['override']) is not bool:_fail('INVALID_INPUT','override must be boolean or null')
        if out['decision'] not in (None,'approve','decline'):_fail('INVALID_INPUT','Decision must be approve, decline or null')
        output.append(out)
    bundle={'rows':output,'target_contract':dict(target_contract),'evaluation_cutoff':evaluation_cutoff}
    bundle['cohort_hash']=_fingerprint(bundle)
    clusters=[r['cluster_id'] for r in output];entities=[r['entity_id'] for r in output]
    return _result(bundle,'monitoring.build_monitoring_cohorts','cohort_bundle',context,diagnostics={'eligible_count':sum(r['eligible'] for r in output),'ignored_not_yet_known_records':ignored,'repeated_cluster_rows':len(clusters)-len(set(clusters)),'repeated_entity_rows':len(entities)-len(set(entities)),'negative_labels_require_complete_through':True},limitations=('Declared coverage must include complete ascertainment of the specified default event. Exits censor subsequent observations; an early default is retained but excluded from metrics until horizon maturity.',))


def _cohort(cohort):
    bundle=cohort.value if isinstance(cohort,CalcResult) else cohort
    if not isinstance(bundle,Mapping) or not isinstance(bundle.get('rows'),list):_fail('INVALID_COHORT','Hashed monitoring cohort bundle required')
    payload={k:v for k,v in bundle.items() if k!='cohort_hash'}
    if bundle.get('cohort_hash')!=_fingerprint(payload):_fail('COHORT_HASH_MISMATCH','Cohort content changed')
    for row in bundle['rows']:
        if row['eligible'] and (not row['mature'] or row['target'] not in (0,1)):_fail('INVALID_COHORT','Eligible outcomes must be mature and observed')
    return bundle


def _metrics(rows,multiplicities=None):
    active=[r for r in rows if r['eligible'] and r['weight']>0 and (multiplicities is None or multiplicities.get(r['cluster_id'],0)>0)]
    if not active:return {'n':0,'weight_sum':0.,'observed_rate':None,'mean_pd':None,'calibration_error':None,'brier':None,'auc':None,'gini':None,'log_loss':None,'log_loss_status':'UNIDENTIFIED'}
    y=np.array([r['target'] for r in active],float);p=np.array([r['pd'] for r in active]);w=np.array([r['weight']*(1 if multiplicities is None else multiplicities[r['cluster_id']]) for r in active]);total=float(w.sum())
    rate=float(w@y/total);mean=float(w@p/total);auc=auc_gini(y,p,w,score_direction='higher_is_risk').value;ll=log_loss(y,p,w).value
    return {'n':len(active),'weight_sum':total,'observed_rate':rate,'mean_pd':mean,'calibration_error':rate-mean,'brier':brier_score(y,p,w).value['brier'],'auc':auc['auc'],'gini':auc['gini'],'log_loss':ll['log_loss'],'log_loss_status':ll['loss_status']}


def _resamples(draws,rows):
    clusters=sorted(set(r['cluster_id'] for r in rows))
    if not isinstance(draws,(list,tuple)) or len(draws)<2:_fail('INVALID_RESAMPLES','At least two supplied cluster-resample paths required')
    result=[]
    for draw in draws:
        if not isinstance(draw,(list,tuple)) or len(draw)!=len(clusters) or any(x not in clusters for x in draw):_fail('INVALID_RESAMPLES','Each draw must sample the original number of known clusters with replacement')
        result.append({c:draw.count(c) for c in clusters})
    return result


def _bands(samples,keys,c):
    out={}
    for key in keys:
        values=[s[key] for s in samples if s.get(key) is not None]
        out[key]={'lower':float(np.quantile(values,(1-c)/2)) if len(values)>=2 else None,'upper':float(np.quantile(values,(1+c)/2)) if len(values)>=2 else None,'identified_resamples':len(values),'degenerate':bool(len(values)>=2 and min(values)==max(values))}
    return out


def monitor_rating_cohort(cohort,*,group_by,reference_window,minimum_count,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Report cohort metrics with explicit IID exact-rate or cluster-bootstrap uncertainty."""
    bundle=_cohort(cohort);rows=bundle['rows'];c=_confidence(confidence_level);minimum=_scalar(minimum_count,'minimum_count',1)
    if minimum!=int(minimum) or not isinstance(group_by,(list,tuple)) or not group_by or len(set(group_by))!=len(group_by):_fail('INVALID_INPUT','Integer minimum and unique grouping fields required')
    if not isinstance(reference_window,Mapping):_fail('INVALID_REFERENCE','Fixed reference window required')
    start=_day(reference_window.get('start'));end=_day(reference_window.get('end'))
    if start>end:_fail('INVALID_REFERENCE','Reference start must precede end')
    if uncertainty not in ('none','iid_homogeneous_binomial','independent_cluster_bootstrap'):_fail('INVALID_UNCERTAINTY','Explicit uncertainty assumptions required')
    draws=_resamples(cluster_resamples,rows) if uncertainty=='independent_cluster_bootstrap' else None
    if uncertainty!='independent_cluster_bootstrap' and cluster_resamples is not None:_fail('INVALID_RESAMPLES','Resamples require cluster bootstrap mode')
    grouped={}
    for row in rows:
        if any(k not in row for k in group_by):_fail('MISSING_FEATURE','Missing grouping field')
        group={k:row[k] for k in group_by};key=_fingerprint(group);grouped.setdefault(key,(group,[]))[1].append(row)
    panel=[]
    for key,(group,part) in sorted(grouped.items()):
        metrics=_metrics(part);eligible=[r for r in part if r['eligible'] and r['weight']>0];intervals={}
        if uncertainty=='iid_homogeneous_binomial' and eligible:
            if any(r['weight']!=1 for r in eligible) or len(set(r['cluster_id'] for r in eligible))!=len(eligible):_fail('INCOMPATIBLE_ASSUMPTIONS','IID homogeneous intervals require unit weights and unique clusters')
            intervals['observed_rate']=_interval(sum(r['target'] for r in eligible),len(eligible),c)
        elif draws is not None:intervals=_bands([_metrics(part,d) for d in draws],['observed_rate','calibration_error','brier','auc'],c)
        migration={};override=[r for r in eligible if r['override'] is not None]
        for row in eligible:
            if row['origin_grade'] is not None and row['outcome_grade'] is not None:
                pair=(str(row['origin_grade']),str(row['outcome_grade']));migration[pair]=migration.get(pair,0)+row['weight']
        reference=[r for r in part if start<=_day(r['forecast_at'])<=end]
        panel.append({'group_id':key,'group':group,'status':'MISSING_OUTCOME' if metrics['n']==0 else 'INSUFFICIENT_INFORMATION' if metrics['n']<minimum else 'DESCRIPTIVE_AVAILABLE','metrics':metrics,'intervals':intervals,'reference_metrics':_metrics(reference),'migration':[{'from':a,'to':b,'weight':w} for (a,b),w in sorted(migration.items())],'override_rate':sum(r['weight']*r['override'] for r in override)/sum(r['weight'] for r in override) if override else None,'excluded_count':sum(not r['eligible'] for r in part),'repeated_cluster_rows':len(eligible)-len(set(r['cluster_id'] for r in eligible))})
    return _result(panel,'monitoring.monitor_rating_cohort','monitoring_panel',context,diagnostics={'cohort_hash':bundle['cohort_hash'],'reference_window':dict(reference_window),'uncertainty':uncertainty,'confidence_level':c,'minimum_count':minimum},limitations=('Exact rate intervals assume independent homogeneous Bernoulli outcomes. Bootstrap assumes independent resampled clusters; zero-width boundary bands are not evidence of perfect calibration. No universal alarm threshold or automatic recalibration.',))


def _weights(weights,n):
    w=np.ones(n) if weights is None else _numeric(weights,'weights',vector_only=True)
    if len(w)!=n or np.any(w<0) or not np.isfinite(w.sum()) or w.sum()<=0:_fail('INVALID_WEIGHTS','Finite nonnegative row weights with positive total required')
    return w


def detect_credit_feature_drift(reference,current,*,feature_order,method,permutations,multiple_testing,dependence_assumption,reference_weights=None,current_weights=None,bandwidth=None,frozen_bins=None,smoothing=.5,permutation_units=None,context=None):
    """Weighted KS/Wasserstein, RBF empirical MMD² or fixed-bin descriptive PSI.

    Numeric resampling uses supplied zero-based permutations of pooled rows;
    cluster permutations must move complete equal-size ordered clusters together.
    """
    ref=_rows(reference);cur=_rows(current)
    if not isinstance(feature_order,(list,tuple)) or not feature_order or len(set(feature_order))!=len(feature_order):_fail('INVALID_FEATURE_ORDER','Unique feature names required')
    if any(any(k not in row for k in feature_order) for row in list(ref)+list(cur)):_fail('MISSING_FEATURE','Feature missing from sample')
    wr=_weights(reference_weights,len(ref));wc=_weights(current_weights,len(cur));n=len(ref);N=n+len(cur);weights=np.r_[wr,wc]
    if method not in ('ks','wasserstein','mmd_rbf','psi'):_fail('INVALID_METHOD','Declare supported drift method')
    missing=lambda v:v is None or isinstance(v,(float,np.floating)) and math.isnan(v)
    findings=[]
    if method=='psi':
        if permutations is not None or multiple_testing!='not_applicable':_fail('INVALID_RESAMPLING','PSI is descriptive; do not attach an uncalibrated significance claim')
        if not isinstance(frozen_bins,Mapping):_fail('INVALID_BINS','Frozen numeric or categorical bins required')
        alpha=_scalar(smoothing,'smoothing',0)
        if alpha==0:_fail('INVALID_INPUT','Positive PSI smoothing required')
        for feature in feature_order:
            spec=frozen_bins.get(feature)
            if not isinstance(spec,Mapping):_fail('INVALID_BINS','Each feature requires frozen bins')
            if spec.get('kind')=='numeric':
                cut=_numeric(spec.get('cutpoints'),'cutpoints',allow_empty=True,vector_only=True)
                if np.any(np.diff(cut)<=0):_fail('INVALID_BINS','Strictly increasing finite cutpoints required')
                support=_numeric(spec.get('support'),'support',vector_only=True)
                if len(support)!=2 or support[0]>=support[1]:_fail('INVALID_BINS','Explicit support bounds required')
                size=len(cut)+2
                def encode(v):return size-1 if missing(v) else int(np.searchsorted(cut,_scalar(v,feature),side='left'))
                new=sum(not missing(r[feature]) and not support[0]<=_scalar(r[feature],feature)<=support[1] for r in cur);label='out_of_reference_support'
            elif spec.get('kind')=='categorical':
                levels=spec.get('levels')
                if not isinstance(levels,(list,tuple)) or any(not isinstance(v,str) for v in levels) or len(set(levels))!=len(levels):_fail('INVALID_BINS','Unique fixed string levels required')
                size=len(levels)+2
                def encode(v):
                    if missing(v):return size-2
                    if not isinstance(v,str):_fail('INVALID_FEATURE_TYPE','Categorical value must be string')
                    return levels.index(v) if v in levels else size-1
                new=sum(not missing(r[feature]) and r[feature] not in levels for r in cur);label='new_category_rows'
            else:_fail('INVALID_BINS','Bin kind numeric/categorical required')
            a=np.bincount([encode(r[feature]) for r in ref],weights=wr,minlength=size);b=np.bincount([encode(r[feature]) for r in cur],weights=wc,minlength=size)
            pa=(a+alpha)/(a.sum()+alpha*size);pb=(b+alpha)/(b.sum()+alpha*size);stat=float(np.sum((pb-pa)*np.log(pb/pa)))
            findings.append({'feature':feature,'statistic':stat,'p_value':None,'adjusted_p_value':None,'reference_missing':sum(missing(r[feature]) for r in ref),'current_missing':sum(missing(r[feature]) for r in cur),label:new,'reference_shares':pa.tolist(),'current_shares':pb.tolist()})
    else:
        if multiple_testing not in ('none','holm','bonferroni'):_fail('INVALID_MULTIPLE_TESTING','Declare none, holm or bonferroni')
        if dependence_assumption not in ('exchangeable_rows','exchangeable_clusters'):_fail('INCOMPATIBLE_ASSUMPTIONS','Permutation exchangeability must be explicit')
        draws=_numeric(permutations,'permutations')
        if draws.ndim!=2 or draws.shape[0]<2 or draws.shape[1]!=N or np.any(draws!=np.floor(draws)) or any(not np.array_equal(np.sort(row),np.arange(N)) for row in draws):_fail('INVALID_PERMUTATIONS','At least two complete pooled-row permutations required')
        draws=draws.astype(int)
        if dependence_assumption=='exchangeable_clusters':
            if not isinstance(permutation_units,(list,tuple)) or len(permutation_units)!=N or any(not isinstance(c,str) or not c for c in permutation_units):_fail('INVALID_CLUSTERS','Pooled cluster IDs required')
            if set(permutation_units[:n])&set(permutation_units[n:]):_fail('INVALID_CLUSTERS','Clusters must belong wholly to one sample')
            units=np.array(permutation_units);positions=[np.flatnonzero(units==c) for c in sorted(set(units))]
            for draw in draws:
                for ix in positions:
                    source=draw[ix]
                    if len(set(units[source]))!=1 or not np.array_equal(source,np.flatnonzero(units==units[source[0]])):_fail('INVALID_PERMUTATIONS','Cluster draws must move complete clusters in their original internal order')
        elif permutation_units is not None:_fail('INVALID_CLUSTERS','Cluster labels require exchangeable_clusters mode')
        data=np.array([[np.nan if missing(row[f]) else _scalar(row[f],f) for f in feature_order] for row in list(ref)+list(cur)])
        if method=='mmd_rbf':
            bw=_scalar(bandwidth,'bandwidth',0)
            if bw==0:_fail('INVALID_KERNEL','Positive predeclared RBF bandwidth required')
        def statistic(indices,columns):
            left=indices[:n];right=indices[n:];left=left[np.all(np.isfinite(data[left][:,columns]),axis=1)&(weights[left]>0)];right=right[np.all(np.isfinite(data[right][:,columns]),axis=1)&(weights[right]>0)]
            if not len(left) or not len(right):return None
            x=data[left][:,columns];y=data[right][:,columns];a=weights[left]/weights[left].sum();b=weights[right]/weights[right].sum()
            if method=='wasserstein':return float(wasserstein_distance(x[:,0],y[:,0],a,b))
            if method=='ks':
                grid=np.unique(np.r_[x[:,0],y[:,0]])
                return float(max(abs(sum(a[x[:,0]<=z])-sum(b[y[:,0]<=z])) for z in grid))
            def kernel(v,u):
                with np.errstate(over='ignore'):distance=np.sum(((v[:,None,:]-u[None,:,:])/bw)**2,axis=2)
                return np.exp(-.5*distance)
            return max(0.,float(a@kernel(x,x)@a+b@kernel(y,y)@b-2*a@kernel(x,y)@b))
        columns=[list(range(len(feature_order)))] if method=='mmd_rbf' else [[i] for i in range(len(feature_order))]
        for cols in columns:
            observed=statistic(np.arange(N),cols);null=[statistic(draw,cols) for draw in draws]
            if observed is None or any(v is None for v in null):_fail('INSUFFICIENT_COVERAGE','All supplied permutations require positive observed sample support')
            exceedances=sum(v>=observed-1e-14*max(1,abs(observed)) for v in null)
            p=(1+exceedances)/(len(null)+1)
            findings.append({'feature':'joint' if method=='mmd_rbf' else feature_order[cols[0]],'statistic':observed,'p_value':p,'adjusted_p_value':p,'permutation_standard_error':math.sqrt(p*(1-p)/(len(null)+1)),'permutation_tail_interval':_interval(exceedances,len(null),.95),'null_statistics':null,'reference_missing':int(np.sum(~np.all(np.isfinite(data[:n][:,cols]),axis=1))),'current_missing':int(np.sum(~np.all(np.isfinite(data[n:][:,cols]),axis=1)))})
        k=len(findings)
        if multiple_testing=='bonferroni':
            for row in findings:row['adjusted_p_value']=min(1,k*row['p_value'])
        if multiple_testing=='holm':
            running=0
            for rank,index in enumerate(sorted(range(k),key=lambda i:(findings[i]['p_value'],i))):
                running=max(running,(k-rank)*findings[index]['p_value']);findings[index]['adjusted_p_value']=min(1,running)
    return _result(findings,'monitoring.detect_credit_feature_drift','distribution_distance',context,diagnostics={'method':method,'feature_order':list(feature_order),'multiple_testing':multiple_testing,'dependence_assumption':dependence_assumption,'kernel_bandwidth':bandwidth,'mmd_variant':'weighted_biased_empirical_embedding_squared' if method=='mmd_rbf' else None,'permutation_index_base':0,'monte_carlo_confidence_level':.95,'missing_policy':'separate_PSI_bin_otherwise_complete_cases','no_concept_drift_claim':True},limitations=('Permutation p-values require exchangeability of the declared rows or intact clusters, including associated weights. Finite shared permutations provide Monte Carlo resolution; PSI has no universal significance threshold. Input drift alone does not establish changed conditional default risk.',))


def detect_pd_calibration_drift(residuals,*,method,parameters,null_paths,null_spec,alpha,validation_null_paths=None,confidence_level=.95,context=None):
    """Finite-horizon two-sided CUSUM/EWMA with supplied null-path calibration."""
    rows=_rows(residuals);times=[_day(r.get('time')) for r in rows];values=np.array([_scalar(r.get('value'),'residual') for r in rows]);c=_confidence(confidence_level);a=_confidence(alpha)
    if any(r.get('mature') is not True for r in rows) or any(x>=y for x,y in zip(times,times[1:])):_fail('INVALID_RESIDUALS','Strictly ordered mature residual aggregates required')
    if method not in ('cusum','ewma') or not isinstance(parameters,Mapping) or not isinstance(null_spec,Mapping):_fail('INVALID_NULL','Explicit method, parameters and null specification required')
    for key in ('model_id','model_version','calibration_id','residual_scale'):_string(null_spec.get(key),key)
    if null_spec.get('dependence') not in ('independent','prespecified_dependent') or null_spec.get('probability_measure')!='PHYSICAL' or _day(null_spec.get('calibrated_before'))>times[0]:_fail('INVALID_NULL','Prespecified physical null and dependence required before monitoring')
    null=_numeric(null_paths,'null_paths')
    if null.ndim!=2 or null.shape[0]<2 or null.shape[1]!=len(rows):_fail('INVALID_NULL','Null paths must share the complete monitoring horizon')
    if method=='cusum':
        allowance=_scalar(parameters.get('allowance'),'allowance',0)
        def path(x):
            up=down=0.;out=[]
            for r in x:up=max(0.,up+r-allowance);down=max(0.,down-r-allowance);out.append(max(up,down))
            return np.array(out)
    else:
        smoothing=_scalar(parameters.get('smoothing'),'smoothing',0,1)
        if smoothing==0:_fail('INVALID_INPUT','EWMA smoothing must be positive')
        def path(x):
            z=0.;out=[]
            for r in x:z=(1-smoothing)*z+smoothing*r;out.append(abs(z))
            return np.array(out)
    maxima=np.array([max(path(x)) for x in null]);rank=math.ceil((len(null)+1)*(1-a))
    if rank>len(null):_fail('INSUFFICIENT_NULL_SIMULATIONS','More null paths required for requested false-alarm level')
    threshold=float(np.sort(maxima)[rank-1]);observed=path(values);cross=np.flatnonzero(observed>threshold);validation=None
    if validation_null_paths is not None:
        independent=_numeric(validation_null_paths,'validation_null_paths')
        if independent.ndim!=2 or independent.shape[0]<2 or independent.shape[1]!=len(rows) or null_spec.get('validation_paths_independent') is not True:_fail('INVALID_NULL','Declare independent validation paths with the same horizon')
        count=sum(max(path(x))>threshold for x in independent);validation={'exceedances':int(count),'paths':len(independent),'estimate':count/len(independent),'interval':_interval(count,len(independent),c)}
    return _result({'statistic_path':observed.tolist(),'threshold':threshold,'first_crossing_index':int(cross[0]) if len(cross) else None,'first_crossing_time':times[cross[0]].isoformat() if len(cross) else None,'validation_false_alarm':validation},'monitoring.detect_pd_calibration_drift','residual_scale',context,diagnostics={'method':method,'parameters':dict(parameters),'null_spec':dict(null_spec),'alpha':a,'null_paths':len(null),'order_statistic_rank':rank,'strict_crossing':True,'horizon':len(rows),'null_paths_hash':_fingerprint(null.tolist()),'calibration_exceedance_fraction':float(np.mean(maxima>threshold)),'validation_performed':validation is not None,'index_base':0},limitations=('Threshold concerns the specified finite horizon under the declared null. Simulated paths must be independent draws of the entire dependence model. Calibration paths do not provide independent false-alarm validation; horizon extension requires new calibration.',))


def compare_challenger_cohorts(champion,challenger,*,cluster_resamples,selection_protocol,confidence_level=.95,context=None):
    """Paired out-of-time assessment without automatic model promotion."""
    a=_cohort(champion);b=_cohort(challenger);c=_confidence(confidence_level)
    if _fingerprint(a['target_contract'])!=_fingerprint(b['target_contract']) or a['evaluation_cutoff']!=b['evaluation_cutoff']:_fail('INCOMPATIBLE_COHORT','Target and evaluation cutoff must match')
    if not isinstance(selection_protocol,Mapping) or selection_protocol.get('evaluation_role')!='OUT_OF_TIME_TEST' or selection_protocol.get('multiple_challenger_policy') not in ('single_prespecified','separate_selection_and_final_test'):_fail('INVALID_SELECTION_PROTOCOL','Declare untouched out-of-time final test and challenger-selection policy')
    _string(selection_protocol.get('protocol_id'),'protocol_id')
    def keyed(rows):
        result={r['case_id']:r for r in rows}
        if len(result)!=len(rows):_fail('DUPLICATE_CASE','One forecast per matched case required')
        return result
    aa=keyed(a['rows']);bb=keyed(b['rows']);left=[];right=[];disagreements=[]
    for key in sorted(set(aa)&set(bb)):
        x,y=aa[key],bb[key]
        if any(x[k]!=y[k] for k in ('entity_id','cluster_id','forecast_at','outcome_end','weight')) or (x['target'] is not None and y['target'] is not None and x['target']!=y['target']):_fail('PAIR_MISMATCH','Paired case identity, outcome window, weights and observed labels must agree')
        if x['eligible']!=y['eligible']:disagreements.append(key)
        if x['eligible'] and y['eligible']:
            if x['prediction_origin']!='OUT_OF_TIME' or y['prediction_origin']!='OUT_OF_TIME':_fail('INVALID_SELECTION_PROTOCOL','Matched predictions must be declared out of time')
            left.append(x);right.append(y)
    if not left:_fail('INSUFFICIENT_INFORMATION','No jointly eligible paired cases')
    draws=_resamples(cluster_resamples,left);keys=['auc','brier','log_loss','calibration_error','mean_pd']
    def difference(x,y):return {k:None if x[k] is None or y[k] is None else y[k]-x[k] for k in keys}
    m1=_metrics(left);m2=_metrics(right);samples=[difference(_metrics(left,d),_metrics(right,d)) for d in draws]
    decisions=[(x['decision'],y['decision']) for x,y in zip(left,right) if x['decision'] is not None and y['decision'] is not None]
    value={'champion_metrics':m1,'challenger_metrics':m2,'differences':difference(m1,m2),'intervals':_bands(samples,keys,c),'paired_count':len(left),'champion_only':sorted(set(aa)-set(bb)),'challenger_only':sorted(set(bb)-set(aa)),'eligibility_disagreements':disagreements,'decision_flips':sum(x!=y for x,y in decisions),'observed_decision_pairs':len(decisions),'recommendation':'REQUIRES_MULTIDIMENSIONAL_REVIEW'}
    return _result(value,'monitoring.compare_challenger_cohorts','paired_metric_difference',context,diagnostics={'champion_hash':a['cohort_hash'],'challenger_hash':b['cohort_hash'],'selection_protocol':dict(selection_protocol),'resamples':len(draws),'resampling_unit':'independent_cluster','confidence_level':c,'difference_direction':'challenger_minus_champion','automatic_deployment':False},limitations=('Paired cluster percentile bands assume independent clusters. One-class resamples cannot identify AUC; infinite exact log loss has no finite difference. A declared protocol does not establish that an undisclosed model search was absent.',))


def diagnose_fairness_slices(cohort,*,group_labels,reference_group,use_basis,minimum_count,risk_threshold,cluster_resamples,confidence_level=.95,context=None):
    """Permitted, supplied group slices with paired uncertainty and coverage counts."""
    bundle=_cohort(cohort);_string(use_basis,'use_basis');_string(reference_group,'reference_group');c=_confidence(confidence_level);threshold=_scalar(risk_threshold,'risk_threshold',0,1);minimum=_scalar(minimum_count,'minimum_count',1)
    if minimum!=int(minimum):_fail('INVALID_INPUT','Integer minimum count required')
    labels={}
    for r in _rows(group_labels):
        case=_string(r.get('case_id'),'case_id');group=_string(r.get('group'),'group')
        if case in labels:_fail('DUPLICATE_CASE','Unique case labels required')
        labels[case]=group
    used=[r for r in bundle['rows'] if r['case_id'] in labels]
    if len({r['case_id'] for r in used})!=len(used):_fail('DUPLICATE_CASE','One forecast per labeled case required')
    groups=sorted({labels[r['case_id']] for r in used})
    if reference_group not in groups:_fail('INVALID_REFERENCE','Reference group must occur in cohort')
    draws=_resamples(cluster_resamples,used);parts={g:[r for r in used if labels[r['case_id']]==g] for g in groups};keys=['selection_rate','true_positive_rate','false_positive_rate','error_rate','observed_rate','mean_pd','calibration_error','brier']
    def metrics(rows,d=None):
        active=[r for r in rows if r['weight']>0 and (d is None or d.get(r['cluster_id'],0)>0)];base=_metrics(rows,d)
        def average(sub,fn):
            den=sum(r['weight']*(1 if d is None else d[r['cluster_id']]) for r in sub)
            return sum(r['weight']*(1 if d is None else d[r['cluster_id']])*fn(r) for r in sub)/den if den else None
        eligible=[r for r in active if r['eligible']];positive=[r for r in eligible if r['target']==1];negative=[r for r in eligible if r['target']==0]
        return {'selection_rate':average([r for r in active if r['decision'] is not None],lambda r:r['decision']=='approve'),'true_positive_rate':average(positive,lambda r:r['pd']>=threshold),'false_positive_rate':average(negative,lambda r:r['pd']>=threshold),'error_rate':average(eligible,lambda r:(r['pd']>=threshold)!=bool(r['target'])),**{k:base[k] for k in ('observed_rate','mean_pd','calibration_error','brier')}}
    point={g:metrics(p) for g,p in parts.items()};replicas={g:[metrics(p,d) for d in draws] for g,p in parts.items()}
    def gap(x,y):return {k:None if x[k] is None or y[k] is None else x[k]-y[k] for k in keys}
    out=[]
    for g in groups:
        eligible=[r for r in parts[g] if r['eligible'] and r['weight']>0];defaults=sum(r['target'] for r in eligible)
        out.append({'group':g,'status':'INSUFFICIENT_INFORMATION' if len(eligible)<minimum else 'DESCRIPTIVE_AVAILABLE','case_count':len(parts[g]),'mature_observed_count':len(eligible),'default_count':defaults,'boundary_outcome_sample':not eligible or defaults in (0,len(eligible)),'metrics':point[g],'intervals':_bands(replicas[g],keys,c),'reference_gaps':gap(point[g],point[reference_group]),'gap_intervals':_bands([gap(x,y) for x,y in zip(replicas[g],replicas[reference_group])],keys,c),'selection_denominator':'all_known_decisions','outcome_denominator':'mature_observed_outcomes'})
    return _result(out,'monitoring.diagnose_fairness_slices','group_diagnostic',context,diagnostics={'cohort_hash':bundle['cohort_hash'],'use_basis':use_basis,'reference_group':reference_group,'risk_threshold':threshold,'positive_prediction':'pd_at_least_threshold','resampling_unit':'independent_cluster','confidence_level':c,'unlabeled_case_count':len(bundle['rows'])-len(used),'compliance_certification':False},limitations=('Group labels are supplied, never inferred. Outcome comparisons can be distorted by selective observed outcomes. Pointwise cluster bootstrap bands assume independent clusters; degenerate boundary bands are flagged and do not establish precision. Different base rates can make fairness criteria incompatible. This diagnostic is not a legal or ethical certification.',))
