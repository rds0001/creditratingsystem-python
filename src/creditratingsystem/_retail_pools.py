"""Frozen retail score pools from deterministic constrained adjacent merging."""
from copy import deepcopy
from bisect import bisect_right
from datetime import timedelta
import math
from .core import CalcResult,_result,_fit_result,_model_artifact,_fingerprint,_fail
from .data import _rows,_required,_date,_days,_unique
from ._loss_paths import _sum
from .retail_features import _text,_strings,_number,_cutoffs


def _target(target):
    if not isinstance(target,dict):_fail('RETAIL_TARGET','Explicit retail default target required')
    _required(target,['event','horizon','time_unit','probability_measure','conditioning','default_unit','default_definition','currency'])
    if target['event']!='default'or target['time_unit']!='days'or target['probability_measure']!='PHYSICAL'or target['conditioning']!='not_defaulted_at_index'or target['default_unit']not in ('facility','obligor'):_fail('RETAIL_TARGET','Require physical day-horizon default target with explicit performing default unit')
    _days(target['horizon'],'target.horizon',1);_text(target['default_definition'],'default_definition');_text(target['currency'],'currency');return target


def _spec(spec):
    if not isinstance(spec,dict):_fail('RETAIL_POOL','Explicit pooling specification required')
    _required(spec,['spec_id','version','known_at','stratum_fields','score_field','segments','minimum_cases','minimum_defaults','minimum_lgd_count','minimum_ead_count','maximum_bin_mean_spread'])
    fields=_strings(spec['stratum_fields'],'stratum_fields',False)
    if not {'product','borrower_type','facility_type','delinquency_state'}<=set(fields):_fail('RETAIL_POOL','Declare product, borrower, facility and delinquency strata explicitly')
    if any(k.lower()in ('target','outcome','label','defaulted','lgd','ead','future_default','default_in_horizon')for k in fields):_fail('RETAIL_LEAKAGE','Outcome labels cannot be stratum features')
    score=_text(spec['score_field'],'score_field')
    if score in fields or score.lower()in ('target','outcome','label','defaulted','lgd','ead','future_default'):_fail('RETAIL_LEAKAGE','Score must be a separate pre-outcome input')
    for k in ('minimum_cases','minimum_defaults','minimum_lgd_count','minimum_ead_count'):_days(spec[k],k,1 if k=='minimum_cases'else 0)
    spreads=spec['maximum_bin_mean_spread']
    if not isinstance(spreads,dict)or set(spreads)!={'pd','lgd','ead'}:_fail('RETAIL_POOL','Separate nullable PD/LGD/EAD homogeneity thresholds required')
    for k,v in spreads.items():
        if v is not None:_number(v,k+' spread',0)
    segments=_rows(spec['segments'],'segments');_unique(segments,'segment_id','segments');seen=set()
    if not segments:_fail('RETAIL_POOL','At least one declared segment required')
    for segment in segments:
        _required(segment,['values','cutpoints','score_min','score_max','monotonicity'])
        if not isinstance(segment['values'],dict)or set(segment['values'])!=set(fields):_fail('RETAIL_POOL','Segment requires one exact value per declared stratum')
        values=tuple(_text(segment['values'][k],k)for k in fields)
        if values in seen:_fail('RETAIL_POOL','Duplicate segment stratum profile')
        seen.add(values);low=_number(segment['score_min'],'score_min');high=_number(segment['score_max'],'score_max')
        if low>=high:_fail('RETAIL_POOL','Finite score domain must have positive width')
        cuts=segment['cutpoints']
        if not isinstance(cuts,(list,tuple)):_fail('RETAIL_POOL','Cutpoints require an ordered array')
        boundaries=[low]+[_number(v,'cutpoint')for v in cuts]+[high]
        if any(a>=b for a,b in zip(boundaries,boundaries[1:])):_fail('RETAIL_POOL','Cutpoints must increase strictly inside declared domain')
        order=segment['monotonicity']
        if not isinstance(order,dict)or set(order)!={'pd','lgd','ead'}or any(v not in ('none','increasing','decreasing')for v in order.values()):_fail('RETAIL_POOL','Monotonicity is separately declared within each score segment')
    return fields,score,segments


def _context_target(ctx,target):
    for key in ('horizon','time_unit','probability_measure','conditioning','currency'):
        if ctx.get(key)is not None and ctx[key]!=target[key]:_fail('TARGET_MISMATCH','Context differs from pool target '+key)


def _features(row,fields,score,segments):
    if any(row.get(k)is None for k in fields+[score]):return None,None,'MISSING_FEATURE'
    values=tuple(_text(row[k],k)for k in fields);segment=next((s for s in segments if tuple(s['values'][k]for k in fields)==values),None)
    if segment is None:return None,None,'UNSEEN_STRATUM'
    x=_number(row[score],score)
    if not segment['score_min']<=x<=segment['score_max']:return segment,None,'OUTSIDE_SCORE_DOMAIN'
    return segment,bisect_right(segment['cutpoints'],x),'COVERED'


def _stats(rows):
    n=len(rows);d=sum(r['defaulted']for r in rows);lgd=[r['lgd']for r in rows if r['lgd']is not None];ead=[r['ead']for r in rows if r['ead']is not None]
    means={'pd':None if n==0 else d/n,'lgd':None if not lgd else _sum(v/len(lgd)for v in lgd),'ead':None if not ead else _sum(v/len(ead)for v in ead)};sd={}
    for key,values in [('pd',[int(r['defaulted'])for r in rows]),('lgd',lgd),('ead',ead)]:
        scale=max((abs(v)for v in values),default=0);sd[key]=None if not values else 0.0 if scale==0 else scale*math.sqrt(_sum((v/scale-means[key]/scale)**2/len(values)for v in values))
        if sd[key]is not None and not math.isfinite(sd[key]):_fail('NUMERICAL_FAILURE','Pool within-outcome standard deviation outside finite range')
    return dict(cases=n,defaults=d,lgd_count=len(lgd),ead_count=len(ead),means=means,within_population_sd=sd)


def _support(stats,spec):
    return [k for k,field in [('minimum_cases','cases'),('minimum_defaults','defaults'),('minimum_lgd_count','lgd_count'),('minimum_ead_count','ead_count')]if stats[field]<spec[k]]


def fit_retail_risk_pools(observations,pooling_spec,*,target,training,model_id,model_version,context=None):
    """Fit on mature training rows; unmet constraints remain unassignable."""
    target=_target(target);fields,score,segments=_spec(pooling_spec)
    if not isinstance(training,dict)or training.get('role')!='TRAIN':_fail('RETAIL_TRAINING','Training metadata must explicitly identify TRAIN role')
    _required(training,['known_at','split_id','data_fingerprint']);cutoff=_date(training['known_at'],'training.known_at');_,_,ctx=_cutoffs(training['known_at'],training['known_at'],context)
    _context_target(ctx,target)
    if _date(pooling_spec['known_at'],'pooling_spec.known_at')>cutoff:_fail('RETAIL_DATE','Pool specification unavailable at training cutoff')
    rows=_rows(observations,'observations');_unique(rows,'observation_id','observations');_unique(rows,'default_unit_id','default units');bins={s['segment_id']:[[]for _ in range(len(s['cutpoints'])+1)]for s in segments}
    for row in rows:
        _required(row,['default_unit_id','role','index_date','feature_observed_at','feature_known_at','label_window_end','label_known_at','defaulted','lgd_status'])
        if any(k not in row for k in ('default_date','lgd','ead','lgd_known_at','ead_known_at')):_fail('RETAIL_TARGET','Default date and LGD/EAD outcomes/knowledge dates require explicit nullable fields')
        if row['role']!='TRAIN'or type(row['defaulted'])is not bool:_fail('RETAIL_TRAINING','Only explicit TRAIN rows with boolean outcomes allowed')
        index=_date(row['index_date'],'index_date');observed=_date(row['feature_observed_at'],'feature_observed_at');known=_date(row['feature_known_at'],'feature_known_at');end=_date(row['label_window_end'],'label_window_end');label_known=_date(row['label_known_at'],'label_known_at')
        if not observed<=known<=index<end<=label_known<=cutoff or(end-index).days!=target['horizon']:_fail('RETAIL_DATE','Training features must predate the mature declared outcome horizon and cutoff')
        if row['lgd_status']not in ('CLOSED','OPEN','NOT_DEFAULTED')or(not row['defaulted']and(row['lgd_status']!='NOT_DEFAULTED'or row['lgd']is not None or row['ead']is not None))or(row['defaulted']and row['lgd_status']=='NOT_DEFAULTED')or(row['lgd_status']!='CLOSED'and row['lgd']is not None):_fail('RETAIL_TARGET','LGD/EAD conditioning and open-workout status inconsistent')
        default=None if row['default_date']is None else _date(row['default_date'],'default_date')
        if row['defaulted']and(default is None or not index<default<=end)or not row['defaulted']and default is not None:_fail('RETAIL_TARGET','Training default date must match the horizon outcome')
        for parameter in ('lgd','ead'):
            known_value=row[parameter+'_known_at']
            if row[parameter]is None:
                if known_value is not None:_fail('RETAIL_TARGET','Missing numeric outcome must have null knowledge date')
            elif known_value is None or not default<=_date(known_value,parameter+'_known_at')<=cutoff:_fail('RETAIL_DATE','Conditional loss/exposure outcome was not known by training cutoff')
        if row['lgd']is not None:_number(row['lgd'],'lgd')
        if row['ead']is not None:_number(row['ead'],'ead',0)
        segment,index,status=_features(row,fields,score,segments)
        if status!='COVERED':_fail('RETAIL_TRAINING','Training observation outside declared feature domain',dict(observation_id=row['observation_id'],status=status))
        bins[segment['segment_id']][index].append(row)
    rule_hash=_fingerprint(pooling_spec);fitted=[];merge_path=[]
    for segment in segments:
        initial=bins[segment['segment_id']];initial_stats=[_stats(cell)for cell in initial];blocks=[dict(first=i,last=i,rows=list(cell))for i,cell in enumerate(initial)]
        def summary(block):
            stats=_stats(block['rows']);spread={}
            for key in ('pd','lgd','ead'):
                values=[s['means'][key]for s in initial_stats[block['first']:block['last']+1]if s['means'][key]is not None];spread[key]=None if not values else _sum((max(values),-min(values)))
            return stats,spread
        def merge_candidate(i):
            left,right=blocks[i],blocks[i+1];merged=dict(first=left['first'],last=right['last'],rows=left['rows']+right['rows']);stats,spread=summary(merged)
            if any(pooling_spec['maximum_bin_mean_spread'][k]is not None and spread[k]is not None and spread[k]>pooling_spec['maximum_bin_mean_spread'][k]for k in spread):return None
            lm,rm=_stats(left['rows'])['means'],_stats(right['rows'])['means'];cost=_sum((lm[k]/max(1,abs(lm[k]),abs(rm[k]))-rm[k]/max(1,abs(lm[k]),abs(rm[k])))**2 for k in lm if lm[k]is not None and rm[k]is not None)
            return cost,i,merged,stats,spread
        while len(blocks)>1:
            candidates=[]
            for i,block in enumerate(blocks):
                if _support(_stats(block['rows']),pooling_spec):
                    for j in (i-1,i):
                        if 0<=j<len(blocks)-1:
                            c=merge_candidate(j)
                            if c is not None:candidates.append((0,c[0],j,'MINIMUM_SUPPORT',c))
            if not candidates:
                for i in range(len(blocks)-1):
                    lm,rm=_stats(blocks[i]['rows'])['means'],_stats(blocks[i+1]['rows'])['means'];violated=[k for k,order in segment['monotonicity'].items()if lm[k]is not None and rm[k]is not None and((order=='increasing'and lm[k]>rm[k])or(order=='decreasing'and lm[k]<rm[k]))]
                    if violated:
                        c=merge_candidate(i)
                        if c is not None:candidates.append((1,c[0],i,'MONOTONICITY:'+','.join(sorted(violated)),c))
            if not candidates:break
            _,_,i,reason,c=min(candidates,key=lambda v:(v[0],v[1],v[2]));_,_,merged,stats,spread=c;merge_path.append(dict(segment_id=segment['segment_id'],left_first=blocks[i]['first'],left_last=blocks[i]['last'],right_first=blocks[i+1]['first'],right_last=blocks[i+1]['last'],reason=reason,merged_statistics=stats,merged_bin_mean_spread=spread));blocks[i:i+2]=[merged]
        pools=[];violations=[]
        for block in blocks:
            stats,spread=summary(block);support=_support(stats,pooling_spec);pool_id=f"{model_id}@{model_version}:{rule_hash[:12]}:{segment['segment_id']}:{block['first']}-{block['last']}";pools.append(dict(pool_id=pool_id,first_bin=block['first'],last_bin=block['last'],statistics=stats,initial_bin_mean_spread=spread,support_violations=support))
            if support:violations.append(dict(kind='INSUFFICIENT_SUPPORT',pool_id=pool_id,constraints=support))
        for left,right in zip(pools,pools[1:]):
            for key,order in segment['monotonicity'].items():
                a,b=left['statistics']['means'][key],right['statistics']['means'][key]
                if a is not None and b is not None and((order=='increasing'and a>b)or(order=='decreasing'and a<b)):violations.append(dict(kind='MONOTONICITY',left_pool=left['pool_id'],right_pool=right['pool_id'],parameter=key,direction=order))
        fitted.append(dict(segment_id=segment['segment_id'],values=segment['values'],score_min=segment['score_min'],score_max=segment['score_max'],cutpoints=segment['cutpoints'],monotonicity=segment['monotonicity'],status='APPLICABLE'if not violations else 'INSUFFICIENT_SUPPORT',pools=pools,violations=violations,initial_bin_statistics=initial_stats))
    model=_model_artifact(dict(schema_version='1.0',model_id=model_id,model_version=model_version,method_id='retail_features.fit_retail_risk_pools',feature_order=fields+[score],parameters=dict(stratum_fields=fields,score_field=score,segments=fitted,ruleset_hash=rule_hash,pooling_spec=deepcopy(pooling_spec),merge_path=merge_path),preprocessing=dict(boundary_convention='LEFT_CLOSED_RIGHT_OPEN_FINAL_CLOSED',missing_policy='UNASSIGNED',unseen_policy='UNASSIGNED'),target=target,training=dict(training,actual_input_fingerprint=_fingerprint(rows),observation_count=len(rows)),diagnostics=dict(algorithm='GREEDY_ADJACENT_CONSTRAINED_MERGING',global_feasibility_claim=False,applicable_segments=sum(s['status']=='APPLICABLE'for s in fitted),insufficient_segments=sum(s['status']!='APPLICABLE'for s in fitted),training_cases=len(rows),training_defaults=sum(r['defaulted']for r in rows)),provenance=dict(data_origin=training.get('data_origin','USER_SUPPLIED'),parameter_origin='ESTIMATED')))
    model['artifact_hash']=_fingerprint(model)
    return _fit_result(model,'retail_features.fit_retail_risk_pools',diagnostics=model['diagnostics'],context=ctx)


def _checked(model):
    model=_model_artifact(model)
    if model['method_id']!='retail_features.fit_retail_risk_pools':_fail('RETAIL_ARTIFACT','Expected frozen retail pooling model')
    unbound=deepcopy(model);digest=unbound.pop('artifact_hash',None)
    if digest!=_fingerprint(unbound):_fail('INTEGRITY_ERROR','Pooling model content differs from its binding')
    _target(model['target']);_spec(model['parameters']['pooling_spec'])
    return model


def assign_retail_risk_pool(observations,fitted_model,*,as_of,information_cutoff,expected_target,uncovered_policy,context=None):
    """Assign exact frozen score intervals; unseen or unsupported profiles stay uncovered."""
    model=_checked(fitted_model);anchor,cutoff,ctx=_cutoffs(as_of,information_cutoff,context)
    _context_target(ctx,model['target'])
    if expected_target!=model['target']:_fail('TARGET_MISMATCH','Pool target differs from expected target')
    if _date(model['training']['known_at'],'training.known_at')>cutoff:_fail('RETAIL_DATE','Pooling model unavailable at scoring cutoff')
    if uncovered_policy not in ('unassigned','error'):_fail('RETAIL_POOL','Declare unassigned or error uncovered-profile policy')
    p=model['parameters'];fields=p['stratum_fields'];score=p['score_field'];rows=_rows(observations,'observations');_unique(rows,'observation_id','observations');outputs=[]
    for row in rows:
        _required(row,['index_date','feature_observed_at','feature_known_at']);index=_date(row['index_date'],'index_date');observed=_date(row['feature_observed_at'],'feature_observed_at');known=_date(row['feature_known_at'],'feature_known_at')
        if not observed<=known<=index<=anchor or known>cutoff:_fail('RETAIL_DATE','Scoring features must be available by their dated index and cutoff')
        segment,bin_index,status=_features(row,fields,score,p['segments']);pool=None
        if status=='COVERED':
            if segment['status']!='APPLICABLE':status='INSUFFICIENT_SUPPORT'
            else:pool=next(s for s in segment['pools']if s['first_bin']<=bin_index<=s['last_bin'])
        if status!='COVERED'and uncovered_policy=='error':_fail('RETAIL_UNCOVERED','Observation not covered by frozen pools',dict(observation_id=row['observation_id'],status=status))
        outputs.append(dict(observation_id=row['observation_id'],index_date=row['index_date'],pool_id=None if pool is None else pool['pool_id'],segment_id=None if segment is None else segment['segment_id'],score_bin=bin_index,status=status,matched_strata=None if segment is None else segment['values'],first_bin=None if pool is None else pool['first_bin'],last_bin=None if pool is None else pool['last_bin']))
    value=dict(schema_version='1.0',artifact_type='RETAIL_POOL_ASSIGNMENTS',model_id=model['model_id'],model_version=model['model_version'],model_hash=model['artifact_hash'],target=model['target'],provenance=model['provenance'],pool_ids=[pool['pool_id']for segment in p['segments']for pool in segment['pools']],as_of=as_of,information_cutoff=information_cutoff,assignments=outputs);value['artifact_hash']=_fingerprint(value)
    return _result(value,'retail_features.assign_retail_risk_pool','pool_assignment',ctx,diagnostics=dict(covered_count=sum(r['status']=='COVERED'for r in outputs),uncovered_count=sum(r['status']!='COVERED'for r in outputs),ruleset_hash=p['ruleset_hash']),provenance=model['provenance'],limitations=('Frozen pool boundaries and strata are never refitted during assignment. Missing features, unseen strata, scores outside the declared finite domain and segments with unmet fitting constraints remain explicitly uncovered. A changed dated delinquency input can legitimately change a pool assignment.',))


def _weighted(values,weights):
    if not values or _sum(weights)==0:return dict(estimate=None,numerator=0.0,denominator=0.0,count=0,independent_episode_standard_error=None,observed_minimum=None,observed_maximum=None)
    from ._loss_paths import _multiply
    pairs=[(v,w)for v,w in zip(values,weights)if w>0];denominator=_sum(w for v,w in pairs);numerator=_sum(float(_multiply(v,w))for v,w in pairs);mean=_sum(v*(w/denominator)for v,w in pairs);n=len(pairs);scale=max((abs(v)for v,w in pairs),default=0);se=None if n<2 else 0.0 if scale==0 else scale*math.sqrt(n/(n-1)*_sum(((w/denominator)*(v/scale-mean/scale))**2 for v,w in pairs))
    if se is not None and not math.isfinite(se):_fail('NUMERICAL_FAILURE','Pool mean uncertainty outside finite range')
    return dict(estimate=mean,numerator=numerator,denominator=denominator,count=n,independent_episode_standard_error=se,observed_minimum=min(v for v,w in pairs),observed_maximum=max(v for v,w in pairs))


def estimate_pool_risk_parameters(pool_assignments,cohort_outcomes,workouts,estimation_profile,*,evaluation_cutoff,calibration_id,calibration_version,context=None):
    """Estimate local mature-cohort PD and default-conditioned loss/exposure separately."""
    from ._calibration_models import calibrate_grade_pd,low_default_bound
    from .lgd import discount_recoveries,workout_lgd
    from .ead import realized_ccf
    assignments=pool_assignments.value if isinstance(pool_assignments,CalcResult)else pool_assignments
    if not isinstance(assignments,dict)or assignments.get('artifact_type')!='RETAIL_POOL_ASSIGNMENTS':_fail('RETAIL_ARTIFACT','Frozen retail pool assignments required')
    unbound=deepcopy(assignments);digest=unbound.pop('artifact_hash',None)
    if digest!=_fingerprint(unbound):_fail('INTEGRITY_ERROR','Pool assignments differ from their binding')
    target=_target(assignments['target']);cutoff=_date(evaluation_cutoff,'evaluation_cutoff');_,_,ctx=_cutoffs(evaluation_cutoff,evaluation_cutoff,context)
    _context_target(ctx,target)
    if _date(assignments['information_cutoff'],'assignment cutoff')>cutoff:_fail('RETAIL_DATE','Assignments not available at evaluation cutoff')
    _text(calibration_id,'calibration_id');_text(calibration_version,'calibration_version');profile=estimation_profile
    if not isinstance(profile,dict):_fail('RETAIL_ESTIMATION','Explicit dated estimation profile required')
    _required(profile,['profile_id','version','known_at','pd_weighting','lgd_weighting','ead_weighting','ccf_weighting','confidence','independence_assumption','minimum_pool_cases'])
    if _date(profile['known_at'],'profile.known_at')>cutoff:_fail('RETAIL_DATE','Estimation profile unavailable at cutoff')
    if profile['pd_weighting']not in ('default_unit','reference_exposure')or profile['lgd_weighting']not in ('episode','ead')or profile['ead_weighting']not in ('episode','reference_exposure')or profile['ccf_weighting']not in ('episode','undrawn')or profile['independence_assumption']!='independent_default_units':_fail('RETAIL_ESTIMATION','Explicit supported parameter weightings and independent-default-unit inference assumption required')
    confidence=_number(profile['confidence'],'confidence')
    if not 1e-8<=confidence<=1-1e-8:_fail('RETAIL_ESTIMATION','Confidence must lie inside [1e-8,1-1e-8]')
    minimum=_days(profile['minimum_pool_cases'],'minimum_pool_cases',1);assigned={r['observation_id']:r for r in assignments['assignments']};rows=_rows(cohort_outcomes,'cohort outcomes');_unique(rows,'observation_id','cohort outcomes');_unique(rows,'default_unit_id','cohort default units');eligible={};excluded=[];pools={k:[]for k in assignments['pool_ids']}
    for row in rows:
        _required(row,['default_unit_id','index_date','label_window_end','label_known_at','outcome_status','reference_exposure','reference_known_at'])
        if 'defaulted'not in row or 'default_date'not in row:_fail('RETAIL_TARGET','Outcome default flag and date require explicit nullable fields')
        assignment=assigned.get(row['observation_id'])
        if assignment is None or assignment['index_date']!=row['index_date']:_fail('RETAIL_TARGET','Outcome lacks same-observation/index-date pool assignment')
        index=_date(row['index_date'],'index_date');end=_date(row['label_window_end'],'label_window_end');known=_date(row['label_known_at'],'label_known_at');reference_known=_date(row['reference_known_at'],'reference_known_at');_number(row['reference_exposure'],'reference_exposure',0)
        if(end-index).days!=target['horizon']or reference_known>index:_fail('RETAIL_DATE','Cohort horizon or frozen exposure weight invalid')
        if row['outcome_status']not in ('MATURE','CENSORED','OPEN_HORIZON'):_fail('RETAIL_TARGET','Explicit cohort maturity required')
        reason='UNASSIGNED_POOL'if assignment['status']!='COVERED'else row['outcome_status']if row['outcome_status']!='MATURE'else 'NOT_MATURE_AT_CUTOFF'if end>cutoff or known>cutoff else None
        if reason:excluded.append(dict(observation_id=row['observation_id'],reason=reason));continue
        if known<end or type(row['defaulted'])is not bool:_fail('RETAIL_TARGET','Mature whole-horizon outcomes require explicit boolean default flag and receipt after horizon')
        if row['defaulted']:
            if row['default_date']is None or not index<_date(row['default_date'],'default_date')<=end:_fail('RETAIL_TARGET','Default date outside performing cohort horizon')
        elif row['default_date']is not None:_fail('RETAIL_TARGET','Nondefault outcome cannot carry default date')
        eligible[row['observation_id']]=row;pools[assignment['pool_id']].append(row)
    episodes=_rows(workouts,'workouts');_unique(episodes,'episode_id','workouts');_unique(episodes,'observation_id','workout observation');episode_results={};workout_exclusions=[]
    for row in episodes:
        _required(row,['observation_id','default_date','default_known_at','ead_at_default','episode_status','valuation_known_at','recovery_cashflows','discount_curve','fx_contract','reference_date','reference_known_at','reference_drawn','reference_limit'])
        cohort=eligible.get(row['observation_id'])
        if cohort is None:workout_exclusions.append(dict(episode_id=row['episode_id'],reason='COHORT_NOT_ELIGIBLE'));continue
        if not cohort['defaulted']or row['default_date']!=cohort['default_date']:_fail('RETAIL_TARGET','Workout must belong to the declared cohort default')
        known=_date(row['valuation_known_at'],'valuation_known_at');default_known=_date(row['default_known_at'],'default_known_at')
        if known>cutoff or default_known>cutoff:workout_exclusions.append(dict(episode_id=row['episode_id'],reason='WORKOUT_NOT_KNOWN'));continue
        if known<default_known or default_known>_date(cohort['label_known_at'],'label_known_at')or row['reference_date']!=cohort['index_date']:_fail('RETAIL_DATE','Workout valuation receipt/reference differs from cohort convention')
        recoveries=discount_recoveries(row['recovery_cashflows'],row['discount_curve'],row['fx_contract'],default_date=row['default_date'],evaluation_cutoff=row['valuation_known_at'],reporting_currency=target['currency']);lgd=workout_lgd(row['ead_at_default'],recoveries,episode_status=row['episode_status'],default_date=row['default_date'],evaluation_cutoff=row['valuation_known_at'],currency=target['currency']);ccf=realized_ccf(row['reference_drawn'],row['reference_limit'],row['ead_at_default'],reference_date=row['reference_date'],default_date=row['default_date'],reference_known_at=row['reference_known_at'],default_known_at=row['default_known_at'],evaluation_cutoff=evaluation_cutoff,currency=target['currency'])
        episode_results[row['observation_id']]=dict(episode_id=row['episode_id'],lgd=lgd.value,ccf=ccf.value,recovery_valuation=recoveries.value)
    table=[]
    for pool_id,cohort in pools.items():
        n=len(cohort);d=sum(r['defaulted']for r in cohort);pd_selected=_weighted([int(r['defaulted'])for r in cohort],[1 if profile['pd_weighting']=='default_unit'else r['reference_exposure']for r in cohort]);interval=None;bound=None
        if n:
            grade=calibrate_grade_pd([n],[d],grades=[pool_id],monotonic=False,confidence=confidence).value['grades'][0];interval=dict(estimate=grade['pd'],lower=grade['lower'],upper=grade['upper'],count=n,defaults=d,weighting='default_unit',method='exact_binomial');bound=low_default_bound([d],[n],confidence=confidence,dependence='independent',pooling='none').value['bounds'][0]['upper_pd']
        lgd_values=[];lgd_weights=[];ead_values=[];ead_weights=[];ccf_values=[];ccf_weights=[];open_count=0;zero_ead=0;undefined_ccf=0;missing_workouts=0;episodes_used=[]
        for row in cohort:
            if not row['defaulted']:continue
            episode=episode_results.get(row['observation_id'])
            if episode is None:missing_workouts+=1;continue
            episodes_used.append(episode['episode_id']);loss=episode['lgd'];conversion=episode['ccf'];ead=loss['ead_at_default'];ead_values.append(ead);ead_weights.append(1 if profile['ead_weighting']=='episode'else row['reference_exposure'])
            if loss['episode_status']=='OPEN':open_count+=1
            elif loss['economic_lgd']is None:zero_ead+=1
            else:lgd_values.append(loss['economic_lgd']);lgd_weights.append(1 if profile['lgd_weighting']=='episode'else ead)
            if conversion['ccf']is None:undefined_ccf+=1
            else:ccf_values.append(conversion['ccf']);ccf_weights.append(1 if profile['ccf_weighting']=='episode'else conversion['undrawn_at_reference'])
        table.append(dict(pool_id=pool_id,status='COHORT_SUPPORT_MET'if n>=minimum else 'INSUFFICIENT_SUPPORT',mature_default_units=n,defaults=d,parameter_status=dict(pd='DEFINED'if pd_selected['denominator']>0 else 'ZERO_WEIGHT_OR_NO_MATURE_UNITS',lgd='CLOSED_EPISODES_ONLY'if _sum(lgd_weights)>0 else 'NO_DEFINED_CLOSED_EPISODES',ead='OBSERVED_DEFAULT_EPISODES'if _sum(ead_weights)>0 else 'NO_POSITIVE_WEIGHT_EXPOSURE_EPISODES',ccf='DEFINED_UNDRAWN_REFERENCES'if _sum(ccf_weights)>0 else 'NO_DEFINED_POSITIVE_WEIGHT_CCF'),pd_weighting=profile['pd_weighting'],pd=pd_selected,unweighted_pd_exact_interval=interval,unweighted_pd_one_sided_upper=bound,lgd_weighting=profile['lgd_weighting'],closed_workout_lgd=_weighted(lgd_values,lgd_weights),ead_weighting=profile['ead_weighting'],default_conditioned_ead=_weighted(ead_values,ead_weights),ccf_weighting=profile['ccf_weighting'],default_conditioned_ccf=_weighted(ccf_values,ccf_weights),open_workouts=open_count,zero_ead_closed_workouts=zero_ead,undefined_ccf_workouts=undefined_ccf,missing_default_workouts=missing_workouts,episode_ids=sorted(episodes_used)))
    value=dict(schema_version='1.0',artifact_type='CalibrationArtifact',calibration_type='RETAIL_POOL_RISK_PARAMETERS',calibration_id=calibration_id,calibration_version=calibration_version,base_model_id=assignments['model_id'],base_model_version=assignments['model_version'],base_model_hash=assignments['model_hash'],assignments_hash=assignments['artifact_hash'],target=target,evaluation_cutoff=evaluation_cutoff,estimation_profile=deepcopy(profile),pool_risk_table=table,aggregate_mature_default_units=sum(r['mature_default_units']for r in table),aggregate_defaults=sum(r['defaults']for r in table),provenance=assignments['provenance']);value['artifact_hash']=_fingerprint(value)
    return _result(value,'retail_features.estimate_pool_risk_parameters','pool_risk_parameters',ctx,provenance=assignments['provenance'],diagnostics=dict(excluded_cohort_outcomes=excluded,excluded_workouts=workout_exclusions,missing_outcome_observation_ids=sorted(set(assigned)-{r['observation_id']for r in rows}),episode_results=episode_results,input_fingerprint=_fingerprint(dict(cohort_outcomes=rows,workouts=episodes))),limitations=('PD inference assumes independent homogeneous default units within frozen pools; exact intervals and upper bounds refer to unweighted unit counts, not exposure-weighted PD. Intervals are conditional on chosen pools and do not account for pool-selection uncertainty or establish long-run/TTC calibration. LGD estimates use only observed closed workouts; open workouts remain separate and no future recoveries are invented. EAD and CCF condition on observed default and available exposure records. Reported episode standard errors describe independent-episode weighted means; they do not resolve censoring, missing-workout selection or dependence. Sparse pools retain descriptive estimates with explicit insufficient-support status.',))
