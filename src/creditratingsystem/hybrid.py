"""Held-out stacking and explicit aggregation of compatible PD estimates."""
from copy import deepcopy
from datetime import timedelta,date
import calendar,math
import numpy as np
from scipy.special import expit,logit
from .core import _result,_fit_result,_model_artifact,_numeric,_fail,_fingerprint,_context_value
from .data import _date
from ._expert_common import _ex_object,_ex_ids,_ex_number,_ex_target


def _hy_rows(x,name):
    if not isinstance(x,(list,tuple))or not x or any(not isinstance(r,dict)for r in x):_fail('HYBRID_CONTRACT',name+' requires nonempty records')
    return x


def _hy_dates(as_of,known_at,context=None):
    a=_date(as_of,'as_of');k=_date(known_at,'known_at');ctx=_context_value(context)
    if k>a:_fail('HYBRID_LOOKAHEAD','Forecast information cannot be known after forecast date')
    for key,value in [('as_of',as_of),('known_at',known_at)]:
        if ctx.get(key)is not None and ctx[key]!=value:_fail('TARGET_MISMATCH','Signal dates differ from context')
    return a,k


def combine_pd_signals(pd_signals,weights,*,combination_scale,combination_contract,context=None):
    """Pool estimates or conditionally mix regimes; never form an event union."""
    signals=_hy_rows(pd_signals,'pd_signals');ids=_ex_ids([r.get('signal_id')for r in signals],'signal_ids');contract=_ex_object(combination_contract,'combination_contract');target=_ex_target(signals[0].get('target'),context);rows=_ex_ids(signals[0].get('row_ids'),'row_ids');n=len(rows);m=len(signals);values=[]
    for signal in signals:
        if signal.get('target')!=target or signal.get('row_ids')!=rows or signal.get('as_of')!=signals[0].get('as_of')or signal.get('known_at')!=signals[0].get('known_at'):_fail('HYBRID_ALIGNMENT','Signals need identical target, ordered row IDs and information dates')
        _hy_dates(signal.get('as_of'),signal.get('known_at'),context)
        for key in ('model_id','model_version'):
            if not isinstance(signal.get(key),str)or not signal[key]:_fail('HYBRID_CONTRACT',key+' required for every signal')
        p=_numeric(signal.get('pd'),'pd',vector_only=True)
        if p.shape!=(n,)or np.any((p<0)|(p>1)):_fail('HYBRID_PROBABILITY','One valid probability per aligned row required')
        values.append(p)
    p=np.column_stack(values);w=_numeric(weights,'weights');interpretation={'linear':'estimator_pool','log_odds':'log_odds_pool','mixture':'latent_regime_mixture'}
    if combination_scale not in interpretation or contract.get('interpretation')!=interpretation.get(combination_scale)or not isinstance(contract.get('reference'),str)or not contract['reference']:_fail('HYBRID_INTERPRETATION','Explicit compatible combination interpretation and reference required')
    if combination_scale=='mixture':
        if w.shape!=(n,m)or contract.get('regime_ids')!=ids or contract.get('conditional_regime_probabilities')is not True:_fail('HYBRID_MIXTURE','Mixture requires aligned row-specific conditional regime probabilities and regime IDs')
    else:
        if w.shape!=(m,):_fail('HYBRID_WEIGHT','Pools require one global weight per signal')
        w=np.tile(w,(n,1))
    if np.any(w<0)or np.any(np.max(w,axis=1)<=0):_fail('HYBRID_WEIGHT','Nonnegative weights and positive total per row required')
    w=w/np.max(w,axis=1)[:,None];w=w/w.sum(axis=1)[:,None];status=['INTERIOR']*n;contributions=None;log_odds=None
    if combination_scale=='log_odds':
        zero=np.any((p==0)&(w>0),axis=1);one=np.any((p==1)&(w>0),axis=1)
        if np.any(zero&one):_fail('HYBRID_BOUNDARY_CONFLICT','Positive-weight zero and one signals give undefined pooled log odds; no hidden clipping is performed')
        lp=np.zeros_like(p);inside=(p>0)&(p<1);lp[inside]=logit(p[inside]);parts=w*lp;eta=parts.sum(axis=1);combined=expit(eta);combined[zero]=0;combined[one]=1
        contributions=[[None if (p[i,j]in(0,1)and w[i,j]>0)else float(parts[i,j])for j in range(m)]for i in range(n)];log_odds=[None if zero[i]or one[i]else float(eta[i])for i in range(n)];status=['ZERO_LIMIT'if zero[i]else'ONE_LIMIT'if one[i]else'INTERIOR'for i in range(n)]
    else:combined=np.sum(w*p,axis=1);contributions=(w*p).tolist()
    between=np.sum(w*(p-combined[:,None])**2,axis=1);within=np.sum(w*p*(1-p),axis=1)
    return _result(dict(pd=combined.tolist(),row_ids=rows,signal_ids=ids,combination_scale=combination_scale,combination_contract=deepcopy(contract),normalized_weights=w.tolist(),contributions=contributions,contribution_scale='log_odds'if combination_scale=='log_odds'else'probability',combined_log_odds=log_odds,boundary_status=status,weighted_signal_dispersion=between.tolist(),mixture_within_regime_variance=within.tolist()if combination_scale=='mixture'else None,mixture_between_regime_variance=between.tolist()if combination_scale=='mixture'else None,target=target,as_of=signals[0]['as_of'],known_at=signals[0]['known_at'],signal_fingerprints=[_fingerprint(r)for r in signals]),
        'hybrid.combine_pd_signals','probability',context,limitations=('Estimates refer to the same event, population rows and horizon; aggregation is not a union of default events. Mixture weights are supplied conditional regime probabilities; pool weights express an explicit modeling choice. Signal dispersion is not a confidence interval or evidence of independent information.',))


def _hy_horizon_end(start,target):
    d=_date(start,'as_of');h=target['horizon'];unit=target['time_unit']
    if unit=='days':
        if h!=int(h):_fail('HYBRID_HORIZON','Date-based OOF windows require integer days')
        return d+timedelta(days=int(h))
    months=h*(12 if unit=='years'else 1)
    if months!=int(months):_fail('HYBRID_HORIZON','Calendar windows require an integer number of months')
    count=d.year*12+d.month-1+int(months);year,month=divmod(count,12);month+=1
    if not 1<=year<=9999:_fail('HYBRID_HORIZON','Horizon outside supported calendar range')
    return date(year,month,min(d.day,calendar.monthrange(year,month)[1]))


def _hy_oof_contract(split_plan,base_model_ids,target,training,outcomes):
    plan=_ex_object(split_plan,'split_plan');rows=_hy_rows(plan.get('rows'),'split rows');folds=_hy_rows(plan.get('folds'),'folds');ids=_ex_ids([r.get('row_id')for r in rows],'row_ids');_ex_ids([f.get('fold_id')for f in folds],'fold_ids');index={v:i for i,v in enumerate(ids)};cutoff=_date(training.get('known_at'),'training known_at')
    if plan.get('horizon_date_rule')!='calendar_clamp':_fail('HYBRID_HORIZON','Declare calendar_clamp horizon rule: integer days or months; month-end day is clamped')
    for row in rows:
        for key in ('entity_id','group_id'):
            if not isinstance(row.get(key),str)or not row[key]:_fail('HYBRID_SPLIT',key+' required')
        if row.get('role')not in ('development','calibration','test'):_fail('HYBRID_SPLIT','Declare development/calibration/test role')
        _date(row.get('as_of'),'as_of')
    for left,right in [('development','calibration'),('development','test'),('calibration','test')]:
        for key in ('entity_id','group_id'):
            if {r[key]for r in rows if r['role']==left}&{r[key]for r in rows if r['role']==right}:_fail('HYBRID_LEAKAGE','Outer role entity/group sets overlap')
    def mature(i,at):
        row=rows[i];value=outcomes[i]
        if value not in (0,1)or isinstance(value,bool):_fail('HYBRID_OUTCOME','Used OOF/train labels must be binary numeric')
        origin=_date(row['as_of'],'as_of');known=_date(row.get('label_known_at'),'label_known_at');end=_hy_horizon_end(row['as_of'],target)
        if known<origin or known>at or end>at:_fail('HYBRID_MATURITY','Used cohort must be horizon-mature and label known at fit cutoff')
        if value==0 and _date(row.get('complete_through'),'complete_through')<end:_fail('HYBRID_MATURITY','Negative labels require declared complete-through coverage for the full window')
    heldout=set();fold_trace=[]
    for fold in folds:
        train=_ex_ids(fold.get('train_row_ids'),'train_row_ids');validation=_ex_ids(fold.get('validation_row_ids'),'validation_row_ids')
        if set(train+validation)-set(ids)or set(train)&set(validation):_fail('HYBRID_SPLIT','Train and validation row IDs must be known and disjoint')
        tr=[index[v]for v in train];va=[index[v]for v in validation]
        if any(rows[i]['role']!='development'for i in tr+va)or heldout&set(va):_fail('HYBRID_LEAKAGE','Only development rows may enter folds; each held-out row is predicted once')
        for key in ('entity_id','group_id'):
            if {rows[i][key]for i in tr}&{rows[i][key]for i in va}:_fail('HYBRID_LEAKAGE','Fold train/held-out entity or group overlap')
        first=min(_date(rows[i]['as_of'],'as_of')for i in va);fit_at=_date(fold.get('fit_known_at'),'fit_known_at')
        if fit_at>=first:_fail('HYBRID_LEAKAGE','Fold fit must precede every held-out forecast')
        for i in tr:mature(i,fit_at)
        for i in va:
            mature(i,cutoff)
            if _date(rows[i].get('prediction_known_at'),'prediction_known_at')>_date(rows[i]['as_of'],'as_of'):_fail('HYBRID_LEAKAGE','Held-out forecast was not known at its forecast date')
        lineage=_ex_object(fold.get('base_lineage'),'base_lineage')
        if set(lineage)!=set(base_model_ids):_fail('HYBRID_LINEAGE','Every base model needs fold-specific lineage')
        for base in base_model_ids:
            record=_ex_object(lineage[base],'base lineage')
            if record.get('model_id')!=base or record.get('target')!=target or record.get('prediction_origin')!='HELD_OUT':_fail('HYBRID_LINEAGE','Base predictions must declare matching target and held-out origin')
            for key in ('model_version','training_fingerprint','reference'):
                if not isinstance(record.get(key),str)or not record[key]:_fail('HYBRID_LINEAGE',key+' required')
            for key in ('fit_row_ids','preprocessing_row_ids','selection_row_ids'):
                used=_ex_ids(record.get(key),key,empty=key=='selection_row_ids')
                if set(used)-set(train):_fail('HYBRID_LEAKAGE','Base fit/preprocessing/selection used rows outside fold training')
            base_known=_date(record.get('known_at'),'base known_at')
            if base_known>fit_at or any(base_known>_date(rows[i]['prediction_known_at'],'prediction_known_at')for i in va):_fail('HYBRID_LEAKAGE','Base artifact unavailable at declared fit or prediction cutoff')
            for row_id in set(record['fit_row_ids']+record['selection_row_ids']):mature(index[row_id],base_known)
        heldout.update(va);fold_trace.append(dict(fold_id=fold['fold_id'],train_row_ids=train,validation_row_ids=validation,lineage_fingerprint=_fingerprint(lineage),fit_known_at=fold['fit_known_at']))
    if not heldout:_fail('HYBRID_SPLIT','At least one mature held-out row required')
    return rows,sorted(heldout),fold_trace


def fit_stacked_pd(out_of_fold_base_predictions,outcomes,*,base_model_ids,split_plan,meta_model_spec,target,training,model_id,model_version,weights=None,context=None):
    """Fit a real penalized-logit meta-model exclusively on declared mature OOF rows."""
    from ._classification_models import fit_penalized_glm
    features=_ex_ids(base_model_ids,'base_model_ids');target=_ex_target(target,context);train=_ex_object(training,'training');spec=_ex_object(meta_model_spec,'meta_model_spec')
    try:x=np.array(out_of_fold_base_predictions,dtype=float);y=np.asarray(outcomes,dtype=object)
    except (ValueError,TypeError):_fail('HYBRID_DATA','Predictions need numeric matrix with null allowed only outside used OOF rows')
    if x.ndim!=2 or x.shape[1]!=len(features)or y.shape!=(len(x),):_fail('HYBRID_DATA','Aligned prediction matrix, base IDs and outcomes required')
    if not isinstance(split_plan,dict)or len(split_plan.get('rows',[]))!=len(x):_fail('HYBRID_DATA','Split rows differ from prediction rows')
    rows,used,trace=_hy_oof_contract(split_plan,features,target,train,y)
    p=x[used];labels=np.asarray(y[used],dtype=float)
    if not np.all(np.isfinite(p))or np.any((p<0)|(p>1)):_fail('HYBRID_PROBABILITY','Used OOF predictions must be finite probabilities')
    transform=spec.get('input_scale')
    if transform not in ('probability','log_odds'):_fail('HYBRID_META_MODEL','input_scale must be probability or log_odds')
    if transform=='log_odds'and np.any((p==0)|(p==1)):_fail('HYBRID_BOUNDARY','Log-odds stacking requires interior OOF probabilities; no implicit clipping')
    design=logit(p)if transform=='log_odds'else p;active=np.flatnonzero(np.ptp(design,axis=0)>0);constant=np.flatnonzero(np.ptp(design,axis=0)==0)
    w=None
    if weights is not None:
        raw=np.asarray(weights,dtype=float)
        if raw.shape!=(len(x),):_fail('HYBRID_WEIGHT','Weights must align with all input rows')
        w=_numeric(raw[used],'used weights',vector_only=True)
        if np.any(w<0)or not np.any(w>0):_fail('HYBRID_WEIGHT','Used weights need nonnegative values and positive support')
    support=np.ones(len(used),dtype=bool)if w is None else w>0
    active=np.flatnonzero(np.ptp(design[support],axis=0)>0);constant=np.flatnonzero(np.ptp(design[support],axis=0)==0)
    allowed={'input_scale','penalty','lambda_path','selected_lambda','l1_ratio','fit_intercept','scale','max_iter','tolerance'}
    if set(spec)-allowed or any(k not in spec for k in ('penalty','lambda_path','selected_lambda','fit_intercept','scale')):_fail('HYBRID_META_MODEL','Declare actual penalized-logit specification without unsupported settings')
    metadata=deepcopy(train);metadata['data_fingerprint']=_fingerprint(dict(X=design[:,active].tolist(),outcomes=labels.tolist(),row_ids=[rows[i]['row_id']for i in used],weights=None if w is None else w.tolist()))
    kwargs={k:v for k,v in spec.items()if k!='input_scale'}
    fitted=fit_penalized_glm(design[:,active],labels,feature_order=[features[i]for i in active],weights=w,target=target,training=metadata,model_id=model_id+'::meta',model_version=model_version,context=context,**kwargs)
    diagnostics=dict(meta_row_ids=[rows[i]['row_id']for i in used],ignored_row_count=len(x)-len(used),fold_trace=trace,meta_data_fingerprint=metadata['data_fingerprint'],constant_base_ids=[features[i]for i in constant],repeated_meta_entities=len(used)-len({rows[i]['entity_id']for i in used}),repeated_meta_groups=len(used)-len({rows[i]['group_id']for i in used}),meta_fit=fitted.diagnostics,uncertainty='NOT_ESTIMATED_POST_SELECTION',lineage_claim='DECLARED_HELD_OUT_PROVENANCE_CHECKED_NOT_AUTHENTICATED')
    model=_model_artifact(dict(schema_version='1.0',model_id=model_id,model_version=model_version,method_id='hybrid.fit_stacked_pd',feature_order=features,parameters=dict(kind='stacked_pd',input_scale=transform,active_indices=active.tolist(),constant_indices=constant.tolist(),constant_training_probabilities=p[np.flatnonzero(support)[0],constant].tolist(),meta_model=fitted.model),preprocessing=dict(training_only=True,constant_rule='EXACT_CONSTANT_ON_USED_OOF_ROWS'),target=target,training=deepcopy(train),diagnostics=diagnostics,provenance=dict(data_origin=train.get('data_origin','USER_SUPPLIED'),parameter_origin='ESTIMATED')))
    return _fit_result(model,'hybrid.fit_stacked_pd',diagnostics=diagnostics,context=context)


def _predict_stacked_pd(fitted_model,X,*,feature_order,expected_target,context=None):
    from .classification import predict_pd
    model=_model_artifact(fitted_model)
    if model['method_id']!='hybrid.fit_stacked_pd' or list(feature_order)!=model['feature_order']or expected_target!=model['target']:_fail('HYBRID_MODEL','Frozen stacking model, base order and expected target must agree')
    _ex_target(model['target'],context);x=_numeric(X,'X');parameters=model['parameters']
    if x.ndim!=2 or x.shape[1]!=len(feature_order)or np.any((x<0)|(x>1)):_fail('HYBRID_PROBABILITY','Aligned probability matrix required at prediction')
    if parameters['input_scale']=='log_odds'and np.any((x==0)|(x==1)):_fail('HYBRID_BOUNDARY','Log-odds stacking requires interior probabilities')
    design=logit(x)if parameters['input_scale']=='log_odds'else x;indices=parameters['active_indices'];meta=parameters['meta_model'];prediction=predict_pd(meta,design[:,indices],feature_order=[feature_order[i]for i in indices],expected_target=expected_target,context=context)
    changed=int(np.sum(np.any(x[:,parameters['constant_indices']]!=np.asarray(parameters['constant_training_probabilities']),axis=1)))if parameters['constant_indices']else 0
    return _result(prediction.value,'classification.predict_pd','probability',context,diagnostics=dict(model_id=model['model_id'],model_version=model['model_version'],prediction_family='stacked_pd',constant_base_departure_rows=changed,meta_model_id=meta['model_id']),provenance=model['provenance'],limitations=('Frozen base probability order and target must match training. Constant base signals receive no empirically fitted slope; departures from their training value are flagged. Meta coefficients are regression coefficients, not automatically convex or causal weights.',))

from ._hybrid_graph import evaluate_model_graph, decompose_hybrid_prediction
