"""Portable CPU neural credit models with explicit gradients and training order."""
import copy
import numpy as np
from scipy.special import expit
from .core import _fail,_numeric,_fingerprint
from .classification import _design
from ._classification_models import _cm_y,_cm_model,_cm_number


def _nn_activation(z,name):
    return np.tanh(z) if name=='tanh' else np.maximum(z,0) if name=='relu' else expit(z) if name=='sigmoid' else z


def _nn_mlp(weights,architecture,x,y=None):
    activations=architecture['activations']+['identity'];cache=[];a=x
    for k,activation in enumerate(activations):
        z=a@weights[f'layer{k}_weight']+weights[f'layer{k}_bias'];out=_nn_activation(z,activation);cache.append((a,z,out,activation));a=out
    eta=a[:,0]
    if y is None:return eta
    delta=(expit(eta)-y)[:,None]/len(y);grad={}
    for k in reversed(range(len(cache))):
        a,z,out,activation=cache[k]
        if activation=='tanh':delta=delta*(1-out*out)
        elif activation=='sigmoid':delta=delta*out*(1-out)
        elif activation=='relu':delta=delta*(z>0)
        grad[f'layer{k}_weight']=a.T@delta;grad[f'layer{k}_bias']=delta.sum(axis=0);delta=delta@weights[f'layer{k}_weight'].T
    return eta,grad


def _nn_recurrent(weights,architecture,sequences,y=None):
    variant=architecture['variant'];hidden=architecture['hidden_size'];etas=[];grad={k:np.zeros_like(v) for k,v in weights.items()}
    for index,sequence in enumerate(sequences):
        h=np.zeros(hidden);c=np.zeros(hidden);cache=[]
        for x in sequence:
            hp=h.copy();cp=c.copy();ai=x@weights['input_weight']+weights['input_bias'];ah=h@weights['hidden_weight']+weights['hidden_bias']
            if variant=='gru':
                r=expit(ai[:hidden]+ah[:hidden]);z=expit(ai[hidden:2*hidden]+ah[hidden:2*hidden]);n=np.tanh(ai[2*hidden:]+r*ah[2*hidden:]);h=(1-z)*n+z*hp;cache.append((x,hp,r,z,n,ah[2*hidden:]))
            else:
                pre=ai+ah;i=expit(pre[:hidden]);f=expit(pre[hidden:2*hidden]);g=np.tanh(pre[2*hidden:3*hidden]);o=expit(pre[3*hidden:]);c=f*cp+i*g;h=o*np.tanh(c);cache.append((x,hp,cp,i,f,g,o,c.copy()))
        eta=float(h@weights['output_weight'][:,0]+weights['output_bias'][0]);etas.append(eta)
        if y is None:continue
        delta=(expit(eta)-y[index])/len(y);grad['output_weight'][:,0]+=h*delta;grad['output_bias'][0]+=delta;dh=weights['output_weight'][:,0]*delta;dc=np.zeros(hidden)
        for values in reversed(cache):
            if variant=='gru':
                x,hp,r,z,n,ah_n=values;dn=dh*(1-z);dz=dh*(hp-n);dan=dn*(1-n*n);dar=dan*ah_n*r*(1-r);daz=dz*z*(1-z);dai=np.r_[dar,daz,dan];dah=np.r_[dar,daz,dan*r];dh=dh*z+weights['hidden_weight']@dah
            else:
                x,hp,cp,i,f,g,o,c=values;tc=np.tanh(c);dc=dc+dh*o*(1-tc*tc);dai=np.r_[dc*g*i*(1-i),dc*cp*f*(1-f),dc*i*(1-g*g),dh*tc*o*(1-o)];dah=dai;dh=weights['hidden_weight']@dah;dc=dc*f
            grad['input_weight']+=np.outer(x,dai);grad['hidden_weight']+=np.outer(hp,dah);grad['input_bias']+=dai;grad['hidden_bias']+=dah
    return np.asarray(etas) if y is None else (np.asarray(etas),grad)


def _nn_loss(eta,y):return float(np.mean(np.logaddexp(0,eta)-y*eta))


def _nn_optimize(weights,architecture,inputs,y,optimizer_spec,validation=None):
    initial_hash=_fingerprint({k:v.tolist() for k,v in weights.items()})
    spec=optimizer_spec
    if not isinstance(spec,dict) or spec.get('algorithm')!='adam':_fail('INVALID_OPTIMIZER','Explicit Adam optimizer specification required')
    rate=_cm_number(spec.get('learning_rate'),'learning_rate',strict=True);b1=_cm_number(spec.get('beta1',.9),'beta1');b2=_cm_number(spec.get('beta2',.999),'beta2');eps=_cm_number(spec.get('epsilon',1e-8),'epsilon',strict=True);l2=_cm_number(spec.get('l2',0),'l2')
    if b1>=1 or b2>=1:_fail('INVALID_OPTIMIZER','Adam beta parameters must be below one')
    batch=spec.get('batch_size');orders=spec.get('epoch_orders');patience=spec.get('early_stopping_patience');clip=spec.get('gradient_clip_norm')
    if isinstance(batch,bool) or not isinstance(batch,(int,np.integer)) or batch<1 or not isinstance(orders,(list,tuple)) or not orders:_fail('INVALID_OPTIMIZER','Positive batch size and explicit nonempty epoch order list required')
    if patience is not None and (isinstance(patience,bool) or not isinstance(patience,(int,np.integer)) or patience<1 or validation is None):_fail('INVALID_SPLIT','Early stopping patience requires explicit inner validation')
    if clip is not None:clip=_cm_number(clip,'gradient_clip_norm',strict=True)
    checked=[]
    for order in orders:
        a=_numeric(order,'epoch_order',vector_only=True)
        if len(a)!=len(y) or not np.array_equal(np.sort(a),np.arange(len(y))):_fail('INVALID_OPTIMIZER','Each epoch order must be a zero-based permutation of all training rows')
        checked.append(a.astype(int))
    forward=_nn_mlp if architecture['kind']=='mlp' else _nn_recurrent
    subset=lambda values,idx:values[idx] if isinstance(values,np.ndarray) else [values[i] for i in idx]
    def loss(values,labels,w):return _nn_loss(forward(w,architecture,values),labels)
    initial_loss=loss(inputs,y,weights);first={k:np.zeros_like(v) for k,v in weights.items()};second=copy.deepcopy(first);best=copy.deepcopy(weights);best_epoch=0;best_val=loss(*validation,weights) if validation is not None else None;stale=0;updates=0;clipped=0;history=[]
    for epoch,order in enumerate(checked,1):
        for start in range(0,len(y),batch):
            index=order[start:start+batch];eta,gradient=forward(weights,architecture,subset(inputs,index),y[index])
            for key in gradient:
                if key.endswith('_weight'):gradient[key]+=l2*weights[key]
            norm=float(np.sqrt(sum(np.sum(g*g) for g in gradient.values())))
            if not np.isfinite(norm):_fail('NUMERICAL_FAILURE','Neural gradient is nonfinite')
            if clip is not None and norm>clip:
                for key in gradient:gradient[key]*=clip/norm
                clipped+=1
            updates+=1
            for key in weights:
                g=gradient[key];first[key]=b1*first[key]+(1-b1)*g;second[key]=b2*second[key]+(1-b2)*g*g;weights[key]-=rate*(first[key]/(1-b1**updates))/(np.sqrt(second[key]/(1-b2**updates))+eps)
                if not np.isfinite(weights[key]).all():_fail('NUMERICAL_FAILURE','Neural parameter update is nonfinite')
        trainloss=loss(inputs,y,weights);valloss=loss(*validation,weights) if validation is not None else None;penalty=.5*l2*sum(float(np.sum(v*v)) for k,v in weights.items() if k.endswith('_weight'));history.append(dict(epoch=epoch,train_log_loss=trainloss,penalized_train_objective=trainloss+penalty,inner_validation_log_loss=valloss))
        if validation is not None:
            if valloss<best_val-1e-12:best_val=valloss;best=copy.deepcopy(weights);best_epoch=epoch;stale=0
            else:stale+=1
            if patience is not None and stale>=patience:break
        else:best=copy.deepcopy(weights);best_epoch=epoch
    _,gradient=forward(best,architecture,inputs,y)
    for k in gradient:
        if k.endswith('_weight'):gradient[k]+=l2*best[k]
    diagnostics=dict(initial_weights_fingerprint=initial_hash,optimization_status='EARLY_STOPPED' if len(history)<len(checked) else 'EPOCH_BUDGET_COMPLETED',initial_train_log_loss=initial_loss,loss_path=history,selected_epoch=best_epoch,executed_epochs=len(history),requested_epochs=len(checked),adam_updates=updates,clipped_updates=clipped,selected_gradient_norm=float(np.sqrt(sum(np.sum(v*v) for v in gradient.values()))),optimizer='ADAM_COUPLED_L2_WEIGHT_PENALTY',optimizer_parameters=dict(learning_rate=rate,beta1=b1,beta2=b2,epsilon=eps,l2=l2,batch_size=int(batch),gradient_clip_norm=clip),epoch_orders=[o.tolist() for o in checked[:len(history)]],early_stopping_patience=patience,global_optimum_claimed=False,uncertainty='NOT_ESTIMATED',calibration='BERNOULLI_MODEL_NOT_HOLDOUT_CALIBRATED',training_only=True)
    return best,diagnostics


def _nn_ids(ids,n):
    ids=list(ids)
    if len(ids)!=n or len(set(ids))!=n or any(not isinstance(v,str) or not v for v in ids):_fail('INVALID_SPLIT','Unique nonempty row identifiers required')
    return ids


def _nn_validation_contract(validation,ids,training):
    from datetime import datetime
    if validation is None:return None
    vi=_nn_ids(validation.get('row_ids',[]),len(validation.get('y',[])))
    if set(vi)&set(ids) or validation.get('role')!='inner_validation' or not validation.get('split_id') or validation['split_id']==training['split_id']:_fail('INVALID_SPLIT','Distinct explicit inner validation role, split and row IDs required')
    try:future=datetime.fromisoformat(validation['known_at'].replace('Z','+00:00'))>datetime.fromisoformat(training['known_at'].replace('Z','+00:00'))
    except (KeyError,TypeError,ValueError):_fail('INVALID_SPLIT','Comparable ISO information cutoffs required')
    if future:_fail('LOOKAHEAD','Inner validation information exceeds training cutoff')
    return {k:validation[k] for k in ('row_ids','role','split_id','known_at')}


def _nn_weights(initial,shapes):
    if not isinstance(initial,dict) or set(initial)!=set(shapes):_fail('INVALID_ARCHITECTURE','Initial parameter keys must match declared architecture')
    weights={}
    for key,shape in shapes.items():
        value=_numeric(initial[key],key)
        if value.shape!=shape:_fail('INVALID_ARCHITECTURE','Initial shape mismatch for '+key)
        weights[key]=value.copy()
    return weights


def fit_neural_pd(X_train,y_train,*,feature_order,architecture,initial_weights,optimizer_spec,train_row_ids,scale=True,inner_validation=None,target,training,model_id,model_version,context=None):
    """Dense MLP trained by actual analytic backpropagation and explicit Adam."""
    x,_=_design(X_train,feature_order,False);y=_cm_y(y_train,len(x));ids=_nn_ids(train_row_ids,len(x));validation_contract=_nn_validation_contract(inner_validation,ids,training)
    if not isinstance(architecture,dict):_fail('INVALID_ARCHITECTURE','Architecture mapping required')
    hidden=list(architecture.get('hidden_units',[]));activations=list(architecture.get('activations',[]))
    if len(activations)!=len(hidden) or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1 for v in hidden) or any(a not in ('tanh','relu','sigmoid') for a in activations):_fail('INVALID_ARCHITECTURE','Positive hidden units and matching tanh/relu/sigmoid activations required')
    arch=dict(kind='mlp',hidden_units=hidden,activations=activations);sizes=[x.shape[1]]+hidden+[1];shapes={}
    for k,(a,b) in enumerate(zip(sizes[:-1],sizes[1:])):shapes[f'layer{k}_weight']=(a,b);shapes[f'layer{k}_bias']=(b,)
    weights=_nn_weights(initial_weights,shapes)
    if not isinstance(scale,bool):_fail('INVALID_INPUT','scale must be boolean')
    center=x.mean(axis=0) if scale else np.zeros(x.shape[1]);scaling=x.std(axis=0) if scale else np.ones(x.shape[1]);scaling[scaling==0]=1;z=(x-center)/scaling;validation=None
    if inner_validation is not None:
        vx,_=_design(inner_validation.get('X'),feature_order,False);vy=_cm_y(inner_validation.get('y'),len(vx),require_both=False);validation=((vx-center)/scaling,vy)
    weights,diagnostics=_nn_optimize(weights,arch,z,y,optimizer_spec,validation);diagnostics.update(inner_validation=validation_contract,train_row_ids=ids)
    return _cm_model('fit_neural_pd',feature_order,dict(kind='neural',architecture=arch,weights={k:v.tolist() for k,v in weights.items()}),dict(transformation='standardize',center=center.tolist(),scale=scaling.tolist()),diagnostics,target,training,model_id,model_version,context)


def _nn_sequences(records,features,include_time_gaps,time_gap_scale):
    if not isinstance(records,(list,tuple)) or not records:_fail('INVALID_SEQUENCE','Nonempty list of sequence records required')
    output=[];metadata=[]
    for record in records:
        if not isinstance(record,dict):_fail('INVALID_SEQUENCE','Every sequence must be a mapping')
        values,_=_design(record.get('values'),features,False);times=_numeric(record.get('observation_times'),'observation_times',vector_only=True);known=_numeric(record.get('known_times'),'known_times',vector_only=True);mask=_numeric(record.get('mask'),'mask',vector_only=True);forecast=_cm_number(record.get('forecast_time'),'forecast_time',lower=-np.inf)
        if any(len(v)!=len(values) for v in (times,known,mask)) or not np.isin(mask,[0,1]).all() or np.any(known<times):_fail('INVALID_SEQUENCE','Aligned values/times/known_times/zero-one mask required; known time cannot precede observation')
        valid=(mask==1)&(times<=forecast)&(known<=forecast);t=times[valid];v=values[valid]
        if len(t)==0 or np.any(np.diff(t)<=0):_fail('INVALID_SEQUENCE','At least one causally available observation, in strictly increasing order, required')
        if include_time_gaps:v=np.c_[v,np.r_[0.,np.diff(t)]/time_gap_scale]
        output.append(v);metadata.append(dict(forecast_time=forecast,used_steps=len(v),excluded_steps=int((~valid).sum()),last_observation_time=float(t[-1])))
    return output,metadata


def fit_sequence_pd(sequences,y_train,*,feature_order,variant,hidden_size,initial_weights,optimizer_spec,train_row_ids,label_known_times,training_as_of_time,include_time_gaps=True,time_gap_scale=1.,scale=True,inner_validation=None,target,training,model_id,model_version,context=None):
    """Single-layer arbitrary-width GRU/LSTM with genuine gated BPTT.

    Sequence records contain values, observation_times, known_times, mask and
    forecast_time in target time units. Padding/future/unavailable observations
    are excluded before scaling or recurrent updates. Negative labels require
    maturity through the full horizon; observed early defaults remain positive.
    """
    if variant not in ('gru','lstm') or isinstance(hidden_size,bool) or not isinstance(hidden_size,(int,np.integer)) or hidden_size<1:_fail('INVALID_ARCHITECTURE','GRU/LSTM and positive integer hidden_size required')
    if not isinstance(include_time_gaps,bool) or not isinstance(scale,bool):_fail('INVALID_INPUT','Time-gap and scaling flags must be boolean')
    gapscale=_cm_number(time_gap_scale,'time_gap_scale',strict=True);inputs,recordmeta=_nn_sequences(sequences,list(feature_order),include_time_gaps,gapscale);y=_cm_y(y_train,len(inputs));ids=_nn_ids(train_row_ids,len(y));asof=_cm_number(training_as_of_time,'training_as_of_time',lower=-np.inf);known=_numeric(label_known_times,'label_known_times',vector_only=True)
    if len(known)!=len(y):_fail('INVALID_SEQUENCE','One label_known_time per training row required')
    forecast=np.array([r['forecast_time'] for r in recordmeta]);horizon=target.get('horizon')
    if np.any(known>asof) or np.any(known<forecast) or np.any((y==0)&(known<forecast+horizon)):_fail('LOOKAHEAD','Labels must be observed by training cutoff; nondefaults need full horizon maturity')
    arch=dict(kind='recurrent',variant=variant,hidden_size=int(hidden_size),gate_order=['r','z','n'] if variant=='gru' else ['i','f','g','o'],gru_reset='AFTER_HIDDEN_AFFINE',initial_state='ZERO_PER_SEQUENCE',include_time_gaps=include_time_gaps,time_gap_scale=gapscale)
    p=len(feature_order)+int(include_time_gaps);h=int(hidden_size);g=3 if variant=='gru' else 4;shapes=dict(input_weight=(p,g*h),hidden_weight=(h,g*h),input_bias=(g*h,),hidden_bias=(g*h,),output_weight=(h,1),output_bias=(1,));weights=_nn_weights(initial_weights,shapes)
    pooled=np.concatenate(inputs);center=pooled.mean(axis=0) if scale else np.zeros(p);scaling=pooled.std(axis=0) if scale else np.ones(p);scaling[scaling==0]=1;inputs=[(v-center)/scaling for v in inputs];validation=None;contract=_nn_validation_contract(inner_validation,ids,training)
    if inner_validation is not None:
        raw,vm=_nn_sequences(inner_validation.get('sequences'),list(feature_order),include_time_gaps,gapscale);vy=_cm_y(inner_validation.get('y'),len(raw),require_both=False);vk=_numeric(inner_validation.get('label_known_times'),'validation label_known_times',vector_only=True);vf=np.array([v['forecast_time'] for v in vm])
        if len(vk)!=len(vy) or np.any(vk>asof) or np.any(vk<vf) or np.any((vy==0)&(vk<vf+horizon)):_fail('LOOKAHEAD','Inner validation sequence labels require cutoff and horizon maturity')
        validation=([(v-center)/scaling for v in raw],vy)
    weights,diagnostics=_nn_optimize(weights,arch,inputs,y,optimizer_spec,validation);diagnostics.update(inner_validation=contract,train_row_ids=ids,sequence_information=recordmeta,label_known_times=known.tolist(),training_as_of_time=asof,temporal_contract='ONLY_OBSERVED_AND_KNOWN_BY_FORECAST; NEGATIVE_LABEL_HORIZON_MATURE')
    return _cm_model('fit_sequence_pd',feature_order,dict(kind='sequence',architecture=arch,weights={k:v.tolist() for k,v in weights.items()}),dict(transformation='standardize',center=center.tolist(),scale=scaling.tolist()),diagnostics,target,training,model_id,model_version,context)


def _neural_raw(model,X,feature_order):
    params=model['parameters'];arch=params['architecture'];weights={k:np.asarray(v) for k,v in params['weights'].items()};prep=model['preprocessing'];center=np.asarray(prep['center']);scaling=np.asarray(prep['scale'])
    if params['kind']=='neural':
        x,_=_design(X,feature_order,False);return _nn_mlp(weights,arch,(x-center)/scaling)
    # JSON numbers are language-independent; widths need integer indexing locally.
    arch=dict(arch,hidden_size=int(arch['hidden_size']));sequences,_=_nn_sequences(X,list(feature_order),arch['include_time_gaps'],arch['time_gap_scale']);return _nn_recurrent(weights,arch,[(v-center)/scaling for v in sequences])
