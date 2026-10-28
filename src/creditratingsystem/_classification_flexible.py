"""Frozen smooth and tree classifiers with explicit training designs."""
import numpy as np
from scipy.special import expit
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from .core import _fail,_numeric
from .classification import _design
from ._classification_models import _cm_y,_cm_model,_cm_number,_cm_controls


def _gam_spline(v,spec,derivative=0):
    knots=np.asarray(spec['full_knots']);degree=int(spec['degree']);nb=len(knots)-degree-1
    spl=BSpline(knots,np.eye(nb),degree,extrapolate=False)
    lo,hi=knots[degree],knots[-degree-1];clamped=np.clip(v,lo,hi);basis=spl(clamped,nu=derivative)
    outside=(v<lo)|(v>hi)
    if np.any(outside):
        if spec['extrapolation']=='error':_fail('OUTSIDE_SUPPORT','Spline predictor outside frozen boundary knots')
        if spec['extrapolation']=='constant':
            if derivative:basis[outside]=0
        elif derivative==0:basis+=((v-clamped)[:,None])*spl(clamped,nu=1)
        elif derivative>1:basis[outside]=0
    transform=np.asarray(spec['coefficient_transform']);return basis@transform


def _gam_design(x,parameters,features,derivative_feature=None):
    n=len(x);columns=[np.ones((n,1)) if derivative_feature is None else np.zeros((n,1))]
    for name in parameters['linear_features']:
        j=features.index(name);columns.append(x[:,j:j+1] if derivative_feature is None else np.full((n,1),float(name==derivative_feature)))
    for s in parameters['smooths']:
        j=features.index(s['feature'])
        columns.append(_gam_spline(x[:,j],s,0 if derivative_feature is None else 1) if derivative_feature in (None,s['feature']) else np.zeros((n,len(s['coefficient_transform'][0]))))
    return np.concatenate(columns,axis=1)



def _gam_recession_check(d,y,penalty,bounds):
    from scipy.optimize import linprog
    values,vectors=np.linalg.eigh(penalty);positive=values>max(float(values.max())*1e-10,0.)
    scaling=np.maximum(np.max(abs(d),axis=0),1.);signed=(2*y-1)[:,None]*d/scaling
    equality=vectors[:,positive].T/scaling
    result=linprog(-signed.mean(axis=0),A_ub=-signed,b_ub=np.zeros(len(y)),A_eq=equality if len(equality) else None,b_eq=np.zeros(len(equality)) if len(equality) else None,bounds=[(-1,1) if b[0] is None else (0,1) for b in bounds],method='highs')
    if not result.success:_fail('NUMERICAL_FAILURE','GAM null-penalty recession LP failed')
    objective=float(np.mean(signed@result.x))
    if objective>1e-8:_fail('SEPARATION','An unpenalized feasible GAM direction separates the outcomes; finite penalized optimum is unavailable')
    return objective

def fit_gam_pd(X_train,y_train,*,feature_order,linear_features,smooth_specs,variant,max_iter=2000,tolerance=1e-9,target,training,model_id,model_version,context=None):
    """Penalized cubic B-spline logistic GAM with optional genuine monotonicity.

    Each smooth spec supplies feature, boundary_knots=[lo,hi], interior_knots,
    degree (1..3), smoothing>=0, monotonicity none/increasing/decreasing and
    extrapolation error/constant/linear. Monotone spline coefficient increments
    are constrained nonnegative during fitting, never repaired after fitting.
    """
    x,_=_design(X_train,feature_order,True);y=_cm_y(y_train,len(x));max_iter,tol=_cm_controls(max_iter,tolerance)
    if variant not in ('unconstrained','monotone'): _fail('INVALID_INPUT','GAM variant unconstrained/monotone required')
    linear=list(linear_features);smooths=[];names=linear.copy();penalties=[];bounds=[(None,None)]*(1+len(linear));has_constraint=False
    for spec in smooth_specs:
        if not isinstance(spec,dict):_fail('INVALID_INPUT','Every smooth spec must be a mapping')
        name=spec.get('feature');names.append(name);degree=spec.get('degree',3)
        if isinstance(degree,bool) or degree not in (1,2,3):_fail('INVALID_INPUT','Spline degree must be 1,2,3')
        boundary=_numeric(spec.get('boundary_knots'),'boundary_knots',vector_only=True);interior=_numeric(spec.get('interior_knots',[]),'interior_knots',vector_only=True,allow_empty=True)
        if len(boundary)!=2 or boundary[0]>=boundary[1] or np.any(np.diff(interior)<=0) or np.any((interior<=boundary[0])|(interior>=boundary[1])):_fail('INVALID_INPUT','Strictly ordered interior knots must lie within two increasing boundaries')
        mono=spec.get('monotonicity','none');extra=spec.get('extrapolation','error');smoothing=_cm_number(spec.get('smoothing'),'smoothing')
        if mono not in ('none','increasing','decreasing') or extra not in ('error','constant','linear'):_fail('INVALID_INPUT','Invalid monotonicity/extrapolation')
        if variant=='unconstrained' and mono!='none':_fail('INVALID_INPUT','Unconstrained variant cannot declare monotone terms')
        knots=np.r_[np.repeat(boundary[0],degree+1),interior,np.repeat(boundary[1],degree+1)];nb=len(knots)-degree-1
        transform=np.vstack((np.zeros((1,nb-1)),np.eye(nb-1))) if mono=='none' else (1 if mono=='increasing' else -1)*np.tril(np.ones((nb,nb-1)),-1)
        penalty=np.diff(np.eye(nb),n=2,axis=0)@transform if nb>2 else np.zeros((0,nb-1))
        penalties.append(smoothing*penalty.T@penalty);bounds.extend([(None,None) if mono=='none' else (0,None)]*(nb-1));has_constraint|=mono!='none'
        smooths.append(dict(feature=name,degree=degree,full_knots=knots.tolist(),boundary_knots=boundary.tolist(),interior_knots=interior.tolist(),monotonicity=mono,extrapolation=extra,smoothing=smoothing,coefficient_transform=transform.tolist()))
    if len(names)!=len(set(names)) or set(names)!=set(feature_order):_fail('FEATURE_MISMATCH','Each input feature must appear exactly once as linear or smooth')
    if variant=='monotone' and not has_constraint:_fail('INVALID_INPUT','Monotone variant requires a genuinely constrained smooth')
    params=dict(kind='gam',variant=variant,linear_features=linear,smooths=smooths);d=_gam_design(x,params,list(feature_order));penalty=np.zeros((d.shape[1],d.shape[1]));offset=1+len(linear)
    for p in penalties:penalty[offset:offset+len(p),offset:offset+len(p)]=p;offset+=len(p)
    if np.linalg.matrix_rank(d)<d.shape[1] and np.linalg.matrix_rank(d.T@d+penalty)<d.shape[1]:_fail('RANK_DEFICIENT','Spline likelihood plus penalty is not identifiable')
    recession=_gam_recession_check(d,y,penalty,bounds)
    def objective(beta):
        eta=d@beta
        return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*beta@penalty@beta),d.T@(expit(eta)-y)/len(y)+penalty@beta
    fitted=minimize(objective,np.zeros(d.shape[1]),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=max_iter,gtol=tol,ftol=tol*1e-4,maxls=50))
    value,grad=objective(fitted.x);free=np.array([b[0] is None or fitted.x[i]>max(tol,1e-8) for i,b in enumerate(bounds)]);kkt=np.where(free,abs(grad),np.maximum(-grad,0));residual=float(kkt.max())
    if not np.isfinite(value) or residual>max(1e-6,np.sqrt(tol)) or (fitted.nit>=max_iter and not fitted.success):_fail('NON_CONVERGENCE','GAM projected score/KKT residual exceeds tolerance',{'kkt_residual':residual})
    prob=expit(d@fitted.x);h=d.T@((prob*(1-prob)/len(y))[:,None]*d);hf=(h+penalty)[np.ix_(free,free)]
    try:inverse=np.linalg.inv(hf)
    except np.linalg.LinAlgError:_fail('NON_IDENTIFIABLE','Active GAM penalized information is singular')
    edf=float(np.trace(inverse@h[np.ix_(free,free)]));params['coefficients']=fitted.x.tolist()
    diag=dict(converged=True,null_penalty_separation_objective=recession,null_penalty_rank_tolerance=1e-10,iterations=int(fitted.nit),objective=value,kkt_residual=residual,effective_degrees_of_freedom=edf,active_constraint_count=int((~free).sum()),smoothing_selection='CALLER_DECLARED_TRAINING_ONLY',uncertainty='NOT_ESTIMATED_CONSTRAINED_OR_SMOOTHING_SELECTION',training_only=True)
    return _cm_model('fit_gam_pd',feature_order,params,dict(transformation='FROZEN_B_SPLINE_DESIGN'),diag,target,training,model_id,model_version,context)


def _gam_raw(model,X,feature_order,derivative_feature=None):
    x,_=_design(X,feature_order,True)
    return _gam_design(x,model['parameters'],list(feature_order),derivative_feature)@np.asarray(model['parameters']['coefficients'])


def _forest_predict_tree(tree,x):
    out=np.empty(len(x))
    for i,row in enumerate(x):
        node=tree
        while 'leaf_probability' not in node:node=node['left'] if row[int(node['feature_index'])]<=node['threshold'] else node['right']
        out[i]=node['leaf_probability']
    return out


def fit_random_forest_pd(X_train,y_train,*,feature_order,sampling_plan,max_depth,min_samples_leaf=1,min_weight_leaf=0.,criterion='gini',weights=None,target,training,model_id,model_version,context=None):
    """Weighted CART forest with completely explicit bootstrap and feature draws.

    sampling_plan is a list of trees: sample_indices (zero-based, repetitions
    represent bootstrap draws), node_features mapping binary paths ('root','L',
    'LR',...) to ordered distinct zero-based candidate features. Every splittable
    visited node must declare its candidate features; deterministic gain ties use
    that order, then increasing threshold. Unvisited candidate entries are unused.
    """
    from .classification import _vector
    x,_=_design(X_train,feature_order,False);y=_cm_y(y_train,len(x))
    if criterion not in ('gini','log_loss'):_fail('INVALID_INPUT','criterion must be gini or log_loss')
    if isinstance(max_depth,bool) or not isinstance(max_depth,(int,np.integer)) or not 1<=max_depth<=20:_fail('INVALID_INPUT','max_depth must be integer 1..20')
    if isinstance(min_samples_leaf,bool) or not isinstance(min_samples_leaf,(int,np.integer)) or min_samples_leaf<1:_fail('INVALID_INPUT','min_samples_leaf must be positive integer')
    minw=_cm_number(min_weight_leaf,'min_weight_leaf');w=_vector(weights,len(x),'weights',1)
    if np.any(w<0) or any(w[y==k].sum()<=0 for k in (0,1)):_fail('INVALID_WEIGHTS','Nonnegative weights with positive support in both classes required')
    if not isinstance(sampling_plan,(list,tuple)) or not sampling_plan:_fail('INVALID_INPUT','Nonempty explicit forest sampling_plan is required')
    def impurity(p):
        return 2*p*(1-p) if criterion=='gini' else (-p*np.log(p)-(1-p)*np.log1p(-p) if 0<p<1 else 0.)
    trees=[];oob_total=np.zeros(len(x));oob_count=np.zeros(len(x),dtype=int);used_plans=[]
    for draw in sampling_plan:
        indices=_numeric(draw.get('sample_indices'),'sample_indices',vector_only=True)
        if np.any(indices!=np.floor(indices)) or np.any((indices<0)|(indices>=len(x))):_fail('INVALID_SAMPLING_PLAN','Bootstrap indices must be zero-based training row indices')
        indices=indices.astype(int);features=draw.get('node_features')
        if not isinstance(features,dict) or w[indices].sum()<=0:_fail('INVALID_SAMPLING_PLAN','Node feature candidates and positive sampled weight required')
        visited={}
        def build(rows,depth,path):
            ww=w[rows];total=float(ww.sum());positive=float(ww@y[rows]);prob=positive/total
            leaf=dict(leaf_probability=prob,weighted_count=total,sample_count=len(rows),default_weight=positive)
            if depth==max_depth or len(rows)<2*min_samples_leaf or prob in (0.,1.):return leaf
            key=path or 'root';candidates=features.get(key)
            if candidates is None:_fail('INVALID_SAMPLING_PLAN','Missing feature candidate draw for visited node '+key)
            cand=_numeric(candidates,'node_features',vector_only=True)
            if np.any(cand!=np.floor(cand)) or np.any((cand<0)|(cand>=x.shape[1])) or len(set(cand))!=len(cand):_fail('INVALID_SAMPLING_PLAN','Each node needs distinct valid zero-based feature candidates')
            visited[key]=cand.astype(int).tolist();best=None;bestgain=0.;parent=impurity(prob)
            for j in cand.astype(int):
                values=np.unique(x[rows,j]);thresholds=values[:-1]+(values[1:]-values[:-1])/2
                for threshold in thresholds:
                    left=x[rows,j]<=threshold;nl=int(left.sum());nr=len(rows)-nl
                    if min(nl,nr)<min_samples_leaf:continue
                    wl=float(ww[left].sum());wr=total-wl
                    if wl<=0 or wr<=0 or min(wl,wr)<minw:continue
                    pl=float(ww[left]@y[rows[left]])/wl;pr=(positive-wl*pl)/wr
                    gain=parent-(wl*impurity(pl)+wr*impurity(pr))/total
                    if gain>bestgain+1e-14:bestgain=gain;best=(int(j),float(threshold),left)
            if best is None:return leaf
            j,threshold,left=best
            return dict(feature_index=j,threshold=threshold,gain=bestgain,weighted_count=total,sample_count=len(rows),left=build(rows[left],depth+1,path+'L'),right=build(rows[~left],depth+1,path+'R'))
        tree=build(indices,0,'');trees.append(tree);used_plans.append(dict(sample_indices=indices.tolist(),node_features=visited));oob=np.ones(len(x),dtype=bool);oob[indices]=False;oob_total[oob]+=_forest_predict_tree(tree,x[oob]);oob_count[oob]+=1
    has=oob_count>0;oob=np.full(len(x),np.nan);oob[has]=oob_total[has]/oob_count[has]
    valid=has&(w>0);brier=float(np.average((oob[valid]-y[valid])**2,weights=w[valid])) if valid.any() else None
    diag=dict(training_only=True,tree_count=len(trees),sampling_plan=used_plans,oob_counts=oob_count.tolist(),oob_probabilities=[None if np.isnan(v) else float(v) for v in oob],oob_brier=brier,oob_coverage=float(has.mean()),oob_assumption='Rows independent; OOB is not a temporal or obligor-group holdout',calibration='UNCALIBRATED_EMPIRICAL_LEAF_PROBABILITIES',uncertainty='NOT_ESTIMATED',split_tie_policy='CANDIDATE_ORDER_THEN_INCREASING_THRESHOLD_GAIN_TOL_1E-14')
    params=dict(kind='random_forest',trees=trees,aggregation='ARITHMETIC_MEAN_PROBABILITY',max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),min_weight_leaf=minw,criterion=criterion)
    return _cm_model('fit_random_forest_pd',feature_order,params,dict(transformation='identity'),diag,target,training,model_id,model_version,context)


def _forest_probability(model,X,feature_order):
    x,_=_design(X,feature_order,False)
    return np.mean([_forest_predict_tree(t,x) for t in model['parameters']['trees']],axis=0)


def _boost_tree(node,row):
    while 'leaf_value' not in node:node=node['left'] if row[int(node['feature_index'])]<node['threshold'] else node['right']
    return node['leaf_value']


def _boost_raw(model,X,feature_order):
    x,_=_design(X,feature_order,False)
    if np.any(abs(x)>np.finfo(np.float32).max):_fail('NUMERICAL_FAILURE','Boosting inputs must be representable in float32')
    x=x.astype(np.float32);p=model['parameters']
    return np.array([p['base_margin']+sum(_boost_tree(t,row) for t in p['trees']) for row in x])


def fit_gradient_boosting_pd(X_train,y_train,*,feature_order,train_row_ids,iterations,max_depth,learning_rate,monotonic_constraints=None,l2_regularization=1.,min_child_weight=1.,min_split_loss=0.,max_bin=256,inner_validation=None,early_stopping_rounds=None,target,training,model_id,model_version,context=None):
    """CPU histogram Bernoulli tree boosting, frozen float32 split semantics.

    No stochastic subsampling. Early stopping reads only an explicitly marked
    disjoint inner_validation split with its own row_ids and information cutoff.
    Dumped leaf weights already contain learning-rate shrinkage.
    """
    import json
    from datetime import datetime
    import xgboost as xgb
    x,_=_design(X_train,feature_order,False);y=_cm_y(y_train,len(x))
    for value,name,lo in [(iterations,'iterations',1),(max_depth,'max_depth',1),(max_bin,'max_bin',2)]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<lo:_fail('INVALID_INPUT',name+' must be integer >= '+str(lo))
    rate=_cm_number(learning_rate,'learning_rate',strict=True)
    if rate>1:_fail('INVALID_INPUT','learning_rate must not exceed one')
    reg=_cm_number(l2_regularization,'l2_regularization');mcw=_cm_number(min_child_weight,'min_child_weight');gamma=_cm_number(min_split_loss,'min_split_loss')
    ids=list(train_row_ids)
    if len(ids)!=len(x) or len(set(ids))!=len(ids) or any(not isinstance(v,str) or not v for v in ids):_fail('INVALID_SPLIT','Distinct nonempty training row_ids are required')
    constraints=np.zeros(x.shape[1],dtype=int) if monotonic_constraints is None else _numeric(monotonic_constraints,'monotonic_constraints',vector_only=True)
    if len(constraints)!=x.shape[1] or not np.isin(constraints,[-1,0,1]).all():_fail('INVALID_INPUT','One -1/0/1 monotonic constraint per feature required')
    if early_stopping_rounds is not None and (isinstance(early_stopping_rounds,bool) or not isinstance(early_stopping_rounds,(int,np.integer)) or early_stopping_rounds<1):_fail('INVALID_INPUT','Early stopping rounds must be positive integer')
    if early_stopping_rounds is not None and inner_validation is None:_fail('INVALID_SPLIT','Early stopping requires explicit inner validation')
    if np.any(abs(x)>np.finfo(np.float32).max):_fail('NUMERICAL_FAILURE','Boosting inputs must be representable in float32')
    train=xgb.DMatrix(x,label=y);evals=[(train,'train')];validation_contract=None
    if inner_validation is not None:
        val=inner_validation;vx,_=_design(val.get('X'),feature_order,False);vy=_cm_y(val.get('y'),len(vx),require_both=False);vid=val.get('row_ids',[])
        if len(vid)!=len(vx) or len(set(vid))!=len(vid) or set(vid)&set(ids) or any(not isinstance(v,str) or not v for v in vid):_fail('INVALID_SPLIT','Inner validation IDs must be unique and disjoint from training')
        if val.get('role')!='inner_validation' or not val.get('split_id') or val['split_id']==training.get('split_id'):_fail('INVALID_SPLIT','Explicit distinct inner_validation role and split_id required')
        try:future=datetime.fromisoformat(val['known_at'].replace('Z','+00:00'))>datetime.fromisoformat(training['known_at'].replace('Z','+00:00'))
        except (KeyError,TypeError,ValueError):_fail('INVALID_SPLIT','Comparable ISO information cutoffs required')
        if future:_fail('LOOKAHEAD','Inner validation labels cannot be known after model training cutoff')
        if np.any(abs(vx)>np.finfo(np.float32).max):_fail('NUMERICAL_FAILURE','Boosting validation inputs must be representable in float32')
        evals.append((xgb.DMatrix(vx,label=vy),'inner_validation'));validation_contract={k:val[k] for k in ('row_ids','role','split_id','known_at')}
    base=float(y.mean());params=dict(objective='binary:logistic',eval_metric='logloss',tree_method='hist',nthread=1,seed=0,max_depth=int(max_depth),eta=rate,reg_lambda=reg,min_child_weight=mcw,gamma=gamma,max_bin=int(max_bin),base_score=base,subsample=1.,colsample_bytree=1.,monotone_constraints='('+','.join(str(int(v)) for v in constraints)+')')
    history={};backend=xgb.train(params,train,num_boost_round=int(iterations),evals=evals,evals_result=history,early_stopping_rounds=early_stopping_rounds,verbose_eval=False)
    selected=int(backend.best_iteration)+1 if early_stopping_rounds is not None else int(iterations)
    def convert(node):
        if 'leaf' in node:return dict(leaf_value=float(node['leaf']))
        index=int(str(node['split']).removeprefix('f'));children={c['nodeid']:c for c in node['children']}
        return dict(feature_index=index,threshold=float(node['split_condition']),left=convert(children[node['yes']]),right=convert(children[node['no']]))
    trees=[convert(json.loads(v)) for v in backend.get_dump(dump_format='json')[:selected]]
    p=dict(kind='gradient_boosting',trees=trees,base_margin=float(np.log(base/(1-base))),base_probability=base,learning_rate=rate,leaf_values_include_shrinkage=True,input_cast='float32',split_rule='STRICT_LESS_THAN',monotonic_constraints=constraints.astype(int).tolist(),max_depth=int(max_depth),l2_regularization=reg,min_child_weight=mcw,min_split_loss=gamma,max_bin=int(max_bin))
    diag=dict(training_only=True,backend='XGBOOST_CPU_HIST',backend_version=xgb.__version__,selected_iterations=selected,requested_iterations=int(iterations),evaluated_iterations=len(history['train']['logloss']),loss_path=history,inner_validation=validation_contract,early_stopping_rounds=early_stopping_rounds,uncertainty='NOT_ESTIMATED',calibration='BERNOULLI_MODEL_NOT_HOLDOUT_CALIBRATED',train_row_ids=ids)
    return _cm_model('fit_gradient_boosting_pd',feature_order,p,dict(transformation='FLOAT32_CAST'),diag,target,training,model_id,model_version,context)
