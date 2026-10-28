"""Portable probabilistic discriminants and explicit uncalibrated model scores."""
import math
import numpy as np
from scipy.special import expit, logsumexp
from .core import _fail, _numeric, _model_artifact, _fit_result, _result, _context_value
from .classification import _design, _vector


def _cm_model(method, features, parameters, preprocessing, diagnostics, target, training, model_id, model_version, context):
    model = _model_artifact(dict(schema_version='1.0', model_id=model_id, model_version=model_version,
        method_id='classification.'+method, feature_order=list(features), parameters=parameters,
        preprocessing=preprocessing, target=target, training=training, diagnostics=diagnostics,
        provenance=dict(data_origin=training.get('data_origin','USER_SUPPLIED'), parameter_origin='ESTIMATED')))
    return _fit_result(model, 'classification.'+method, diagnostics=diagnostics, context=context)


def _cm_y(y, n, require_both=True):
    y = _vector(y, n, 'y_train', np.nan)
    if n == 0 or not np.all(np.isin(y, [0, 1])):
        _fail('INVALID_OUTCOME', 'Training requires nonempty observed numeric zero/one labels')
    if require_both and len(np.unique(y)) != 2:
        _fail('INSUFFICIENT_CLASS_SUPPORT', 'Both outcome classes must be observed in training')
    return y.astype(int)


def _cm_priors(priors, y):
    p = np.bincount(y, minlength=2)/len(y) if priors is None else _numeric(priors, 'priors', vector_only=True)
    if p.shape != (2,) or np.any(p<=0) or abs(p.sum()-1)>1e-12:
        _fail('INVALID_PRIORS', 'priors must be positive probabilities for classes [0,1] summing to one')
    return p


def _cm_number(value, name, lower=0, strict=False):
    a=_numeric(value,name)
    if a.ndim or (float(a)<=lower if strict else float(a)<lower):
        _fail('INVALID_INPUT',name+' is outside its permitted range')
    return float(a)


def fit_discriminant_pd(X_train,y_train,*,feature_order,variant,covariance_estimator='mle',covariance_ridge=0.,priors=None,target,training,model_id,model_version,context=None):
    """Gaussian LDA/QDA with explicit MLE/unbiased covariance and additive ridge.

    Ridge is an absolute variance on the supplied feature scale, not an implicit
    data-dependent shrinkage. No feature is removed to hide singularity.
    """
    x,_=_design(X_train,feature_order,False); y=_cm_y(y_train,len(x)); p=_cm_priors(priors,y)
    if variant not in ('lda','qda') or covariance_estimator not in ('mle','unbiased'):
        _fail('INVALID_INPUT','variant must be lda/qda and covariance_estimator mle/unbiased')
    ridge=_cm_number(covariance_ridge,'covariance_ridge'); counts=np.bincount(y,minlength=2)
    if np.any(counts<2): _fail('INSUFFICIENT_CLASS_SUPPORT','Covariance estimation requires at least two rows per class')
    means=np.array([x[y==k].mean(axis=0) for k in range(2)])
    scatters=[(x[y==k]-means[k]).T@(x[y==k]-means[k]) for k in range(2)]
    if variant=='lda':
        cov=sum(scatters)/(len(x)-(2 if covariance_estimator=='unbiased' else 0))
        covs=[cov,cov]
    else: covs=[scatters[k]/(counts[k]-(covariance_estimator=='unbiased')) for k in range(2)]
    covs=[c+ridge*np.eye(x.shape[1]) for c in covs]; conditions=[]
    for c in covs:
        try: np.linalg.cholesky(c)
        except np.linalg.LinAlgError: _fail('SINGULAR_COVARIANCE','Covariance is singular; choose an explicit positive ridge or revise features')
        conditions.append(float(np.linalg.cond(c)))
        if conditions[-1]>1e14: _fail('SINGULAR_COVARIANCE','Covariance condition number exceeds 1e14')
    diag=dict(class_counts=counts.tolist(),covariance_estimator=covariance_estimator,covariance_condition_numbers=conditions,
        calibration='MODEL_BASED_POSTERIOR_NOT_EMPIRICALLY_CALIBRATED',uncertainty='NOT_ESTIMATED',training_only=True)
    return _cm_model('fit_discriminant_pd',feature_order,dict(kind='discriminant',variant=variant,priors=p.tolist(),means=means.tolist(),covariances=[c.tolist() for c in covs],covariance_ridge=ridge),dict(transformation='identity'),diag,target,training,model_id,model_version,context)


def _cm_table(X, features):
    if not isinstance(features,(list,tuple)) or not features or len(set(features))!=len(features) or any(not isinstance(f,str) or not f.strip() for f in features):
        _fail('FEATURE_MISMATCH','Distinct nonempty ordered feature names are required')
    if hasattr(X,'to_numpy'): X=X.to_numpy(dtype=object)
    x=np.asarray(X,dtype=object)
    if x.ndim!=2 or x.shape[1]!=len(features): _fail('FEATURE_MISMATCH','Table columns must match feature_order')
    return x


def _cm_missing(v):
    return v is None or (isinstance(v,(float,np.floating)) and np.isnan(v))


def _cm_category(v):
    if not isinstance(v,str) or not v: _fail('INVALID_CATEGORY','Categorical values must be nonempty strings, with null reserved for missing')
    return v


def fit_naive_bayes_pd(X_train,y_train,*,feature_order,feature_types,variant,alpha=1.,variance_floor=0.,missing_policy='error',unknown_policy='error',priors=None,target,training,model_id,model_version,context=None):
    """Gaussian/categorical/mixed conditional-independent Bayes classifier.

    Categorical missing and unknown values occupy distinct reserved bins when
    requested. Missing Gaussian values are marginalized (likelihood factor one)
    only under missing_policy='marginalize'; no prediction data enter estimates.
    """
    x=_cm_table(X_train,feature_order); y=_cm_y(y_train,len(x)); p=_cm_priors(priors,y)
    types=list(feature_types)
    if len(types)!=x.shape[1] or any(t not in ('gaussian','categorical') for t in types): _fail('INVALID_INPUT','One gaussian/categorical type is required per feature')
    if variant not in ('gaussian','categorical','mixed') or (variant!='mixed' and any(t!=variant for t in types)) or (variant=='mixed' and len(set(types))!=2): _fail('INVALID_INPUT','Variant must agree with feature types')
    if missing_policy not in ('error','marginalize','category') or unknown_policy not in ('error','reserved'): _fail('INVALID_INPUT','Unsupported missing/unknown policy')
    alpha=_cm_number(alpha,'alpha',strict=True); floor=_cm_number(variance_floor,'variance_floor'); specs=[]
    for j,t in enumerate(types):
        vals=x[:,j]; missing=np.array([_cm_missing(v) for v in vals])
        if missing.any() and (missing_policy=='error' or (t=='gaussian' and missing_policy=='category')): _fail('MISSING_FEATURE','Missing feature incompatible with declared policy')
        if t=='gaussian':
            means=[]; variances=[]; counts=[]
            for k in range(2):
                z=_numeric(list(vals[(y==k)&~missing]),'Gaussian feature',vector_only=True)
                if len(z)<2: _fail('INSUFFICIENT_CLASS_SUPPORT','At least two observed Gaussian values per class are needed')
                v=float(np.var(z))+floor
                if v<=0: _fail('SINGULAR_COVARIANCE','Zero Gaussian variance requires explicit positive variance_floor')
                means.append(float(z.mean())); variances.append(v); counts.append(len(z))
            specs.append(dict(type=t,means=means,variances=variances,observed_counts=counts))
        else:
            categories=sorted({_cm_category(v) for v,m in zip(vals,missing) if not m})
            has_missing=missing_policy=='category'; has_unknown=unknown_policy=='reserved'; size=len(categories)+has_missing+has_unknown
            if size==0: _fail('INSUFFICIENT_CLASS_SUPPORT','Categorical feature has no supported values')
            counts=np.zeros((2,size))
            for v,m,k in zip(vals,missing,y):
                if m and not has_missing: continue
                index=len(categories) if m else categories.index(v); counts[k,index]+=1
            probs=(counts+alpha)/(counts.sum(axis=1)[:,None]+alpha*size)
            specs.append(dict(type=t,categories=categories,missing_index=len(categories) if has_missing else None,unknown_index=len(categories)+has_missing if has_unknown else None,counts=counts.tolist(),probabilities=probs.tolist()))
    diag=dict(class_counts=np.bincount(y,minlength=2).tolist(),conditional_independence_assumed=True,calibration='MODEL_BASED_POSTERIOR_NOT_EMPIRICALLY_CALIBRATED',uncertainty='NOT_ESTIMATED',training_only=True)
    return _cm_model('fit_naive_bayes_pd',feature_order,dict(kind='naive_bayes',variant=variant,priors=p.tolist(),features=specs,alpha=alpha,variance_floor=floor,missing_policy=missing_policy,unknown_policy=unknown_policy),dict(transformation='identity'),diag,target,training,model_id,model_version,context)


def fit_svm_score(X_train,y_train,*,feature_order,kernel,C,gamma=None,scale=True,class_weights=None,tolerance=1e-7,max_iter=None,target,training,model_id,model_version,context=None):
    """LIBSVM C-SVC; the signed decision function is deliberately not a PD."""
    from sklearn.svm import SVC
    x,_=_design(X_train,feature_order,False); y=_cm_y(y_train,len(x)); C=_cm_number(C,'C',strict=True)
    if kernel not in ('linear','rbf') or not isinstance(scale,bool): _fail('INVALID_INPUT','kernel linear/rbf and boolean scale are required')
    g=None if kernel=='linear' else _cm_number(gamma,'gamma',strict=True)
    if kernel=='linear' and gamma is not None: _fail('INVALID_INPUT','Linear kernel does not use gamma')
    tol=_cm_number(tolerance,'tolerance',strict=True)
    if max_iter is not None and (isinstance(max_iter,bool) or not isinstance(max_iter,(int,np.integer)) or max_iter<1): _fail('INVALID_INPUT','max_iter must be a positive integer')
    cw=np.ones(2) if class_weights is None else _numeric(class_weights,'class_weights',vector_only=True)
    if cw.shape!=(2,) or np.any(cw<=0): _fail('INVALID_WEIGHTS','class_weights must be positive [nondefault,default] weights')
    center=x.mean(axis=0) if scale else np.zeros(x.shape[1]); scaling=x.std(axis=0) if scale else np.ones(x.shape[1]); scaling[scaling==0]=1
    z=(x-center)/scaling
    backend=SVC(C=C,kernel=kernel,gamma=g if g is not None else 1.,class_weight={0:float(cw[0]),1:float(cw[1])},tol=tol,max_iter=-1 if max_iter is None else int(max_iter),probability=False,shrinking=False).fit(z,y)
    if backend.fit_status_!=0: _fail('NON_CONVERGENCE','LIBSVM did not converge')
    params=dict(kind='svm',kernel=kernel,C=C,gamma=g,support_vectors=backend.support_vectors_.tolist(),dual_coefficients=backend.dual_coef_[0].tolist(),intercept=float(backend.intercept_[0]),class_weights=cw.tolist())
    score=backend.decision_function(z); margin=(2*y-1)*score
    diag=dict(converged=True,iterations=int(backend.n_iter_[0]),support_vector_count=len(backend.support_),hinge_loss=float(np.maximum(0,1-margin).mean()),training_only=True,score_orientation='HIGHER_MEANS_DEFAULT',calibration='UNCALIBRATED_SCORE',uncertainty='NOT_ESTIMATED')
    return _cm_model('fit_svm_score',feature_order,params,dict(transformation='standardize',center=center.tolist(),scale=scaling.tolist()),diag,target,training,model_id,model_version,context)


def _extended_model_scores(fitted_model,X,feature_order,expected_target):
    model=_model_artifact(fitted_model)
    if list(feature_order)!=model['feature_order']: _fail('FEATURE_MISMATCH','Feature order must exactly match frozen artifact')
    if expected_target!=model['target']: _fail('TARGET_MISMATCH','Prediction target must exactly match frozen artifact')
    p=model['parameters']; kind=p.get('kind')
    if kind in ('neural','sequence'):
        from ._classification_neural import _neural_raw
        return _neural_raw(model,X,feature_order),'LINEAR_PREDICTOR_LOGIT'
    if kind=='bayesian':
        from ._classification_bayesian import _bayesian_prediction
        return _bayesian_prediction(model,X,feature_order)[0],'POSTERIOR_MEAN_PROBABILITY_SCORE'
    if kind in ('gam','random_forest','gradient_boosting'):
        from ._classification_flexible import _gam_raw,_forest_probability,_boost_raw
        if kind=='random_forest': return _forest_probability(model,X,feature_order),'LEAF_PROBABILITY_SCORE'
        return (_gam_raw if kind=='gam' else _boost_raw)(model,X,feature_order),'LINEAR_PREDICTOR_LOGIT'
    if kind=='naive_bayes':
        x=_cm_table(X,feature_order); scores=np.tile(np.log(p['priors']),(len(x),1))
        for j,s in enumerate(p['features']):
            for i,v in enumerate(x[:,j]):
                missing=_cm_missing(v)
                if missing:
                    if p['missing_policy']=='marginalize': continue
                    if s['type']=='gaussian' or s.get('missing_index') is None: _fail('MISSING_FEATURE','Missing feature is not supported')
                    idx=s['missing_index']
                elif s['type']=='categorical':
                    _cm_category(v); idx=s['categories'].index(v) if v in s['categories'] else s['unknown_index']
                    if idx is None: _fail('UNKNOWN_CATEGORY','Previously unseen category is not supported')
                if s['type']=='gaussian':
                    v=float(_numeric(v,'Gaussian feature')); var=np.asarray(s['variances']); scores[i]+=-.5*(np.log(2*np.pi*var)+(v-np.asarray(s['means']))**2/var)
                else: scores[i]+=np.log(np.array(s['probabilities'])[:,idx])
        return scores[:,1]-scores[:,0],'LOG_POSTERIOR_ODDS'
    if kind=='ordinal':
        x,_=_design(X,feature_order,True)
        return x@np.asarray(p['coefficients']),'ORDINAL_LATENT_RISK_'+p['link'].upper()
    if kind in ('penalized_logit','firth_logit'):
        x=_numeric(X,'X',allow_empty=True)
        if model['preprocessing']['transformation']=='standardize':
            x=(x-np.asarray(model['preprocessing']['center']))/np.asarray(model['preprocessing']['scale'])
        _,d=_design(x,feature_order,p['fit_intercept'])
        return d@np.asarray(p['coefficients']),'LINEAR_PREDICTOR_LOGIT'
    x,_=_design(X,feature_order,False)
    if kind=='discriminant':
        scores=[]
        for k in range(2):
            c=np.asarray(p['covariances'][k]); delta=x-np.asarray(p['means'][k]); chol=np.linalg.cholesky(c)
            v=np.linalg.solve(chol,delta.T)
            scores.append(np.log(p['priors'][k])-np.log(np.diag(chol)).sum()-.5*np.sum(v*v,axis=0))
        return scores[1]-scores[0],'LOG_POSTERIOR_ODDS'
    if kind=='svm':
        prep=model['preprocessing']; z=(x-np.asarray(prep['center']))/np.asarray(prep['scale']); sv=np.asarray(p['support_vectors']); coef=np.asarray(p['dual_coefficients'])
        kernel=z@sv.T if p['kernel']=='linear' else np.exp(-p['gamma']*np.sum((z[:,None,:]-sv[None,:,:])**2,axis=2))
        return kernel@coef+p['intercept'],'SVM_MARGIN'
    _fail('UNSUPPORTED_MODEL','Frozen model kind is not supported by extended score prediction')


def predict_model_score(fitted_model,X,*,feature_order,expected_target,context=None):
    """Frozen uncalibrated score with orientation and explicit scale metadata."""
    model=_model_artifact(fitted_model)
    ctx=_context_value(context)
    for key in ('horizon','time_unit','probability_measure','conditioning'):
        if ctx.get(key) is not None and ctx[key]!=model['target'].get(key): _fail('TARGET_MISMATCH','Context '+key+' conflicts with frozen target')
    if model['method_id']=='classification.fit_binary_glm':
        p=model['parameters']
        if p['offset_required']: _fail('OFFSET_MISMATCH','Use explicit GLM prediction for offset-dependent models')
        if list(feature_order)!=model['feature_order'] or expected_target!=model['target']: _fail('MODEL_MISMATCH','Features and target must match frozen artifact')
        _,d=_design(X,feature_order,p['fit_intercept']); score=d@np.asarray(p['coefficients']); scale='LINEAR_PREDICTOR_'+p['link'].upper()
    else: score,scale=_extended_model_scores(model,X,feature_order,expected_target)
    diagnostics=dict(scale=scale,orientation='HIGHER_MEANS_DEFAULT',model_id=model['model_id'],model_version=model['model_version'],calibration='NO_CALIBRATION_APPLIED')
    if model['parameters'].get('kind')=='ordinal':
        diagnostics.update(category_probabilities=_ordinal_probabilities(model,X,feature_order,expected_target).tolist(),ordered_categories=model['parameters']['ordered_categories'])
    return _result(score.tolist(),'classification.predict_model_score','score',context,diagnostics=diagnostics,provenance=model['provenance'])


def _predict_extended_pd(fitted_model,X,*,feature_order,expected_target,context=None):
    model=_model_artifact(fitted_model)
    if model['parameters'].get('kind')=='bayesian':
        if list(feature_order)!=model['feature_order'] or expected_target!=model['target']: _fail('MODEL_MISMATCH','Features and target must match frozen model')
        from ._classification_bayesian import _bayesian_prediction
        probability,diagnostics=_bayesian_prediction(model,X,feature_order)
        diagnostics.update(model_id=model['model_id'],model_version=model['model_version'],target=model['target'])
        return _result(probability.tolist(),'classification.predict_pd','probability',context,diagnostics=diagnostics,provenance=model['provenance'])
    if model['parameters'].get('kind')=='ordinal':
        probabilities=_ordinal_probabilities(model,X,feature_order,expected_target);p=model['parameters']
        if not p['default_categories']: _fail('UNMAPPED_DEFAULT','Ordinal categories require explicit default-category mapping before PD prediction')
        pd=probabilities[:,-len(p['default_categories']):].sum(axis=1)
        return _result(pd.tolist(),'classification.predict_pd','probability',context,diagnostics=dict(model_id=model['model_id'],model_version=model['model_version'],target=model['target'],category_probabilities=probabilities.tolist(),ordered_categories=p['ordered_categories']),provenance=model['provenance'])
    score,scale=_extended_model_scores(model,X,feature_order,expected_target)
    if scale=='SVM_MARGIN': _fail('UNCALIBRATED_SCORE','SVM margins require a separately fitted calibration model before becoming PDs')
    probabilities=score if scale=='LEAF_PROBABILITY_SCORE' else expit(score)
    return _result(probabilities.tolist(),'classification.predict_pd','probability',context,diagnostics=dict(model_id=model['model_id'],model_version=model['model_version'],target=model['target'],calibration='MODEL_BASED_POSTERIOR'),provenance=model['provenance'])


def _cm_controls(max_iter,tolerance):
    if isinstance(max_iter,bool) or not isinstance(max_iter,(int,np.integer)) or max_iter<1: _fail('INVALID_INPUT','max_iter must be a positive integer')
    t=_cm_number(tolerance,'tolerance',strict=True)
    if t>=1: _fail('INVALID_INPUT','tolerance must be below one')
    return int(max_iter),t


def _firth_objective(beta,d,y):
    eta=d@beta; p=expit(eta); w=p*(1-p); info=d.T@(w[:,None]*d)
    try:
        chol=np.linalg.cholesky(info); inv=np.linalg.solve(chol,d.T); h=w*np.sum(inv*inv,axis=0)
    except np.linalg.LinAlgError: return np.inf,np.zeros(d.shape[1])
    ll=np.sum(y*eta-np.logaddexp(0,eta)); objective=-(ll+np.log(np.diag(chol)).sum())
    score=d.T@(y-p+h*(.5-p))
    return float(objective),-score


def fit_firth_logit(X_train,y_train,*,feature_order,fit_intercept=True,max_iter=1000,tolerance=1e-9,target,training,model_id,model_version,context=None):
    """Firth logistic regression: log likelihood + 1/2 log determinant Fisher I.

    Full rank is required. The reported inverse expected information is a local
    curvature diagnostic, not a bias-corrected interval or profile-likelihood CI.
    """
    from scipy.optimize import minimize
    x,d=_design(X_train,feature_order,fit_intercept); y=_cm_y(y_train,len(x),require_both=False); max_iter,tol=_cm_controls(max_iter,tolerance)
    if np.linalg.matrix_rank(d,tol=np.linalg.svd(d,compute_uv=False)[0]*1e-10)<d.shape[1]: _fail('RANK_DEFICIENT','Firth requires full rank; Jeffreys determinant is otherwise zero')
    opt=minimize(_firth_objective,np.zeros(d.shape[1]),args=(d,y),jac=True,method='BFGS',options=dict(maxiter=max_iter,gtol=tol))
    objective,gradient=_firth_objective(opt.x,d,y); residual=float(np.max(abs(gradient)))
    if not np.isfinite(objective) or residual>max(1e-7,np.sqrt(tol)) or (opt.nit>=max_iter and not opt.success): _fail('NON_CONVERGENCE','Firth adjusted score has not converged',{'adjusted_score_max':residual})
    prob=expit(d@opt.x); info=d.T@((prob*(1-prob))[:,None]*d)
    diagnostics=dict(converged=True,iterations=int(opt.nit),penalized_log_likelihood=-objective,adjusted_score_max=residual,rank=d.shape[1],jeffreys_logdet_multiplier=.5,uncertainty='EXPECTED_INFORMATION_LOCAL_APPROXIMATION_NOT_PROFILE_INTERVAL',training_only=True)
    return _cm_model('fit_firth_logit',feature_order,dict(kind='firth_logit',link='logit',fit_intercept=fit_intercept,coefficients=opt.x.tolist(),expected_information_inverse=np.linalg.inv(info).tolist()),dict(transformation='identity'),diagnostics,target,training,model_id,model_version,context)


def fit_penalized_glm(X_train,y_train,*,feature_order,penalty,lambda_path,selected_lambda,l1_ratio=None,fit_intercept=True,scale=True,weights=None,max_iter=10000,tolerance=1e-9,target,training,model_id,model_version,context=None):
    """Logistic elastic-net path with explicitly selected training-only lambda.

    Minimize weighted mean negative Bernoulli log likelihood plus lambda times
    ((1-alpha)/2 * sum(beta^2) + alpha * sum(abs(beta))). Intercept is unpenalized.
    No validation-set selection is performed implicitly.
    """
    import warnings
    import statsmodels.api as sm
    x,_=_design(X_train,feature_order,fit_intercept); y=_cm_y(y_train,len(x)); max_iter,tol=_cm_controls(max_iter,tolerance)
    if penalty not in ('ridge','lasso','elasticnet') or not isinstance(scale,bool): _fail('INVALID_INPUT','Explicit penalty ridge/lasso/elasticnet and boolean scale required')
    alpha=0. if penalty=='ridge' else 1. if penalty=='lasso' else _cm_number(l1_ratio,'l1_ratio',strict=True)
    if penalty=='elasticnet' and alpha>=1: _fail('INVALID_INPUT','Elastic net l1_ratio must be strictly between zero and one')
    if penalty!='elasticnet' and l1_ratio is not None: _fail('INVALID_INPUT','l1_ratio is supplied only for elasticnet')
    lambdas=_numeric(lambda_path,'lambda_path',vector_only=True)
    if np.any(lambdas<0) or len(set(lambdas))!=len(lambdas) or np.any(np.diff(lambdas)>=0): _fail('INVALID_INPUT','lambda_path must be unique strictly decreasing nonnegative values')
    selected=_cm_number(selected_lambda,'selected_lambda')
    if selected not in lambdas: _fail('INVALID_INPUT','selected_lambda must be an exact member of lambda_path')
    w=_vector(weights,len(x),'weights',1)
    if np.any(w<0) or w.sum()<=0 or any(w[y==k].sum()==0 for k in (0,1)): _fail('INVALID_WEIGHTS','Positive total weight in both classes is required')
    center=np.average(x,weights=w,axis=0) if scale else np.zeros(x.shape[1]); scaling=np.sqrt(np.average((x-center)**2,weights=w,axis=0)) if scale else np.ones(x.shape[1]); scaling[scaling==0]=1
    z=(x-center)/scaling; _,d=_design(z,feature_order,fit_intercept); mask=np.ones(d.shape[1]); mask[0]=0 if fit_intercept else 1
    path=[]; beta=np.zeros(d.shape[1]); backend=sm.GLM(y,d,family=sm.families.Binomial(),freq_weights=w)
    for lam in lambdas:
        if lam==0:
            from .classification import _separation
            active=w>0
            if np.linalg.matrix_rank(d[active])<d.shape[1]: _fail('RANK_DEFICIENT','Zero penalty needs identifiable full-rank design')
            _separation(d[active],y[active]); fit=backend.fit(maxiter=max_iter,tol=tol); beta=np.asarray(fit.params)
        elif alpha==0:
            from scipy.optimize import minimize
            def ridge_objective(b):
                e=d@b
                return float(np.sum(w*(np.logaddexp(0,e)-y*e))/w.sum()+lam/2*np.sum(mask*b*b)), d.T@(w*(expit(e)-y))/w.sum()+lam*mask*b
            fit=minimize(ridge_objective,beta,jac=True,method='BFGS',options=dict(gtol=tol,maxiter=max_iter)); beta=np.asarray(fit.x)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                args=dict(alpha=lam*w.sum()/len(x)*mask,L1_wt=alpha,start_params=beta,maxiter=max_iter)
                if alpha>0: args.update(cnvrg_tol=tol,zero_tol=tol*.1)
                else: args.update(opt_method='bfgs',gtol=tol)
                fit=backend.fit_regularized(**args); beta=np.asarray(fit.params)
        eta=d@beta; smoothgrad=d.T@(w*(expit(eta)-y))/w.sum()+lam*(1-alpha)*mask*beta
        kkt=np.where(mask==0,abs(smoothgrad),np.where(abs(beta)>tol,abs(smoothgrad+lam*alpha*np.sign(beta)),np.maximum(abs(smoothgrad)-lam*alpha,0)))
        residual=float(kkt.max()); objective=float(np.sum(w*(np.logaddexp(0,eta)-y*eta))/w.sum()+lam*((1-alpha)/2*np.sum(mask*beta**2)+alpha*np.sum(mask*abs(beta))))
        if not np.all(np.isfinite(beta)) or residual>max(1e-6,np.sqrt(tol)): _fail('NON_CONVERGENCE','Elastic-net KKT tolerance not met',{'lambda':float(lam),'kkt_residual':residual})
        path.append(dict(lambda_value=float(lam),coefficients=beta.tolist(),objective=objective,kkt_residual=residual))
    chosen=path[list(lambdas).index(selected)]
    diagnostics=dict(converged=True,path=path,selection='CALLER_DECLARED_LAMBDA_NO_AUTOMATIC_HOLDOUT_ACCESS',intercept_penalized=False,objective_normalization='WEIGHTED_MEAN_NEGATIVE_LOG_LIKELIHOOD',weight_type='NONNEGATIVE_CASE_WEIGHT',training_only=True,uncertainty='NOT_ESTIMATED_POST_SELECTION')
    return _cm_model('fit_penalized_glm',feature_order,dict(kind='penalized_logit',link='logit',penalty=penalty,l1_ratio=alpha,selected_lambda=selected,fit_intercept=fit_intercept,coefficients=chosen['coefficients']),dict(transformation='standardize',center=center.tolist(),scale=scaling.tolist()),diagnostics,target,training,model_id,model_version,context)


def _ordinal_parallel_diagnostic(x,y,link):
    """Stacked binary estimating-equation Wald diagnostic with shared-row sandwich."""
    import statsmodels.api as sm
    from scipy.linalg import block_diag
    from scipy.stats import chi2,norm
    from .classification import _separation
    if x.shape[1]==0: return dict(status='NOT_APPLICABLE',reason='No slopes')
    d=np.c_[np.ones(len(x)),x]; k=len(np.unique(y))-1; b=[]; influences=[]
    if k<2: return dict(status='NOT_APPLICABLE',reason='Only one cumulative threshold')
    try:
        for j in range(k):
            binary=(y>j).astype(int); _separation(d,binary)
            fam=sm.families.Binomial(link=sm.families.links.Logit() if link=='logit' else sm.families.links.Probit())
            fitted=sm.GLM(binary,d,family=fam).fit(maxiter=1000,tol=1e-11)
            if not fitted.converged: return dict(status='NOT_ESTIMABLE',reason='Threshold regression failed')
            eta=d@fitted.params; p=np.clip(fitted.fittedvalues,1e-12,1-1e-12); dp=p*(1-p) if link=='logit' else norm.pdf(eta)
            bread=d.T@((dp*dp/(p*(1-p)))[:,None]*d)
            scores=d*((binary-p)*dp/(p*(1-p)))[:,None]
            influences.append(scores@np.linalg.inv(bread));b.append(fitted.params)
    except (ValueError,np.linalg.LinAlgError): return dict(status='NOT_ESTIMABLE',reason='Separated or singular cumulative binary regression')
    influence=np.concatenate(influences,axis=1); covariance=influence.T@influence; beta=np.concatenate(b); contrast=np.zeros(((k-1)*x.shape[1],k*d.shape[1]))
    for j in range(1,k):
        for h in range(x.shape[1]): contrast[(j-1)*x.shape[1]+h,h+1]=-1;contrast[(j-1)*x.shape[1]+h,j*d.shape[1]+h+1]=1
    diff=contrast@beta; variance=contrast@covariance@contrast.T; rank=np.linalg.matrix_rank(variance)
    if rank<len(diff): return dict(status='NOT_ESTIMABLE',reason='Singular joint slope-contrast covariance')
    statistic=float(diff@np.linalg.solve(variance,diff))
    return dict(status='ESTIMATED',method='STACKED_BINARY_SANDWICH_WALD',statistic=statistic,degrees_of_freedom=len(diff),p_value=float(chi2.sf(statistic,len(diff))),threshold_slopes=[v[1:].tolist() for v in b],assumptions='IID rows; asymptotic joint estimating-equation sandwich; exploratory diagnostic, not proof of proportionality')


def fit_ordinal_model(X_train,y_train,*,feature_order,ordered_categories,link,default_categories=None,max_iter=1000,tolerance=1e-9,target,training,model_id,model_version,context=None):
    """Cumulative-logit/probit model P(Y<=j|x)=F(threshold_j-x beta).

    Categories must be ordered from lower to higher credit risk. Mapping one or
    more terminal categories to default is caller-declared at the target horizon.
    """
    from scipy.stats import norm
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    x,_=_design(X_train,feature_order,True);max_iter,tol=_cm_controls(max_iter,tolerance)
    cats=list(ordered_categories)
    if len(cats)<2 or len(set(cats))!=len(cats) or any(not isinstance(c,str) or not c for c in cats): _fail('INVALID_INPUT','Distinct ordered category labels are required')
    if link not in ('logit','probit'): _fail('INVALID_INPUT','Ordinal link must be logit or probit')
    if len(y_train)!=len(x) or any(v not in cats for v in y_train): _fail('INVALID_OUTCOME','Every observed outcome must belong to ordered_categories')
    y=np.array([cats.index(v) for v in y_train]);counts=np.bincount(y,minlength=len(cats))
    if np.any(counts==0): _fail('INSUFFICIENT_CLASS_SUPPORT','Every ordered category must be represented in training')
    defaults=[] if default_categories is None else list(default_categories)
    if defaults and (len(set(defaults))!=len(defaults) or defaults!=cats[len(cats)-len(defaults):]): _fail('INVALID_TARGET','Default categories must be a nonempty ordered terminal tail of risk grades')
    if x.shape[1]:
        if np.linalg.matrix_rank(np.c_[np.ones(len(x)),x])<x.shape[1]+1: _fail('RANK_DEFICIENT','Ordinal slopes must be full rank without intercept')
        backend=OrderedModel(y,x,distr='logit' if link=='logit' else 'probit')
        fit=backend.fit(method='bfgs',maxiter=max_iter,gtol=tol,disp=False)
        gradient=float(np.max(abs(backend.score(fit.params))))
        if not fit.mle_retvals['converged'] and gradient>max(1e-6,np.sqrt(tol)): _fail('NON_CONVERGENCE','Ordinal likelihood failed to converge')
        beta=np.asarray(fit.params[:x.shape[1]]); thresholds=backend.transform_threshold_params(fit.params)[1:-1];cov=np.asarray(fit.cov_params());ll=float(fit.llf)
    else:
        cumulative=np.cumsum(counts)[:-1]/len(x);thresholds=np.log(cumulative/(1-cumulative)) if link=='logit' else norm.ppf(cumulative);beta=np.array([]);cov=None;gradient=0.;ll=float(np.sum(counts*np.log(counts/len(x))))
    if np.any(np.diff(thresholds)<=0) or not np.isfinite(thresholds).all(): _fail('NON_IDENTIFIABLE','Ordinal thresholds must be finite strictly increasing')
    diag=dict(converged=True,log_likelihood=ll,score_max=gradient,class_counts=counts.tolist(),parallel_slopes_diagnostic=_ordinal_parallel_diagnostic(x,y,link),covariance_parameterization='SLOPES_FIRST_THRESHOLD_LOG_THRESHOLD_SPACINGS' if cov is not None else 'NOT_ESTIMATED',uncertainty='MODEL_BASED_ASYMPTOTIC' if cov is not None else 'NOT_ESTIMATED',training_only=True)
    return _cm_model('fit_ordinal_model',feature_order,dict(kind='ordinal',link=link,coefficients=beta.tolist(),thresholds=thresholds.tolist(),ordered_categories=cats,default_categories=defaults,covariance=None if cov is None else cov.tolist()),dict(transformation='identity'),diag,target,training,model_id,model_version,context)


def _ordinal_probabilities(model,X,feature_order,expected_target):
    from scipy.special import ndtr
    if list(feature_order)!=model['feature_order'] or expected_target!=model['target']: _fail('MODEL_MISMATCH','Feature order and target must match frozen ordinal model')
    x,_=_design(X,feature_order,True);p=model['parameters'];eta=x@np.asarray(p['coefficients']);thresholds=np.asarray(p['thresholds'])
    cdf=(expit if p['link']=='logit' else ndtr)(thresholds[None,:]-eta[:,None])
    return np.diff(np.c_[np.zeros(len(x)),cdf,np.ones(len(x))],axis=1)
