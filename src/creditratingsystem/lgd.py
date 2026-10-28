"""Recovery cashflows and conditional loss-given-default models."""
from copy import deepcopy
import math
import numpy as np
from .core import CalcResult, FitResult, _result, _fail, _numeric, _probability, _context_value, _fingerprint, _model_artifact, _fit_result
from .data import _rows, _required, _unique, _date
from ._loss_paths import _multiply, _sum


def _scalar(value,name,minimum=None):
    a=_numeric(value,name)
    if a.ndim or (minimum is not None and float(a)<minimum):_fail('LGD_INPUT',name+' requires a scalar in its stated domain')
    return float(a)


def _currency(currency,context):
    if not isinstance(currency,str) or not currency.strip():_fail('INVALID_CURRENCY','Explicit reporting currency required')
    ctx=_context_value(context)
    if ctx.get('currency') is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Context currency conflicts with reporting currency')
    return ctx


def discount_recoveries(cashflows,discount_curve,fx_contract,*,default_date,evaluation_cutoff,
                        reporting_currency,context=None):
    """Discount known realized recoveries/costs using explicitly dated factors and FX.

    Curve points are exact-date factors to the default-date anchor; there is no
    interpolation or implicit interest-rate convention. FX conversion is selected
    at cashflow date or at default date by the supplied contract.
    """
    ctx=_currency(reporting_currency,context);default=_date(default_date,'default_date');cutoff=_date(evaluation_cutoff,'evaluation_cutoff')
    if default>cutoff:_fail('RECOVERY_DATE','Default follows evaluation cutoff')
    if not isinstance(discount_curve,dict) or not isinstance(fx_contract,dict):_fail('RECOVERY_CONTRACT','Explicit discount and FX objects required')
    _required(discount_curve,['anchor_date','known_at','discount_basis','points']);_required(fx_contract,['reporting_currency','known_at','conversion_basis','rates'])
    if _date(discount_curve['anchor_date'],'anchor_date')!=default or _date(discount_curve['known_at'],'curve known_at')>cutoff:_fail('RECOVERY_CONTRACT','Curve must be anchored at default and known at cutoff')
    if not isinstance(discount_curve['discount_basis'],str) or not discount_curve['discount_basis'].strip():_fail('RECOVERY_CONTRACT','Document the economic discount basis')
    if fx_contract['reporting_currency']!=reporting_currency or fx_contract['conversion_basis'] not in ('cashflow_date','default_date') or _date(fx_contract['known_at'],'FX known_at')>cutoff:_fail('RECOVERY_CONTRACT','FX convention or knowledge cutoff incompatible')
    points=_rows(discount_curve['points'],'discount points');_unique(points,'cashflow_date','discount points');df={}
    for row in points:
        _required(row,['discount_factor']);date=_date(row['cashflow_date'],'cashflow_date');factor=_scalar(row['discount_factor'],'discount_factor',0)
        if date<default or (date==default and factor!=1):_fail('RECOVERY_CONTRACT','Discount anchor factor must be one and points cannot precede default')
        df[row['cashflow_date']]=factor
    rates=_rows(fx_contract['rates'],'FX rates');_unique(rates,['currency','date'],'FX rates');fx={}
    for row in rates:
        _required(row,['rate']);_date(row['date'],'FX date');rate=_scalar(row['rate'],'FX rate',0)
        if rate<=0 or (row['currency']==reporting_currency and rate!=1):_fail('RECOVERY_CONTRACT','Positive FX rates and reporting-currency identity required')
        fx[(row['currency'],row['date'])]=rate
    flows=_rows(cashflows,'cashflows');_unique(flows,'cashflow_id','cashflows');included=[];excluded=[];recoveries=[];costs=[]
    for row in flows:
        _required(row,['cashflow_date','known_at','currency','kind','amount'])
        date,known=(_date(row[k],k)for k in ('cashflow_date','known_at'));amount=_scalar(row['amount'],'cashflow amount',0)
        if not isinstance(row['currency'],str) or not row['currency'].strip():_fail('INVALID_CURRENCY','Cashflow currency must be an explicit string')
        if date<default or known<date:_fail('RECOVERY_DATE','Realized cashflow cannot precede default or knowledge precede realization')
        if row['kind'] not in ('RECOVERY','DIRECT_COST','INDIRECT_COST'):_fail('RECOVERY_KIND','Declare RECOVERY, DIRECT_COST or INDIRECT_COST')
        if date>cutoff or known>cutoff:
            excluded.append(dict(cashflow_id=row['cashflow_id'],reason='FUTURE_CASHFLOW' if date>cutoff else 'NOT_YET_KNOWN'));continue
        fx_date=default_date if fx_contract['conversion_basis']=='default_date' else row['cashflow_date'];key=(row['currency'],fx_date)
        if row['cashflow_date'] not in df:_fail('MISSING_DISCOUNT','Exact cashflow-date discount factor missing')
        if key not in fx:_fail('MISSING_FX','Explicit currency/date FX rate missing')
        value=float(_multiply(amount,fx[key],df[row['cashflow_date']]))
        (recoveries if row['kind']=='RECOVERY' else costs).append(value)
        included.append(dict(cashflow_id=row['cashflow_id'],cashflow_date=row['cashflow_date'],known_at=row['known_at'],kind=row['kind'],original_currency=row['currency'],amount=amount,fx_date=fx_date,fx_rate=fx[key],discount_factor=df[row['cashflow_date']],discounted_contribution=value if row['kind']=='RECOVERY' else -value))
    gross=_sum(recoveries);cost=_sum(costs);net=_sum((gross,-cost))
    return _result(dict(discounted_gross_recoveries=gross,discounted_costs=cost,discounted_net_recoveries=net,contributions=included,excluded_cashflows=excluded),
        'lgd.discount_recoveries',reporting_currency,ctx,
        diagnostics=dict(default_date=default_date,evaluation_cutoff=evaluation_cutoff,discount_basis=discount_curve['discount_basis'],fx_conversion_basis=fx_contract['conversion_basis'],input_fingerprint=_fingerprint(dict(cashflows=flows,discount_curve=discount_curve,fx_contract=fx_contract))),
        limitations=('Only realized and known cashflows are included. Explicit curve/FX data determine the economic valuation convention; no regulatory discount rate or future recovery is supplied.',))


def workout_lgd(ead_at_default,discounted_net_recoveries,*,episode_status,default_date,
                evaluation_cutoff,currency,context=None):
    """Keep raw economic loss and open-workout incompleteness separate."""
    ctx=_currency(currency,context);ead=_scalar(ead_at_default,'ead_at_default',0)
    if _date(default_date,'default_date')>_date(evaluation_cutoff,'evaluation_cutoff'):_fail('RECOVERY_DATE','Default follows evaluation cutoff')
    if episode_status not in ('OPEN','CLOSED'):_fail('WORKOUT_STATUS','Declare OPEN or CLOSED')
    if isinstance(discounted_net_recoveries,CalcResult):
        r=discounted_net_recoveries
        if r.method_id!='lgd.discount_recoveries' or r.unit!=currency or r.diagnostics.get('default_date')!=default_date or r.diagnostics.get('evaluation_cutoff')!=evaluation_cutoff:_fail('TARGET_MISMATCH','Recovery result and workout valuation convention differ')
        net=_scalar(r.value['discounted_net_recoveries'],'discounted_net_recoveries')
    else:net=_scalar(discounted_net_recoveries,'discounted_net_recoveries')
    loss=_sum((ead,-net));ratio=None if ead==0 else loss/ead
    if ratio is not None and not math.isfinite(ratio):_fail('NUMERICAL_FAILURE','LGD ratio outside finite range')
    return _result(dict(ead_at_default=ead,discounted_net_recoveries=net,economic_loss=loss,economic_lgd_to_date=ratio,
        economic_lgd=ratio if episode_status=='CLOSED' else None,episode_status=episode_status,complete=episode_status=='CLOSED',
        denominator_status='ZERO_EAD' if ead==0 else 'DEFINED',remaining_recovery_estimated=False),
        'lgd.workout_lgd','loss_fraction',ctx,diagnostics=dict(currency=currency,default_date=default_date,evaluation_cutoff=evaluation_cutoff),
        limitations=('Raw economic LGD may be negative or above one. An open workout has a loss-to-date ratio, not an observed final LGD. Zero EAD leaves the ratio undefined.',))

# Frozen conditional regression models. Native R implements the same likelihoods
# with its own numerical routines; no fitted callable objects are persisted.
def _fit_inputs(X_train,lgd_targets,feature_order,fit_intercept,weights,weight_type,target,training,model_id,model_version,method_id):
    from .classification import _design
    x,d=_design(X_train,feature_order,fit_intercept);y=_numeric(lgd_targets,'lgd_targets',vector_only=True)
    if len(y)!=len(x):_fail('INCOMPATIBLE_SHAPE','One target per training row required')
    if weight_type!='frequency':_fail('INVALID_WEIGHT_TYPE','Declare integer frequency weights')
    w=np.ones(len(y)) if weights is None else _numeric(weights,'weights',vector_only=True)
    if len(w)!=len(y) or np.any(w<0) or np.any(w!=np.floor(w)) or not np.any(w>0) or not math.isfinite(float(w.sum())) or w.sum()>2**53-1:_fail('INVALID_WEIGHTS','Nonnegative exact integer frequencies with positive finite total required')
    if not isinstance(training,dict) or training.get('workout_population')!='closed_defaulted_episodes':_fail('LGD_POPULATION','Training must declare workout_population=closed_defaulted_episodes')
    if not isinstance(target,dict) or (target.get('probability_measure')!='PHYSICAL' or target.get('conditioning')!='given_default'):_fail('INCOMPATIBLE_TARGET','Empirical LGD requires a physical target conditional on default')
    model=_model_artifact(dict(schema_version='1.0',model_id=model_id,model_version=model_version,method_id=method_id,
        feature_order=list(feature_order),parameters={},preprocessing=dict(transformation='identity'),target=target,training=training,diagnostics={},
        provenance=dict(data_origin=training.get('data_origin','USER_SUPPLIED'),parameter_origin='ESTIMATED')))
    active=w>0;xa,da,ya,wa=x[active],d[active],y[active],w[active]
    singular=np.linalg.svd(da,compute_uv=False)
    if len(singular)<da.shape[1] or singular[-1]<=singular[0]*1e-10:_fail('RANK_DEFICIENT','LGD mean design requires full rank at relative tolerance1e-10')
    model['training_input_fingerprint']=_fingerprint(dict(X=x.tolist(),y=y.tolist(),weights=w.tolist()))
    model['parameters']=dict(fit_intercept=fit_intercept,default_condition='given_default',feature_min=xa.min(axis=0).tolist(),feature_max=xa.max(axis=0).tolist(),weight_type=weight_type)
    return model,x,d,y,w,active


def _controls(max_iter,tolerance):
    if isinstance(max_iter,bool) or not isinstance(max_iter,int) or max_iter<1:_fail('INVALID_INPUT','max_iter must be a positive integer')
    t=_scalar(tolerance,'tolerance')
    if not 0<t<1:_fail('INVALID_INPUT','tolerance must lie in (0,1)')
    return t


def _fractional_separation(d,y):
    from scipy.optimize import linprog
    boundary=(y==0)|(y==1)
    if not boundary.any():return 0.
    scaled=d/np.max(np.abs(d),axis=0);signed=(2*y[boundary]-1)[:,None]*scaled[boundary]
    interior=scaled[~boundary]
    fit=linprog(-signed.mean(axis=0),A_ub=-signed,b_ub=np.zeros(len(signed)),A_eq=interior if len(interior)else None,b_eq=np.zeros(len(interior)) if len(interior)else None,bounds=[(-1,1)]*d.shape[1],method='highs')
    if not fit.success:_fail('NUMERICAL_FAILURE','Fractional separation certificate failed')
    objective=float(np.mean(signed@fit.x))
    if objective>1e-8:_fail('SEPARATION','Endpoint observations permit an unbounded coefficient direction while interior rows remain fixed')
    return objective


def _fractional_solution(d,y,w,fit_intercept,max_iter,tolerance):
    from scipy.special import expit
    separation=_fractional_separation(d,y);total=float(w.sum());beta=np.zeros(d.shape[1]);mean=float(np.dot(w,y)/total)
    if fit_intercept:
        if not 0<mean<1:_fail('NUMERICAL_FAILURE','Fractional initial weighted mean rounded to a boundary')
        beta[0]=math.log(mean/(1-mean))
    objective=lambda b:float(np.dot(w,np.logaddexp(0,d@b)-y*(d@b))/total)
    for iteration in range(max_iter+1):
        eta=d@beta;p=expit(eta);gradient=d.T@(w*(p-y))/total;h=d.T@((w*p*(1-p))[:,None]*d)/total
        if np.max(np.abs(gradient))<=tolerance:break
        if iteration==max_iter:_fail('NON_CONVERGENCE','Fractional likelihood score exceeds tolerance')
        try:step=np.linalg.solve(h,gradient)
        except np.linalg.LinAlgError:_fail('NON_IDENTIFIABLE','Fractional information singular')
        before=objective(beta);scale=1.
        while scale>2**-40 and objective(beta-scale*step)>before-1e-4*scale*float(gradient@step):scale/=2
        if scale<=2**-40:_fail('NON_CONVERGENCE','Fractional Newton line search failed')
        beta-=scale*step
    p=expit(d@beta);information=d.T@((w*p*(1-p))[:,None]*d)
    try:bread=np.linalg.inv(information);np.linalg.cholesky(information)
    except np.linalg.LinAlgError:_fail('NON_IDENTIFIABLE','Fractional fitted information singular')
    if np.any((p<=0)|(p>=1)) or not np.all(np.isfinite(bread)):_fail('NUMERICAL_FAILURE','Fractional fit reached a numerical probability boundary')
    return beta,p,bread,iteration,separation


def fit_fractional_lgd(X_train,lgd_targets,*,feature_order,link,covariance,target,training,
                       model_id,model_version,weights=None,weight_type='frequency',fit_intercept=True,
                       max_iter=200,tolerance=1e-10,context=None):
    """Fractional logit QMLE with exact endpoints and robust frequency covariance."""
    if link!='logit':_fail('UNSUPPORTED_LINK','Fractional LGD currently uses the explicit logit mean link')
    if covariance not in ('HC0','HC1'):_fail('INVALID_COVARIANCE','Use HC0 or HC1 under independent observations')
    tolerance=_controls(max_iter,tolerance)
    model,x,d,y,w,active=_fit_inputs(X_train,lgd_targets,feature_order,fit_intercept,weights,weight_type,target,training,model_id,model_version,'lgd.fit_fractional_lgd')
    if np.any((y<0)|(y>1)):_fail('INVALID_LGD_TARGET','Fractional targets must lie in [0,1]; no clipping is performed')
    d,y,w=d[active],y[active],w[active];beta,p,bread,iterations,separation=_fractional_solution(d,y,w,fit_intercept,max_iter,tolerance)
    n=float(w.sum());k=d.shape[1]
    if covariance=='HC1' and n<=k:_fail('NON_IDENTIFIABLE','HC1 requires frequency total greater than parameter count')
    # Integer weights represent repeated independent rows, so meat uses w, not w^2.
    meat=d.T@((w*(y-p)**2)[:,None]*d);cov=bread@meat@bread*(n/(n-k) if covariance=='HC1'else 1.)
    diagnostics=dict(converged=True,iterations=iterations,quasi_log_likelihood=float(np.sum(w*(y*np.log(p)+(1-y)*np.log1p(-p)))),
        frequency_total=n,n_active_rows=len(y),rank=k,maximum_mean_score=float(np.max(np.abs(d.T@(w*(y-p))/n))),
        separation_mean_margin=separation,covariance_type=covariance,weight_type=weight_type,solver='damped_newton',max_iter=max_iter,tolerance=tolerance,
        distribution_identified=False)
    model['parameters'].update(link=link,coefficients=beta.tolist(),covariance=cov.tolist(),covariance_type=covariance)
    model['diagnostics']=diagnostics;model=_model_artifact(model)
    return _fit_result(model,'lgd.fit_fractional_lgd',diagnostics=diagnostics,context=context)


def _precision_design(precision_design,precision_feature_order,n):
    from .classification import _design
    if precision_design is None:
        if precision_feature_order not in (None,[],()):_fail('FEATURE_MISMATCH','Constant precision has no predictor names')
        return np.empty((n,0)),np.ones((n,1)),[]
    z,d=_design(precision_design,precision_feature_order,True)
    if len(z)!=n:_fail('INCOMPATIBLE_SHAPE','Precision design must match mean-design rows')
    return z,d,list(precision_feature_order)


def _beta_statistics(theta,d,z,y,w):
    from scipy.special import expit,gammaln,digamma,polygamma
    k=d.shape[1];eta=d@theta[:k];tau=z@theta[k:]
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        mu=expit(eta);phi=np.exp(tau);a=mu*phi;b=(1-mu)*phi;ly=np.log(y);ln=np.log1p(-y)
        ll=gammaln(phi)-gammaln(a)-gammaln(b)+(a-1)*ly+(b-1)*ln
        u=ly-ln-digamma(a)+digamma(b);v=digamma(phi)-mu*digamma(a)-(1-mu)*digamma(b)+mu*ly+(1-mu)*ln
        derivative=mu*(1-mu);gb=phi*derivative*u;gg=phi*v
        score=np.column_stack((d*gb[:,None],z*gg[:,None]));gradient=np.sum(w[:,None]*score,axis=0)
        trig_a=polygamma(1,a);trig_b=polygamma(1,b)
        hbb=phi**2*derivative**2*(trig_a+trig_b)-phi*derivative*(1-2*mu)*u
        hbg=-derivative*phi*(u+phi*(-mu*trig_a+(1-mu)*trig_b))
        hgg=-phi*v-phi**2*(polygamma(1,phi)-mu**2*trig_a-(1-mu)**2*trig_b)
        information=np.block([[d.T@((w*hbb)[:,None]*d),d.T@((w*hbg)[:,None]*z)],
                              [z.T@((w*hbg)[:,None]*d),z.T@((w*hgg)[:,None]*z)]])
    return float(np.dot(w,ll)),gradient,information,mu,phi


def fit_beta_lgd(X_train,lgd_targets,*,feature_order,mean_link,precision_link,target,training,
                 model_id,model_version,precision_design=None,precision_feature_order=None,
                 weights=None,weight_type='frequency',fit_intercept=True,max_iter=500,tolerance=1e-10,context=None):
    """Beta MLE with logit mean and independent log-precision design; no endpoint adjustment."""
    from scipy.optimize import minimize
    if mean_link!='logit' or precision_link!='log':_fail('UNSUPPORTED_LINK','Declare logit mean and log precision links')
    tolerance=_controls(max_iter,tolerance)
    model,x,d,y,w,active=_fit_inputs(X_train,lgd_targets,feature_order,fit_intercept,weights,weight_type,target,training,model_id,model_version,'lgd.fit_beta_lgd')
    if np.any((y<=0)|(y>=1)):_fail('INVALID_LGD_TARGET','Beta targets must lie strictly inside (0,1); boundaries need a separate model')
    raw_z,z,znames=_precision_design(precision_design,precision_feature_order,len(y));d,y,w,z=d[active],y[active],w[active],z[active]
    singular=np.linalg.svd(z,compute_uv=False)
    if len(singular)<z.shape[1] or singular[-1]<=singular[0]*1e-10:_fail('RANK_DEFICIENT','Precision design requires full rank')
    beta,p,_,_,_=_fractional_solution(d,y,w,fit_intercept,max_iter,tolerance)
    residual=float(np.dot(w,(y-p)**2)/w.sum())
    if residual<=1e-24:_fail('NON_IDENTIFIABLE','No residual interior dispersion; beta precision has no finite estimate')
    initial_phi=max(.1,float(np.dot(w,p*(1-p))/w.sum())/residual-1);theta=np.r_[beta,np.log(initial_phi),np.zeros(z.shape[1]-1)];total=float(w.sum())
    def objective(b):
        ll,g,_,_,_=_beta_statistics(b,d,z,y,w)
        if not math.isfinite(ll) or not np.all(np.isfinite(g)):return np.inf,np.zeros_like(b)
        return -ll/total,-g/total
    fit=minimize(objective,theta,jac=True,method='BFGS',options=dict(maxiter=max_iter,gtol=tolerance))
    ll,gradient,information,mu,phi=_beta_statistics(fit.x,d,z,y,w);score=float(np.max(np.abs(gradient)/total))
    if not math.isfinite(ll) or not math.isfinite(score) or score>max(np.sqrt(tolerance),1e-7):_fail('NON_CONVERGENCE','Beta likelihood score exceeds tolerance',dict(solver_status=int(fit.status),maximum_mean_score=score))
    try:np.linalg.cholesky(information);covariance=np.linalg.inv(information)
    except np.linalg.LinAlgError:_fail('NON_IDENTIFIABLE','Observed beta information is not positive definite')
    if not np.all(np.isfinite(covariance)) or np.any(mu<=0) or np.any(mu>=1) or not np.all(np.isfinite(phi)):_fail('NUMERICAL_FAILURE','Beta fit reached numerical parameter boundary')
    k=d.shape[1];diagnostics=dict(converged=True,log_likelihood=ll,frequency_total=total,n_active_rows=len(y),mean_rank=k,precision_rank=z.shape[1],maximum_mean_score=score,
        covariance_type='OBSERVED_INFORMATION_IID',solver='BFGS',solver_success=bool(fit.success),iterations=int(fit.nit),max_iter=max_iter,tolerance=tolerance,effective_score_tolerance=max(np.sqrt(tolerance),1e-7))
    model['parameters'].update(mean_link=mean_link,precision_link=precision_link,mean_coefficients=fit.x[:k].tolist(),precision_coefficients=fit.x[k:].tolist(),
        precision_feature_order=znames,precision_feature_min=raw_z[active].min(axis=0).tolist(),precision_feature_max=raw_z[active].max(axis=0).tolist(),covariance=covariance.tolist(),covariance_type='OBSERVED_INFORMATION_IID')
    model['precision_input_fingerprint']=_fingerprint(raw_z.tolist());model['diagnostics']=diagnostics;model=_model_artifact(model)
    return _fit_result(model,'lgd.fit_beta_lgd',diagnostics=diagnostics,context=context)


def predict_lgd(fitted_model,X,*,feature_order,expected_target,default_condition,
                precision_design=None,precision_feature_order=None,quantile_levels=None,context=None):
    """Replay a frozen conditional LGD mean or supported distribution."""
    from .classification import _design
    from scipy.special import expit
    from scipy.stats import beta as beta_distribution
    model=_model_artifact(fitted_model);p=model['parameters'];method=model['method_id'];context=_lgd_target_context(model['target'],context,model['model_version'])
    if method=='lgd.fit_recovery_hurdle':
        from ._lgd_workout import _hurdle_prediction
        if precision_design is not None or precision_feature_order is not None:_fail('FEATURE_MISMATCH','Recovery hurdle has no beta precision design')
        return _hurdle_prediction(model,X,feature_order,expected_target,default_condition,quantile_levels,context)
    if method not in ('lgd.fit_fractional_lgd','lgd.fit_beta_lgd','lgd.fit_zero_one_inflated_lgd'):_fail('UNSUPPORTED_MODEL','Unsupported frozen LGD estimator')
    if list(feature_order)!=model['feature_order'] or expected_target!=model['target'] or default_condition!='given_default' or p.get('default_condition')!=default_condition:_fail('TARGET_MISMATCH','Feature order and physical conditional target must match the frozen model')
    x,d=_design(X,feature_order,p.get('fit_intercept'));n=len(x)
    levels=None if quantile_levels is None else _numeric(quantile_levels,'quantile_levels',vector_only=True)
    if levels is not None and (np.any((levels<=0)|(levels>=1)) or np.any(np.diff(levels)<=0)):_fail('INVALID_QUANTILE','Quantile levels must be strictly increasing inside (0,1)')
    lo=_numeric(p.get('feature_min'),'feature_min',allow_empty=True,vector_only=True);hi=_numeric(p.get('feature_max'),'feature_max',allow_empty=True,vector_only=True)
    if lo.shape!=(x.shape[1],) or hi.shape!=lo.shape or np.any(lo>hi):_fail('INVALID_MODEL','Frozen feature support is malformed')
    outside=np.any((x<lo)|(x>hi),axis=1)
    if method=='lgd.fit_zero_one_inflated_lgd':
        for key,required in [('boundary_gate','lgd.fit_fractional_lgd'),('one_gate','lgd.fit_fractional_lgd'),('interior_model','lgd.fit_beta_lgd')]:
            if _model_artifact(p.get(key))['method_id']!=required:_fail('INVALID_MODEL','Inflated beta component method mismatch')
        gate_results={key:predict_lgd(p[key],X,feature_order=feature_order,expected_target=expected_target,default_condition=default_condition) for key in ('boundary_gate','one_gate')}
        interior_result=predict_lgd(p['interior_model'],X,feature_order=feature_order,expected_target=expected_target,default_condition=default_condition,precision_design=precision_design,precision_feature_order=precision_feature_order)
        for result in list(gate_results.values())+[interior_result]:outside|=np.asarray(result.diagnostics['outside_marginal_training_range'])
        gates={key:r.value['mean_lgd']for key,r in gate_results.items()};inside=interior_result.value
        pb=np.asarray(gates['boundary_gate']);po=np.asarray(gates['one_gate']);pi=1-pb;p0=pb*(1-po);p1=pb*po;mu=np.asarray(inside['mean_lgd']);phi=np.asarray(inside['precision']);mean=p1+pi*mu
        quantiles=None
        if levels is not None:
            quantiles=[]
            for a,b,c,m,f in zip(p0,p1,pi,mu,phi):
                quantiles.append([0. if u<=a else 1. if u>=1-b else float(beta_distribution.ppf((u-a)/c,m*f,(1-m)*f)) for u in levels])
        value=dict(mean_lgd=mean.tolist(),distribution='ZERO_ONE_INFLATED_BETA',zero_probability=p0.tolist(),one_probability=p1.tolist(),interior_probability=pi.tolist(),interior_mean=mu.tolist(),precision=phi.tolist(),quantile_levels=None if levels is None else levels.tolist(),quantiles=quantiles)
    else:
        key='coefficients' if method=='lgd.fit_fractional_lgd' else 'mean_coefficients';coef=_numeric(p.get(key),key,vector_only=True)
        if coef.shape!=(d.shape[1],):_fail('INVALID_MODEL','Frozen mean coefficient dimension mismatch')
        eta=d@coef
        if not np.all(np.isfinite(eta)):_fail('NUMERICAL_FAILURE','LGD linear predictor outside finite range')
        mean=expit(eta)
        if method=='lgd.fit_fractional_lgd':
            if precision_design is not None or precision_feature_order is not None:_fail('FEATURE_MISMATCH','Fractional mean has no precision design')
            if levels is not None:_fail('DISTRIBUTION_NOT_IDENTIFIED','A fractional conditional mean does not identify loss quantiles')
            if p.get('link')!='logit':_fail('INVALID_MODEL','Frozen fractional mean link must be logit')
            value=dict(mean_lgd=mean.tolist(),distribution=None,quantile_levels=None,quantiles=None)
        else:
            if p.get('mean_link')!='logit' or p.get('precision_link')!='log':_fail('INVALID_MODEL','Frozen beta links must be logit/log')
            raw_z,z,znames=_precision_design(precision_design,precision_feature_order,n)
            if znames!=p.get('precision_feature_order'):_fail('FEATURE_MISMATCH','Precision feature order must match frozen model')
            gamma=_numeric(p.get('precision_coefficients'),'precision_coefficients',vector_only=True)
            zl=_numeric(p.get('precision_feature_min'),'precision_feature_min',allow_empty=True,vector_only=True);zh=_numeric(p.get('precision_feature_max'),'precision_feature_max',allow_empty=True,vector_only=True)
            if gamma.shape!=(z.shape[1],) or zl.shape!=(raw_z.shape[1],) or zh.shape!=zl.shape or np.any(zl>zh):_fail('INVALID_MODEL','Frozen precision coefficient or support dimension mismatch')
            outside|=np.any((raw_z<zl)|(raw_z>zh),axis=1)
            with np.errstate(over='ignore',under='ignore'):phi=np.exp(z@gamma)
            if not np.all(np.isfinite(phi)) or np.any(phi<=0) or np.any(mean<=0) or np.any(mean>=1):_fail('NUMERICAL_FAILURE','Beta distribution reached a numerical parameter boundary')
            quantiles=None if levels is None else beta_distribution.ppf(levels[None,:],(mean*phi)[:,None],((1-mean)*phi)[:,None]).tolist()
            value=dict(mean_lgd=mean.tolist(),distribution='BETA',precision=phi.tolist(),quantile_levels=None if levels is None else levels.tolist(),quantiles=quantiles)
    return _result(value,'lgd.predict_lgd','loss_fraction',context,provenance=model['provenance'],diagnostics=dict(model_id=model['model_id'],model_version=model['model_version'],model_fingerprint=_fingerprint(model),default_condition=default_condition,outside_marginal_training_range=outside.tolist()),limitations=('Conditional on default and the frozen target. Marginal training ranges do not establish joint support; fractional means alone do not identify a loss distribution.',))


def fit_zero_one_inflated_lgd(X_train,lgd_targets,*,feature_order,component_spec,target,training,
                              model_id,model_version,precision_design=None,precision_feature_order=None,
                              weights=None,weight_type='frequency',fit_intercept=True,max_iter=500,tolerance=1e-10,context=None):
    """Structural endpoint gates and an interior beta likelihood fitted separately."""
    if component_spec!={'boundary_origin':'structural','gate_link':'logit'}:_fail('COMPONENT_SPEC','Declare structural (not rounded/censored) endpoints and logit gates')
    model,x,d,y,w,active=_fit_inputs(X_train,lgd_targets,feature_order,fit_intercept,weights,weight_type,target,training,model_id,model_version,'lgd.fit_zero_one_inflated_lgd')
    if np.any((y<0)|(y>1)):_fail('INVALID_LGD_TARGET','Inflated beta targets must lie in [0,1]')
    boundary=(y==0)|(y==1);interior=~boundary
    if not all(np.any(active&m)for m in (y==0,y==1,interior)):_fail('INSUFFICIENT_COMPONENT_SUPPORT','Positive frequency support required for zero, one and interior components')
    raw_z,_,znames=_precision_design(precision_design,precision_feature_order,len(y))
    common=dict(feature_order=feature_order,target=target,training=training,model_version=model_version,weight_type=weight_type,fit_intercept=fit_intercept,max_iter=max_iter,tolerance=tolerance)
    gate=fit_fractional_lgd(x,boundary.astype(float),weights=w,link='logit',covariance='HC0',model_id=model_id+':boundary',**common)
    one=fit_fractional_lgd(x[boundary],y[boundary],weights=w[boundary],link='logit',covariance='HC0',model_id=model_id+':one',**common)
    inside=fit_beta_lgd(x[interior],y[interior],weights=w[interior],mean_link='logit',precision_link='log',model_id=model_id+':interior',precision_design=None if precision_design is None else raw_z[interior],precision_feature_order=znames,**common)
    model['parameters'].update(component_spec=deepcopy(component_spec),boundary_gate=gate.model,one_gate=one.model,interior_model=inside.model)
    diagnostics=dict(converged=True,component_frequency=dict(zero=float(w[y==0].sum()),one=float(w[y==1].sum()),interior=float(w[interior].sum())),log_likelihood=gate.diagnostics['quasi_log_likelihood']+one.diagnostics['quasi_log_likelihood']+inside.diagnostics['log_likelihood'],conditional_independence='Likelihood factors by observed endpoint membership; component scores use their own observation subsets')
    model['diagnostics']=diagnostics;return _fit_result(_model_artifact(model),'lgd.fit_zero_one_inflated_lgd',diagnostics=diagnostics,context=context)


def _lgd_target_context(target,context,model_version):
    ctx=_context_value(context)
    for key in ('horizon','time_unit','probability_measure','conditioning','currency'):
        if ctx.get(key) is not None and target.get(key) is not None and ctx[key]!=target[key]:_fail('TARGET_MISMATCH','Calculation context conflicts with frozen LGD '+key)
    if ctx.get('model_version') is not None and ctx['model_version']!=model_version:_fail('TARGET_MISMATCH','Calculation context model version differs')
    return ctx


def estimate_downturn_lgd(fitted_lgd_model,baseline_covariates,adverse_covariates,aggregation_weights,*,
                          feature_order,expected_target,downturn_definition,
                          baseline_precision_design=None,adverse_precision_design=None,precision_feature_order=None,
                          model_draws=None,uncertainty_contract=None,interval_level=.95,context=None):
    """Compare matched portfolio scenarios under a declared adverse-condition policy."""
    model=_model_artifact(fitted_lgd_model)
    if not isinstance(downturn_definition,dict):_fail('DOWNTURN_POLICY','Explicit dated scenario-selection policy required')
    policy=deepcopy(downturn_definition);_required(policy,['definition_id','version','selection_basis','baseline_scenario_id','adverse_scenario_id','known_at','as_of','unit_ids','extrapolation_policy'])
    for key in ('definition_id','version','selection_basis','baseline_scenario_id','adverse_scenario_id'):
        if not isinstance(policy[key],str)or not policy[key].strip():_fail('DOWNTURN_POLICY','Nonempty '+key+' required')
    if _date(policy['known_at'],'known_at')>_date(policy['as_of'],'as_of') or _date(model['training']['known_at'],'training.known_at')>_date(policy['as_of'],'as_of'):_fail('DOWNTURN_DATE','Selection policy and fitted model must be known at scenario evaluation')
    if policy['extrapolation_policy']not in ('error','report'):_fail('DOWNTURN_POLICY','Declare error or report for marginal support violations')
    ids=policy['unit_ids']
    if not isinstance(ids,list)or not ids or any(not isinstance(v,str)or not v for v in ids)or len(set(ids))!=len(ids):_fail('DOWNTURN_POLICY','Distinct matched portfolio unit IDs required')
    w=_numeric(aggregation_weights,'aggregation_weights',vector_only=True)
    if len(w)!=len(ids)or np.any(w<0)or not np.any(w>0):_fail('INVALID_WEIGHTS','Nonnegative matched portfolio weights with positive total required')
    w=w/np.max(w);w=w/_sum(w)
    ctx=_lgd_target_context(model['target'],context,model['model_version'])
    if ctx.get('as_of') is not None and ctx['as_of']!=policy['as_of']:_fail('DOWNTURN_DATE','Context and policy evaluation dates differ')
    def pair(frozen):
        if _date(frozen['training']['known_at'],'training.known_at')>_date(policy['as_of'],'as_of'):_fail('DOWNTURN_DATE','Model draw was not known at scenario evaluation')
        results=[predict_lgd(frozen,x,feature_order=feature_order,expected_target=expected_target,default_condition='given_default',precision_design=z,precision_feature_order=precision_feature_order,context=context)for x,z in ((baseline_covariates,baseline_precision_design),(adverse_covariates,adverse_precision_design))]
        if any(len(r.value['mean_lgd'])!=len(ids)for r in results):_fail('INCOMPATIBLE_SHAPE','Scenario rows must match portfolio unit IDs and weights')
        support=[r.diagnostics['outside_marginal_training_range']for r in results]
        if policy['extrapolation_policy']=='error'and any(any(b and weight>0 for b,weight in zip(s,w))for s in support):_fail('DOWNTURN_SUPPORT','Positive-weight scenario rows exceed frozen marginal training ranges')
        means=[_sum(_multiply(w,np.asarray(r.value['mean_lgd'])))for r in results]
        return means,results,support
    means,results,support=pair(model);uplift=_sum((means[1],-means[0]));uncertainty=dict(status='NOT_ESTIMATED',interval_level=None,baseline_interval=None,adverse_interval=None,uplift_interval=None,draw_count=0)
    if model_draws is None:
        if uncertainty_contract is not None:_fail('DOWNTURN_UNCERTAINTY','Uncertainty contract without paired model draws is not an estimate')
    else:
        if not isinstance(model_draws,list)or len(model_draws)<2 or not isinstance(uncertainty_contract,dict):_fail('DOWNTURN_UNCERTAINTY','At least two paired frozen model draws and a declared uncertainty contract required')
        _required(uncertainty_contract,['source','interpretation'])
        if uncertainty_contract['source']not in ('bootstrap','posterior','user_supplied')or not isinstance(uncertainty_contract['interpretation'],str)or not uncertainty_contract['interpretation']:_fail('DOWNTURN_UNCERTAINTY','Declare bootstrap/posterior/user_supplied draws and their inferential interpretation')
        level=_scalar(interval_level,'interval_level')
        if not 0<level<1:_fail('DOWNTURN_UNCERTAINTY','Interval level must be inside (0,1)')
        draws=[]
        for draw in model_draws:
            frozen=_model_artifact(draw)
            if frozen['method_id']!=model['method_id'] or frozen['target']!=model['target'] or frozen['feature_order']!=model['feature_order']:_fail('TARGET_MISMATCH','Uncertainty draws must share model family and target/feature contract')
            if model['method_id']=='lgd.fit_recovery_hurdle' and frozen['parameters'].get('stage_spec')!=model['parameters'].get('stage_spec'):_fail('RECOVERY_STAGE','Paired uncertainty draws must share the frozen residual payout stage')
            m,_,_=pair(frozen);draws.append(m+[_sum((m[1],-m[0]))])
        q=np.quantile(np.array(draws),[(1-level)/2,(1+level)/2],axis=0,method='linear')
        uncertainty=dict(status='SUPPLIED_PAIRED_MODEL_DRAWS',interval_level=level,baseline_interval=q[:,0].tolist(),adverse_interval=q[:,1].tolist(),uplift_interval=q[:,2].tolist(),draw_count=len(draws),contract=deepcopy(uncertainty_contract))
    return _result(dict(baseline_lgd=means[0],downturn_lgd=means[1],difference=uplift,unit_ids=ids,normalized_weights=w.tolist(),baseline_unit_lgd=results[0].value['mean_lgd'],adverse_unit_lgd=results[1].value['mean_lgd'],uncertainty=uncertainty),'lgd.estimate_downturn_lgd','loss_fraction',context,provenance=model['provenance'],diagnostics=dict(model_fingerprint=_fingerprint(model),policy_fingerprint=_fingerprint(policy),downturn_definition=policy,baseline_outside_marginal_range=support[0],adverse_outside_marginal_range=support[1],adverse_effect_nonnegative=uplift>=0),limitations=('Scenario selection and economic adversity are explicit user policy. The model may imply a negative adverse-minus-baseline difference; no regulatory floor or automatic surcharge is imposed. Marginal ranges do not prove joint support or causal identification. Supplied paired draws retain their declared interpretation; no confidence guarantee is manufactured.',))


from ._lgd_workout import fit_recovery_hurdle, estimate_open_workout_lgd
