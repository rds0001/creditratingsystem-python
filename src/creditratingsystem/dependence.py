"""Explicit dependence matrices and portable, addressed simulation innovations."""
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np
from .core import _numeric, _fail, _result, _fingerprint, _context_value


def _ids(values,name):
    if not isinstance(values,(list,tuple)) or not values or any(not isinstance(v,str) or not v for v in values) or len(set(values))!=len(values):
        _fail('DEPENDENCE_IDS',name+' requires distinct nonempty string IDs')
    return list(values)


def validate_dependence_matrix(matrix,variable_ids,*,matrix_kind,repair_policy='none',tolerance=1e-10,context=None):
    """Check PSD and report an explicitly requested spectral repair, not nearest PSD."""
    a=_numeric(matrix,'matrix');ids=_ids(variable_ids,'variable_ids');tol=_numeric(tolerance,'tolerance')
    if tol.ndim or not 0<float(tol)<1:_fail('DEPENDENCE_TOLERANCE','Tolerance must be scalar in (0,1)')
    tol=float(tol)
    if a.shape!=(len(ids),len(ids)):_fail('INCOMPATIBLE_SHAPE','Square matrix must match ordered variable IDs')
    if matrix_kind not in ('correlation','covariance') or repair_policy not in ('none','spectral'):_fail('DEPENDENCE_CONTRACT','Declare correlation/covariance and none/spectral repair')
    scale=float(np.max(np.abs(a))) or 1.;asymmetry=float(np.max(np.abs(a/scale-a.T/scale)))
    if asymmetry>tol and repair_policy=='none':_fail('ASYMMETRIC_MATRIX','Dependence matrix is not symmetric at declared tolerance')
    # Eigenvalue diagnostics use the symmetric part; no-repair output is unchanged.
    symmetric=a/2+a.T/2
    relative_values,vectors=np.linalg.eigh(symmetric/scale)
    with np.errstate(over='ignore'):values=relative_values*scale
    if not np.all(np.isfinite(values)):_fail('NUMERICAL_FAILURE','Absolute matrix eigenvalues exceed finite range')
    if matrix_kind=='correlation' and np.max(np.abs(np.diag(a)-1))>tol and repair_policy=='none':_fail('INVALID_CORRELATION','Correlation diagonal must equal one')
    if matrix_kind=='covariance' and np.any(np.diag(a)<0) and repair_policy=='none':_fail('INVALID_COVARIANCE','Negative marginal variance')
    if values[0]<-tol*scale and repair_policy=='none':_fail('NON_PSD_MATRIX','Negative eigenvalue exceeds declared numerical tolerance')
    out=a.copy()
    if repair_policy=='spectral':
        out=(vectors*np.maximum(values,0))@vectors.T
        if matrix_kind=='correlation':
            diagonal=np.diag(out)
            if np.any(diagonal<=0):_fail('NON_REPAIRABLE_MATRIX','Spectral projection has a zero marginal variance; correlation scaling is undefined')
            out=out/np.sqrt(diagonal[:,None])/np.sqrt(diagonal[None,:]);np.fill_diagonal(out,1.)
        out=out/2+out.T/2
    with np.errstate(over='ignore',invalid='ignore'):
        repaired_values=np.linalg.eigvalsh((out/2+out.T/2)/scale)*scale
        difference=out-a
        change_max=float(np.max(np.abs(difference)))
        change_norm=0. if change_max==0 else change_max*float(np.linalg.norm(difference/change_max))
    if not np.all(np.isfinite(out)) or not np.all(np.isfinite(repaired_values)) or not math.isfinite(change_norm):_fail('NUMERICAL_FAILURE','Matrix repair or absolute diagnostics exceed finite range')
    return _result(dict(matrix=out.tolist(),variable_ids=ids,matrix_kind=matrix_kind,eigenvalues=repaired_values.tolist()),
        'dependence.validate_dependence_matrix','dependence_matrix',context,
        diagnostics=dict(repair_policy=repair_policy,input_eigenvalues=values.tolist(),relative_asymmetry=asymmetry,
            frobenius_change=change_norm,maximum_absolute_change=change_max,
            numerical_rank=int(np.sum(repaired_values>tol*scale)),tolerance=tol,
            strictly_positive_definite=bool(repaired_values[0]>tol*scale)),
        limitations=('Numerical PSD acceptance does not guarantee suitability for an inverse or Cholesky factor. Tiny negative eigenvalues within tolerance remain explicit when no repair is requested.',
                    'Spectral repair truncates negative eigenvalues and rescales correlations; it is not claimed to be the nearest correlation matrix.'))


def _addressed_uniform(seed,scenario,time,factor,stream,lane):
    address=_fingerprint(['creditratingsystem-innovations-v1',seed,scenario,time,factor,stream,lane])
    # First 32 bits of a canonical SHA256 address. Midpoints avoid 0 and 1.
    return (int(address[:8],16)+.5)/4294967296.


def make_shared_innovations(scenario_ids,time_grid,factor_ids,stream_spec,seed,*,output_directory,chunk_rows=1024,context=None):
    """Write bounded-memory float64LE streams keyed by IDs, independent of row order."""
    scenarios=_ids(scenario_ids,'scenario_ids');factors=_ids(factor_ids,'factor_ids');times=_numeric(time_grid,'time_grid',vector_only=True)
    if np.any(times<0) or np.any(np.diff(times)<=0):_fail('INVALID_TIME_GRID','Times must be nonnegative and strictly increasing')
    if not isinstance(seed,str) or not seed:_fail('INNOVATION_SEED','Seed is an explicit nonempty string')
    if isinstance(chunk_rows,bool) or not isinstance(chunk_rows,int) or chunk_rows<1:_fail('INNOVATION_CHUNK','chunk_rows must be a positive integer')
    if not isinstance(stream_spec,list) or not stream_spec or any(not isinstance(s,dict) for s in stream_spec):_fail('INNOVATION_STREAM','Named stream specifications required')
    streams=_ids([s.get('stream_id') for s in stream_spec],'stream_ids')
    for s in stream_spec:
        if set(s)!={'stream_id','distribution'} or s['distribution'] not in ('uniform','normal'):_fail('INNOVATION_STREAM','Each stream declares stream_id and distribution uniform/normal')
    root=Path(output_directory)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):_fail('OUTPUT_EXISTS','Output directory must be absent or empty')
    root.mkdir(parents=True,exist_ok=True)
    entries=[]
    for number,s in enumerate(stream_spec):
        filename=f'stream-{number:04d}.f64le';digest=hashlib.sha256();count=0
        with (root/filename).open('wb') as handle:
            for offset in range(0,len(scenarios),chunk_rows):
                for scenario in scenarios[offset:offset+chunk_rows]:
                    for time in times:
                        for factor in factors:
                            u=_addressed_uniform(seed,scenario,float(time),factor,s['stream_id'],0)
                            if s['distribution']=='normal':
                                v=_addressed_uniform(seed,scenario,float(time),factor,s['stream_id'],1)
                                u=math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v)
                            payload=struct.pack('<d',u);handle.write(payload);digest.update(payload);count+=1
        entries.append(dict(stream_id=s['stream_id'],distribution=s['distribution'],file=filename,sha256=digest.hexdigest(),values=count,shape=[len(scenarios),len(times),len(factors)]))
    manifest=dict(format='creditratingsystem-f64le-addressed-v1',seed=seed,scenario_ids=scenarios,time_grid=times.tolist(),factor_ids=factors,
        order='scenario,time,factor (factor varies fastest)',streams=entries,
        algorithm='canonical SHA256 address first32bits midpoint; normal Box-Muller cosine with separately addressed lanes0,1')
    path=root/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return _result(dict(directory=str(root),manifest=manifest,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        'dependence.make_shared_innovations','innovations_bundle',context,
        diagnostics=dict(chunk_rows=chunk_rows,address_order_invariant=True),provenance=dict(data_origin='SYNTHETIC'),
        limitations=('Addressed streams are deterministic simulation inputs, not a cryptographic randomness service. Uniform draws have 32-bit resolution.',
                    'Uniform binary files are exact across runtimes. Normal transcendental transforms may differ by floating-point rounding; share one frozen file for exact path replay.'))


def _copula_model(model):
    from .core import _model_artifact
    model=_model_artifact(model);p=model['parameters']
    if model['method_id']!='dependence.fit_copula' or p.get('family') not in ('gaussian','t','clayton','gumbel'):_fail('DEPENDENCE_MODEL','Frozen supported copula model required')
    d=len(model['feature_order'])
    if d<2 or p.get('dimension')!=d:_fail('DEPENDENCE_MODEL','Copula dimension must match variable order')
    if p['family'] in ('gaussian','t'):
        r=validate_dependence_matrix(p.get('correlation'),model['feature_order'],matrix_kind='correlation').value['matrix']
        a=np.asarray(r)
        if not np.array_equal(a,a.T) or np.any(np.diag(a)!=1) or np.any(np.abs(a)>1):_fail('DEPENDENCE_MODEL','Frozen correlation must be symmetric with exact unit diagonal and entries in [-1,1]')
        p=dict(p,correlation=r)
        if p['family']=='t' and (not isinstance(p.get('df'),(int,float)) or not math.isfinite(p['df']) or p['df']<=0):_fail('DEPENDENCE_MODEL','Positive finite t degrees of freedom required')
    else:
        theta=_numeric(p.get('theta'),'theta')
        if theta.ndim or theta<(0 if p['family']=='clayton' else 1):_fail('DEPENDENCE_MODEL','Supported Clayton theta>=0 or Gumbel theta>=1 required')
    return model,p


def fit_copula(observations,*,variable_ids,family,input_kind,tie_policy,correlation_repair='none',
               df=None,df_bounds=(1.,100.),target,training,model_id,model_version,context=None):
    """Rank dependence fit; t degrees of freedom use conditional likelihood profiling."""
    from scipy.stats import rankdata,kendalltau,norm,t,multivariate_t
    from scipy.optimize import minimize_scalar
    from statsmodels.distributions.copula.api import GaussianCopula,StudentTCopula,ClaytonCopula,GumbelCopula
    from .core import _model_artifact,_fit_result
    ids=_ids(variable_ids,'variable_ids');x=_numeric(observations,'observations');d=len(ids)
    if x.ndim!=2 or x.shape[1]!=d or d<2 or len(x)<3:_fail('DEPENDENCE_DATA','At least three aligned rows and two named variables required')
    if family not in ('gaussian','t','clayton','gumbel') or input_kind not in ('observations','uniforms') or tie_policy not in ('error','average'):_fail('DEPENDENCE_CONTRACT','Declare family, observations/uniforms and error/average tie policy')
    ties=[len(x)-len(np.unique(x[:,j])) for j in range(d)]
    if any(ties) and tie_policy=='error':_fail('TIED_OBSERVATIONS','Continuous copula estimation requires explicit handling of tied observations')
    if any(np.all(x[:,j]==x[0,j]) for j in range(d)):_fail('NON_IDENTIFIABLE','Constant marginal observations do not identify dependence')
    if input_kind=='observations':u=np.column_stack([rankdata(x[:,j],method='average')/(len(x)+1) for j in range(d)])
    else:
        if np.any((x<=0)|(x>=1)):_fail('INVALID_PROBABILITY','Copula uniforms must lie strictly inside (0,1)')
        u=x.copy()
    tau=np.eye(d)
    for j in range(d):
        for k in range(j):tau[j,k]=tau[k,j]=kendalltau(x[:,j],x[:,k]).statistic
    parameters=dict(family=family,dimension=d);diagnostics=dict(estimator='INVERSION_OF_KENDALL_TAU',n_rows=len(x),tie_counts=ties,tie_policy=tie_policy,
        tied_continuous_approximation=bool(any(ties)),input_kind=input_kind,kendall_tau=tau.tolist(),uncertainty='NOT_ESTIMATED')
    if family in ('gaussian','t'):
        raw=np.sin(np.pi*tau/2);np.fill_diagonal(raw,1)
        checked=validate_dependence_matrix(raw,ids,matrix_kind='correlation',repair_policy=correlation_repair)
        r=np.asarray(checked.value['matrix'])
        if np.linalg.eigvalsh(r)[0]<=1e-12:_fail('SINGULAR_COPULA','Fitted likelihood requires strictly positive definite dependence; perfect ranks need a declared singular scenario model')
        parameters['correlation']=r.tolist();diagnostics['correlation_validation']=checked.diagnostics
        if family=='t':
            bounds=_numeric(df_bounds,'df_bounds',vector_only=True)
            if bounds.shape!=(2,) or bounds[0]<=0 or bounds[1]<=bounds[0]:_fail('INVALID_DF','Positive increasing profile bounds required')
            def objective(nu):
                z=t.ppf(u,nu);ll=np.sum(multivariate_t.logpdf(z,shape=r,df=nu)-np.sum(t.logpdf(z,nu),axis=1));return -float(ll)
            if df is None:
                fit=minimize_scalar(objective,bounds=tuple(bounds),method='bounded',options=dict(xatol=1e-8,maxiter=500))
                if not fit.success or not math.isfinite(fit.fun):_fail('NON_CONVERGENCE','t copula degrees-of-freedom profile failed')
                candidates=[(float(fit.x),float(fit.fun))]+[(float(v),objective(float(v))) for v in bounds]
                nu,nll=min(candidates,key=lambda z:z[1]);boundary=nu==bounds[0] or nu==bounds[1]
                diagnostics.update(estimator='KENDALL_CORRELATION_THEN_DF_PROFILE',df_bounds=bounds.tolist(),df_on_boundary=bool(boundary),df_identification='BOUND_LIMITED' if boundary else 'INTERIOR_PROFILE_ESTIMATE',conditional_log_likelihood=-nll)
            else:
                value=_numeric(df,'df')
                if value.ndim or value<=0:_fail('INVALID_DF','Fixed t degrees of freedom must be positive')
                nu=float(value);diagnostics.update(df_identification='FIXED_BY_USER',conditional_log_likelihood=-objective(nu))
            parameters['df']=nu
        elif df is not None:_fail('INVALID_DF','Degrees of freedom only apply to the t family')
    else:
        if df is not None:_fail('INVALID_DF','Degrees of freedom only apply to the t family')
        tau_mean=float(np.mean(tau[np.triu_indices(d,1)]))
        if tau_mean<0 or tau_mean>=1:_fail('ARCHIMEDEAN_DOMAIN','This exchangeable Archimedean fit requires mean Kendall tau in [0,1); negative dependence needs a different family')
        theta=2*tau_mean/(1-tau_mean) if family=='clayton' else 1/(1-tau_mean)
        parameters['theta']=theta;diagnostics.update(mean_kendall_tau=tau_mean,pair_tau_range=[float(np.min(tau[np.triu_indices(d,1)])),float(np.max(tau[np.triu_indices(d,1)]))],exchangeable=True)
    pair_ll=[];tail=[]
    for j in range(d):
        for k in range(j):
            if family=='gaussian':cop=GaussianCopula(corr=r[j,k]);lower=upper=0.
            elif family=='t':
                cop=StudentTCopula(corr=r[j,k],df=nu);lower=upper=float(2*t.cdf(-math.sqrt((nu+1)*(1-r[j,k])/(1+r[j,k])),nu+1))
            elif family=='clayton':
                cop=None if theta==0 else ClaytonCopula(theta=theta);lower=0. if theta==0 else 2**(-1/theta);upper=0.
            else:cop=None if theta==1 else GumbelCopula(theta=theta);lower=0.;upper=2-2**(1/theta)
            ll=0. if cop is None else float(np.sum(cop.logpdf(u[:,[k,j]])))
            if not math.isfinite(ll):_fail('NUMERICAL_FAILURE','Pairwise copula density outside representable range')
            pair_ll.append(ll);tail.append(dict(variables=[ids[k],ids[j]],lower=lower,upper=upper))
    diagnostics.update(pairwise_composite_log_likelihood=float(sum(pair_ll)),tail_dependence=tail,
        likelihood_note='Pairwise composite likelihood is a descriptive fit diagnostic, not a joint MLE or an IID standard-error estimator.')
    model=_model_artifact(dict(schema_version='1.0',model_id=model_id,model_version=model_version,method_id='dependence.fit_copula',feature_order=ids,
        parameters=parameters,preprocessing=dict(input_kind=input_kind,tie_policy=tie_policy,rank_denominator=len(x)+1 if input_kind=='observations' else None),
        target=target,training=training,diagnostics=diagnostics,provenance=dict(data_origin=training.get('data_origin','USER_SUPPLIED'),parameter_origin='ESTIMATED'),input_fingerprint=_fingerprint(x.tolist())))
    return _fit_result(model,'dependence.fit_copula',diagnostics=diagnostics,context=context)


def _psd_cholesky(r,tolerance=1e-12):
    """Ordered semidefinite Cholesky with explicit numerical zero pivots."""
    n=len(r);l=np.zeros((n,n))
    for j in range(n):
        pivot=r[j,j]-np.dot(l[j,:j],l[j,:j])
        if pivot < -tolerance:_fail('NON_PSD_MATRIX','Semidefinite factorization encountered a negative pivot')
        if pivot<=tolerance:
            for i in range(j+1,n):
                if abs(r[i,j]-np.dot(l[i,:j],l[j,:j]))>tolerance:_fail('NON_PSD_MATRIX','Zero pivot has a nonzero residual covariance')
        else:
            l[j,j]=math.sqrt(pivot)
            for i in range(j+1,n):l[i,j]=(r[i,j]-np.dot(l[i,:j],l[j,:j]))/l[j,j]
    return l


def simulate_copula(dependence_model,independent_uniforms,*,variable_ids,auxiliary_uniforms=None,context=None):
    """Transform shared open uniforms; frailty/chi-square draws are explicit inputs."""
    from scipy.stats import norm,t,chi2,gamma
    model,p=_copula_model(dependence_model);_copula_context(model,context);ids=_ids(variable_ids,'variable_ids');u=_numeric(independent_uniforms,'independent_uniforms')
    if ids!=model['feature_order'] or u.ndim!=2 or u.shape[1]!=len(ids):_fail('FEATURE_MISMATCH','Uniform columns must match frozen variable order')
    if np.any((u<=0)|(u>=1)):_fail('INVALID_PROBABILITY','Innovation uniforms must be strictly inside (0,1)')
    family=p['family'];need=1 if family=='t' or (family=='clayton' and p['theta']>0) else 2 if family=='gumbel' and p['theta']>1 else 0
    aux=None if auxiliary_uniforms is None else _numeric(auxiliary_uniforms,'auxiliary_uniforms')
    if need:
        if aux is None or aux.shape!=(len(u),need) or np.any((aux<=0)|(aux>=1)):_fail('AUXILIARY_DRAWS','Required open auxiliary uniforms have wrong shape or domain')
    elif aux is not None:_fail('AUXILIARY_DRAWS','No auxiliary innovations apply at this family/boundary')
    if family in ('gaussian','t'):
        root=_psd_cholesky(np.asarray(p['correlation']));z=norm.ppf(u)@root.T
        if family=='gaussian':out=norm.cdf(z)
        else:
            scaling=chi2.ppf(aux[:,0],p['df'])/p['df']
            if np.any(scaling<=0) or not np.all(np.isfinite(scaling)):_fail('NUMERICAL_FAILURE','t scale quantile is outside representable positive range')
            out=t.cdf(z/np.sqrt(scaling)[:,None],p['df'])
        factor_error=float(np.max(np.abs(root@root.T-np.asarray(p['correlation']))))
    elif not need:out=u.copy();factor_error=None
    elif family=='clayton':
        frailty=gamma.ppf(aux[:,0],a=1/p['theta'])
        if np.any(frailty<=0) or not np.all(np.isfinite(frailty)):_fail('NUMERICAL_FAILURE','Gamma frailty quantile is outside representable positive range')
        out=np.exp(-np.log1p(-np.log(u)/frailty[:,None])/p['theta']);factor_error=None
    else:
        alpha=1/p['theta'];angle=np.pi*aux[:,0];exponential=-np.log(aux[:,1])
        log_s=np.log(np.sin(alpha*angle))-np.log(np.sin(angle))/alpha+(1-alpha)/alpha*(np.log(np.sin((1-alpha)*angle))-np.log(exponential))
        out=np.exp(-np.exp(alpha*(np.log(-np.log(u))-log_s[:,None])));factor_error=None
    if not np.all(np.isfinite(out)):_fail('NUMERICAL_FAILURE','Copula transform produced a nonfinite value')
    return _result(out,'dependence.simulate_copula','uniform_probability',context,
        diagnostics=dict(family=family,variable_ids=ids,model_fingerprint=_fingerprint(model),draw_fingerprint=_fingerprint(dict(uniforms=u.tolist(),auxiliary=None if aux is None else aux.tolist())),
            numerical_boundary_count=int(np.sum((out==0)|(out==1))),factorization_maximum_residual=factor_error),provenance=model['provenance'],
        limitations=('Only the supplied innovation distribution and frozen copula establish these marginal/dependence assumptions. Empirical dependence and scenario realism require separate validation.',))


def joint_default_probability(p_a,p_b,*,event_order,dependence_model=None,joint_probability=None,context=None):
    """Bivariate lower-tail default intersection, with deterministic quadrature."""
    from scipy.stats import norm,t
    from scipy.integrate import quad
    from .core import _probability
    a=_probability(p_a,name='p_a');b=_probability(p_b,name='p_b');ids=_ids(event_order,'event_order')
    if a.ndim or b.ndim or len(ids)!=2:_fail('DEPENDENCE_EVENTS','Two ordered event IDs and scalar marginal probabilities required')
    a=float(a);b=float(b);lo=max(0.,math.fsum((a,b,-1.)));hi=min(a,b);error=0.
    if (dependence_model is None)==(joint_probability is None):_fail('DEPENDENCE_CONTRACT','Provide exactly one copula model or explicit intersection')
    if joint_probability is not None:
        j=_probability(joint_probability,name='joint_probability')
        if j.ndim:_fail('DEPENDENCE_EVENTS','Intersection must be scalar')
        j=float(j);family='explicit'
    else:
        model,p=_copula_model(dependence_model);_copula_context(model,context)
        if model['feature_order']!=ids:_fail('FEATURE_MISMATCH','Bivariate event order must exactly match model')
        family=p['family']
        if hi==0:j=0.
        elif a==1 or b==1:j=hi
        elif family in ('gaussian','t'):
            rho=p['correlation'][0][1]
            if rho==1:j=hi
            elif rho==-1:j=lo
            elif family=='gaussian' and rho==0:j=a*b
            else:
                dist=norm if family=='gaussian' else t(p['df']);threshold=dist.ppf(b)
                def integrand(v):
                    x=dist.ppf(a*v)
                    if family=='gaussian':return norm.cdf((threshold-rho*x)/math.sqrt(1-rho*rho))
                    nu=p['df'];return t.cdf((threshold-rho*x)*math.sqrt((nu+1)/((nu+x*x)*(1-rho*rho))),nu+1)
                integral,err=quad(integrand,0,1,epsabs=1e-10,epsrel=1e-10,limit=300);j=a*integral;error=a*err
        elif family=='clayton':
            theta=p['theta']
            if theta==0:j=a*b
            else:
                x=-math.log(a);y=-math.log(b);m=max(x,y)
                exponent=math.log1p(math.expm1(theta*x)+math.expm1(theta*y))/theta if theta*m<500 else m+math.log(math.exp(theta*(x-m))+math.exp(theta*(y-m))-math.exp(-theta*m))/theta
                j=math.exp(-exponent)
        else:
            theta=p['theta'];x=math.log(-math.log(a));y=math.log(-math.log(b));m=max(x,y);j=math.exp(-math.exp(m)*(math.exp(theta*(x-m))+math.exp(theta*(y-m)))**(1/theta))
    if not math.isfinite(j) or not math.isfinite(error):_fail('NUMERICAL_FAILURE','Copula intersection or quadrature error is nonfinite')
    unadjusted=j
    if j<lo or j>hi:
        if dependence_model is not None and min(abs(j-lo),abs(j-hi))<=max(error,8*np.finfo(float).eps):j=min(hi,max(lo,j))
        else:_fail('INVALID_JOINT_PROBABILITY','Intersection lies outside Frechet bounds')
    return _result(j,'dependence.joint_default_probability','probability',context,diagnostics=dict(family=family,event_order=ids,frechet_lower=lo,frechet_upper=hi,quadrature_absolute_error=error,frechet_roundoff_adjustment=j-unadjusted),limitations=('Defaults are lower-tail uniform events U<=PD. Latent correlation is not Bernoulli default correlation. A finite-df t copula with zero correlation is generally dependent.',))


def _copula_context(model,context):
    ctx=_context_value(context)
    for key in ("horizon","time_unit","probability_measure","conditioning"):
        if ctx.get(key) is not None and ctx[key]!=model["target"].get(key):_fail("TARGET_MISMATCH","Context conflicts with frozen copula target")
    return ctx
