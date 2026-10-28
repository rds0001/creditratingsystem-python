"""Explicit random-walk posterior simulation for proper-prior binary models."""
import numpy as np
from scipy.special import expit,ndtr,log_ndtr
from scipy.stats import rankdata,norm
from .core import _numeric,_fail,_fingerprint
from .classification import _design
from ._classification_models import _cm_y,_cm_model,_cm_number


def _bayes_rhat_ess(draws):
    # Split chains; rank-normalized bulk and folded split-Rhat; tail ESS indicators.
    a=np.asarray(draws,float);m,n=a.shape
    if n<8 or m<2:_fail('INSUFFICIENT_DRAWS','At least two chains with eight retained draws required')
    half=n//2;a=np.concatenate([a[:,:half],a[:,-half:]],axis=0);m,n=a.shape
    def statistics(z):
        W=float(np.var(z,axis=1,ddof=1).mean());B=float(n*np.var(z.mean(axis=1),ddof=1));varplus=(n-1)/n*W+B/n
        if np.any(np.var(z,axis=1,ddof=1)==0):return (None,0.)
        rhat=float(np.sqrt(varplus/W));center=z-z.mean(axis=1)[:,None];size=1<<(2*n-1).bit_length();fourier=np.fft.rfft(center,n=size,axis=1);autocov=np.fft.irfft(fourier*np.conjugate(fourier),n=size,axis=1)[:,:n]/n;rho=1-(W-autocov.mean(axis=0))/varplus;rho[0]=1.
        pairs=[]
        for k in range(0,n-1,2):
            pair=float(rho[k]+rho[k+1])
            if pair<0:break
            pairs.append(min(pair,pairs[-1]) if pairs else pair)
        tau=max(1/np.log10(max(m*n,10)),-1+2*sum(pairs));ess=float(min(m*n,m*n/tau))
        return rhat,ess
    def ranks(z):return norm.ppf((rankdata(z,method='average').reshape(z.shape)-3/8)/(z.size+1/4))
    rhat,bulk=statistics(ranks(a));folded,_=statistics(ranks(abs(a-np.median(a))));lo,hi=np.quantile(a,[.05,.95]);_,low=statistics((a<=lo).astype(float));_,high=statistics((a>=hi).astype(float))
    _,ess_mean=statistics(a)
    return dict(rhat=None if rhat is None or folded is None else max(rhat,folded),ess_bulk=bulk,ess_tail=min(low,high),ess_mean=ess_mean,split_chains=m,draws_per_split=n)


def _bayes_logposterior(beta,d,y,link,mean,sd):
    eta=d@beta
    likelihood=np.sum(y*eta-np.logaddexp(0,eta)) if link=='logit' else np.sum(np.where(y==1,log_ndtr(eta),log_ndtr(-eta)))
    return float(likelihood-.5*np.sum(((beta-mean)/sd)**2)-np.log(sd).sum())


def fit_bayesian_pd(X_train,y_train,*,feature_order,link,prior_mean,prior_sd,sampler_spec,group_spec=None,credible_level=.95,target,training,model_id,model_version,context=None):
    """Proper-normal-prior logit/probit posterior using supplied MH innovations.

    sampler_spec: initial_positions[chain,param], proposal_cholesky[param,param],
    normal_innovations[chain,step,param], uniforms[chain,step], warmup, thin,
    innovation_contract='INDEPENDENT_STANDARD_NORMAL_AND_UNIFORM'. No adaptivity or
    hidden random state. Optional known groups have proper N(0,group_prior_sd²)
    effects; their scale is fixed and uncertainty conditional on that choice.
    """
    x,d=_design(X_train,feature_order,True);y=_cm_y(y_train,len(x),require_both=False)
    if link not in ('logit','probit'):_fail('INVALID_INPUT','Bayesian link logit/probit required')
    level=_cm_number(credible_level,'credible_level',strict=True)
    if level>=1:_fail('INVALID_INPUT','credible_level must be below one')
    mean=_numeric(prior_mean,'prior_mean',vector_only=True);sd=_numeric(prior_sd,'prior_sd',vector_only=True)
    if len(mean)!=d.shape[1] or len(sd)!=d.shape[1] or np.any(sd<=0):_fail('INVALID_PRIOR','Proper mean and positive prior SD for intercept and slopes required')
    groups=None
    if group_spec is not None:
        levels=list(group_spec.get('levels',[]));labels=list(group_spec.get('labels',[]));gsd=_cm_number(group_spec.get('prior_sd'),'group prior_sd',strict=True)
        if not levels or len(set(levels))!=len(levels) or any(not isinstance(v,str) or not v for v in levels) or len(labels)!=len(x) or any(v not in levels for v in labels) or set(labels)!=set(levels):_fail('INVALID_GROUP','Distinct observed group levels and one known label per training row required')
        z=np.array([[float(v==level) for level in levels] for v in labels]);d=np.c_[d,z];mean=np.r_[mean,np.zeros(len(levels))];sd=np.r_[sd,np.full(len(levels),gsd)];groups=dict(levels=levels,prior_sd=gsd,unknown_policy='ERROR',hyperparameter_inference='FIXED_GROUP_PRIOR_SD')
    if not isinstance(sampler_spec,dict) or sampler_spec.get('innovation_contract')!='INDEPENDENT_STANDARD_NORMAL_AND_UNIFORM':_fail('INVALID_SAMPLER','Explicit independent innovation contract required')
    initial=_numeric(sampler_spec.get('initial_positions'),'initial_positions');proposal=_numeric(sampler_spec.get('proposal_cholesky'),'proposal_cholesky');noise=_numeric(sampler_spec.get('normal_innovations'),'normal_innovations');uniform=_numeric(sampler_spec.get('uniforms'),'uniforms')
    p=d.shape[1]
    if initial.ndim!=2 or initial.shape[1]!=p or len(initial)<2 or proposal.shape!=(p,p) or np.any(np.diag(proposal)<=0) or not np.array_equal(proposal,np.tril(proposal)):_fail('INVALID_SAMPLER','Multiple chain initial positions and positive-diagonal lower proposal Cholesky required')
    if noise.ndim!=3 or noise.shape[0]!=len(initial) or noise.shape[2]!=p or uniform.shape!=noise.shape[:2] or np.any((uniform<=0)|(uniform>=1)):_fail('INVALID_SAMPLER','Innovations must align chain/step/parameter with open-interval uniforms')
    warmup=sampler_spec.get('warmup');thin=sampler_spec.get('thin',1)
    if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in (warmup,thin)) or warmup<0 or warmup>=noise.shape[1] or thin<1:_fail('INVALID_SAMPLER','Integer warmup and positive thinning required')
    retained=list(range(warmup,noise.shape[1],thin))
    if len(retained)<8:_fail('INSUFFICIENT_DRAWS','At least eight retained draws per chain required')
    draws=np.empty((len(initial),len(retained),p));accept=[];positions=initial.copy();post=np.array([_bayes_logposterior(b,d,y,link,mean,sd) for b in positions]);retained_index={v:i for i,v in enumerate(retained)}
    if not np.isfinite(post).all():_fail('INVALID_SAMPLER','Initial posterior densities must be finite')
    for c in range(len(initial)):
        accepted=0
        for step in range(noise.shape[1]):
            candidate=positions[c]+proposal@noise[c,step];logp=_bayes_logposterior(candidate,d,y,link,mean,sd)
            if np.log(uniform[c,step])<logp-post[c]:positions[c]=candidate;post[c]=logp;accepted+=1
            if step in retained_index:draws[c,retained_index[step]]=positions[c]
        accept.append(accepted/noise.shape[1])
    metrics=[_bayes_rhat_ess(draws[:,:,j]) for j in range(p)];converged=all(v['rhat'] is not None and v['rhat']<=1.01 and v['ess_bulk']>=400 and v['ess_tail']>=400 for v in metrics)
    diag=dict(sampler='FIXED_PROPOSAL_RANDOM_WALK_METROPOLIS',initial_positions=initial.tolist(),innovation_fingerprint=_fingerprint(dict(normal_innovations=noise.tolist(),uniforms=uniform.tolist())),convergence_status='DIAGNOSTICS_PASSED' if converged else 'DIAGNOSTICS_FAILED_OR_INSUFFICIENT',parameter_diagnostics=metrics,acceptance_rates=accept,warmup=int(warmup),thin=int(thin),steps=noise.shape[1],chains=len(initial),retained_per_chain=len(retained),diagnostic_thresholds=dict(rhat_max=1.01,ess_bulk_min=400,ess_tail_min=400),innovation_assumption=sampler_spec['innovation_contract'],prior_sensitivity='REQUIRES_REFITTING_DECLARED_ALTERNATIVE_PRIORS',uncertainty='POSTERIOR_CONDITIONAL_ON_PRIOR_AND_MODEL',training_only=True)
    params=dict(kind='bayesian',link=link,fit_intercept=True,prior_mean=mean.tolist(),prior_sd=sd.tolist(),posterior_draws=draws.tolist(),groups=groups,credible_level=level,proposal_cholesky=proposal.tolist())
    return _cm_model('fit_bayesian_pd',feature_order,params,dict(transformation='identity'),diag,target,training,model_id,model_version,context)


def _bayesian_prediction(model,X,feature_order):
    params=model['parameters'];groups=params['groups']
    if groups is not None:
        if not isinstance(X,dict):_fail('INVALID_GROUP','Grouped Bayesian prediction requires {X,groups}')
        x,d=_design(X.get('X'),feature_order,True);labels=X.get('groups',[])
        if len(labels)!=len(x) or any(v not in groups['levels'] for v in labels):_fail('UNKNOWN_GROUP','Known trained group label required per prediction row')
        d=np.c_[d,[[float(v==level) for level in groups['levels']] for v in labels]]
    else:x,d=_design(X,feature_order,True)
    draws=np.asarray(params['posterior_draws']);linear=np.einsum('np,ctp->nct',d,draws);prob=(expit if params['link']=='logit' else ndtr)(linear);q=(1-params['credible_level'])/2;flat=prob.reshape(len(d),-1);means=flat.mean(axis=1);limits=np.quantile(flat,[q,1-q],axis=1).T;metrics=[_bayes_rhat_ess(a) for a in prob];mcse=[float(np.std(a,ddof=1)/np.sqrt(m['ess_mean'])) if m['ess_mean']>0 else None for a,m in zip(flat,metrics)]
    return means,dict(credible_level=params['credible_level'],pd_credible_intervals=limits.tolist(),posterior_sd=flat.std(axis=1,ddof=1).tolist(),posterior_predictive_event_variance=(means*(1-means)).tolist(),mean_pd_mcse=mcse,prediction_chain_diagnostics=metrics,fit_convergence_status=model['diagnostics']['convergence_status'],interpretation='Intervals for latent conditional event probability, not prediction intervals for individual Bernoulli outcome')
