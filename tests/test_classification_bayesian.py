import os
import numpy as np
import pytest
from creditratingsystem import predict_pd
from scipy.integrate import quad
from scipy.special import expit,ndtr
from scipy.stats import norm
from creditratingsystem._classification_bayesian import fit_bayesian_pd,_bayes_rhat_ess,_bayesian_prediction
extended = pytest.mark.skipif(os.environ.get('CREDITRATINGSYSTEM_EXTENDED_TESTS', '').lower() != 'true', reason='Extended posterior recovery oracle')
T=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing');M=dict(target=T,training=dict(data_fingerprint='bayes-analytic',split_id='train',known_at='2025-01-01',data_origin='SYNTHETIC'),model_id='bayes',model_version='1')
def sampler(p=1,steps=7000,seed=981):
 rng=np.random.default_rng(seed)
 return dict(initial_positions=np.zeros((4,p)),proposal_cholesky=np.eye(p)*.9,normal_innovations=rng.normal(size=(4,steps,p)),uniforms=rng.uniform(size=(4,steps)),warmup=1000,thin=2,innovation_contract='INDEPENDENT_STANDARD_NORMAL_AND_UNIFORM')
@extended
@pytest.mark.parametrize('link',['logit','probit'])
def test_bayesian_intercept_posterior_against_independent_quadrature(link):
 spec=sampler();m=fit_bayesian_pd(np.zeros((12,0)),[0]*11+[1],feature_order=[],link=link,prior_mean=[-1.],prior_sd=[1.5],sampler_spec=spec,**M)
 probability,diag=_bayesian_prediction(m.model,np.zeros((1,0)),[]);cdf=expit if link=='logit' else ndtr
 def density(b):
  p=cdf(b)
  return p*(1-p)**11*norm.pdf(b,-1,1.5)
 z=quad(density,-15,15,epsabs=1e-12)[0];mean=quad(lambda b:cdf(b)*density(b),-15,15,epsabs=1e-12)[0]/z
 assert probability[0]==pytest.approx(mean,abs=4*diag['mean_pd_mcse'][0]+.001)
 assert diag['pd_credible_intervals'][0][0]<mean<diag['pd_credible_intervals'][0][1]
 assert m.diagnostics['parameter_diagnostics'][0]['rhat']<1.03
 assert m.diagnostics['parameter_diagnostics'][0]['ess_bulk']>400
 assert abs(probability[0]-cdf(np.asarray(m.model['parameters']['posterior_draws']).mean()))>.005 # not inverse link at posterior mean

@extended
def test_bayesian_low_default_prior_sensitivity_and_chain_failure_visible():
 spec=sampler(steps=5000)
 priors=[]
 for sd in [.3,2.]:
  m=fit_bayesian_pd(np.zeros((5,0)),np.zeros(5),feature_order=[],link='logit',prior_mean=[-2],prior_sd=[sd],sampler_spec=spec,**M);p,d=_bayesian_prediction(m.model,np.zeros((1,0)),[]);priors.append((p[0],np.diff(d['pd_credible_intervals'][0])[0]))
 assert priors[1][1]>2*priors[0][1]
 stuck=_bayes_rhat_ess(np.zeros((4,20)));assert stuck['rhat'] is None and stuck['ess_bulk']==0
 one_stuck=_bayes_rhat_ess(np.r_[np.zeros((1,100)),np.random.default_rng(9).normal(size=(3,100))]);assert one_stuck['rhat'] is None and one_stuck['ess_bulk']==0
 nonmixed=np.random.default_rng(1).normal(size=(4,100))+np.arange(4)[:,None]*4
 assert _bayes_rhat_ess(nonmixed)['rhat']>1.1

@extended
def test_bayesian_known_groups_and_unknown_group_rejected():
 spec=sampler(p=3,steps=2000)
 m=fit_bayesian_pd(np.zeros((8,0)),[0,0,0,1,0,1,1,1],feature_order=[],link='logit',prior_mean=[0],prior_sd=[1],group_spec=dict(levels=['a','b'],labels=['a']*4+['b']*4,prior_sd=.7),sampler_spec=spec,**M)
 p,d=_bayesian_prediction(m.model,dict(X=np.zeros((2,0)),groups=['a','b']),[])
 assert p[0]<p[1]
 with pytest.raises(ValueError,match='UNKNOWN_GROUP'):_bayesian_prediction(m.model,dict(X=np.zeros((1,0)),groups=['c']),[])

@pytest.mark.parametrize('link',['logit','probit'])
def test_bayesian_small_real_fit_predict_and_nonmixing_diagnostics(link):
 spec=sampler(steps=80);spec['warmup']=20;spec['thin']=1
 m=fit_bayesian_pd(np.zeros((4,0)),[0,0,0,1],feature_order=[],link=link,prior_mean=[-1],prior_sd=[1],sampler_spec=spec,**M)
 p,d=_bayesian_prediction(m.model,np.zeros((1,0)),[])
 assert predict_pd(m,np.zeros((1,0)),feature_order=[],expected_target=T).value==pytest.approx(p)
 for kwargs,code in [(dict(offset=[0]),'OFFSET_MISMATCH'),(dict(context={'horizon':2}),'TARGET_MISMATCH')]:
  with pytest.raises(ValueError,match=code):predict_pd(m,np.zeros((1,0)),feature_order=[],expected_target=T,**kwargs)
 assert 0<p[0]<1 and d['pd_credible_intervals'][0][0]<=p[0]<=d['pd_credible_intervals'][0][1]
 stuck=_bayes_rhat_ess(np.zeros((4,20)))
 assert stuck['rhat'] is None and stuck['ess_bulk']==0
