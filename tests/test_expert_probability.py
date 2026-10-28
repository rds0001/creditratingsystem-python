import numpy as np
import pytest
import math
from creditratingsystem.expert import fit_elicited_beta_prior,update_expert_beta_binomial,pool_expert_probability_distributions,update_odds_from_evidence,check_expert_preference_consistency,assess_expert_rating_agreement
T=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
C=dict(population_id='retail',reference='documented synthetic elicitation',known_at='2024-01-01',expert_ids=['e1'],evidence_ids=['prior-evidence'],target=T)
O=dict(population_id='retail',known_at='2025-01-01',evidence_ids=['cohort-2024'],target=T,conditional_iid=True,outcome_complete=True)

def test_beta_moments_update_predictive_and_no_reused_evidence():
 p=fit_elicited_beta_prior(dict(method='moments',mean=.1,concentration=20),elicitation_contract=C)
 assert [p.value['alpha'],p.value['beta']]==pytest.approx([2,18])
 q=update_expert_beta_binomial(p,3,100,observation_contract=O,predictive_exposures=2).value
 assert [q['posterior']['alpha'],q['posterior']['beta']]==[5,115]
 assert q['predictive_probabilities']==pytest.approx([115*116/(120*121),2*5*115/(120*121),5*6/(120*121)])
 assert q['prior_weight']==pytest.approx(1/6)
 with pytest.raises(ValueError,match='DOUBLE_COUNT'):update_expert_beta_binomial(q['posterior'],3,100,observation_contract=O)
 with pytest.raises(ValueError,match='EXPERT_ELICITATION'):fit_elicited_beta_prior(dict(method='moments',mean=.1,variance=.1),elicitation_contract=C)
 with pytest.raises(ValueError,match='OBSERVATIONS'):update_expert_beta_binomial(p,3,100,observation_contract=dict(O,conditional_iid=False))
 zero=update_expert_beta_binomial(p,0,100,observation_contract=O).value
 assert zero['posterior_mean']>0

@pytest.mark.parametrize('shapes',[(2,18),(3,7)])
def test_beta_quantile_fit_against_closed_form_cdf(shapes):
 alpha,beta=shapes;x=np.array([.03,.1,.2])if shapes==(2,18)else np.array([.1,.3,.6]);n=alpha+beta-1
 prob=sum(math.comb(n,k)*x**k*(1-x)**(n-k)for k in range(alpha,n+1))
 p=fit_elicited_beta_prior(dict(method='quantiles',quantiles=x,probabilities=prob,weights=[1,2,1]),elicitation_contract=C,max_cdf_error=1e-6).value
 assert [p['alpha'],p['beta']]==pytest.approx(shapes,rel=1e-6)
 with pytest.raises(ValueError,match='EXPERT_ELICITATION'):fit_elicited_beta_prior(dict(method='quantiles',quantiles=[.2,.1],probabilities=[.1,.9],weights=[1,1]),elicitation_contract=C)

@pytest.mark.parametrize('rule',['linear','logarithmic'])
def test_discrete_and_histogram_pool_exact_normalization(rule):
 a=np.array([[.5,.25],[.2,.4]])
 result=pool_expert_probability_distributions(a,[1,1],support=dict(kind='histogram',breaks=[0,1,3]),rule=rule).value
 expected=np.mean(a,axis=0)if rule=='linear'else np.sqrt(a.prod(axis=0));expected/=expected@[1,2]
 assert result['density_or_mass']==pytest.approx(expected)
 assert result['cumulative_probabilities'][-1]==pytest.approx(1)
 same=pool_expert_probability_distributions([[.2,.8],[.2,.8]],[2,3],support=dict(kind='discrete',values=['low','high']),rule=rule).value
 assert same['density_or_mass']==pytest.approx([.2,.8])
 if rule=='logarithmic':
  with pytest.raises(ValueError,match='DISJOINT_SUPPORT'):pool_expert_probability_distributions([[1,0],[0,1]],[1,1],support=dict(kind='discrete',values=['a','b']),rule=rule)

def test_odds_joint_evidence_and_duplicate_independence_guards():
 g=dict(group_id='g',evidence_ids=['warning1','warning2'],joint_likelihood_ratio=2,reference='joint likelihood study')
 assert update_odds_from_evidence(.1,[g],dependence_contract={}).value['posterior_pd']==pytest.approx(2/11)
 assert update_odds_from_evidence(0,[g],dependence_contract={}).value['posterior_pd']==0
 with pytest.raises(ValueError,match='DOUBLE_COUNT'):update_odds_from_evidence(.1,[g],dependence_contract={},previous_evidence_ids=['warning1'])
 with pytest.raises(ValueError,match='DEPENDENCE'):update_odds_from_evidence(.1,[g,dict(g,group_id='h',evidence_ids=['warning3'])],dependence_contract={})

def test_preference_consistent_ratios_cycle_and_disconnected_identification():
 w=np.array([1.,2.,4.]);result=check_expert_preference_consistency(w[:,None]/w,items=['a','b','c'],incomplete_policy='error').value
 assert result['components'][0]['priorities']==pytest.approx(w/w.sum())
 assert abs(result['triad_residuals'][0]['log_residual'])<1e-14
 cycle=check_expert_preference_consistency([[1,2,.5],[.5,1,2],[2,.5,1]],items=['a','b','c'],incomplete_policy='error').value
 assert cycle['cycle_witnesses'] and not cycle['consistent_with_tolerance']
 missing=check_expert_preference_consistency([[1,None],[None,1]],items=['a','b'],incomplete_policy='connected_components').value
 assert not missing['globally_identified'] and len(missing['components'])==2

def test_paired_agreement_hand_kappa_missing_and_degenerate_resamples():
 kw=dict(ordered_grades=['A','B'],rater_pair=['one','two'],weighting='unweighted',missing_policy='pairwise_complete')
 a=['A','A','A','B','B','B',None];b=['A','A','B','A','B','B','A']
 result=assess_expert_rating_agreement(a,b,**kw).value
 assert result['kappa']==pytest.approx(1/3) and result['complete_cases']==6 and result['missing_case_indices']==[6]
 degenerate=assess_expert_rating_agreement(a,b,bootstrap_indices=[[0]*6,list(range(6))],**kw).value
 assert degenerate['bootstrap_percentile_interval'] is None and degenerate['bootstrap_degenerate']==1
 assert assess_expert_rating_agreement(['A'],['A'],**kw).value['status']=='DEGENERATE_MARGINALS'
 for weighting in ['linear','quadratic']:
  same=assess_expert_rating_agreement(['A','B'],['A','B'],**dict(kw,weighting=weighting)).value
  assert same['kappa']==1
