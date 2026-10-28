from copy import deepcopy
from itertools import product
import numpy as np
import pytest
from creditratingsystem._validation_numerics import simulation_convergence,sensitivity_analysis
from creditratingsystem.core import _fingerprint,build_calculation_context,CreditRiskError


def design():return dict(model_id='frozen',model_version='1',parameter_version='1',data_fingerprint='synthetic-oracle',population_id='oracle',split_id='external',prediction_origin='EXTERNAL',evaluation_cutoff='2024-01-01',target=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing'))


def simulation_fixture(binary=False):
 innovations=dict(r1=dict(path_ids=['p0','p1','p2','p3'],values=[[-1],[1],[-2],[2]]));runs=[]
 for dt in [1,.5]:
  for N in [2,4]:
   values=[0]*N if binary else(np.array([1,3,1,3][:N])+dt).tolist()
   runs.append(dict(replicate_id='r1',time_step=dt,sample_size=N,path_ids=innovations['r1']['path_ids'][:N],values=values,innovation_fingerprint=_fingerprint(innovations['r1']['values'][:N])))
 contract=dict(sampling_unit='iid_paths_with_independent_replicates',metric='bernoulli_probability'if binary else'mean',reference='independent-path exercise',finest_time_step=.5,time_unit='years',analytic_reference=dict(value=0 if binary else 2,source='known mean'))
 return runs,innovations,dict(sampling_contract=contract,precision=dict(confidence_level=.95,mc_half_width=.01,grid_difference_tolerance=.1),design=design())


def test_mc_and_grid_error_separated_with_shared_prefix_oracle():
 runs,innovations,kw=simulation_fixture();r=simulation_convergence(runs,innovations,**kw).value
 assert [s['estimate']for s in r['metrics']['sampling']]==[2.5,2.5,3,3]
 assert r['metrics']['sampling'][1]['mc_standard_error']==pytest.approx(1/np.sqrt(3))
 assert all(s['estimate']==.5 and s['mc_standard_error']==0 and not s['observed_grid_tolerance_met']for s in r['metrics']['paired_grid_differences'])
 runs[0]['innovation_fingerprint']='wrong'
 with pytest.raises(CreditRiskError):simulation_convergence(runs,innovations,**kw)


def test_zero_default_uncertainty_positive():
 runs,innovations,kw=simulation_fixture(True);r=simulation_convergence(runs,innovations,**kw).value
 assert all(s['upper']>0 and s['maximum_interval_radius']>0 and not s['mc_precision_met']for s in r['metrics']['sampling'])


def graph_fixture():
 d=design();ctx=build_calculation_context('2024-01-01','2024-01-01',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing').value;c=dict(unit='probability',target=d['target'],row_ids=['r'],as_of='2024-01-01',known_at='2024-01-01');data={s:dict(value=[.5],contract=c)for s in ['a','b']};nodes=[dict(node_id=s,operation='input',inputs={},parameters=dict(input_id=s),output_contract=c)for s in ['a','b']];nodes.append(dict(node_id='pool',operation='combine_pd_signals',inputs=dict(a='a',b='b'),parameters=dict(signal_order=['a','b'],weights=[.25,.75],combination_scale='linear',combination_contract=dict(interpretation='estimator_pool',reference='fixed linear combination')),output_contract=c));graph=dict(graph_id='linear-oracle',graph_version='1',nodes=nodes,output_node='pool');model=dict(kind='model_graph',model_graph=graph,input_data=data,parameter_snapshots={},execution_context=ctx,output_row=0);specs=[dict(name=s,location='input',path=[s,'value',0],lower=0,upper=1,step=.01)for s in ['a','b']]
 return model,[.5,.5],specs,d


def test_actual_graph_local_slopes_global_exact_additive_oracle():
 model,base,specs,d=graph_fixture();r=sensitivity_analysis(model,base,specs,method='finite_difference',shared_samples=None,design=d).value
 assert [s['derivative']for s in r['metrics']['local_derivatives']]==pytest.approx([.25,.75],abs=1e-12)
 levels=list(product([.2,.8],repeat=2));pairs=list(product(levels,levels));samples=dict(A=[a for a,b in pairs],B=[b for a,b in pairs],distribution_contract=dict(independent_inputs=True,reference='Cartesian exact integration oracle'))
 r=sensitivity_analysis(model,base,specs,method='variance_based',shared_samples=samples,design=d).value
 assert r['metrics']['first_order']==pytest.approx(dict(a=.1,b=.9),abs=1e-12)
 assert r['metrics']['total_order']==pytest.approx(dict(a=.1,b=.9),abs=1e-12)
 assert r['metrics']['pairwise_interactions'][0]['second_order']==pytest.approx(0,abs=1e-12)
 samples['distribution_contract']['independent_inputs']=False
 with pytest.raises(CreditRiskError):sensitivity_analysis(model,base,specs,method='variance_based',shared_samples=samples,design=d)


def test_graph_parameter_intervention_and_boundary_second_order_difference():
 model,base,specs,d=graph_fixture();model['input_data']['a']['value']=[.2];model['input_data']['b']['value']=[.8];spec=[dict(name='weight_a',location='parameter',path=['nodes',2,'parameters','weights',0],lower=.1,upper=.9,step=.0001)]
 r=sensitivity_analysis(model,[.25],spec,method='finite_difference',shared_samples=None,design=d).value
 assert r['metrics']['local_derivatives'][0]['derivative']==pytest.approx(-.45,abs=2e-9)
 model,base,specs,d=graph_fixture();model['input_data']['a']['value']=[0];base[0]=0;r=sensitivity_analysis(model,base,specs,method='finite_difference',shared_samples=None,design=d).value
 assert r['metrics']['local_derivatives'][0]['scheme']=='forward'
 assert r['metrics']['local_derivatives'][0]['derivative']==pytest.approx(.25,abs=1e-12)


def test_real_frozen_pd_model_derivative_against_grouped_logistic_mle():
 from creditratingsystem.classification import fit_binary_glm
 from scipy.special import expit,logit
 d=design();fit=fit_binary_glm([[0],[0],[1],[1]],[0,1,0,1],link='logit',feature_order=['group'],fit_intercept=True,target=d['target'],training=dict(data_fingerprint='group-counts',split_id='train',known_at='2023-01-01',data_origin='SYNTHETIC'),model_id='glm',model_version='1',weights=[8,2,4,6])
 spec=[dict(name='group',location='feature',lower=0,upper=1,step=.001)];r=sensitivity_analysis(dict(kind='pd_model',fitted_model=fit),[.5],spec,method='finite_difference',shared_samples=None,design=d).value
 p=expit((logit(.2)+logit(.6))/2);slope=(logit(.6)-logit(.2))*p*(1-p)
 assert r['metrics']['baseline_value']==pytest.approx(p,abs=1e-8)
 assert r['metrics']['local_derivatives'][0]['derivative']==pytest.approx(slope,abs=3e-8)
