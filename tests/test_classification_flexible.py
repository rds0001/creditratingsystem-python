import numpy as np
import pytest
from scipy.special import expit
from creditratingsystem._classification_flexible import fit_gam_pd,fit_random_forest_pd,fit_gradient_boosting_pd,_gam_raw,_boost_raw
from creditratingsystem._classification_models import _predict_extended_pd
from creditratingsystem.classification import fit_binary_glm, predict_pd
T=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
M=dict(target=T,training=dict(data_fingerprint='flexible-analytic',split_id='train',known_at='2025-01-01',data_origin='SYNTHETIC'),model_id='flexible',model_version='1')
def pred(m,x):return np.array(predict_pd(m,x,feature_order=['x'],expected_target=T).value)
def test_gam_linear_reduction_to_glm():
 x=np.arange(-3,4.)[:,None];y=[0,1,0,1,0,1,1]
 g=fit_gam_pd(x,y,feature_order=['x'],linear_features=['x'],smooth_specs=[],variant='unconstrained',**M)
 m=fit_binary_glm(x,y,feature_order=['x'],fit_intercept=True,link='logit',**M)
 assert g.model['parameters']['coefficients']==pytest.approx(m.model['parameters']['coefficients'],abs=1e-6)
 assert g.diagnostics['effective_degrees_of_freedom']==pytest.approx(2)

def smooth(mono='increasing',extra='error'):
 return dict(feature='x',boundary_knots=[-2,2],interior_knots=[-1,0,1],degree=3,smoothing=.02,monotonicity=mono,extrapolation=extra)
def test_gam_monotonic_constraints_derivatives_and_active_bounds():
 x=np.repeat(np.linspace(-2,2,9),8)[:,None];y=np.array([j<i for i in [1,1,2,3,4,5,6,7,7] for j in range(8)],int)
 m=fit_gam_pd(x,y,feature_order=['x'],linear_features=[],smooth_specs=[smooth()],variant='monotone',**M)
 grid=np.linspace(-2,2,301)[:,None];deriv=_gam_raw(m.model,grid,['x'],derivative_feature='x')
 assert deriv.min()>=-1e-12 and np.diff(pred(m,grid)).min()>=-1e-12
 assert np.max(abs(( _gam_raw(m.model,[[.2+1e-5]],['x'])-_gam_raw(m.model,[[.2-1e-5]],['x']))/2e-5-_gam_raw(m.model,[[.2]],['x'],derivative_feature='x')))<1e-8
 decreasing=fit_gam_pd(x,y,feature_order=['x'],linear_features=[],smooth_specs=[smooth('decreasing')],variant='monotone',**M)
 assert np.max(abs(pred(decreasing,grid)-y.mean()))<1e-6
 assert decreasing.diagnostics['active_constraint_count']>0
 with pytest.raises(ValueError,match='OUTSIDE_SUPPORT'):pred(m,[[3]])
 for extra in ['linear','constant']:
  em=fit_gam_pd(x,y,feature_order=['x'],linear_features=[],smooth_specs=[smooth(extra=extra)],variant='monotone',**M)
  assert pred(em,[[3]])[0]>=pred(em,[[2]])[0]

def plan(indices,features={'root':[0],'L':[0],'R':[0]}):return dict(sample_indices=indices,node_features=features)
def test_random_forest_hand_split_weighted_leaf_and_oob():
 x=[[0],[1],[2],[3]];y=[0,0,1,1]
 for criterion in ['gini','log_loss']:
  m=fit_random_forest_pd(x,y,feature_order=['x'],sampling_plan=[plan([0,1,2,3]),plan([0,0,2,2])],max_depth=1,criterion=criterion,**M)
  assert pred(m,[[0],[1],[2],[3]])==pytest.approx([0,0,1,1])
  assert m.model['parameters']['trees'][0]['threshold']==1.5
  assert m.diagnostics['oob_counts']==[0,1,0,1]
  assert m.diagnostics['oob_brier']==0
 m=fit_random_forest_pd(x,[0,1,0,1],feature_order=['x'],sampling_plan=[plan([0,1,2,3])],max_depth=1,min_samples_leaf=3,weights=[1,3,2,2],**M)
 assert pred(m,[[0],[3]])==pytest.approx([5/8,5/8])
 with pytest.raises(ValueError,match='INVALID_SAMPLING_PLAN'):
  fit_random_forest_pd(x,y,feature_order=['x'],sampling_plan=[dict(sample_indices=[0,1,2,3],node_features={})],max_depth=2,**M)

def test_forest_controlled_one_tree_matches_independent_cart():
 from sklearn.tree import DecisionTreeClassifier
 x=np.array([[0,1],[0,2],[1,1],[1,3],[2,0],[2,4],[3,2],[4,1]],float);y=[0,0,1,0,1,1,0,1];w=[1,2,1,3,1,1,2,1]
 features={'root':[0,1],'L':[0,1],'R':[0,1]}
 m=fit_random_forest_pd(x,y,feature_order=['x','z'],sampling_plan=[plan(list(range(8)),features)],max_depth=2,weights=w,**M)
 ref=DecisionTreeClassifier(max_depth=2,criterion='gini',random_state=0).fit(x,y,sample_weight=w)
 assert _predict_extended_pd(m,x,feature_order=['x','z'],expected_target=T).value==pytest.approx(ref.predict_proba(x)[:,1])

def test_boosting_one_tree_newton_leaf_hand_calculation():
 x=[[-1]]*4+[[1]]*4;y=[0]*4+[1]*4
 m=fit_gradient_boosting_pd(x,y,feature_order=['x'],train_row_ids=[str(i) for i in range(8)],iterations=1,max_depth=1,learning_rate=.3,l2_regularization=1,min_child_weight=0,**M)
 # initial p=.5, sum gradient +/-2, Hessian1, L2=1 -> leaf +/-1 times eta.
 assert _boost_raw(m.model,[[-1],[1]],['x'])==pytest.approx([-.3,.3],abs=2e-8)
 assert pred(m,[[-1],[1]])==pytest.approx(expit([-.3,.3]),abs=1e-8)

def test_boosting_monotonic_and_inner_early_stop_contract():
 x=np.linspace(-2,2,40)[:,None];y=np.array([0,1]*20)
 inner=dict(X=x,y=1-y,row_ids=['v'+str(i) for i in range(40)],role='inner_validation',split_id='inner',known_at='2024-12-31')
 kw=dict(feature_order=['x'],train_row_ids=[str(i) for i in range(40)],iterations=30,max_depth=2,learning_rate=.3,min_child_weight=0,inner_validation=inner,early_stopping_rounds=2,monotonic_constraints=[1])
 m=fit_gradient_boosting_pd(x,y,**kw,**M)
 assert m.diagnostics['selected_iterations']<30
 assert np.diff(pred(m,np.linspace(-2,2,101)[:,None])).min()>=-1e-12
 inner['row_ids'][0]='0'
 with pytest.raises(ValueError,match='INVALID_SPLIT'):fit_gradient_boosting_pd(x,y,**kw,**M)

def test_cubic_basis_independent_bernstein_polynomial_oracle():
 from creditratingsystem._classification_flexible import _gam_spline
 t=np.linspace(0,1,17)
 spec=dict(full_knots=[0]*4+[1]*4,degree=3,coefficient_transform=np.vstack([np.zeros(3),np.eye(3)]).tolist(),extrapolation='error')
 basis=_gam_spline(t,spec)
 assert basis==pytest.approx(np.c_[3*t*(1-t)**2,3*t*t*(1-t),t**3],abs=1e-15)

@pytest.mark.parametrize('monotonicity',['none','increasing'])
def test_gam_rejects_separation_in_unpenalized_affine_null_space(monotonicity):
 with pytest.raises(ValueError,match='SEPARATION'):
  fit_gam_pd([[-2],[-1],[1],[2]],[0,0,1,1],feature_order=['x'],linear_features=[],smooth_specs=[smooth(monotonicity)],variant='unconstrained' if monotonicity=='none' else 'monotone',**M)
