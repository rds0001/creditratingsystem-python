import numpy as np
import pytest
from scipy.stats import multivariate_normal
from creditratingsystem._classification_models import (fit_discriminant_pd,fit_naive_bayes_pd,fit_svm_score,predict_model_score,_predict_extended_pd)
TARGET={'event':'default','horizon':1,'time_unit':'years','probability_measure':'PHYSICAL','conditioning':'performing'}
TRAIN={'data_fingerprint':'analytic-classification','split_id':'train','known_at':'2025-01-01','data_origin':'SYNTHETIC'}
META=dict(target=TARGET,training=TRAIN,model_id='classification-oracle',model_version='1')
def pred(m,x,features):return np.array(_predict_extended_pd(m,x,feature_order=features,expected_target=TARGET).value)
def test_discriminant_normal_density_and_equal_covariance():
 x=np.array([[-2,-1],[-2,1],[0,-1],[0,1],[1,-2],[1,0],[3,-2],[3,0.]])
 y=[0]*4+[1]*4
 for estimator in ['mle','unbiased']:
  lda=fit_discriminant_pd(x,y,feature_order=['x','z'],variant='lda',covariance_estimator=estimator,**META)
  qda=fit_discriminant_pd(x,y,feature_order=['x','z'],variant='qda',covariance_estimator=estimator,**META)
  z=np.array([[0,0],[1,.5],[-1,.2]])
  a=multivariate_normal.pdf(z,mean=[-1,0],cov=np.eye(2)*(1 if estimator=='mle' else 4/3))
  b=multivariate_normal.pdf(z,mean=[2,-1],cov=np.eye(2)*(1 if estimator=='mle' else 4/3))
  assert pred(lda,z,['x','z'])==pytest.approx(b/(a+b),abs=1e-14)
  assert pred(qda,z,['x','z'])==pytest.approx(pred(lda,z,['x','z']),abs=1e-14)
 with pytest.raises(ValueError,match='SINGULAR_COVARIANCE'):fit_discriminant_pd([[0,0],[1,1],[2,2],[3,3]],[0,0,1,1],feature_order=['x','z'],variant='qda',**META)
 m=fit_discriminant_pd([[0,0],[1,1],[2,2],[3,3]],[0,0,1,1],feature_order=['x','z'],variant='qda',covariance_ridge=.1,**META)
 assert np.isfinite(pred(m,[[0,0]],['x','z'])).all()

def test_categorical_naive_bayes_hand_bayes_unknown_missing():
 x=[['a','yes'],['a','no'],['b','no'],['b','yes'],['b','yes'],['b','no']]; y=[0,0,0,1,1,1]
 m=fit_naive_bayes_pd(x,y,feature_order=['a','b'],feature_types=['categorical']*2,variant='categorical',alpha=1,unknown_policy='reserved',missing_policy='category',**META)
 # six rows, each class 3; each variable 2 levels + missing + unknown => denominator7.
 # P(b,yes|1)=4*3/49, P(b,yes|0)=2*2/49 => posterior3/4.
 assert pred(m,[['b','yes']],['a','b'])==pytest.approx([.75])
 assert pred(m,[['unseen',None]],['a','b'])==pytest.approx([.5])
 err=fit_naive_bayes_pd(x,y,feature_order=['a','b'],feature_types=['categorical']*2,variant='categorical',**META)
 with pytest.raises(ValueError,match='UNKNOWN_CATEGORY'):pred(err,[['unseen','no']],['a','b'])

def test_gaussian_mixed_missing_marginalizes_and_frozen():
 from sklearn.naive_bayes import GaussianNB
 x=np.array([[-2,-1],[-1,1],[0,0],[1,0],[2,2],[3,1.]]) ;y=[0,0,0,1,1,1]
 m=fit_naive_bayes_pd(x,y,feature_order=['a','b'],feature_types=['gaussian']*2,variant='gaussian',missing_policy='marginalize',**META)
 ref=GaussianNB(var_smoothing=0).fit(x,y)
 assert pred(m,x,['a','b'])==pytest.approx(ref.predict_proba(x)[:,1],abs=1e-14)
 assert pred(m,[[None,None]],['a','b'])==pytest.approx([.5])
 mixed=[[r[0],'a' if i<3 else 'b'] for i,r in enumerate(x)]
 mm=fit_naive_bayes_pd(mixed,y,feature_order=['a','b'],feature_types=['gaussian','categorical'],variant='mixed',missing_policy='marginalize',**META)
 assert pred(mm,[[None,'a']],['a','b'])==pytest.approx([1/5])
 assert pred(m,[[1,0]],['a','b'])[0]==pred(m,[[100,100],[1,0]],['a','b'])[1]

def test_svm_analytic_two_point_margin_and_frozen_kernel():
 for kernel,gamma in [('linear',None),('rbf',.5)]:
  m=fit_svm_score([[-1],[1]],[0,1],feature_order=['x'],kernel=kernel,C=10,gamma=gamma,scale=False,**META)
  points=[[-1],[0],[1]]
  score=predict_model_score(m,points,feature_order=['x'],expected_target=TARGET)
  assert score.value==pytest.approx([-1,0,1],abs=1e-7)
  assert score.diagnostics['scale']=='SVM_MARGIN'
  with pytest.raises(ValueError,match='UNCALIBRATED_SCORE'):pred(m,points,['x'])

def test_firth_separation_and_exact_intercept_jeffreys():
 from creditratingsystem._classification_models import fit_firth_logit,_firth_objective
 from scipy.optimize._numdiff import approx_derivative
 x=np.zeros((8,0)); y=[0]*7+[1]
 m=fit_firth_logit(x,y,feature_order=[],**META)
 assert m.model['parameters']['coefficients']==pytest.approx([np.log(1.5/7.5)],abs=1e-7)
 m=fit_firth_logit([[-2],[-1],[1],[2]],[0,0,1,1],feature_order=['x'],**META)
 assert np.isfinite(m.model['parameters']['coefficients']).all()
 d=np.c_[np.ones(4),[-2,-1,1,2]];y=np.array([0,0,1,1]); beta=np.array([.4,.8])
 analytic=_firth_objective(beta,d,y)[1];numeric=approx_derivative(lambda b:_firth_objective(b,d,y)[0],beta).ravel()
 assert analytic==pytest.approx(numeric,abs=1e-9)

def test_penalized_path_zero_and_kkt_weight_replication():
 from creditratingsystem._classification_models import fit_penalized_glm
 from creditratingsystem.classification import fit_binary_glm
 x=np.array([[-2],[-1],[0],[1],[2],[3.]]) ;y=np.array([0,1,0,1,0,1]);weights=[2,1,1,2,1,1]
 ordinary=fit_binary_glm(x,y,link='logit',feature_order=['x'],fit_intercept=True,**META)
 for penalty in ['ridge','lasso','elasticnet']:
  kw=dict(penalty=penalty,lambda_path=[10,.1,0],selected_lambda=0,feature_order=['x'],scale=False)
  if penalty=='elasticnet':kw['l1_ratio']=.4
  m=fit_penalized_glm(x,y,**kw,**META)
  assert m.model['parameters']['coefficients']==pytest.approx(ordinary.model['parameters']['coefficients'],abs=1e-7)
  if penalty=='lasso':assert m.diagnostics['path'][0]['coefficients'][1]==0
  kw['selected_lambda']=.1
  weighted=fit_penalized_glm(x,y,weights=weights,**kw,**META)
  replicated=fit_penalized_glm(np.repeat(x,weights,axis=0),np.repeat(y,weights),**kw,**META)
  assert pred(weighted,x,['x'])==pytest.approx(pred(replicated,x,['x']),abs=1e-7)

def test_ordinal_intercept_exact_frequencies_and_default_mapping():
 from creditratingsystem._classification_models import fit_ordinal_model,_ordinal_probabilities
 x=np.zeros((10,0));labels=['good']*5+['weak']*3+['default']*2
 for link in ['logit','probit']:
  m=fit_ordinal_model(x,labels,feature_order=[],ordered_categories=['good','weak','default'],link=link,default_categories=['default'],**META)
  probs=_ordinal_probabilities(m.model,np.zeros((1,0)),[],TARGET)
  assert probs[0]==pytest.approx([.5,.3,.2],abs=1e-14)
  assert pred(m,np.zeros((1,0)),[])==pytest.approx([.2])
  m=fit_ordinal_model(x,labels,feature_order=[],ordered_categories=['good','weak','default'],link=link,**META)
  with pytest.raises(ValueError,match='UNMAPPED_DEFAULT'):pred(m,np.zeros((1,0)),[])

def test_ordinal_nonzero_fit_probs_and_joint_diagnostic():
 from creditratingsystem._classification_models import fit_ordinal_model,_ordinal_probabilities
 x=np.repeat(np.arange(-2,3.),12)[:,None]
 # Fixed category frequencies by x, all thresholds nonseparated.
 counts=[[9,2,1],[7,3,2],[4,4,4],[2,3,7],[1,2,9]]
 labels=[c for row in counts for c,n in zip(['good','weak','default'],row) for _ in range(n)]
 for link in ['logit','probit']:
  m=fit_ordinal_model(x,labels,feature_order=['x'],ordered_categories=['good','weak','default'],link=link,default_categories=['default'],**META)
  assert m.model['parameters']['coefficients'][0]>0
  p=_ordinal_probabilities(m.model,[[-1],[0],[1]],['x'],TARGET)
  assert p.sum(axis=1)==pytest.approx(np.ones(3),abs=1e-14)
  assert m.diagnostics['parallel_slopes_diagnostic']['status']=='ESTIMATED'
  assert m.diagnostics['parallel_slopes_diagnostic']['p_value']>.5

def test_firth_zero_default_sample_is_finite():
 from creditratingsystem._classification_models import fit_firth_logit
 m=fit_firth_logit(np.zeros((8,0)),np.zeros(8),feature_order=[],**META)
 assert pred(m,np.zeros((1,0)),[])==pytest.approx([.5/9],abs=1e-9)

def test_svm_orientation_reversed_first_class_and_control():
 m=fit_svm_score([[1],[-1]],[1,0],feature_order=['x'],kernel='linear',C=10,scale=False,**META)
 assert predict_model_score(m,[[-1],[1]],feature_order=['x'],expected_target=TARGET).value==pytest.approx([-1,1])
