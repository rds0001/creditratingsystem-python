import math
from copy import deepcopy
import numpy as np
import pytest
from creditratingsystem.pd_term_structure import (conditional_pd_from_cumulative,hazard_to_pd,estimate_transition_matrix,project_transition_matrix,cumulative_pd_from_conditional)
from creditratingsystem.core import CreditRiskError

STATES=['A','B','D']
RATINGS=[dict(entity_id='x',at='2020-01-01',known_at='2020-01-01',state='A'),dict(entity_id='x',at='2020-07-01',known_at='2020-07-02',state='B'),dict(entity_id='x',at='2020-10-01',known_at='2020-10-02',state='D'),dict(entity_id='y',at='2020-01-01',known_at='2020-01-01',state='A'),dict(entity_id='z',at='2020-01-01',known_at='2020-01-01',state='B')]
WINDOWS=[dict(entity_id=e,start='2020-01-01',end='2021-01-01',complete_until='2021-01-01',known_at='2021-01-02') for e in ['x','y','z']]

def estimate(estimator='cohort',**kwargs):
 args=dict(ratings=RATINGS,observation_windows=WINDOWS,states=STATES,default_state='D',estimator=estimator,knowledge_date='2021-02-01',observation_scheme='exact_event_times',cohort_period_days=366 if estimator=='cohort' else None,days_per_year=366)
 args.update(kwargs)
 return estimate_transition_matrix(**args)

def project(m,t,kind='generator',**kw):
 return project_transition_matrix(m,t,matrix_kind=kind,states=STATES,default_state='D',**kw)

def test_curve_inverse_absorption_and_tiny_hazard():
 result=conditional_pd_from_cumulative([.1,.28,1,1]).value
 np.testing.assert_allclose(result['conditional_pd'][:3],[.1,.2,1])
 assert result['conditional_pd'][3] is None
 assert result['undefined_after_full_default']==[False,False,False,True]
 assert conditional_pd_from_cumulative([]).value['conditional_pd']==[]
 with pytest.raises(CreditRiskError,match='NONMONOTONE_CURVE'):conditional_pd_from_cumulative([.2,.1])
 result=hazard_to_pd([1e-20,.2,.2],[1,1,2]).value
 assert result['conditional_pd'][0]==pytest.approx(1e-20,rel=1e-12,abs=0)
 assert result['cumulative_pd'][-1]==pytest.approx(-math.expm1(-.6))
 assert sum(result['marginal_pd'])==pytest.approx(result['cumulative_pd'][-1])
 np.testing.assert_allclose(cumulative_pd_from_conditional(result['conditional_pd']).value,result['cumulative_pd'])

def test_discrete_power_analytic():
 m=np.array([[.8,.1,.1],[0,.9,.1],[0,0,1.]])
 result=project(m,2,'transition',step_length=1)
 np.testing.assert_allclose(result.value['matrix'],[[.64,.17,.19],[0,.81,.19],[0,0,1]],atol=1e-14)
 np.testing.assert_array_equal(project(m,0,'transition',step_length=1).value['matrix'],np.eye(3))
 with pytest.raises(CreditRiskError,match='INCOMPATIBLE_HORIZON'):project(m,.5,'transition',step_length=1)

def test_generator_repeated_eigenvalue_analytic_and_semigroup():
 # A->B->D at equal intensity gives Erlang(2) default from A.
 rate=.3;m=np.array([[-rate,rate,0],[0,-rate,rate],[0,0,0.]])
 result=project(m,2).value['matrix'];e=math.exp(-.6)
 np.testing.assert_allclose(result,[[e,.6*e,1-1.6*e],[0,e,1-e],[0,0,1]],atol=3e-15)
 np.testing.assert_allclose(project(m,.7).value['matrix']@project(m,1.3).value['matrix'],result,atol=2e-15)
 np.testing.assert_array_equal(project(m,0).value['matrix'],np.eye(3))

@pytest.mark.parametrize('matrix,kind,code',[([[-.2,.1,0],[0,-.1,.1],[0,0,0]],'generator','INVALID_GENERATOR'),([[.8,.1,.1],[0,.9,.1],[.1,0,.9]],'transition','DEFAULT_NOT_ABSORBING'),([[.8,.1,.2],[0,.9,.1],[0,0,1]],'transition','INVALID_TRANSITION_MATRIX')])
def test_invalid_matrices(matrix,kind,code):
 with pytest.raises(CreditRiskError,match=code):project(matrix,1,kind,step_length=1 if kind=='transition' else None)

def test_cohort_counts_and_absorption():
 result=estimate()
 np.testing.assert_array_equal(result.value['matrix'],[[.5,0,.5],[0,1,0],[0,0,1]])
 assert result.diagnostics['step_length_years']==1
 np.testing.assert_array_equal(result.value['counts'],[[1,0,1],[0,1,0],[0,0,0]])

def test_duration_exposure_analytic():
 result=estimate('duration')
 # x contributes 182 days in A, 92 B, 92 D. y contributes 366 A; z 366 B.
 np.testing.assert_allclose(result.value['exposure_years'],np.array([548,458,92])/366)
 np.testing.assert_allclose(result.value['matrix'],[[-366/548,366/548,0],[0,-366/458,366/458],[0,0,0]])

def test_censoring_and_already_observed_default():
 exits=[dict(entity_id='x',at='2020-11-01',known_at='2020-11-02'),dict(entity_id='y',at='2020-08-01',known_at='2020-08-02')]
 result=estimate(exits=exits)
 assert result.diagnostics['excluded_entities']==[{'entity_id':'y','reason':'censored_before_endpoint'}]
 assert result.value['matrix'][0,2]==1 # x has already defaulted before exiting
 duration=estimate('duration',exits=exits)
 assert duration.value['exposure_years'][0]==pytest.approx((182+213)/366)

def test_empty_rows_and_illegal_cure():
 with pytest.raises(CreditRiskError,match='EMPTY_TRANSITION_ROW'):estimate(states=['A','B','C','D'])
 explicit=estimate(states=['A','B','C','D'],empty_row_policy='identity')
 assert explicit.diagnostics['unidentified_rows']==['C']
 assert explicit.value['matrix'][2,2]==1
 ratings=deepcopy(RATINGS)+[dict(entity_id='x',at='2020-12-01',known_at='2020-12-02',state='A')]
 with pytest.raises(CreditRiskError,match='DEFAULT_NOT_ABSORBING'):estimate(ratings=ratings)
 with pytest.raises(CreditRiskError,match='INVALID_OBSERVATION_SCHEME'):estimate('duration',observation_scheme='state_carried_until_change')

def test_unknown_events_do_not_leak_and_duplicate_dates_rejected():
 ratings=deepcopy(RATINGS)+[dict(entity_id='y',at='2020-09-01',known_at='2022-01-01',state='D')]
 result=estimate(ratings=ratings)
 assert result.diagnostics['ignored_not_yet_known_records']==1
 assert result.value['matrix'][0,2]==.5
 with pytest.raises(CreditRiskError,match='AMBIGUOUS_EVENT'):estimate(ratings=RATINGS+[RATINGS[0]])
