import numpy as np
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._explainability_ale import compute_credit_ale


def contract():
    return dict(model_id='ale',model_version='1',feature_order=['x','y'],output_scale='score',deterministic_frozen=True,target=dict(event='score',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing'))


def test_linear_weighted_center_and_correlated_reference():
    r=compute_credit_ale(lambda z:2*z[:,0]+3*z[:,1],[[0,0],[1,10],[2,20],[3,30]],contract(),selected_features=['x'],bins={'x':[0,1,2,3]},feature_types={'x':'continuous'},weights=[1,1,2,2]).value
    assert r['status']=='ESTIMABLE';assert r['local_differences']==[2,2,2]
    assert r['centered_ale_grid']==pytest.approx([-11/3,-5/3,1/3,7/3])
    assert np.dot(r['reference_ale'],[1,1,2,2])==pytest.approx(0,abs=1e-14)
    assert r['positive_weight_counts']==[1,1,2]


def test_genuine_purified_interaction_and_additive_zero_oracles():
    x=[[-1,-1],[-1,1],[1,-1],[1,1]];kw=dict(selected_features=['x','y'],bins={'x':[-1,0,1],'y':[-1,0,1]},feature_types={'x':'continuous','y':'continuous'})
    r=compute_credit_ale(lambda z:z[:,0]*z[:,1],x,contract(),**kw).value
    assert r['local_differences']==[[1,1],[1,1]]
    assert r['centered_ale_grid']==[[1,0,-1],[0,0,0],[-1,0,1]]
    assert r['reference_ale']==[1,-1,-1,1]
    assert r['purification']['first_axis_accumulated']==[0,1,2]
    r=compute_credit_ale(lambda z:2*z[:,0]+3*z[:,1],x,contract(),**kw).value
    assert r['centered_ale_grid']==[[0,0,0],[0,0,0],[0,0,0]]


def test_unsupported_correlated_cells_remain_non_estimable():
    r=compute_credit_ale(lambda z:z[:,0]*z[:,1],[[-1,-1],[1,1]],contract(),selected_features=['x','y'],bins={'x':[-1,0,1],'y':[-1,0,1]},feature_types={'x':'continuous','y':'continuous'}).value
    assert r['status']=='NON_ESTIMABLE_SUPPORT';assert r['unsupported_cells']==[[0,1],[1,0]]
    assert r['centered_ale_grid']is None;assert r['reference_ale']is None
    assert r['local_differences']==[[1,None],[None,1]]
    assert r['weighted_selected_correlation']==[[1,1],[1,1]]


def test_categorical_interpolation_and_insufficient_count_are_explicit():
    with pytest.raises(CreditRiskError,match='ALE_FEATURE_TYPE'):
        compute_credit_ale(lambda z:z[:,0],[[0,0],[1,1]],contract(),selected_features=['x'],bins={'x':[0,1]},feature_types={'x':'categorical'})
    r=compute_credit_ale(lambda z:z[:,0],[[0,0],[1,1]],contract(),selected_features=['x'],bins={'x':[0,1]},feature_types={'x':'continuous'},weights=[1,0],min_bin_count=2).value
    assert r['positive_weight_counts']==[1];assert r['unsupported_cells']==[[0]];assert r['centered_ale_grid']is None
