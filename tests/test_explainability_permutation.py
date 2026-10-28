from copy import deepcopy
import numpy as np
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._explainability_permutation import credit_permutation_importance


def fixture():
    m=dict(model_id='heldout',model_version='1',feature_order=['x','y'],output_scale='probability',deterministic_frozen=True,target=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing'))
    design=dict(split='OUT_OF_TIME',training_end='2022-12-31',data_cutoff='2024-02-01',train_row_ids=['train'],train_subject_ids=['trained'],model_and_calibration_frozen=True,selection_use='FORBIDDEN',permutation_design='FIXED_PERMUTATIONS')
    meta=[dict(row_id=str(i),subject_id='s'+str(i),observation_time='2023-01-01',target_end='2024-01-01',label_known_time='2024-01-01')for i in range(2)]
    return dict(predictor=lambda z:z[:,0],X=[[0,0],[1,1]],y=[0,1],model_contract=m,feature_groups={'x':['x'],'y':['y']},permutations=[[0,1],[1,0]],design=design,loss_contract=dict(loss_id='brier',version='1',kind='BRIER'),row_metadata=meta)


def test_known_brier_importance_unused_feature_and_shared_plan():
    r=credit_permutation_importance(**fixture()).value
    assert r['baseline_loss']==0
    assert r['groups'][0]['loss_increases']==[0,1];assert r['groups'][0]['mean_loss_increase']==.5
    assert r['groups'][1]['mean_loss_increase']==0
    assert r['groups'][0]['permutation_standard_error']is None
    assert r['weighted_reference_correlation']==[[1,1],[1,1]]


def test_iid_permutation_uncertainty_and_joint_block_support():
    a=fixture();a['design']['permutation_design']='IID_RANDOM_PERMUTATIONS';a['support_evaluator']=lambda z:z[:,0]==z[:,1];a['support_contract']=dict(support_id='equal',version='1')
    r=credit_permutation_importance(**a).value
    assert r['groups'][0]['permutation_standard_error']==.5
    assert r['groups'][0]['support_findings'][1]['unsupported_weight']==1
    # Unsupported rows retain their original loss contribution.
    assert r['groups'][0]['permuted_losses'][1]==1
    a['feature_groups']={'joint':['x','y']};r=credit_permutation_importance(**a).value
    assert r['groups'][0]['mean_loss_increase']==.5
    assert r['groups'][0]['support_findings'][1]['unsupported_weight']==0


def test_leakage_timing_and_stratum_restrictions():
    a=fixture();a['strata']=['one','two']
    with pytest.raises(CreditRiskError,match='PERMUTATION_STRATA'):credit_permutation_importance(**a)
    a=fixture();a['design']['train_subject_ids']=['s0']
    with pytest.raises(CreditRiskError,match='PERMUTATION_LEAKAGE'):credit_permutation_importance(**a)
    a=fixture();a['row_metadata'][0]['target_end']='2025-01-01'
    with pytest.raises(CreditRiskError,match='PERMUTATION_TIMING'):credit_permutation_importance(**a)
    a=fixture();a['permutations']=[[1,1]]
    with pytest.raises(CreditRiskError,match='PERMUTATION_PLAN'):credit_permutation_importance(**a)


def test_log_loss_does_not_silently_clip_and_custom_loss_is_explicit():
    a=fixture();a['loss_contract']['kind']='LOG_LOSS'
    with pytest.raises(CreditRiskError,match='PERMUTATION_INFINITE_LOSS'):credit_permutation_importance(**a)
    a['loss_contract']['kind']='CUSTOM';a['custom_loss']=lambda y,p:2*(y-p)**2
    r=credit_permutation_importance(**a).value
    assert r['groups'][0]['mean_loss_increase']==1
