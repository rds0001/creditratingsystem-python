import pytest
from creditratingsystem import fit_naive_bayes_pd,fit_svm_score,predict_pd,CreditRiskError
TARGET=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
META=dict(target=TARGET,training=dict(data_fingerprint='dispatch',split_id='train',known_at='2024-01-01',data_origin='SYNTHETIC'),model_id='dispatch',model_version='1')

def test_public_prediction_dispatch_preserves_semantics():
    model=fit_naive_bayes_pd([['a'],['b'],['a'],['b']],[0,0,1,1],feature_order=['x'],feature_types=['categorical'],variant='categorical',**META)
    p=predict_pd(model,[['a']],feature_order=['x'],expected_target=TARGET)
    assert p.value==pytest.approx([.5])
    assert p.provenance['data_origin']=='SYNTHETIC'
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):
        predict_pd(model,[['a']],feature_order=['x'],expected_target=TARGET,context={'probability_measure':'RISK_NEUTRAL'})
    with pytest.raises(CreditRiskError,match='OFFSET_MISMATCH'):
        predict_pd(model,[['a']],feature_order=['x'],expected_target=TARGET,offset=[0])
    svm=fit_svm_score([[-1],[1]],[0,1],feature_order=['x'],kernel='linear',C=1,**META)
    with pytest.raises(CreditRiskError,match='UNCALIBRATED_SCORE'):
        predict_pd(svm,[[0]],feature_order=['x'],expected_target=TARGET)
