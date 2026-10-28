from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._explainability_reasons import rank_credit_reason_codes


def fixture():
    rows=[dict(feature_id=f,contribution=v)for f,v in [('a',3),('a_squared',-1),('b',2),('c',-4),('other',1)]]
    basis=dict(attribution_id='a',model_id='m',model_version='1',reference_id='r',explained_quantity='default_log_odds',output_scale='log_odds',baseline=-2,prediction=-1,adverse_direction='HIGHER')
    dictionary=dict(dictionary_id='d',version='1',model_reasons=[dict(reason_code='A',description='A grouped contribution',priority=2,feature_ids=['a','a_squared']),dict(reason_code='B',description='B contribution',priority=1,feature_ids=['b']),dict(reason_code='C',description='C contribution',priority=0,feature_ids=['c'])],policy_reasons=[])
    return rows,basis,dictionary


def test_group_netting_ties_coverage_and_limit():
    r=rank_credit_reason_codes(*fixture(),max_reasons=1).value
    assert [x['reason_code']for x in r['model_reasons']]==['B']
    assert r['omitted_by_limit'][0]['reason_code']=='A'
    assert r['omitted_by_limit'][0]['contribution']==2
    assert r['suppressed_groups'][0]['reason_code']=='C'
    assert r['coverage']['mapped_absolute_fraction']==pytest.approx(10/11)
    assert r['unmapped_contributions']==[dict(feature_id='other',contribution=1)]
    assert r['reconstruction_residual']==0


def test_direction_reversal_and_reconstruction_validation():
    rows,basis,dictionary=fixture();basis['adverse_direction']='LOWER'
    r=rank_credit_reason_codes(rows,basis,dictionary).value
    assert [x['reason_code']for x in r['model_reasons']]==['C']
    basis['prediction']=0
    with pytest.raises(CreditRiskError,match='REASON_RECONSTRUCTION'):rank_credit_reason_codes(rows,basis,dictionary)


def test_duplicate_transformed_feature_mapping_rejected():
    rows,basis,dictionary=fixture();dictionary['model_reasons'][1]['feature_ids']=['a']
    with pytest.raises(CreditRiskError,match='REASON_GROUPS'):rank_credit_reason_codes(rows,basis,dictionary)


def test_policy_evidence_stays_separate_and_sealed():
    from creditratingsystem.core import _fingerprint
    rows,basis,dictionary=fixture()
    trace=dict(rule_id='eligibility',reason_code='NOT_ELIGIBLE',matched=True,selected=True,action='DECLINE',sources=['eligibility'])
    d=dict(schema_version='1.0',kind='DecisionArtifact',action='DECLINE',findings=['eligibility.check'],rule_trace=[trace]);d['sha256']=_fingerprint(d)
    dictionary['policy_reasons']=[dict(reason_code='NOT_ELIGIBLE',description='Eligibility condition failed',priority=0,rule_ids=['eligibility'])]
    r=rank_credit_reason_codes(rows,basis,dictionary,policy_decision=d).value
    assert len(r['model_reasons'])==2;assert len(r['policy_reasons'])==1
    assert r['policy_reasons'][0]['rules']==[trace]
    assert r['policy_decision_hash']==d['sha256']
    broken=deepcopy(d);broken['action']='APPROVE'
    with pytest.raises(CreditRiskError):rank_credit_reason_codes(rows,basis,dictionary,policy_decision=broken)
