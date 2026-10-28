from copy import deepcopy
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import creditratingsystem as cr

path=Path(__file__).resolve().parents[1]/'examples'/'selection_transport.py'
spec=importlib.util.spec_from_file_location('selection_example',path);example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)


def fixture():
    target=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='potential_own_contract')
    fit=cr.fit_binary_glm(np.empty((4,0)),[0,1,0,1],feature_order=[],link='logit',fit_intercept=True,target=target,training=dict(data_fingerprint='independent2018four-observations',known_at='2019-01-01',split_id='prior-training',data_origin='SYNTHETIC'),model_id='prior',model_version='1')
    pd=cr.predict_pd(fit,np.empty((1,0)),feature_order=[],expected_target=target).value[0];assert abs(pd-.5)<1e-12
    records=[dict(application_id=str(i),customer_id='customer-'+str(i),facility_id='loan-'+str(i)if i<2 else None,application_at='2024-01-01',decision_at='2024-01-01',first_draw_at='2024-01-02'if i<2 else None,funded=i<2,observed=i<2,outcome=i if i<2 else None,offered=i!=2,policy_version='1',pd=pd)for i in range(4)]
    return dict(product='own-data',records=records,target=target,model=fit.model,calibrator=None,training_entities=['old0','old1','old2','old3'],cutoff='2024-02-02',assumptions=dict(observation_propensity=.5,log_odds_shifts=[-1,0,1],reference='Explicit scenario'))


def test_actual_fit_and_independent_selection_oracles():
    out=example.audit(**fixture());s=out['summary'];assert s['bounds']['lower']==.25 and s['bounds']['upper']==.75
    assert s['brier']==.25 and s['mar_effective_sample_size']==2
    assert s['mnar_population_pd'][1]['population_pd']==.5
    assert s['stage_counts']=={'funded_observed':2,'rejected':1,'offered_not_taken_up':1}
    assert not s['pseudo_labels_change_factual_count'] and s['zero_support_status']=='NOT_IDENTIFIED_ZERO_SUPPORT'


def test_false_unfunded_label_and_future_model_rejected():
    kw=fixture();kw['records'][2].update(observed=True,outcome=0)
    with pytest.raises(cr.CreditRiskError):example.audit(**kw)
    kw=fixture();kw['model']['training']['known_at']='2024-03-01'
    with pytest.raises(cr.CreditRiskError):example.audit(**kw)


def test_reported_arithmetic_mutation_blocked():
    out=example.audit(**fixture());args=deepcopy(out['validation_inputs']);args['selection_results'][1]['result']['lower']=0
    assert cr.selection_bias_sensitivity(**args).value['decision']['status']=='BLOCKED'
