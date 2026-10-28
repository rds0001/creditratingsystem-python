from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import importlib.util
from importlib.resources import files
import json
import math
import pytest
import creditratingsystem as cr

path=Path(__file__).parents[1]/'examples'/'retail_application_journey.py'
spec=importlib.util.spec_from_file_location('retail_application_journey',path)
journey=importlib.util.module_from_spec(spec);spec.loader.exec_module(journey)
P='RETAIL_UNSECURED_INSTALLMENT'
@lru_cache(None)
def fixture():return journey.run(product_ids=[P],return_artifacts=True)

def test_real_application_funding_identity_and_maturity():
    out=fixture();p=out['products'][P];rows=p['applications'];labels=p['targets']
    assert [r['target']for r in labels]==[0,1,None,None,None]
    assert [r['decision']['action']for r in rows]==['APPROVE','APPROVE','DECLINE','APPROVE','APPROVE']
    assert len(p['performing']['accounts'])==3
    assert {r['account_id']for r in p['performing']['accounts']}=={r['facility_id']for r in labels if r['funded']}
    assert p['performing']['validation']['decision']['status']=='PASS'
    m=p['monitoring']['monitor'][0];assert m['metrics']['n']==2 and m['excluded_count']==1 and m['metrics']['observed_rate']==.5
    assert m['metrics']['brier']==pytest.approx(1/16,abs=1e-10)and m['status']=='INSUFFICIENT_INFORMATION'
    assert p['monitoring']['replay_validation']['decision']['status']=='PASS'
    assert p['monitoring']['replay_validation']['metrics']['duplicate_identical_deliveries']==1

def test_independent_fit_and_actual_cash_payment_oracles():
    p=fixture()['products'][P];model=p['fitted']['model'];beta=model['parameters']['coefficients']
    assert beta==pytest.approx([0,math.log(3)],abs=1e-8)
    assert p['fitted']['calibrator']['parameters']['intercept']==pytest.approx(0,abs=1e-8)
    assert p['fitted']['calibrator']['parameters']['slope']==pytest.approx(1,abs=1e-8)
    assert p['policy_validation']['result']['metrics']['factual']['funded_rate']==.6
    assert p['policy_validation']['result']['metrics']['factual']['observed_pd']==.5
    assert p['policy_validation']['result']['metrics']['factual']['observed_mean_loss']==200
    assert p['affordability_validation']['result']['metrics']['factual_scheduled_minus_paid']['bias']==50
    assert [r['dpd']['dpd']for r in p['performing']['payment_history']]==[0,5]
    assert all(a['result']['decision']['status']=='PASS'for a in fixture()['sources']['audits'])
    assert all(not r['release_id'].startswith('late-')for r in fixture()['sources']['rows'])

def test_pseudolabel_and_replay_mutations_are_detected():
    p=fixture()['products'][P];kw=deepcopy(p['performing']['validation_inputs']);kw['application_targets'][2].update(target=0,label_observed=True,label_origin='PSEUDO')
    r=cr.validate_retail_cohort(**kw).value;assert r['decision']['status']=='FAIL'and 'UNFUNDED_OWN_CONTRACT_OUTCOME'in r['violation_counts']
    kw=deepcopy(p['monitoring']['replay_inputs']);kw['replayed_checkpoints'][0]['state']['ratings'][0]['row_id']='unrelated-account'
    assert cr.replay_consistency(**kw).value['decision']['status']!='PASS'

def test_training_cutoffs_and_joint_identity():
    data=json.loads(files('creditratingsystem.datasets').joinpath('retail_application_journey.json').read_text());case=deepcopy(data['products'][P]);case['training'][0]['feature_known_at']='2024-01-01'
    with pytest.raises(ValueError,match='chronology'):journey.fit_product_models(P,case,data['target'])
    p=fixture()['products'][P];assert p['applications'][0]['bureau']['file_state']=='NO_HIT'
    assert p['applications'][3]['bureau']['file_state']=='STALE'
    assert p['applications'][0]['risk']['pd']==pytest.approx(.25,abs=1e-8)
