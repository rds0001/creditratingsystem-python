"""Independent matched-cell oracles for chronological applicant development."""
import importlib.util
from pathlib import Path
from functools import lru_cache
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

_spec=importlib.util.spec_from_file_location('retail_development_example',Path(__file__).parents[1]/'examples'/'retail_development.py')
_example=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_example)

@lru_cache(maxsize=1)
def result():
    return _example.run()

def test_independent_holdout_and_selection_cell_algebra():
    v=result();a=v['selection_analysis'];m=v['holdout']['benchmark']['metrics']['by_model']['scorecard']
    assert abs(m['brier']-3/16)<1e-12 and abs(m['auc']-.75)<1e-12
    assert abs(m['log_loss']-(-.25*math.log(.25)-.75*math.log(.75)))<1e-12
    assert abs(a['ipw']['effective_sample_size']-64**2/(16*2**2+24*(4/3)**2))<1e-7
    assert abs(a['aipw']['population_pd']-.5)<1e-8
    assert a['bounds']['lower']==20/64 and a['bounds']['upper']==44/64
    assert a['overlap_failure']=='POSITIVITY_FAILURE'
    beta=v['models']['scorecard']['parameters']['coefficients']
    np.testing.assert_allclose(beta,[0,math.log(3)],atol=1e-8)
    np.testing.assert_allclose([v['logistic_calibrator']['parameters'][k]for k in ['intercept','slope']],[0,1],atol=1e-8)

def test_thin_evidence_and_independent_firth_integral():
    v=result();thin=v['thin_file'];beta=thin['model']['parameters']['coefficients'];slope=2*math.log(21/61)
    np.testing.assert_allclose(beta,[-slope,slope],atol=2e-7)
    for name in ['no_hit','stale']:
        c=thin['cases'][name]
        assert c['income']['total_selected_income']==100 and c['cashflow_ratio']==1
        assert abs(c['raw_application_risk']['pd']-.5)<1e-8
    assert thin['cases']['outside_support']['raw_application_risk'] is None
    params=thin['hierarchical_calibrator']['parameters']
    integral=quad(lambda z:expit(params['intercept']+math.sqrt(params['variance'])*z)*math.exp(-z*z/2)/math.sqrt(2*math.pi),-12,12,epsabs=1e-12)[0]
    assert abs(thin['unknown_group_reference'][0]-integral)<2e-7
    assert thin['population_bounds']['BUREAU_NO_HIT']['lower']==13/16
    assert thin['population_bounds']['BUREAU_STALE']['lower']==7/48
    assert thin['population_bounds']['BUREAU_STALE']['upper']==31/48

def test_chronological_roles_and_unknown_outcomes_remain_unknown():
    v=result();rows=v['prepared']['rows'];roles={role:[r for r in rows if r['role']==role]for role in ['train','calibration','test']}
    assert [len(roles[r])for r in roles]==[128,64,64]
    groups=[{r['entity_id']for r in roles[role]}for role in roles]
    assert not(groups[0]&groups[1]or groups[0]&groups[2]or groups[1]&groups[2])
    for row in rows:
        if not row['selected']:assert row['outcome'] is None
        assert row['features_known_at']<=row['as_of']
        assert row['snapshot']['features']['x']!=99
    assert sum(r['selected']for r in roles['test'])==40

def test_untouched_test_outcomes_cannot_refit_models():
    import json
    from importlib.resources import files
    data=json.loads(files('creditratingsystem.datasets').joinpath('retail_development.json').read_text())
    data['defaults']=[r for r in data['defaults'] if not r['default_at'].startswith('2022-')]
    changed=_example.run(data=data)
    reference=result()
    assert changed['models']==reference['models']
    assert changed['logistic_calibrator']==reference['logistic_calibrator']
    assert changed['thin_file']['hierarchical_calibrator']==reference['thin_file']['hierarchical_calibrator']
    assert changed['holdout']['benchmark']['metrics']['by_model']['scorecard']['observed_rate']==0
    assert changed['holdout']['benchmark']['metrics']['by_model']['scorecard']['auc'] is None

def test_selection_assumptions_and_pseudo_rows_not_truth():
    from scipy.stats import norm
    v=result();p=v['models']['joint']['parameters'];q=norm.ppf(.75)
    np.testing.assert_allclose(p['outcome_coefficients'],[0,q],atol=1e-7)
    np.testing.assert_allclose(p['selection_coefficients'],[q/2,q/2],atol=1e-7)
    assert p['rho']==0
    np.testing.assert_allclose(v['selection_analysis']['parcel']['assumed_reject_bad_rate'],[2/5,6/7],atol=1e-12)
    assert not v['selection_analysis']['parcel']['validation_eligible']
    weights={}
    for row in v['selection_analysis']['fuzzy']['rows']:
        if row['label_origin']=='HYPOTHETICAL':
            assert not row['validation_eligible']
            weights[row['application_id']]=weights.get(row['application_id'],0)+row['weight']
    assert len(weights)==48 and all(abs(w-1)<1e-12 for w in weights.values())
    pseudo=[r for r in v['holdout']['evaluation_roles']['rows']if r['label_origin']=='PSEUDO']
    assert len(pseudo)==1 and not pseudo[0]['validation_eligible']
