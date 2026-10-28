from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json,math,runpy
import pytest
import creditratingsystem as cr


def load():return json.loads(files('creditratingsystem.datasets').joinpath('market_credit_review.json').read_text())


def test_quotes_identification_and_distinct_probability_measures():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples/market_credit_review.py'))['run']
    result=run(return_artifacts=True)
    assert [c['definition']['probability_measure']for c in result['assessment_components']]==['PHYSICAL','RISK_NEUTRAL']
    assert result['bridge']['physical_pd']>result['bridge']['q_pd']
    assert result['identification']['rank_without_recovery']==2
    assert result['identification']['rank_with_recovery']==3
    assert result['bridge']['warmup_count']==100
    restored=json.loads(json.dumps(result['artifacts']['bridge']))
    p=cr.predict_pd(restored,[[.01],[.05]],feature_order=['q_pd'],expected_target=load()['bridge_contract']['target']).value
    assert math.isclose(p[0],.02,abs_tol=1e-10)and math.isclose(p[1],.1,abs_tol=1e-10)
    wrong=dict(restored['target'],probability_measure='RISK_NEUTRAL')
    with pytest.raises(cr.CreditRiskError):cr.predict_pd(restored,[[.01]],feature_order=['q_pd'],expected_target=wrong)


def test_future_market_release_and_reused_issuer_are_rejected():
    d=load();rows=deepcopy(d['bridge_observations']);rows[0]['market_curve_known_at']='2019-01-01'
    with pytest.raises(cr.CreditRiskError,match='LOOKAHEAD_BIAS'):
        cr.fit_market_to_physical_pd_bridge(rows,['q_pd'],d['bridge_contract'],d['folds'],model_id='bridge',model_version='1')
    rows=deepcopy(d['bridge_observations']);rows[100]['entity_id']=rows[0]['entity_id']
    with pytest.raises(cr.CreditRiskError,match='LEAKAGE'):
        cr.fit_market_to_physical_pd_bridge(rows,['q_pd'],d['bridge_contract'],d['folds'],model_id='bridge',model_version='1')
