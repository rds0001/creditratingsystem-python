from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json
import math
import runpy
import pytest
import creditratingsystem as cr
E=runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/retail_collections_review.py'))
def data():return json.loads(files('creditratingsystem.datasets').joinpath('retail_collections_review.json').read_text())
@pytest.fixture(scope='module')
def result():return E['run'](return_artifacts=True)
def test_same_claim_prior_probation_models_exclusive_cash_mixture(result):
 r=result;residual=100-1000/120
 assert r['remaining_claim']==pytest.approx(residual)
 assert r['action_value']['expected_net_recovery']==pytest.approx(.25*(7/12*residual-5)+.75*20-2)
 assert r['no_uptake_value']['expected_net_recovery']==18
 assert r['landmark_prediction']['sustained_cure_probability'][-1]==pytest.approx(2/3)
 assert r['landmark_prediction']['cure_then_redefault_probability'][-1]==pytest.approx(1/3)
 assert r['prior_first_cure_forecasts']['fine_gray']['cumulative_incidence']['cure'][-1]==pytest.approx(1-math.exp(-1/3.75-1/2.75))
 assert r['cure_targets'][0]['cure_at']=='2024-03-16'and r['cure_targets'][0]['redefault_at']=='2024-04-15'
 assert r['observed_episodes']['episodes'][1]['previous_episode_id']==r['observed_episodes']['episodes'][0]['episode_id']
 assert all(t['cure_label']is None for t in r['missing_probation_targets'])
def test_offer_not_acceptance_and_frozen_value_replay(result):
 raw=result['raw_artifacts'];d=data();a=result['action_history']['rows'][0]
 assert not a['accepted']and not a['performed']and not a['causal_effect_identified']
 replay=cr.expected_collections_value([raw['action_value_input']],d['discount_factors'],ead=result['remaining_claim'],currency='EUR',period_end_dates=d['period_end_dates']).value[0]
 assert replay==result['action_value']
 assert raw['multistate_model']['training']['known_at']<'2024-02-15'
 assert d['open_workout']['episode']['resolution_date']is None
 assert all(c['known_at']<=d['valuation_date']for c in d['open_workout']['cashflows'])
def test_future_model_and_post_cure_valuation_are_rejected():
 d=data();d['training']['known_at']='2025-01-01'
 with pytest.raises(ValueError,match='genuinely prior'):E['run'](d)
 d=data();d['valuation_date']='2024-03-29';d['action_history']['actions'][0]['occurred_at']='2024-03-29';d['action_history']['actions'][0]['known_at']='2024-03-29'
 with pytest.raises(ValueError,match='precede observed cure'):E['run'](d)
