from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json
import math
import runpy
import pytest

EXAMPLE=runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/lifetime_risk_review.py'))
def data():return json.loads(files('creditratingsystem.datasets').joinpath('lifetime_risk_review.json').read_text())
def test_competing_first_default_conditional_loss_and_observed_errors():
 r=EXAMPLE['run']();p=r['cause_specific_curves']['cumulative_incidence']['default']
 assert p[-1]<r['ignored_prepayment_probability']
 assert r['marginal_first_default']==pytest.approx([0,p[1],p[2]-p[1]])
 assert r['ead_validation']['coverage']['included_count']==3
 assert r['lgd_validation']['coverage']['included_count']==2
 assert r['survival_validation']['metrics']['by_horizon'][-1]['censored_by_horizon']==1
 assert r['open_workout']['episodes'][0]['expected_final_lgd']is not None
 assert r['unidentified_tail']['episodes'][0]['expected_final_lgd']is None
 assert r['fine_gray_curves']['cumulative_incidence']['default'][-1]==pytest.approx(1-math.exp(-1/3.75-1/2.75))
def test_temporal_and_terminal_noncase_contracts_reject_false_history():
 d=data();d['training']['known_at']='2025-01-01'
 with pytest.raises(ValueError,match='Training event support'):EXAMPLE['run'](d)
 d=data();d['holdout'][3]['coverage_end']=None
 with pytest.raises(ValueError,match='Terminal prepaid'):EXAMPLE['run'](d)
def test_open_claim_cannot_turn_into_closed_observed_loss():
 d=data();d['holdout'][2]['lgd']=-999
 r=EXAMPLE['run'](d,verify_reference=False)
 assert r['lgd_validation']['coverage']['included_count']==2
 assert r['lgd_validation']['metrics']['overall']['ead_weighted']['bias']==pytest.approx(4/180)
