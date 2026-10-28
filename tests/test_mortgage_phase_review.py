from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json,math,runpy
import pytest
import creditratingsystem as cr
E=runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/mortgage_phase_review.py'))
def test_same_mortgage_phase_economics_and_unknown_cure():
 r=E['run'](return_artifacts=True);raw=r['raw_artifacts']
 assert r['facility_id']=='facility_A'and r['freshness']['status']=='STALE'
 assert r['observed_cure_targets'][0]['cure_label']is None
 assert r['cure_prediction']['expected_recovery']==.5*.8*95000
 assert r['collections_value'][0]['expected_net_recovery']==.25*(38000-100)+.75*19000-10
 assert r['event_conditioned_realized_loss']['expected_loss']==.25*(95000-.9*70000)
 assert r['default_count']['overall']['observed_count']==1
 assert math.isclose(r['competing_exit_curves']['curves'][0]['cumulative_incidence']['exit'][1],.5*(1-math.exp(-.5)),abs_tol=1e-12)
 assert raw['open_inputs']['cashflows']==[] and raw['open_inputs']['episodes'][0]['resolution_date']is None
 assert r['unidentified_tail']['episodes'][0]['expected_final_lgd']is None
 assert cr.estimate_open_workout_lgd(**raw['open_inputs']).value==r['open_recovery_scenario']
 assert cr.evaluate_treatment_candidates(**raw['treatment_inputs']).value==r['treatment']
 ratio=(1+math.sqrt(.25*.75)*math.exp((math.log(3)/2)**2/2))/3
 assert math.isclose(r['open_recovery_scenario']['episodes'][0]['expected_final_loss'],95000-.95*95000*ratio,abs_tol=1e-7)
 d=json.loads(files('creditratingsystem.datasets').joinpath('mortgage_phase_review.json').read_text());d['cure_training']['training']['known_at']='2025-01-01'
 with pytest.raises(ValueError,match='Cure training'):E['run'](d)
