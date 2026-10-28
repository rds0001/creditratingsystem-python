from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json,runpy
import pytest
import creditratingsystem as cr
E=runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/counterparty_public_review.py'))
def test_actual_claim_and_missing_external_evidence():
 r=E['run'](return_artifacts=True);raw=r['raw_artifacts']
 assert r['opening_balance']==80 and r['workout']['closed_workout']['economic_loss']==48
 assert r['benchmark']['applicability']=='NOT_APPLICABLE'and r['benchmark']['coverage']['included_count']==0
 assert r['support_transfer']['metrics']['by_stage']['entity']['brier']==.25
 assert r['support_transfer']['metrics']['by_stage']['transfer']['brier']is None
 assert r['actual_monitoring'][0]['metrics']['observed_rate']==1
 assert r['replay_status']=='PASS'and r['negative_replay_status']=='BLOCKED'
 assert cr.benchmark_ratings(**raw['benchmark_inputs']).value==r['benchmark']
 assert cr.support_transfer_backtest(**raw['support_inputs']).value==r['support_transfer']
 assert cr.validate_source_lineage(**raw['lineage']['audit_inputs']).value==raw['lineage']['audit']
 bad=deepcopy(raw['support_inputs']);bad['records'][0]['stage_outcomes']['transfer']=dict(value=0,known_at='2024-02-15',complete_through='2024-02-14',event_ids=[],event_times=[])
 with pytest.raises(cr.CreditRiskError):cr.support_transfer_backtest(**bad)
 d=json.loads(files('creditratingsystem.datasets').joinpath('counterparty_public_review.json').read_text());d['subject_id']='wrong-claim'
 with pytest.raises(ValueError):E['run'](d)
