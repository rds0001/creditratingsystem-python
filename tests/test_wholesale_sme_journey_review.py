from pathlib import Path
from importlib.resources import files
import json,os,runpy
import pytest

def test_sme_actual_historical_reference_and_replay():
    e=runpy.run_path(str(Path(__file__).parents[1]/'examples/wholesale_sme_journey_review.py'));d=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_sme_journey_review.json').read_text())
    if os.getenv('CREDITRATINGSYSTEM_EXTENDED_TESTS')!='true':d['cases']=d['cases'][:1]
    v=e['run'](d)
    for c in v['cases']:
        assert c['replay']['decision']['status']=='PASS' and c['negative_replay']['decision']['status']=='BLOCKED'
        assert c['prediction_review']['override']['metrics']['overridden_cases']==0
        assert c['prediction_review']['benchmark']['metrics']['by_model']['raw']['brier']==pytest.approx(c['observed_brier_oracle'])
