from pathlib import Path
from importlib.resources import files
import json,os,runpy
import pytest

def test_sector_source_replay_and_observed_benchmark():
    data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_sector_journey_review.json').read_text())
    if os.getenv('CREDITRATINGSYSTEM_EXTENDED_TESTS')!='true':data['governance']['cases']=[c for c in data['governance']['cases']if c['product']in ('SOVEREIGN_DEBT','PUBLIC_SERVICE_FINANCE')]
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples/wholesale_sector_journey_review.py'))['run'];v=run(data)
    assert v['lineage']['decision']['status']=='PASS'
    for c in v['cases']:
        assert c['replay']['decision']['status']=='PASS'
        assert c['negative_replay']['decision']['status']=='BLOCKED'
        assert c['rating_benchmark']['applicability']=='NOT_APPLICABLE'
        if c['product']=='SOVEREIGN_DEBT':assert c['original_target']['event']=='pre_transfer_sovereign_payment_default'
        if c['prediction_review']:
            q=c['prediction_review'];p=q['raw_pd'];scores=q['benchmark']['metrics']['by_model']['raw']
            assert scores['brier']==pytest.approx(.1*(1-p)**2+.9*p*p)
            assert q['override']['metrics']['overridden_cases']==0
