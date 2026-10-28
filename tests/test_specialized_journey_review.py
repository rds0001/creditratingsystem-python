from pathlib import Path
from importlib.resources import files
import json,os,runpy

def test_specialist_actual_asset_kernel_and_nested_grid():
    e=runpy.run_path(str(Path(__file__).parents[1]/'examples/specialized_journey_review.py'));d=json.loads(files('creditratingsystem.datasets').joinpath('specialized_journey_review.json').read_text());v=e['_convergence'](d['cases'][0],d['convergence']);m=v['result']['metrics']
    assert len(m['sampling'])==4 and len(m['paired_grid_differences'])==2
    assert all(x['estimate']==0 for x in m['paired_grid_differences'])
    assert not v['result']['discretization_bias_identified']
    if os.getenv('CREDITRATINGSYSTEM_EXTENDED_TESTS')=='true':
        report=e['run']();assert len(report['cases'])==5
        assert all(c['replay']['decision']['status']=='PASS' and c['negative_replay']['decision']['status']=='BLOCKED' for c in report['cases'])
