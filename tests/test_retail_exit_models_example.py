from importlib.resources import files
from pathlib import Path
import json
import runpy


def test_prior_observed_exit_fits_at_three_time_scales():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples/retail_exit_models.py'))['run']
    d=json.loads(files('creditratingsystem.datasets').joinpath('retail_exit_models.json').read_text())
    names=['RETAIL_BNPL','RETAIL_RESIDENTIAL_MORTGAGE','RETAIL_EDUCATION_LOAN']
    d['cases']={k:d['cases'][k] for k in names}
    v=run(d)['products']
    assert all(v[k]['aft']['likelihood_oracle_pass'] for k in names)
    assert [v[k]['competing']['horizon_grid'][-1] for k in names]==[14,60,180]
