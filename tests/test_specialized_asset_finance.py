"""Fast representative contracts; full sixteen-variant replay is opt-in."""
from importlib.resources import files
from pathlib import Path
import json
import math
import os
import runpy


def test_specialized_contract_ledgers_and_cycle_law():
    example=runpy.run_path(str(Path(__file__).parents[1]/'examples/specialized_asset_finance.py'))
    data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_asset_finance.json').read_text())
    if os.getenv('CREDITRATINGSYSTEM_EXTENDED_TESTS','').lower()!='true':
        selected={'commercial_real_estate-development','project_finance-merchant','aircraft_finance-portfolio','ship_finance-spot_exposure','equipment_leasing-vendor_supported'}
        data['cases']=[c for c in data['cases']if c['case_id']in selected]
        data['parameter_snapshot']['parameters']={k:v for k,v in data['parameter_snapshot']['parameters'].items()if k in selected}
        data['entity_innovations']=[v for v in data['entity_innovations']if any(v['channel_id'].startswith(k+'-')for k in selected)]
    result=example['run'](data)
    assert len(result['variants'])==len(data['cases'])
    vendor=result['variants']['equipment_leasing-vendor_supported']
    assert math.isclose(vendor['guarantee']['expected_loss_after'],vendor['guarantee']['expected_loss_before'],abs_tol=1e-12)
    assert vendor['lessee_pd']==.1
    assert result['pit_sampling_contract']['conditioning_scope']=='PIT'
    assert result['ttc_sampling_contract']['conditioning_scope']=='TTC_STATIONARY'
