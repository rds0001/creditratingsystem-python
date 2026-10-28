"""Packaged product contracts execute full economic chains with hand oracles."""
from pathlib import Path
import runpy


def test_six_distinct_securitization_contracts_and_failure_routes():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples'/'securitization_products.py'))['run']
    result=run()
    assert set(result['variants'])=={'ABS','RMBS','CMBS','CLO','revolving_pool','static_pool'}
    assert len(result['contract_error_codes'])==6
