"""Packaged legal-netting and joint-exposure reference journeys."""
from pathlib import Path
import runpy


def test_four_counterparty_contracts_and_observed_exposure_validation():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples'/'counterparty_credit_products.py'))['run']
    result=run()
    assert set(result['variants'])=={'netting_set','collateralized','uncollateralized','wrong_way'}
    for case in result['variants'].values():
        assert case['validation']['coverage']['included_count']==1
        assert case['validation']['uncertainty']['method'] is None
