"""Covered products execute issuer/cover joint states and capped cash claims."""
from pathlib import Path
import runpy


def test_three_covered_contracts_joint_cash_and_invalid_extensions():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples'/'covered_bond_products.py'))['run']
    result=run()
    assert set(result['variants'])=={'mortgage_cover_pool','public_sector_cover_pool','maturity_extension'}
    for case in result['variants'].values():
        assert case['risk']['pd_interval'] is None
        assert case['covered_recourse']['ending_states'][1]['estate_payment']==0
