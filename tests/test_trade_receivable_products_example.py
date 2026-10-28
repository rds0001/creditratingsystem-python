"""Packaged invoice variants run independent hand references."""
from pathlib import Path
import runpy


def test_invoice_variants_separate_dilution_credit_and_recourse():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples'/'trade_receivable_products.py'))['run']
    result=run()
    assert set(result['variants'])=={'recourse','nonrecourse','invoice_pool'}
    for case in result['variants'].values():
        assert all(abs(e['conservation_residual'])<1e-10 for e in case['paths']['ending_states'])
