"""Independent borrowing-base and cash-rights oracles for three fund structures."""
from pathlib import Path
import runpy,json,copy
import pytest


def test_wholesale_fund_finance():
    root=Path(__file__).parents[1]
    run=runpy.run_path(str(root/'examples/wholesale_fund_finance.py'))['run']
    data=json.loads((root/'src/creditratingsystem/datasets/wholesale_fund_finance.json').read_text())
    result=run(data)
    rows={x['variant']:x for x in result['cases']}
    assert [rows[v]['borrowing_bases'][0]['borrowing_base'] for v in ['NAV_backed','subscription_line','levered_fund']]==[36,36,70]
    assert rows['subscription_line']['borrowing_bases'][1]['borrowing_base']==0
    assert rows['subscription_line']['borrowing_bases'][1]['liquidity_gap']==25
    assert abs(rows['levered_fund']['borrowing_bases'][1]['liquidity_gap']-23.3)<1e-12
    assert [x['underlying_asset_loss']['portfolios'][0]['path_losses'][1] for x in result['cases']]==[38,38,39]
    assert all(x['reported_fund_pd']<.2 for x in result['cases'])
    bad=copy.deepcopy(data);bad['cases'][1]['borrowing_base_basis']='ELIGIBLE_NAV'
    with pytest.raises(ValueError,match='incompatible'):run(bad,verify_reference=False)
