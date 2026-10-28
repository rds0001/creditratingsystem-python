"""Independent obligor counts and claim loss accounting, with leakage controls."""
from pathlib import Path
from importlib.resources import files
import json
import runpy
import pytest
from creditratingsystem import CreditRiskError


def test_portfolio_unit_deduplication_closed_losses_and_undefined_ccf():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples'/'structured_monitoring.py'))['run']
    result=run()
    expected={'SECURITIZATION_TRANCHE':-.1875,'COVERED_BOND':.175,'TRADE_RECEIVABLES':.2,'COUNTERPARTY_EXPOSURE':-.1}
    for product,r in result['variants'].items():
        assert r['lgd_backtest']['metrics']['overall']['count_weighted']['bias']==pytest.approx(expected[product])
        assert r['default_counts']['overall']['observed_expected_ratio']==pytest.approx(4/3)
    assert result['variants']['COUNTERPARTY_EXPOSURE']['ead_backtest']is None
    data=json.loads(files('creditratingsystem.datasets').joinpath('structured_monitoring.json').read_text())
    first=next(iter(data['variants'].values()));first['frozen_forecasts'][1]['pd']=.6
    with pytest.raises(CreditRiskError,match='DUPLICATE_UNIT_CONFLICT'):run(data)
    first['frozen_forecasts'][1]['pd']=.5;first['loss_records'][0]['forecast_known_at']='2024-02-02'
    with pytest.raises(CreditRiskError):run(data)
