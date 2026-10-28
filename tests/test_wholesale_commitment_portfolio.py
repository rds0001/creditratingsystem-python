"""Conditional exposure timing, scale concentration and actual loss errors."""
from pathlib import Path
import runpy
import pytest

def test_commitment_portfolio_financial_oracles():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples/wholesale_commitment_portfolio.py'))['run'];v=run()
    for i,p in enumerate(v['portfolios']):
        r=p['portfolio_review'];assert r['independent_oracle']['hhi']==pytest.approx(5/9)
        assert r['independent_oracle']['ead_bias']==pytest.approx([-7.5,0][i])
        x=r['exposures'][0]
        assert x['before']['schedule'][0]['conditional_ead']==[0,132][i]
        assert x['after']['schedule'][0]['conditional_ead']==[50,114][i]
        assert len(p['event_losses'])==2
