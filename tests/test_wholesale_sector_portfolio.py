"""Independent exposure event order, translation and loss-review algebra."""
from pathlib import Path
import runpy,math


def test_wholesale_sector_portfolio():
    root=Path(__file__).parents[1];r=runpy.run_path(str(root/'examples/wholesale_sector_portfolio.py'))['run']();portfolio=r['portfolio_review']
    assert len(portfolio['prepared_portfolio']['positions'])==7
    for source,exposure,loss in zip(portfolio['forecast_sources'],portfolio['exposures'],r['event_losses']):
        b=source['observed_ead_bridge'];p=b['principal_at_origin'];interest=p*.05*61/365
        assert math.isclose(b['observed_ead'],.9*p+interest,abs_tol=1e-12)
        assert math.isclose(exposure['before']['schedule'][0]['conditional_ead']-exposure['after']['schedule'][0]['conditional_ead'],.2*p,abs_tol=1e-12)
        assert math.isclose(loss['event_loss']['expected_loss'],source['forecast_pd']*(exposure['after']['schedule'][0]['conditional_ead']-source['conditional_loss_scenario']['net_recovery']/1.05**(123/365)),abs_tol=1e-12)
    assert portfolio['forecast_sources'][3]['currency']=='USD' and portfolio['forecast_sources'][3]['fx_rate']==.9
    assert portfolio['independent_oracle']['ead_bias']<0
