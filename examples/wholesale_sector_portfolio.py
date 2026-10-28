"""Sector exposure timing, closed-loss backtests and explicit event losses."""
from importlib.resources import files
from pathlib import Path
import json,runpy,math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_sector_portfolio.json').read_text())
    engine=runpy.run_path(str(Path(__file__).with_name('specialized_portfolio_review.py')))['run'];portfolio=engine(data,verify_reference=verify_reference)
    losses=[]
    for case in data['event_losses']:
        loss=cr.event_conditioned_loss(**case['arguments']).value
        if verify_reference:assert math.isclose(loss['expected_loss'],case['reference_expected_loss'],abs_tol=1e-12)
        losses.append(dict(case_id=case['case_id'],event_loss=loss))
    return dict(data_origin='SYNTHETIC',portfolio_review=portfolio,event_losses=losses)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
