"""Product cashflow amounts, conditional terminal forecasts and incomplete tails."""
from pathlib import Path
from importlib.resources import files
import json
import runpy


def test_product_workout_claims_and_cash_not_counted_twice():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples'/'structured_workout.py'))['run']
    result=run()
    losses={p:r['closed_workout']['economic_loss']for p,r in result['variants'].items()}
    assert losses=={'SECURITIZATION_TRANCHE':55,'COVERED_BOND':32.5,'TRADE_RECEIVABLES':30,'COUNTERPARTY_EXPOSURE':48}
    data=json.loads(files('creditratingsystem.datasets').joinpath('structured_workout.json').read_text())
    for case in data['variants'].values():case['episode']['resolution_known_at']='2024-06-01'
    delayed=run(data,verify_reference=False)
    assert all(r['closed_workout']['economic_lgd']is None and r['event_conditioned_loss']is None for r in delayed['variants'].values())
