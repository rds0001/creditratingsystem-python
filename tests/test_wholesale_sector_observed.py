"""Observed outcomes stay separate from frozen forecast probabilities."""
from pathlib import Path
import runpy,json,copy
import pytest


def test_wholesale_sector_observed():
    root=Path(__file__).parents[1];run=runpy.run_path(str(root/'examples/wholesale_sector_observed.py'))['run']
    data=json.loads((root/'src/creditratingsystem/datasets/wholesale_sector_observed.json').read_text());rows=run(data)['journeys']
    assert [x['default_backtest']['overall']['observed_defaults']for x in rows]==[1,2,3,2,1,1,2]
    for row in rows:
        assert row['default_backtest']['overall']['observed_count']==20
        assert row['windows'][19]['drawn_balance_at_origin']==0 and row['windows'][19]['eligible']
        assert row['closed_workout']['economic_lgd']is not None and row['incomplete_workout']['economic_lgd']is None
        assert row['independent_oracle']['discount_days']==123
    bad=copy.deepcopy(data);bad['journeys'][0]['monitoring']['predictions'][0]['pd']=.9
    with pytest.raises(ValueError,match='pinned model'):run(bad,verify_reference=False)
