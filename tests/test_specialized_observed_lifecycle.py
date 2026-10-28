"""Fast independent cohort and dated settlement oracles for all five products."""
from pathlib import Path
import runpy


def test_specialized_observed_lifecycle():
    result=runpy.run_path(str(Path(__file__).parents[1]/'examples/specialized_observed_lifecycle.py'))['run']()
    assert len(result['journeys'])==5
    for row in result['journeys']:
        zero=[w for w in row['windows']if w.get('drawn_balance_at_origin')==0]
        assert len(zero)==1 and zero[0]['eligible'] and zero[0]['target']==0
        assert row['closed_workout']['economic_lgd'] is not None
        assert row['incomplete_workout']['economic_lgd'] is None
