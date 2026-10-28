"""Actual censored hurdle fits and independent ultimate recovery algebra."""
from pathlib import Path
import runpy


def test_specialized_open_workout():
    result=runpy.run_path(str(Path(__file__).parents[1]/'examples/specialized_open_workout.py'))['run']()
    assert len(result['journeys'])==5
    assert len({row['expected_final_lgd'] for row in result['journeys']})==5
    assert all(row['ratio_status']=='DEFINED' for row in result['journeys'])
