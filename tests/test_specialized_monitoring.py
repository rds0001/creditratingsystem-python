"""First-hit versus state occupancy and measured mature-cohort drift."""
from pathlib import Path
import runpy


def test_specialized_monitoring():
    result=runpy.run_path(str(Path(__file__).parents[1]/'examples/specialized_monitoring.py'))['run']()
    assert len(result['journeys'])==5
    for row in result['journeys']:
        assert [source['eligible_count'] for source in row['residual_sources']]==[20,20,20]
        curve=row['multistate']['curves'][0]
        assert curve['first_hit_probability'][2]>0 and curve['state_occupancy'][2][1]==0
