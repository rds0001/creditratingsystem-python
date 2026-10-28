"""Five fast actual cause-specific fits with hand hazard and cashflow oracles."""
from pathlib import Path
import runpy


def test_specialized_risk_review():
    value=runpy.run_path(str(Path(__file__).parents[1]/'examples/specialized_risk_review.py'))['run']()
    assert len(value['journeys'])==5
    for row in value['journeys']:
        assert len(row['treatment']['feasible_ranking'])==2
        assert row['historical_training']['known_at']=='2021-01-01'
        assert row['treatment']['profile']['identification']=='CONDITIONAL_SCENARIO_NOT_CAUSAL'
