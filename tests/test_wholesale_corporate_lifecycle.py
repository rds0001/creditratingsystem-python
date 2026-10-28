"""Corporate phase oracles including observed cohorts and cross-language artifacts."""
from pathlib import Path
import runpy
import pytest


def test_corporate_lifecycle_observed_losses_and_pinned_replay():
    result=runpy.run_path(str(Path(__file__).parents[1]/"examples"/"wholesale_corporate_lifecycle.py"))["run"]()
    assert result["governance"]["replay_after_new_champion"]=="REPRODUCED"
    assert result["governance"]["tampered_replay"]=="NOT_REPRODUCIBLE"
    assert result["closed_workout"]["economic_lgd"]==pytest.approx(.3)
    assert result["incomplete_workout"]["economic_lgd"] is None
    assert [x["defaults"] for x in result["drift_sources"]]==[2,3,4]
    for language in ("python","r"):
        frozen=result["frozen_workout_replay"][language]
        assert len(frozen["model_hash"])==64
        assert frozen["forecast"]["episodes"][0]["expected_final_lgd"]==pytest.approx(.33116041251551276)
