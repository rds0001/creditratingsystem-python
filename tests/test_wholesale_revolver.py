"""Active zero-drawn credit lines and explicit legal cancellation."""
from pathlib import Path
import runpy
import pytest


def test_revolver_conditional_exposure_and_evidence_gate():
    result=runpy.run_path(str(Path(__file__).parents[1]/"examples"/"wholesale_revolver.py"))["run"]()
    v=result["variants"]
    assert all(x["reference_drawn"]==0 and x["subject_count"]==1 for x in v.values())
    assert [v[k]["conditional_ead"] for k in ("working_capital_line","contingent_commitment","cancellable","uncancellable")]==pytest.approx([50,100,50,50])
    assert v["cancellable"]["contractual_cancellation"]["schedule"][0]["conditional_ead"]==0
    assert v["uncancellable"]["contractual_cancellation"] is None
    assert result["missing_evidence_status"]=="REFER"
