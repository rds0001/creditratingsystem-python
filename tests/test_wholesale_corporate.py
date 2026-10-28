"""Financial oracles for the packaged corporate component."""
from pathlib import Path
import runpy
import pytest


def test_corporate_native_fits_inverse_and_recovery():
    result = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "wholesale_corporate.py"))["run"]()
    variants = result["variants"]
    assert set(variants) == {"SME", "large_corporate", "senior", "subordinated", "secured"}
    assert result["calibrated"] == pytest.approx([0.17912878474779] * 2)
    assert [variants[x]["economic_loss_at_default"] for x in ("senior", "subordinated", "secured")] == [0, 20, 10]
    assert all(variants[x]["negative_status"] == "NO_ROUTE" for x in ("senior", "subordinated", "secured"))
    assert result["target"]["conditioning"] == "PIT"
