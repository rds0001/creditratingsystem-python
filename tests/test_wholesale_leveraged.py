"""Unfunded maturity and declared covenant oracles."""
from pathlib import Path
import runpy
import pytest


def test_leveraged_variants_positive_cashflow_does_not_fund_principal():
    result=runpy.run_path(str(Path(__file__).parents[1]/"examples"/"wholesale_leveraged.py"))["run"]()
    v=result["variants"]
    assert [v[k]["refinancing_shortfall"] for k in ("sponsor_backed","high_leverage","covenant_lite")]==pytest.approx([114,156,114])
    assert v["sponsor_backed"]["supported_pd"]<v["sponsor_backed"]["standalone_pd"]
    assert v["covenant_lite"]["supported_pd"] is None
    assert all(x["cashflow_default"]["first_default_events"][0]["first_default_date"]=="2026-01-01" for x in v.values())
