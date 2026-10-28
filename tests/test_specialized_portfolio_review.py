"""Same-day exposure timing and independent observed loss bias oracles."""
from pathlib import Path
import runpy


def test_specialized_portfolio_review():
    value=runpy.run_path(str(Path(__file__).parents[1]/'examples/specialized_portfolio_review.py'))['run']()
    assert len(value['prepared_portfolio']['positions'])==5
    assert len(value['exposures'])==5
    for exposure in value['exposures']:
        assert exposure['before']['schedule'][0]['conditional_ead']>exposure['after']['schedule'][0]['conditional_ead']
