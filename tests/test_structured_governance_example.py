"""Domain gates, maturity, exit and historical replay for four products."""
from pathlib import Path
import runpy


def test_dated_investment_governance_and_immutable_exit_evidence():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples'/'structured_governance.py'))['run']
    result=run()
    assert set(result['variants'])=={'SECURITIZATION_TRANCHE','COVERED_BOND','TRADE_RECEIVABLES','COUNTERPARTY_EXPOSURE'}
    for case in result['variants'].values():
        assert case['performing_assessment']['as_of']=='2024-01-15'
        assert case['retired']['history'][0]['previous_assessment_hash']==case['assessment']['assessment_hash']
        assert case['assessment']['state']=='DRAFT'
        assert case['exit_cohort']['rows'][-1]['target'] is None
