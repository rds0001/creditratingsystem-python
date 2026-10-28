"""Actual sector fits, dated evidence and independently checked support effects."""
from pathlib import Path
import runpy


def test_wholesale_sector_financial():
    result=runpy.run_path(str(Path(__file__).parents[1]/'examples/wholesale_sector_financial.py'))['run'](return_fitted_models=True)
    assert len(result['report']['cases'])==len(result['raw_native_fit_details'])==11
    for case,raw in zip(result['report']['cases'],result['raw_native_fit_details']):
        assert raw['scorecard']['value']['model']['model_id']==case['case_id']+'-scorecard'
        if case['variant']in ['financial_holding','public_agency','reinsurer']:
            assert case['supported']['pd']==case['reported_standalone_pd']
        if case['variant']=='financial_holding':
            assert case['constrained']['grade']=='C' and case['constrained']['binding']
        assert case['override']['current']['pd']==case['constrained']['pd']
