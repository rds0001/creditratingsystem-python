"""Actual model-bound drift and first-hit versus cured-state occupancy."""
from pathlib import Path
import runpy,json,copy,math
import pytest


def test_wholesale_sector_monitoring():
    root=Path(__file__).parents[1];run=runpy.run_path(str(root/'examples/wholesale_sector_monitoring.py'))['run']
    data=json.loads((root/'src/creditratingsystem/datasets/wholesale_sector_monitoring.json').read_text());rows=run(data)['journeys']
    for i,row in enumerate(rows):
        curve=row['multistate']['curves'][0];p=[1,2,3,2,1,1,2][i]/20
        assert curve['first_hit_probability']==[0,p,p,p]
        assert [x[1]for x in curve['state_occupancy']]==[0,p,0,p]
        assert [x['eligible_count']for x in row['residual_sources']]==[20,20,20]
        assert all(x['evaluation_cutoff']>'2025-01-01'for x in row['residual_sources'])
        assert all(math.isclose(a,b,abs_tol=1e-12)for a,b in zip(row['pd_drift']['statistic_path'],row['independent_ewma']))
    bad=copy.deepcopy(data);bad['journeys'][0]['cohorts'][1]['predictions'][0]['pd']=.9
    with pytest.raises(ValueError,match='pinned model'):run(bad,verify_reference=False)
