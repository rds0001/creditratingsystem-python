"""Independent 30/731-day counts and discounted settlement algebra."""
from pathlib import Path
from copy import deepcopy
import json,runpy
import pytest

def test_observed_commitment_horizons_and_frozen_prediction():
    root=Path(__file__).parents[1];run=runpy.run_path(str(root/'examples/wholesale_commitment_observed.py'))['run'];data=json.loads((root/'src/creditratingsystem/datasets/wholesale_commitment_observed.json').read_text());result=run(data)
    for i,cohort in enumerate(result['cohorts']):
        j=cohort['journeys'][0];overall=j['default_backtest']['overall']
        assert overall['observed_count']==20 and overall['observed_defaults']==[2,6][i]
        assert j['closed_workout']['economic_lgd']==pytest.approx(j['independent_oracle']['lgd'],abs=1e-12)
        assert j['incomplete_workout']['economic_lgd']is None
        assert sum(w['censored']for w in j['windows'])==2
    bad=deepcopy(data);bad['cohorts'][0]['journeys'][0]['monitoring']['predictions'][0]['pd']=.9
    with pytest.raises(ValueError,match='pinned inference'):run(bad,verify_reference=False)
    bad=deepcopy(data);bad['cohorts'][1]['cohort_contract']['horizon_days']=730
    with pytest.raises(ValueError,match='horizons differ'):run(bad,verify_reference=False)
