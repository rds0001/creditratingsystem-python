"""Real competing-risk/workout fits and independent monitoring oracles."""
from pathlib import Path
from copy import deepcopy
import json,runpy,math
import pytest

def test_commitment_review_and_monitoring():
    root=Path(__file__).parents[1];run=runpy.run_path(str(root/'examples/wholesale_commitment_review.py'))['run'];data=json.loads((root/'src/creditratingsystem/datasets/wholesale_commitment_review.json').read_text());v=run(data)
    for i,row in enumerate(v['review']['journeys']):
        d,e=[(2,4),(6,2)][i];expected=(1-math.exp(-(d+e)/20))*d/(d+e)
        assert row['independent_oracle']['competing']['default_incidence']==pytest.approx(expected)
        assert row['treatment']['candidates']
    for row in v['monitoring']['journeys']:
        curve=row['multistate']['curves'][0]
        assert curve['first_hit_probability'][1:]==pytest.approx([curve['first_hit_probability'][1]]*3)
        assert curve['state_occupancy'][2][1]==0
        assert row['pd_drift']['statistic_path']==pytest.approx(row['independent_ewma'])
    bad=deepcopy(data);bad['monitoring']['journeys'][0]['cohorts'][0]['predictions'][0]['pd']=.99
    with pytest.raises(ValueError,match='pinned inference'):run(bad,verify_reference=False)
