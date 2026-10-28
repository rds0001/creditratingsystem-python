"""Independent risk-set and chronology oracles for all four structured products."""
from pathlib import Path
from importlib.resources import files
import json
import math
import runpy
import pytest


def test_transition_first_hit_competing_exit_and_information_cutoff():
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples'/'structured_lifecycle.py'))['run']
    result=run()
    for case in result['variants'].values():
        curve=case['competing_curves']['curves'][0]
        assert curve['cumulative_incidence']['exit'][1]==pytest.approx((1-math.exp(-.5))/2)
        assert sum([curve['survival'][1],curve['cumulative_incidence']['exit'][1],curve['cumulative_incidence']['default'][1]])==pytest.approx(1)
        assert case['transition_curves']['curves'][0]['first_hit_probability'][-1]==.5
        assert case['restructuring_event']['state']['processed'][-1]['event']['payload']['execution']=='PROPOSAL_ONLY'
    data=json.loads(files('creditratingsystem.datasets').joinpath('structured_lifecycle.json').read_text())
    data['training']['known_at']='2024-01-17'
    with pytest.raises(ValueError,match='before the decision'):run(data)
