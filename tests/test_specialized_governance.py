"""One complete frozen history by default; all five product histories opt in."""
from importlib.resources import files
from pathlib import Path
import json
import os
import runpy


def test_specialized_frozen_local_history():
    data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_governance.json').read_text())
    if os.getenv('CREDITRATINGSYSTEM_EXTENDED_TESTS','').lower()!='true':data['journeys']=data['journeys'][:1]
    result=runpy.run_path(str(Path(__file__).parents[1]/'examples/specialized_governance.py'))['run'](data)
    for row in result['journeys'].values():
        assert row['replay']==row['replay_after_later_fit']=='REPRODUCED'
        assert row['tampered_replay']=='NOT_REPRODUCIBLE'
        assert row['observed_servicing']['default_days_past_due']>90
