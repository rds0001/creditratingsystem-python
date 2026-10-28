"""Dated delinquency, conditional restructuring and competing exit evidence."""
from copy import deepcopy
from datetime import date, timedelta
from importlib.resources import files
import json
import math
import numpy as np
import creditratingsystem as cr


def run(data=None, verify_reference=True):
    if data is None:
        data=json.loads(files('creditratingsystem.datasets').joinpath('structured_lifecycle.json').read_text())
    data=deepcopy(data);out={}
    for product,c in data['variants'].items():
        calendar=c['history_calendar']
        if not calendar['origin']<=calendar['complete_through']<=calendar['known_at']<=data['training']['known_at']<'2024-01-16':
            raise ValueError('Historical transition sample must be complete and known before the decision')
        last_day=date.fromisoformat(calendar['origin'])+timedelta(days=max(float(row['stop']) for row in c['history']+c['competing_panel']))
        if last_day>date.fromisoformat(calendar['complete_through']):
            raise ValueError('Historical transitions exceed the declared complete-through date')
        meta=dict(target=data['target'],training=data['training'],model_id=product+'-historical-transitions',model_version='1')
        model=cr.estimate_multistate_transitions(c['history'],states=c['states'],allowed_transitions=c['allowed_transitions'],initial_state='performing',**meta)
        curves=cr.predict_event_curves(model,[],[0,14,15,31,121],feature_order=[],target=data['target'],event_states=['default']).value
        conditional=cr.predict_event_curves(model,[],[0,16,106],feature_order=[],target=data['target'],initial_state='restructured',start_time=15,event_states=['default']).value
        target=dict(data['target'],horizon=60)
        competing=cr.fit_cause_specific_hazards(c['competing_panel'],np.empty((4,0)),c['event_types'],feature_order=[],causes=['default','exit'],ties='breslow',target=target,training=data['training'],model_id=product+'-historical-competing',model_version='1')
        economic_parameters=deepcopy(competing.model['parameters'])
        for cause_model in economic_parameters['cause_models']:cause_model.pop('diagnostics')
        exit_curves=cr.predict_event_curves(competing,[dict(path_id='claim',start=0,stop=60,covariates=[],stratum='all',covariate_time=0)],[0,30,60],feature_order=[],target=target).value
        windows=cr.build_observation_windows(c['ratings'],c['default_events'],c['exits'],30,'2024-02-15').value
        immature=cr.build_observation_windows(c['ratings'],c['default_events'],c['exits'],30,'2024-01-31').value
        dependencies=[dict(assessment_id=c['assessment_id'],dependency_type=c['subject_type'],dependency_id=c['subject_id'])]
        def event(kind,occurred,known):
            return dict(event_id=product+'-'+kind,revision='1',occurred_at=occurred,known_at=known,dependencies=[dict(type=c['subject_type'],id=c['subject_id'])],payload=dict(kind=kind,execution='OBSERVED' if kind=='payment_delay' else 'PROPOSAL_ONLY'))
        delay=event('payment_delay','2024-01-15','2024-01-16')
        deferred=cr.apply_assessment_event(None,delay,dependencies,'2024-01-15').value
        applied=cr.apply_assessment_event(None,delay,dependencies,'2024-01-16').value
        proposed=cr.apply_assessment_event(applied['state'],event('restructure_proposal','2024-01-16','2024-01-16'),dependencies,'2024-01-16').value
        out[product]=dict(governance_assessment_hash=c['governance_assessment_hash'],transition_model=model.model,transition_curves=curves,conditional_restructured_curves=conditional,competing_parameters=economic_parameters,competing_target=target,competing_curves=exit_curves,observation_windows=windows,immature_windows=immature,deferred_event=deferred,delinquency_event=applied,restructuring_event=proposed)
        if verify_reference:
            v=curves['curves'][0];assert v['first_hit_probability']==[0,0,0,.5,.5]
            assert v['state_occupancy'][-1][3]==0 and v['state_occupancy'][-1][4]==.5
            assert conditional['curves'][0]['first_hit_probability']==[0,1,1]
            v=exit_curves['curves'][0]
            assert math.isclose(v['survival'][1],math.exp(-.5),abs_tol=1e-12)
            for cause in ['default','exit']:assert math.isclose(v['cumulative_incidence'][cause][1],(1-math.exp(-.5))/2,abs_tol=1e-12)
            assert windows[0]['target']==1 and windows[1]['target']is None and windows[1]['label_status']=='exit'
            assert immature[0]['target']is None and deferred['status']=='DEFERRED'
            assert proposed['affected_assessment_ids']==[c['assessment_id']]
    return dict(data_origin=data['data_origin'],version=data['version'],product=data['product'],variants=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
