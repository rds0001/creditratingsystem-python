"""Actual residual settlement fits; ultimate-value oracle under declared flat curve."""
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_open_workout.json').read_text())
    out=[]
    for journey in data['journeys']:
        fitted=cr.fit_recovery_hurdle(**journey['fit'])
        prediction=cr.estimate_open_workout_lgd(fitted_recovery_model=fitted,**journey['inference']).value
        episode=prediction['episodes'][0]
        if verify_reference:
            assert math.isclose(episode['expected_final_lgd'],journey['reference']['expected_final_lgd'],abs_tol=1e-10)
            assert len(journey['fit']['episodes'])==16 and sum(x['resolution_date']is None for x in journey['fit']['episodes'])==4
        # Ultimate quantities deliberately integrate the full timing tail; grid
        # probabilities are not substituted for ultimate resolution probability.
        out.append(dict(product=journey['product'],case_id=journey['case_id'],training=journey['fit']['training'],stage_spec=journey['fit']['stage_spec'],reference=journey['reference'],ead_at_default=episode['ead_at_default'],known_discounted_net_cashflows=episode['known_discounted_net_cashflows'],expected_final_loss=episode['expected_final_loss'],expected_final_lgd=episode['expected_final_lgd'],ratio_status=episode['ratio_status']))
    return dict(data_origin='SYNTHETIC',scope='Ultimate flat-curve financial quantities from actual hurdle and censored Weibull fits; timing-grid acceptance is separate',journeys=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
