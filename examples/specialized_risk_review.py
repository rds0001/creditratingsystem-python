"""Actual competing-exit fits and conditional specialized workout choices."""
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_risk_review.json').read_text())
    out=[]
    for journey in data['journeys']:
        fit=cr.fit_cause_specific_hazards(**journey['survival'])
        curve=cr.predict_event_curves(fit,[dict(path_id=journey['case_id'],start=0,stop=2,covariates=[],stratum='all',covariate_time=0)],[0,1,2],feature_order=[],target=journey['survival']['target']).value
        treatment=cr.evaluate_treatment_candidates(**journey['treatment']).value
        ref=journey['survival_reference'];h=(ref['defaults']+ref['exits'])/ref['risk_set'];mass=1-math.exp(-h)
        oracle=dict(default_incidence=mass*ref['defaults']/(ref['defaults']+ref['exits']),exit_incidence=mass*ref['exits']/(ref['defaults']+ref['exits']),survival=math.exp(-h))
        values=[]
        for candidate in journey['treatment']['treatment_candidates']:
            accepted=sum((flow['expected_recovery']-flow['expected_cost'])*flow['discount_factor']for flow in candidate['cashflows'])-candidate['accepted_upfront_cost']
            values.append(candidate['uptake_probability']*accepted+(1-candidate['uptake_probability'])*candidate['rejected_present_value']-candidate['offer_cost'])
        if verify_reference:
            c=curve['curves'][0]
            assert math.isclose(c['cumulative_incidence']['default'][1],oracle['default_incidence'],abs_tol=1e-12)
            assert math.isclose(c['cumulative_incidence']['exit'][1],oracle['exit_incidence'],abs_tol=1e-12)
            assert math.isclose(oracle['default_incidence']+oracle['exit_incidence']+oracle['survival'],1,abs_tol=1e-12)
            expected_by_id={candidate['treatment_id']:value for candidate,value in zip(journey['treatment']['treatment_candidates'],values)}
            for result in treatment['candidates']:assert math.isclose(result['expected_present_value'],expected_by_id[result['treatment_id']],abs_tol=1e-12)
        out.append(dict(product=journey['product'],case_id=journey['case_id'],historical_training=journey['survival']['training'],competing_curves=curve,treatment=treatment,independent_oracle=dict(competing=oracle,treatment_present_values=values)))
    return dict(data_origin='SYNTHETIC',scope='Historical competing-exit fits and conditional treatment economics, not identified causal effects or executed arrangements',journeys=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
