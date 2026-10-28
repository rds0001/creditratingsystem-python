"""Fit prior-observed retail default/exit cohorts with independent likelihood checks."""
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('retail_exit_models.json').read_text())
    out={}
    for product,c in data['cases'].items():
        s=c['step_days'];target=c['target']
        meta=dict(target=target,training=c['training'],model_id=product+'-events',model_version='1')
        paths=[dict(path_id=product,start=0,stop=4*s,covariates=[],stratum='all',covariate_time=0)]
        cause=cr.fit_cause_specific_hazards(c['cause_panel'],[[] for _ in c['cause_panel']],c['cause_events'],feature_order=[],causes=['default','exit'],ties='breslow',**meta)
        competing=cr.predict_event_curves(cause,paths,[0,s,2*s],feature_order=[],target=target).value
        fine=cr.fit_fine_gray(c['fine_gray_times'],c['fine_gray_events'],[[] for _ in c['fine_gray_times']],cause_of_interest='default',feature_order=[],censoring_spec=dict(estimator='kaplan_meier',independence_assumption='independent_within_censoring_group'),**meta)
        subdistribution=cr.predict_event_curves(fine,paths,[0,3*s,4*s],feature_order=[],target=target).value
        grouped=cr.fit_discrete_hazard(c['grouped_panel'],[[] for _ in c['grouped_panel']],feature_order=[],period_grid=[0,s],link='cloglog',partial_period_policy='exclude_censored',**meta)
        discrete=cr.predict_event_curves(grouped,paths,[0,s/2,s],feature_order=[],target=target).value
        exit_target=dict(target,event='exit')
        aft=cr.fit_aft_model(c['exit_durations'],[1]*len(c['exit_durations']),[[] for _ in c['exit_durations']],feature_order=[],distribution='lognormal',target=exit_target,training=c['training'],model_id=product+'-exit-time',model_version='1')
        aft_prediction=cr.predict_event_curves(aft,paths,[s/2,s,2*s],feature_order=[],target=exit_target).value
        # Uncensored intercept-only lognormal likelihood has an independent
        # closed-form maximum: mean(log T), population variance(log T).
        logs=[math.log(x) for x in c['exit_durations']];mu=sum(logs)/len(logs)
        sigma=math.sqrt(sum((z-mu)**2 for z in logs)/len(logs))
        oracle_ll=sum(-math.log(t)-math.log(sigma)-.5*math.log(2*math.pi)-.5*((math.log(t)-mu)/sigma)**2 for t in c['exit_durations'])
        fit_gap=abs(aft.diagnostics['log_likelihood']-oracle_ll)
        assert fit_gap<1e-7,'Independent lognormal likelihood maximum not reproduced'
        oracle_sf=[.5*math.erfc((math.log(t)-mu)/(sigma*math.sqrt(2))) for t in [s/2,s,2*s]]
        assert max(abs(x-y) for x,y in zip(oracle_sf,aft_prediction['curves'][0]['survival']))<1e-5
        if verify_reference:
            cf=competing['curves'][0]
            assert math.isclose(cf['cumulative_incidence']['default'][1],(1-math.exp(-3/8))/3,abs_tol=1e-12)
            assert math.isclose(cf['cumulative_incidence']['exit'][1],2*(1-math.exp(-3/8))/3,abs_tol=1e-12)
            assert math.isclose(cf['survival'][1]+sum(x[1] for x in cf['cumulative_incidence'].values()),1,abs_tol=1e-12)
            assert math.isclose(subdistribution['curves'][0]['cumulative_incidence']['default'][2],1-math.exp(-1/3.75-1/2.75),abs_tol=1e-12)
            assert all(math.isclose(x,y,abs_tol=1e-10) for x,y in zip(discrete['curves'][0]['survival'],[1,math.sqrt(.8),.8]))
        # Economic fitted parameters are retained. Backend-specific covariance
        # and optimizer diagnostics are not a common serialization contract.
        out[product]=dict(time_unit='days',interpretation=c['interpretation'],competing=competing,fine_gray=subdistribution,discrete=discrete,aft=dict(coefficients=aft.model['parameters']['coefficients'],log_time_scale=aft.model['parameters']['log_time_scale'],prediction=aft_prediction,closed_form_coefficients=[mu],closed_form_scale=sigma,likelihood_oracle_pass=True))
    return dict(data_origin='SYNTHETIC',products=out,training_known_at='2023-12-31')


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
