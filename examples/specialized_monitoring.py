"""Observed default/cure histories, feature drift and mature cohort residuals."""
from importlib.resources import files
from copy import deepcopy
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_monitoring.json').read_text())
    out=[]
    for journey in data['journeys']:
        model=cr.estimate_multistate_transitions(**journey['multistate'])
        curves=cr.predict_event_curves(model,[],[0,1,2,3],feature_order=[],target=journey['multistate']['target'],event_states=['default']).value
        feature=cr.detect_credit_feature_drift(**journey['feature_drift']).value
        args=deepcopy(journey['pd_drift']);sources=[];expected=[];state=0
        for observed in journey['cohorts']:
            cohort=cr.build_monitoring_cohorts(**observed).value
            rows=[row for row in cohort['rows']if row['eligible']]
            residual=sum(row['target']for row in rows)/len(rows)-math.fsum(row['pd']for row in rows)/len(rows)
            args['residuals'].append(dict(time=observed['evaluation_cutoff'],value=residual,mature=True))
            sources.append(dict(evaluation_cutoff=observed['evaluation_cutoff'],eligible_count=len(rows),defaults=sum(row['target']for row in rows),mean_pd=math.fsum(row['pd']for row in rows)/len(rows),residual=residual))
        drift=cr.detect_pd_calibration_drift(**args).value
        for count in journey['reference']['defaults']:
            state=.5*state+.5*(count/20-journey['reference'].get('cohort_pd',.1));expected.append(abs(state))
        if verify_reference:
            curve=curves['curves'][0];p=journey['reference']['first_hit']
            assert all(math.isclose(a,b,abs_tol=1e-12)for a,b in zip(curve['first_hit_probability'],[0,p,p,p]))
            assert all(math.isclose(a,b,abs_tol=1e-12)for a,b in zip([x[1]for x in curve['state_occupancy']],[0,p,0,p]))
            assert math.isclose(feature[0]['statistic'],journey['reference']['feature_wasserstein'],abs_tol=1e-12)
            assert all(math.isclose(a,b,abs_tol=1e-12)for a,b in zip(drift['statistic_path'],expected))
        out.append(dict(product=journey['product'],case_id=journey['case_id'],multistate=curves,feature_drift=feature,pd_drift=drift,residual_sources=sources,independent_ewma=expected))
    return dict(data_origin='SYNTHETIC',forecast_reference=data['forecast_reference'],journeys=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
