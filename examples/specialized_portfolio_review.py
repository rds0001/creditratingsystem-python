"""Specialized exposure timing, portfolio preparation and observed loss review."""
from importlib.resources import files
from copy import deepcopy
import math
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_portfolio_review.json').read_text())
    graph=cr.build_entity_graph([dict(entity_id=p['entity_id'])for p in data['positions']],[],data['graph_policy'],as_of=data['portfolio_contract']['as_of'],known_at=data['portfolio_contract']['known_at'])
    portfolio=cr.prepare_credit_portfolio(data['positions'],graph,data['default_link_policy'],data['portfolio_contract']).value
    concentration=cr.credit_concentration_measures(data['positions'],weight_field='ead',group_fields=['sector'],weight_unit='EUR').value
    lgd=cr.lgd_backtest(data['loss_records'],design=data['design'],workout_policy=data['workout_policy'],uncertainty='none').value
    ead=cr.ead_backtest(data['loss_records'],design=data['design'],weighting_policy=data['weighting_policy'],uncertainty='none').value
    exposures=[]
    for record in data['exposures']:
        before=cr.exposure_schedule(**record['arguments']).value
        args=deepcopy(record['arguments']);args['default_timing']='AFTER_PERIOD_END_EVENTS'
        after=cr.exposure_schedule(**args).value
        if verify_reference:
            assert math.isclose(before['schedule'][0]['conditional_ead'],record['reference_ead_before'],abs_tol=1e-12)
            assert math.isclose(after['schedule'][0]['conditional_ead'],record['reference_ead_after'],abs_tol=1e-12)
        exposures.append(dict(product=record['product'],before=before,after=after))
    total=sum(p['ead']for p in data['positions']);hhi=sum((p['ead']/total)**2 for p in data['positions'])
    lgd_bias=sum((x['predicted_lgd']-x['realized_lgd'])*x['ead_at_default']for x in data['loss_records'])/sum(x['ead_at_default']for x in data['loss_records'])
    ead_bias=sum(x['predicted_ead']-x['realized_ead']for x in data['loss_records'])/len(data['loss_records'])
    if verify_reference:
        assert math.isclose(concentration['dimensions'][0]['hhi'],hhi,abs_tol=1e-12)
        assert math.isclose(lgd['metrics']['overall']['ead_weighted']['bias'],lgd_bias,abs_tol=1e-12)
        assert math.isclose(ead['metrics']['ead']['overall']['selected_weighting']['bias'],ead_bias,abs_tol=1e-12)
    return dict(data_origin='SYNTHETIC',forecast_sources=data['forecast_sources'],prepared_portfolio=portfolio,concentration=concentration,exposures=exposures,lgd_backtest=lgd,ead_backtest=ead,independent_oracle=dict(hhi=hhi,lgd_bias=lgd_bias,ead_bias=ead_bias))


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
