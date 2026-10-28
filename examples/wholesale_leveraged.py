"""Leveraged loan variants with explicit unfunded maturity and covenant contracts."""
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None, verify_reference=True):
    if data is None:
        data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_leveraged.json').read_text())
    target=data['target'];tr=data['score_training'];cal=data['calibration'];ec=data['economic_contract']
    model=cr.fit_scorecard(tr['X'],tr['y'],feature_order=tr['feature_order'],fit_spec=tr['fit_spec'],weights=tr['weights'],target=target,training=tr['training'],model_id='leveraged-fundamental',model_version='1')
    calibrator=cr.calibrate_logistic(cal['values'],cal['outcomes'],cal['weights'],metadata=cal['metadata'])
    st=data['survival_training']
    survival=cr.estimate_survival_curve(st['entry'],st['exit'],st['event'],weight_policy='frequency',target=target,training=st['training'],model_id='leveraged-observed-vintage',model_version='1')
    curves=cr.predict_event_curves(survival,[],[0,1,2],feature_order=[],target=target).to_dict()['value']
    variants={}
    for v in data['variants']:
        identity=v['id']
        statement=dict(entity_id=identity,currency='EUR',as_of='2023-12-31',period_start='2023-01-01',period_end='2023-12-31',values={k:v[k]for k in ['debt','cash','ebitda']},units={k:'EUR'for k in ['debt','cash','ebitda']})
        ratio=cr.build_financial_ratios([statement],data['ratio_spec']).value[0]['ratios']['net_debt_ebitda']['value']
        scored=cr.score_scorecard(model,[[ratio]],feature_order=tr['feature_order'],expected_target=target).value['pd'][0]
        pd=cr.apply_calibrator(calibrator,[scored],target=target,base_model_version='1').value[0]
        facts=dict(variant=identity,terms_verified=True,sponsor_present=v['sponsor'],maintenance_covenant=v['maintenance_covenant'])
        snapshot=dict(subject_id=identity,subject_type='facility',as_of=data['as_of'],known_at=data['known_at'],attributes={k:dict(value=x,observed_at=data['as_of'],known_at=data['known_at'])for k,x in facts.items()})
        rule=dict(rule_id=identity,profile_id='leveraged-'+identity,profile_version='1',priority=1,effective_from='2023-01-01',effective_to=None,known_at='2023-01-01',conditions=[dict(field=k,operator='eq',value=x)for k,x in facts.items()])
        route=cr.route_rating_profile(snapshot,[rule],data['as_of'],data['known_at']).value
        snapshot['attributes']['terms_verified']['value']=False
        negative=cr.route_rating_profile(snapshot,[rule],data['as_of'],data['known_at']).value
        debt=cr.build_debt_schedule(dict(data['debt_contract'],facility_id=identity,principal=v['debt']),data['calendar']).value
        initial=dict(scenario_id=identity,entity_id=identity,facility_id=identity,date=data['as_of'],currency='EUR',cfads=0,debt_service_due=0,debt_outstanding=v['debt'],asset_value=v['asset_value'],liquidity_buffer_available=0,materiality_met=False,material_days_past_due=0,utp_indicators=[])
        rows=[initial]+[dict(initial,date=p['payment_date'],cfads=c,debt_service_due=p['debt_service_due'])for p,c in zip(debt['schedule'],v['cfads'])]
        defaults=cr.detect_cashflow_default(rows,data['default_policy'],currency='EUR').value
        risk=dict(entity_id=identity,pd=pd,stage='STANDALONE',target=target,applied_effects=[])
        supported=cr.apply_parent_support(risk,dict(entity_id='SPONSOR',pd=.05,stage='STANDALONE',target=target,applied_effects=[]),data['support'],joint_default_probability=.01).value if v['sponsor']else None
        covenant_breach=bool(v['maintenance_covenant']and ratio>ec['maintenance_max_net_debt_ebitda'])
        shortfall=debt['schedule'][-1]['debt_service_due']-v['cfads'][-1]
        variants[identity]=dict(profile_id=route['profile_id'],negative_status=negative['status'],net_debt_ebitda=ratio,standalone_pd=pd,supported_pd=None if supported is None else supported['pd'],debt_schedule=debt,cashflow_default=defaults,maintenance_covenant_breach=covenant_breach,refinancing_shortfall=shortfall,committed_refinancing=ec['refinancing_committed'])
        if verify_reference:
            assert route['status']=='ROUTED' and negative['status']=='NO_ROUTE'
            assert math.isclose(debt['schedule'][0]['debt_service_due'],v['debt']*.1*366/365,abs_tol=1e-10)
            assert math.isclose(shortfall,v['debt']*1.1-v['cfads'][-1],abs_tol=1e-10)
            assert defaults['first_default_events'][0]['first_default_date']=='2026-01-01'
            assert all(c>0 for c in v['cfads'])
            expected=1/(1+math.exp(-(math.log(.2/.8)+(ratio-4)/4*(math.log(.4/.6)-math.log(.2/.8)))))
            assert math.isclose(pd,expected,rel_tol=1e-7)
            if supported:assert math.isclose(supported['pd'],pd-.4*(pd-.01),rel_tol=1e-7)
    if verify_reference:
        assert all(math.isclose(x,y,abs_tol=1e-12)for x,y in zip(curves['curves'][0]['survival'],[1,.9,.8]))
        assert variants['high_leverage']['maintenance_covenant_breach']
        assert not variants['covenant_lite']['maintenance_covenant_breach']
        assert variants['covenant_lite']['refinancing_shortfall']==variants['sponsor_backed']['refinancing_shortfall']
    return dict(product=data['product'],data_origin='SYNTHETIC',target=target,variants=variants,event_curves=curves,economic_contract=ec)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
