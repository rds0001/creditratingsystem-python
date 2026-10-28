"""Restricted project funds, service cash generation and applicable rating branches."""
from importlib.resources import files
import json,math
import creditratingsystem as cr


def run(data=None,verify_reference=True,return_fitted_models=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_public_service.json').read_text())
    rows=[];raw=[]
    for c in data['cases']:
        sid=c['case_id'];variant=c['variant'];expected={'housing_association':'RENTAL_SERVICE','education_health':'ATTENDANCE_AND_GRANT','utility_service':'REGULATED_TARIFF'}
        if c['cashflow_type']!=expected[variant]or c['restricted_purpose']!='CAPITAL_PROJECT':raise ValueError('Service variant or restricted-purpose contract is incompatible')
        restricted_used=min(c['restricted_grant'],c['capex']);unfunded=c['capex']-restricted_used;cf=[];flows=[];oracles=[]
        for scenario in ['base','stress']:
            stress=scenario=='stress';quantity=c['stress_quantity']if stress else c['quantity'];price=c['stress_unit_price']if stress else c['unit_price'];revenue=quantity*price;cfads=revenue+c['general_grant']-c['operating_cost']-unfunded
            cf.append(dict(scenario_id=scenario,period_start='2024-03-01',period_end='2024-06-01',currency='EUR',base_quantity=quantity,base_unit_price=price,base_unit_variable_cost=0,base_fixed_cost=c['operating_cost'],volume_multiplier=1,price_multiplier=1,variable_cost_multiplier=1,fixed_cost_multiplier=1,working_capital_revenue_ratio=0,cash_taxes=0,capex=unfunded,asset_disposals=0,other_operating_cash=c['general_grant']))
            flows.append(dict(scenario_id=scenario,date='2024-06-01',currency='EUR',cash_available=c['opening_unrestricted_cash']+revenue+c['general_grant']-unfunded,reserve_target=0,amounts_due=dict(operating=c['operating_cost'],debt=c['debt_service'])))
            oracles.append(dict(scenario_id=scenario,service_revenue=revenue,general_grant=c['general_grant'],restricted_capex_funding=restricted_used,restricted_remaining=c['restricted_grant']-restricted_used,unrestricted_capex=unfunded,cfads=cfads,debt_service_gap=max(c['debt_service']-c['opening_unrestricted_cash']-cfads,0)))
        operating=cr.simulate_operating_cashflows(cf,dict(contract_id=sid,version='1',currency='EUR',period_unit='quarter',quantity_unit=c['cashflow_type'],initial_working_capital=0,include_asset_disposals_in_cfads=False)).value
        waterfall=cr.apply_cash_waterfall(flows,dict(policy_id=sid,version='1',initial_reserve=0,reserve_target=0,reserve_priority=3,excess_reserve_policy='retain',arrears_policy='carry_without_interest',claims=[dict(claim_id='operating',priority=1,reserve_eligible=False),dict(claim_id='debt',priority=2,reserve_eligible=False)]),currency='EUR').value
        statement=dict(entity_id=sid,currency='EUR',as_of='2024-03-01',period_start='2023-12-01',period_end='2024-03-01',values=dict(debt_service=c['debt_service'],unrestricted_cfads=oracles[0]['cfads']),units=dict(debt_service='EUR',unrestricted_cfads='EUR'))
        ratio=cr.build_financial_ratios([statement],[dict(name='debt_service_to_unrestricted_cfads',numerator={'debt_service':1},denominator={'unrestricted_cfads':1},denominator_policy='positive',output_unit='dimensionless')]).value[0]['ratios']['debt_service_to_unrestricted_cfads']['value']
        t=c['score_training'];z=2*(ratio-1)
        model=cr.fit_scorecard(t['X'],t['y'],feature_order=t['feature_order'],fit_spec=t['fit_spec'],target=data['target'],training=t['training'],model_id=sid,model_version='1',weights=t['weights'])
        pred=cr.score_scorecard(model,[[z]],feature_order=t['feature_order'],expected_target=data['target'])
        a=c['calibration'];cal=cr.calibrate_logistic(a['values'],a['outcomes'],a['weights'],metadata=a['metadata'],**a['fit_controls']);calibrated=cr.apply_calibrator(cal,pred.value['pd'],target=data['target'],base_model_version='1')
        fundamental=round(calibrated.value[0],data['precision_contract']['pd_decimal_places']);market=None;raw_market=None;pooled=None;standalone=fundamental
        if c['listed']:
            inversion=cr.infer_merton_assets(**data['market_inputs']).value;asset=inversion['asset_value'];sigma=inversion['asset_volatility'];barrier=data['market_inputs']['debt_face'];mu=data['physical_drift']
            dd=cr.distance_to_default(asset,barrier,sigma,mu,1).value['distance_to_default'];physical=cr.merton_default_probability(asset,barrier,sigma,mu,1,measure='PHYSICAL').value
            mapped=cr.calibrate_distance_pd_mapping(**data['distance_calibration'],evaluation_distances=[dd]).to_dict()['value']
            empirical=mapped['pd'][0];raw_market=dict(inversion=inversion,distance=dd,physical_pd=physical,mapping=mapped)
            market={k:round(v,data['precision_contract']['market_output_decimal_places'])for k,v in dict(asset_value=asset,asset_volatility=sigma,distance=dd,physical_pd=physical,empirical_pd=empirical).items()}
            signals=[dict(signal_id=name,model_id=name,model_version='1',row_ids=[sid],pd=[p],target=data['target'],as_of='2024-03-01',known_at='2024-03-01')for name,p in [('fundamental',fundamental),('empirical_distance',round(empirical,6))]]
            pooled=cr.combine_pd_signals(signals,[.7,.3],combination_scale='linear',combination_contract=dict(interpretation='estimator_pool',reference='Declared synthetic utility estimator weights, not independent event union')).value;standalone=pooled['pd'][0]
            if verify_reference:
                oracle_dd=(math.log(100/80)+(.1-.5*.2**2))/.2
                assert abs(asset-100)<1e-7 and abs(sigma-.2)<1e-7 and abs(dd-oracle_dd)<1e-7
                assert abs(physical-.5*math.erfc(oracle_dd/math.sqrt(2)))<1e-7 and 1<oracle_dd<2 and abs(empirical-.2)<1e-7
        base=dict(entity_id=sid,pd=standalone,stage='STANDALONE',target=data['target'],applied_effects=[]);parent=dict(entity_id=sid+'-sponsor',pd=.02,stage='STANDALONE',target=data['target'],applied_effects=[])
        supported=cr.apply_parent_support(base,parent,c['support'],joint_default_probability=min(.01,standalone*.25)).value
        facts=dict(product='PUBLIC_SERVICE_FINANCE',variant=variant,borrower_class='PUBLIC_SERVICE',legal_claim=c['legal_claim'],cashflow_type=c['cashflow_type'],listed=c['listed'])
        snapshot=dict(subject_id=sid,subject_type='obligor',as_of='2024-03-01',known_at='2024-03-01',attributes={k:dict(value=v,observed_at='2024-01-01',known_at='2024-01-01')for k,v in facts.items()})
        rule=dict(rule_id=sid,profile_id=sid,profile_version='1',priority=1,effective_from='2023-01-01',effective_to=None,known_at='2023-01-01',conditions=[dict(field=k,operator='eq',value=v)for k,v in facts.items()]);route=cr.route_rating_profile(snapshot,[rule],'2024-03-01','2024-03-01').value
        del snapshot['attributes']['legal_claim'];missing=cr.route_rating_profile(snapshot,[rule],'2024-03-01','2024-03-01').value
        if verify_reference:
            logit=lambda p:math.log(p/(1-p));oracle=1/(1+math.exp(-((1-z)/2*logit(a['values'][0])+(1+z)/2*logit(a['values'][2]))))
            assert abs(calibrated.value[0]-oracle)<1e-7
            for j,prefix in enumerate(['base','stress']):
                assert abs(oracles[j]['cfads']-c['reference']['cfads'if j==0 else'stress_cfads'])<1e-12
                assert abs(oracles[j]['debt_service_gap']-c['reference'][prefix+'_gap'])<1e-12
                assert abs(waterfall['periods'][j]['payments'][1]['closing_arrears']-oracles[j]['debt_service_gap'])<1e-12
            assert missing['status']=='NO_ROUTE'
            rescue=.4 if c['support']['states']['parent_alive']['support_permitted']else 0
            assert abs(supported['pd']-(standalone-(standalone-min(.01,standalone*.25))*rescue))<1e-12
        rows.append(dict(variant=variant,case_id=sid,cash_oracles=oracles,operating=operating,waterfall=waterfall,ratio=ratio,reported_fundamental_pd=fundamental,market_branch='EMPIRICAL_DISTANCE_MAPPING'if c['listed']else'NOT_APPLICABLE_UNLISTED',market=market,pooled=pooled,supported=supported,grade=cr.map_pd_to_grade(supported['pd'],data['rating_scale']).value,profile=route['profile_id'],missing_claim_status=missing['status']))
        if return_fitted_models:raw.append(dict(case_id=sid,scorecard=model.to_dict(),calibrator=cal.to_dict(),prediction=pred.to_dict(),calibrated=calibrated.to_dict(),market=raw_market))
    report=dict(data_origin='SYNTHETIC',product='PUBLIC_SERVICE_FINANCE',precision_contract=data['precision_contract'],cases=rows)
    return dict(report=report,raw_native_fit_details=raw)if return_fitted_models else report


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
