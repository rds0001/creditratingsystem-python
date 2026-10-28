"""Fund default score, legally distinct borrowing bases and joint asset losses."""
from importlib.resources import files
import math
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True,return_fitted_models=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_fund_finance.json').read_text())
    output=[];raw=[]
    for case in data['cases']:
        expected_basis={'NAV_backed':'ELIGIBLE_NAV','subscription_line':'UNCALLED_COMMITMENTS','levered_fund':'ELIGIBLE_NAV'}[case['variant']]
        if case['borrowing_base_basis']!=expected_basis:raise ValueError('Variant and borrowing-base contract are incompatible')
        sid=case['case_id'];bases=[];funding=[];ops=[]
        for scenario in data['scenario_ids']:
            stress=scenario=='stress'
            if case['borrowing_base_basis']=='UNCALLED_COMMITMENTS':
                eligible=math.fsum((p['committed']-p['paid_in'])*p['haircut']for p in case['commitments']if p['eligible']and(not stress or p['stress_performing']))
            else:eligible=math.fsum(p['value']*(p['stress_price']if stress else 1)*p['haircut']for p in case['assets']if p['eligible'])
            borrowing=eligible*case['advance_rate'];margin=max(case['drawn']-borrowing,0);redemptions=case['stress_redemptions']if stress else 0;cash=case['cash']+case['distributions']-case['operating_cost'];gap=max(margin+redemptions-cash,0)
            bases.append(dict(scenario_id=scenario,basis=case['borrowing_base_basis'],eligible_amount=eligible,borrowing_base=borrowing,margin_due=margin,redemptions_due=redemptions,available_cash=cash,liquidity_gap=gap))
            funding.append(dict(scenario_id=scenario,date='2024-06-01',currency='EUR',cash_available=case['cash']+case['distributions'],reserve_target=0,amounts_due=dict(operating=case['operating_cost'],margin=margin,redemptions=redemptions)))
            ops.append(dict(scenario_id=scenario,period_start='2024-03-01',period_end='2024-06-01',currency='EUR',base_quantity=1,base_unit_price=case['distributions'],base_unit_variable_cost=0,base_fixed_cost=case['operating_cost'],volume_multiplier=1,price_multiplier=1,variable_cost_multiplier=1,fixed_cost_multiplier=1,working_capital_revenue_ratio=0,cash_taxes=0,capex=0,asset_disposals=0,other_operating_cash=0))
        statement=dict(entity_id=sid,currency='EUR',as_of='2024-03-01',period_start='2023-03-01',period_end='2024-03-01',values=dict(drawn=case['drawn'],borrowing_base=bases[0]['borrowing_base']),units=dict(drawn='EUR',borrowing_base='EUR'))
        ratio=cr.build_financial_ratios([statement],[dict(name='borrowing_base_utilization',numerator={'drawn':1},denominator={'borrowing_base':1},denominator_policy='positive',output_unit='dimensionless')]).value[0]['ratios']['borrowing_base_utilization']['value']
        tr=case['score_training'];prep=cr.fit_preprocessor(case['preprocessor_training'],[dict(name='borrowing_base_utilization',type='numeric',imputation='error',scaling='standard')],target=data['target'],training=tr['training'],model_id=sid+'-preprocessor',model_version='1')
        design=cr.transform_features([dict(borrowing_base_utilization=ratio)],prep).value
        model=cr.fit_scorecard(tr['X'],tr['y'],weights=tr['weights'],feature_order=tr['feature_order'],fit_spec=tr['fit_spec'],target=data['target'],training=tr['training'],model_id=sid+'-score',model_version='1')
        score=cr.score_scorecard(model,design['design_matrix'],feature_order=tr['feature_order'],expected_target=data['target'])
        c=case['calibration'];cal=cr.calibrate_logistic(c['values'],c['outcomes'],c['weights'],metadata=c['metadata'],**c['fit_controls']);prediction=cr.apply_calibrator(cal,score.value['pd'],target=data['target'],base_model_version='1')
        pd=round(prediction.value[0],data['precision_contract']['pd_decimal_places'])
        operating=cr.simulate_operating_cashflows(ops,dict(contract_id=sid+'-distributions',version='1',currency='EUR',period_unit='quarter',quantity_unit='portfolio_distribution',initial_working_capital=0,include_asset_disposals_in_cfads=False)).value
        waterfall=cr.apply_cash_waterfall(funding,dict(policy_id=sid+'-cash-rights',version='1',initial_reserve=0,reserve_priority=4,excess_reserve_policy='retain',arrears_policy='carry_without_interest',claims=[dict(claim_id=name,priority=i+1,reserve_eligible=False)for i,name in enumerate(['operating','margin','redemptions'])]),currency='EUR').value
        assetloss=[dict(facility_id=a['asset_id'],currency='EUR',scenario_ids=data['scenario_ids'],losses=[0,a['value']*(1-a['stress_price'])])for a in case['assets']]
        losses=cr.aggregate_portfolio_loss(assetloss,[dict(facility_id=a['asset_id'],portfolio_id=sid,share=1)for a in case['assets']],data['scenario_weights'],{'EUR':1},scenario_ids=data['scenario_ids'],reporting_currency='EUR',probability_measure='PHYSICAL',quantile_levels=[.9]).value
        facts=dict(product='FUND_FINANCE',variant=case['variant'],borrower_class='FUND',legal_claim=case['legal_claim'],borrowing_base_basis=case['borrowing_base_basis'])
        snapshot=dict(subject_id=sid,subject_type='facility',as_of='2024-03-01',known_at='2024-03-01',attributes={k:dict(value=v,observed_at='2024-03-01',known_at='2024-03-01')for k,v in facts.items()})
        rule=dict(rule_id=sid,profile_id=sid,profile_version='1',priority=1,effective_from='2023-01-01',effective_to=None,known_at='2023-01-01',conditions=[dict(field=k,operator='eq',value=v)for k,v in facts.items()])
        route=cr.route_rating_profile(snapshot,[rule],'2024-03-01','2024-03-01').value
        snapshot['attributes']['borrowing_base_basis']['value']='ELIGIBLE_NAV'if case['borrowing_base_basis']=='UNCALLED_COMMITMENTS'else'UNCALLED_COMMITMENTS'
        incompatible=cr.route_rating_profile(snapshot,[rule],'2024-03-01','2024-03-01').value
        z=2*(ratio-1);p0=c['values'][0];p1=c['values'][2];logit=lambda p:math.log(p/(1-p));oracle=1/(1+math.exp(-((1-z)/2*logit(p0)+(1+z)/2*logit(p1))))
        if verify_reference:
            assert abs(prediction.value[0]-oracle)<data['precision_contract']['raw_fit_oracle_absolute_tolerance']
            for actual,key in [(bases[0]['borrowing_base'],'initial_base'),(bases[1]['borrowing_base'],'stress_base'),(bases[1]['margin_due'],'stress_margin'),(bases[1]['liquidity_gap'],'stress_liquidity_gap')]:assert math.isclose(actual,case['reference'][key],abs_tol=1e-12)
            portfolio=losses['portfolios'][0];assert math.isclose(portfolio['path_losses'][1],case['reference']['underlying_stress_loss'],abs_tol=1e-12)
            assert math.isclose(portfolio['expected_loss'],.2*case['reference']['underlying_stress_loss'],abs_tol=1e-12)
            assert math.isclose(sum(p['closing_arrears']for p in waterfall['periods'][1]['payments']),bases[1]['liquidity_gap'],abs_tol=1e-12)
            assert incompatible['status']=='NO_ROUTE'
        output.append(dict(variant=case['variant'],case_id=sid,borrowing_bases=bases,ratio=ratio,reported_fund_pd=pd,grade=cr.map_pd_to_grade(pd,data['rating_scale']).value,operating=operating,liquidity=waterfall,underlying_asset_loss=losses,profile=route['profile_id'],incompatible_basis_status=incompatible['status']))
        if return_fitted_models:raw.append(dict(case_id=sid,preprocessor=prep.to_dict(),scorecard=model.to_dict(),calibrator=cal.to_dict(),fundamental=score.to_dict(),calibrated=prediction.to_dict()))
    report=dict(product=data['product'],data_origin='SYNTHETIC',precision_contract=data['precision_contract'],scenario_law=data['scenario_law'],cases=output)
    return dict(report=report,raw_native_fit_details=raw)if return_fitted_models else report


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
