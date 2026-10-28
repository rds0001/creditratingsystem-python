"""Prior-known conditional 30-day losses for the six original application accounts."""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import importlib.util
import json
import math
import numpy as np
import creditratingsystem as cr


def helper():
    s=importlib.util.spec_from_file_location('risk_application',Path(__file__).with_name('retail_application_journey.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def evaluate(product,c,r,target,chain=None,verify_reference=True):
    app=next(a for a in c['applications']if a['application_id']==r['application_id']);D=app['information_cutoff'];key=app['application_id']
    if app['customer_id']!=r['customer_id']or r['account_id']!='facility-'+key or r['assumptions_known_at']>D:raise ValueError('Same application identity and prior-known assumptions required')
    if r['horizon_end']!='2025-01-30'or r['proposed_funding_at']!='2024-12-31'or target['horizon']!=30:raise ValueError('Exact30day proposed funding window required')
    expected_ead=c['terms']['rent']if c['terms']['kind']=='OPERATING_RENT'else app['requested_terms']['amount']
    if r['conditional_ead']!=expected_ead:raise ValueError('Claim differs from requested principal or unpaid rent')
    if 'asset'in r and c['terms']['kind']=='OPERATING_RENT' and r['waterfall']['claims'][0]['subject_id']==app['customer_id']:raise ValueError('Lessor investment cannot become borrower debt')
    # No observed booking/outcome or later settlement enters this forecast.
    if chain is None:
        fitted=helper().fit_product_models(product,c,target);feature=next(f for f in c['features']if f['application_id']==key and f['feature']=='driver')
        if feature['known_at']>D:raise ValueError('Future feature')
        snapshot=cr.build_application_snapshot([app],[feature],required_features=['driver']).value[0]
        pd=cr.predict_application_risk(snapshot,fitted['model'],target,calibrator=fitted['calibrator']).value['pd']
    else:
        origin=chain['products'][product];row=next(a for a in origin['applications']if a['application']['application_id']==key)
        if row['application']!=app or row['risk']['target']!=target:raise ValueError('Upstream origin identity/target mismatch')
        fitted=origin['fitted'];pd=row['risk']['pd']
    prior=[]
    for w in r['prior_workouts']:
        if not w['default_date']<=w['recovery_date']<=w['closed_at']<=w['known_at']<D or w['entity_id']==app['customer_id']:raise ValueError('Independent completed prior workouts required')
        discounted=cr.discount_recoveries([dict(cashflow_id=w['episode_id'],cashflow_date=w['recovery_date'],known_at=w['known_at'],currency='EUR',kind='RECOVERY',amount=w['recovery'])],dict(anchor_date=w['default_date'],known_at=w['default_date'],discount_basis='Declared synthetic zero-rate prior valuation',points=[dict(cashflow_date=w['recovery_date'],discount_factor=w['discount_factor'])]),dict(reporting_currency='EUR',known_at=w['default_date'],conversion_basis='cashflow_date',rates=[dict(currency='EUR',date=w['recovery_date'],rate=1)]),default_date=w['default_date'],evaluation_cutoff=w['known_at'],reporting_currency='EUR')
        prior.append(cr.workout_lgd(w['ead'],discounted,episode_status='CLOSED',default_date=w['default_date'],evaluation_cutoff=w['known_at'],currency='EUR').value)
    lt=dict(event='economic_loss',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='given_default');X=np.empty((len(prior),0))
    model=cr.fit_fractional_lgd(X,[w['economic_lgd']for w in prior],feature_order=[],link='logit',covariance='HC0',target=lt,training=dict(known_at=max(w['known_at']for w in r['prior_workouts']),data_fingerprint=cr.content_fingerprint(r['prior_workouts']).value,split_id=product+'-prior-workouts',data_origin='SYNTHETIC',workout_population='closed_defaulted_episodes'),model_id=product+'-claim-lgd',model_version='1')
    lgd=cr.predict_lgd(model,np.empty((1,0)),feature_order=[],expected_target=lt,default_condition='given_default').value['mean_lgd'][0]
    schedule=cr.exposure_schedule(dict(facility_id=r['account_id'],currency='EUR',as_of=r['proposed_funding_at'],drawn_principal=r['conditional_ead'],credit_limit=r['conditional_ead'],accrued_interest=0),[],[0],period_end_dates=[r['horizon_end']],day_count='ACT/365F',default_timing='BEFORE_PERIOD_END_EVENTS',allow_overlimit=False,currency='EUR')
    ead=cr.predict_ead(schedule,None,feature_order=[],expected_target=None,reference_dates=[r['proposed_funding_at']],default_dates=[r['horizon_end']],currency='EUR').value['conditional_ead'][0]
    loss=cr.lifetime_expected_loss([pd],[ead*lgd],[r['loss_discount_factor']],[30/365],currency='EUR',probability_measure='PHYSICAL').value
    out=dict(application_id=key,customer_id=app['customer_id'],account_id=r['account_id'],household_id=r['household_id'],forecast_at=D,proposed_funding_at=r['proposed_funding_at'],horizon_days=30,application_pd=pd,conditional_ead=ead,conditional_lgd=lgd,thirty_day_expected_loss=loss['lifetime_expected_loss'],ead_scope=r['ead_scope'],prior_workout_lgd=[w['economic_lgd']for w in prior],prior_known_at=model.model['training']['known_at'])
    raw=dict(application_fit=fitted,lgd_model=model.model,prior_workouts=prior,exposure_schedule=schedule.value,loss_result=loss)
    if 'asset'in r:
        asset=cr.simulate_asset_values(**r['asset']).value;args=deepcopy(r['waterfall']);args['realization_paths'][0]['gross_value']=asset['paths'][0]['asset_value'];recovery=cr.simulate_recovery_waterfall(**args).value
        out['asset_scenario']=dict(scope=r['asset_scope'],gross_value=asset['paths'][0]['asset_value'],cost=50,economic_loss=recovery['claim_results'][0]['economic_loss'],not_added_to_credit_expected_loss=True)
        raw.update(asset=asset,waterfall=recovery)
    if 'bnpl'in r:
        obligations=cr.build_bnpl_obligation_features(**r['bnpl']).value;graph=cr.build_customer_household_graph(**r['household_graph']).value
        household=cr.aggregate_household_obligations(r['obligations'],graph,r['bnpl']['provider_coverage'],currency='EUR',period='month',guarantee_budget_policy='exclude').value
        ht=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='initially_performing');tr=r['hazard_training']
        if tr['outcome_end']>tr['known_at']or tr['known_at']>=D:raise ValueError('Hazard cohort must mature before origin')
        hm=cr.fit_discrete_hazard(r['hazard_panel'],np.empty((10,0)),feature_order=[],period_grid=[0,30],link='logit',partial_period_policy='exclude_censored',target=ht,training=tr,model_id=product+'-prior-discrete',model_version='1')
        hc=cr.predict_event_curves(hm,[dict(path_id=r['account_id'],start=0,stop=30,covariates=[],stratum='all',covariate_time=0)],[30],feature_order=[],target=ht).value['curves'][0]
        out['bnpl']=dict(outstanding=obligations['total_outstanding'],near_term_due=obligations['known_near_term_due'],active_plans=obligations['active_plan_count'],household_debt=household['households'][0]['total_linked_debt_stock'],household_service=household['households'][0]['total_linked_debt_service'],prior_hazard_pd=1-hc['survival'][0],hazard_role='INDEPENDENT_PRIOR_BENCHMARK_NOT_REPLACEMENT_APPLICATION_PD')
        raw.update(bnpl=obligations,household=household,hazard_model=hm.model,hazard_curves=hc)
    if 'expert'in r:
        prior_beta=cr.fit_elicited_beta_prior(**r['expert']).value;update=cr.update_expert_beta_binomial(prior_beta,**r['expert_update']).value
        offers=deepcopy(r['limit_offers'])
        for o in offers:o['pd']=update['posterior_mean']
        frontier=cr.compute_limit_frontier(offers,r['limit_profile'],as_of=D).value
        out['micro']=dict(expert_prior_alpha=prior_beta['alpha'],expert_prior_beta=prior_beta['beta'],posterior_alpha=update['posterior']['alpha'],posterior_beta=update['posterior']['beta'],posterior_mean=update['posterior_mean'],maximum_feasible_amount=frontier['maximum_feasible_amount'],expert_role='SEPARATE_POPULATION_OFFER_SCENARIO_NOT_APPLICATION_PD')
        raw.update(expert_prior=prior_beta,expert_update=update,frontier=frontier)
    if verify_reference:
        calibration_cell=[a for a in c['calibration']if a['driver']==1];pd_oracle=sum(a['outcome']for a in calibration_cell)/len(calibration_cell)
        assert math.isclose(pd,pd_oracle,abs_tol=1e-9)
        assert math.isclose(loss['lifetime_expected_loss'],pd_oracle*ead*r['expected_prior_mean_lgd'],rel_tol=0,abs_tol=1e-8)
        assert math.isclose(lgd,r['expected_prior_mean_lgd'],abs_tol=1e-9)
        assert ead==r['conditional_ead'] and math.isclose(out['thirty_day_expected_loss'],pd*ead*r['expected_prior_mean_lgd'],abs_tol=1e-8)
        if 'asset_scenario'in out:assert out['asset_scenario']['gross_value']==600 and out['asset_scenario']['economic_loss']==450
        if 'bnpl'in out:
            assert out['bnpl']['outstanding']==out['bnpl']['household_debt']==1000 and out['bnpl']['near_term_due']==out['bnpl']['household_service']==500
            assert math.isclose(out['bnpl']['prior_hazard_pd'],.2,abs_tol=1e-8)
        if 'micro'in out:assert out['micro']['posterior_alpha']==4 and out['micro']['posterior_beta']==16 and out['micro']['maximum_feasible_amount']==1000
    out['thirty_day_expected_loss']=round(out['thirty_day_expected_loss'],2)
    return out,raw


def run(data=None,application_data=None,application_chain=None,product_ids=None,verify_reference=True,return_artifacts=False):
    resource=files('creditratingsystem.datasets')
    if data is None:data=json.loads(resource.joinpath('retail_application_risk.json').read_text())
    if application_data is None:application_data=json.loads(resource.joinpath('retail_application_journey.json').read_text())
    rows={};raw={}
    for p,r in data['products'].items():
        if product_ids is None or p in product_ids:rows[p],raw[p]=evaluate(p,application_data['products'][p],r,application_data['target'],application_chain,verify_reference)
    result=dict(data_origin=data['data_origin'],forecast_scope=data['forecast_scope'],presentation_contract='Expected loss displayed to cents; original unrounded calculation remains in raw_artifacts.loss_result.',products=rows)
    if return_artifacts:result['raw_artifacts']=raw
    return result


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
