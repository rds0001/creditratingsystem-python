"""Observed retail cohorts and the same facilities in an explicit portfolio DGP."""
from copy import deepcopy
from importlib.resources import files
from itertools import product as cartesian_product
import json
import math
import creditratingsystem as cr


def _portfolio(case,product):
    positions=case['positions'];graph=cr.build_entity_graph([dict(entity_id=e)for e in sorted({p['entity_id']for p in positions})],[],dict(policy_id='no-inferred-cross-default',version='1',edge_types=['CROSS_DEFAULT']),as_of='2024-01-01',known_at='2024-01-01')
    contract=dict(portfolio_id=product,version='1',currency='EUR',as_of='2024-01-01',known_at='2024-01-01',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='initially_performing',event='first_default')
    prepared=cr.prepare_credit_portfolio(positions,graph,dict(policy_id='legal-groups',version='1',edge_types=['CROSS_DEFAULT'],direction_policy='mutual_only',transitive=True),contract).value
    groups=prepared['default_group_ids'];forecast={r['prediction_id']:r for r in case['predictions']};pds={r['default_group_id']:forecast[r['position_id']]['pd']for r in prepared['positions']};ids=[r['position_id']for r in positions];ead=[p['ead']for p in positions];lgd=[forecast[p['position_id']]['lgd']for p in positions]
    for g in groups:
        if len({forecast[r['position_id']]['pd']for r in prepared['positions']if r['default_group_id']==g})!=1:raise ValueError('Frozen facility PDs must agree within a legal default group')
    uniforms=[];weights=[]
    for state in cartesian_product([0,1],repeat=len(groups)):
        uniforms.append([pds[g]/2 if bit else(1+pds[g])/2 for g,bit in zip(groups,state)])
        weights.append(round(math.prod(pds[g]if bit else 1-pds[g]for g,bit in zip(groups,state)),12))
    scenario_ids=['state-'+str(i)for i in range(len(weights))];dep=dict(kind='SUPPLIED_DEPENDENT_UNIFORMS',group_ids=groups);draws=dict(scenario_ids=scenario_ids,weights=weights,group_ids=groups,sampling_design='EXACT_STATES',group_uniforms=uniforms)
    def paths(dependence,innovations):return dict(scenario_ids=innovations['scenario_ids'],position_ids=ids,driver_hash=cr.content_fingerprint(dict(dependence_contract=dependence,shared_innovations=innovations)).value,ead=[ead for _ in innovations['scenario_ids']],lgd=[lgd for _ in innovations['scenario_ids']],guarantee_treatment='NOT_YET_APPLIED')
    simulation=cr.simulate_credit_portfolio(prepared,pds,dep,draws,paths(dep,draws),levels=[.8,.95]).value
    distribution=cr.credit_loss_distribution(simulation['portfolio_loss_paths'],weights,scenario_ids=scenario_ids,levels=[.8,.95],currency='EUR').value
    allocation=cr.allocate_expected_shortfall(simulation['net_position_losses'],weights,scenario_ids=scenario_ids,position_ids=ids,level=.8,currency='EUR',groups={p['position_id']:p['household_id']for p in positions}).value
    removal=cr.incremental_credit_risk(simulation['portfolio_loss_paths'],[-r[0]for r in simulation['net_position_losses']],weights,scenario_ids=scenario_ids,levels=[.8,.95],currency='EUR',change_id='remove-first-held-facility').value
    bygroup={g:[r for r in prepared['positions']if r['default_group_id']==g]for g in groups};initial={g:'A'if pds[g]==.1 else'B'for g in groups};value_contract=dict(valuation_id=product+'-marks',version='1',currency='EUR',valuation_basis='SUPPLIED_FROZEN_MARKED_VALUES',group_ids=groups,states=['A','B','D'],initial_values=[sum(r['ead']for r in bygroup[g])for g in groups],terminal_values_by_grade=[[sum(r['ead']for r in bygroup[g]),sum(r['ead']for r in bygroup[g]),sum(r['ead']*(1-forecast[r['position_id']]['lgd'])for r in bygroup[g])]for g in groups],cash_distributions=[[0]*len(groups)for _ in scenario_ids])
    migration=cr.simulate_portfolio_migrations(initial,[[.9,0,.1],[0,.5,.5],[0,0,1]],dict(scenario_ids=scenario_ids,group_ids=groups,values=[[[1-u for u in row]]for row in uniforms],weights=weights),dict(contract_id=product+'-migration',version='1',group_ids=groups,states=['A','B','D'],default_state='D',allow_cure=False,probability_measure='PHYSICAL',time_grid=[30],time_unit='days',matrix_mode='HOMOGENEOUS',innovation_law='CONDITIONALLY_UNIFORM_GIVEN_PRIOR_STATES'),valuation_contract=value_contract).value
    # Weighted cells integrate the exact threshold events for loadings zero and one.
    # They are not a generic quadrature rule for intermediate loadings.
    factor=[];residual=[];stress_weights=[]
    for q,w in [(.05,.1),(.3,.4),(.75,.5)]:
        for u,weight in zip(uniforms,weights):factor.append([case['normal_coordinates'][format(q,'.2f')]]);residual.append([case['normal_coordinates'][format(x,'.2f')]for x in u]);stress_weights.append(round(w*weight,12))
    normal_draws=dict(scenario_ids=['normal-cell-'+str(i)for i in range(len(factor))],weights=stress_weights,group_ids=groups,sampling_design='FIXED_WEIGHTED_SCENARIOS',factor_normals=factor,group_normals=residual)
    base_dep=dict(kind='GAUSSIAN_FACTORS',group_ids=groups,factor_ids=['common'],factor_correlation=[[1]],loadings=[[0]for _ in groups]);stress_dep=dict(base_dep,loadings=[[1]for _ in groups]);stress_case=dict(case_id='perfect-common-factor',dependence_contract=stress_dep,group_pds=pds,conditional_loss_paths=paths(stress_dep,normal_draws),recovery_response='FIXED_PATHS')
    stress=cr.stress_portfolio_factor_structure(prepared,pds,base_dep,normal_draws,paths(base_dep,normal_draws),[stress_case],levels=[.8,.95]).value
    expected=sum(p['ead']*forecast[p['position_id']]['pd']*forecast[p['position_id']]['lgd']for p in positions);total=sum(ead)
    benchmark=cr.asymptotic_factor_loss(sum(p['ead']*forecast[p['position_id']]['pd']for p in positions)/total,.4,0,[-2,0,2]).value
    target={k:contract[k]for k in ['event','horizon','time_unit','probability_measure','conditioning']};binding=dict(profile_id='retail-portfolio',profile_version='1',model_id=product+'-portfolio-dgp',model_version='1',model_hash=cr.content_fingerprint(dict(pds=pds,dependence=dep,lgd=lgd)).value,input_hash=prepared['portfolio_hash']);assessment=cr.compose_risk_assessment(product+'-portfolio-assessment',dict(subject_id=product,subject_type='portfolio'),[dict(component_id=key,quantity=key,value=value,subject_id=product,subject_type='portfolio',variant='standalone',definition=target,currency='EUR')for key,value in [('EL',round(distribution['expected_loss'],2)),('EAD',total)]],binding,'2024-01-01','2024-01-01','2024-02-01').value
    return dict(prepared=prepared,group_pds=pds,simulation=simulation,distribution=distribution,expected_shortfall_allocation=allocation,paired_facility_removal=removal,migration=migration,asymptotic_zero_correlation_benchmark=benchmark,stress=dict(base_distribution=stress['base']['distribution'],stressed_distribution=stress['stress_cases'][0]['stressed']['distribution'],paired_comparison=stress['stress_cases'][0]['paired_comparison'],shared_innovations_hash=stress['shared_innovations_hash'],exact_endpoint_scope='Threshold event cells exact only for loadings zero and one; not a continuous-normal simulation sample'),assessment=assessment,assessment_monetary_rounding='EUR_CENTS',analytic_expected_loss=expected)


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('retail_portfolio_cases.json').read_text())
    data=deepcopy(data);out={}
    for product,case in data['cases'].items():
        fitted=cr.fit_retail_risk_pools(**case['pool_fit']);assignments=cr.assign_retail_risk_pool(fitted_model=fitted,**case['assignment_inputs']);poolargs=dict(fitted_model=fitted,pool_assignments=assignments,assignment_inputs=case['assignment_inputs'],predictions=case['predictions'],cohort_outcomes=case['cohort_outcomes'],workouts=case['workouts'],links=case['links'],estimation_profile=case['estimation_profile'],design=case['design'])
        clusters=sorted({r['cluster_id']for r in case['predictions']});resamples=[clusters,list(reversed(clusters)),[clusters[0]]*len(clusters),[clusters[-1]]*len(clusters)]
        diagnostics=cr.retail_pool_diagnostics(**poolargs,uncertainty='cluster_bootstrap',cluster_resamples=resamples).value
        estimated=cr.estimate_pool_risk_parameters(assignments,case['cohort_outcomes'],case['workouts'],case['estimation_profile'],evaluation_cutoff='2024-02-01',calibration_id=product+'-observed',calibration_version='1').value
        # Retain observed arithmetic, not independent-unit intervals for linked households.
        estimated_summary=[{k:row[k]for k in ['pool_id','pd','closed_workout_lgd','default_conditioned_ead','default_conditioned_ccf','open_workouts','missing_default_workouts']}for row in estimated['pool_risk_table']]
        for row in estimated_summary:
            for value in row.values():
                if isinstance(value,dict):value.pop('independent_episode_standard_error',None)
        drift=cr.detect_credit_feature_drift([dict(score=r['score'])for r in case['pool_fit']['observations']],[dict(score=r['score'])for r in case['assignment_inputs']['observations']],feature_order=['score'],method='psi',permutations=None,multiple_testing='not_applicable',dependence_assumption='descriptive',frozen_bins={'score':dict(kind='numeric',cutpoints=[25,50,75],support=[0,100])}).value
        monitoring=cr.build_monitoring_cohorts(**case['monitoring']);early=deepcopy(case['monitoring']);early['evaluation_cutoff']='2024-01-20';immature=cr.build_monitoring_cohorts(**early).value
        missing=deepcopy(case['monitoring']);missing['coverage']=missing['coverage'][:-1];incomplete=cr.build_monitoring_cohorts(**missing).value
        summary=cr.monitor_rating_cohort(monitoring,group_by=['cohort'],reference_window=dict(start='2024-01-01',end='2024-01-31'),minimum_count=10,uncertainty='independent_cluster_bootstrap',cluster_resamples=resamples).value
        counts=cr.default_count_backtest(**case['count_backtest']).value
        loss_design=deepcopy(case['design']);loss_design['target']['conditioning']='default'
        lgdtest=cr.lgd_backtest(case['loss_records'],design=loss_design,workout_policy=dict(method='closed_only',reference='Independent actual retail settlements; open claims excluded'),uncertainty='none').value
        eadtest=cr.ead_backtest(case['loss_records'],design=loss_design,weighting_policy=dict(method='equal',reference='Each observed defaulted facility once'),uncertainty='none').value
        vintage=cr.build_vintage_performance(**case['vintage']).value
        risk=_portfolio(case,product);concentration=cr.credit_concentration_measures(case['positions'],weight_field='ead',group_fields=['household_id'],weight_unit='EUR').value
        out[product]=dict(mechanism=case['mechanism'],feature_drift=drift,pool_diagnostics=diagnostics,observed_pool_parameters=estimated_summary,pool_parameter_interval_policy='Independent-unit intervals omitted; household links retained in diagnostics',monitoring=monitoring.value,cohort_summary=summary,immature_cohort=immature,incomplete_negative_cohort=incomplete,default_count_backtest=counts,lgd_backtest=lgdtest,ead_backtest=eadtest,vintage=vintage,concentration=concentration,portfolio_risk=risk)
        if verify_reference:
            pa=[(x+.5)/12.5 for x in [2,2,4,2,0]];pb=[(x+.5)/8.5 for x in [4,0,0,2,0]]
            assert math.isclose(drift[0]['statistic'],sum((b-a)*math.log(b/a)for a,b in zip(pa,pb)),abs_tol=1e-12)
            assert counts['overall']['observed_defaults']==3 and counts['overall']['observed_count']==6
            assert not any(r['eligible']for r in immature['rows'])and sum(r['eligible']for r in incomplete['rows'])==5
            assert vintage[0]['cumulative_default_rate']==.5 and vintage[1]['cumulative_default_rate']is None
            assert math.isclose(lgdtest['metrics']['overall']['ead_weighted']['bias'],.1,abs_tol=1e-12)
            assert math.isclose(eadtest['metrics']['ead']['overall']['selected_weighting']['bias'],-50*case['exposure_scale'],abs_tol=1e-8)
            assert math.isclose(risk['distribution']['expected_loss'],risk['analytic_expected_loss'],rel_tol=1e-12)
            assert math.isclose(sum(r['contribution']for r in risk['expected_shortfall_allocation']['contributions']),risk['expected_shortfall_allocation']['expected_shortfall'],rel_tol=1e-12)
            assert all(math.isclose(x,y,abs_tol=1e-8)for x,y in zip(risk['simulation']['portfolio_loss_paths'],risk['migration']['marked_values']['portfolio_marked_loss_paths']))
            assert math.isclose(risk['stress']['base_distribution']['expected_loss'],risk['stress']['stressed_distribution']['expected_loss'],rel_tol=1e-12)
            assert all(math.isclose(rate*risk['prepared']['total_ead'],risk['analytic_expected_loss'],rel_tol=1e-12)for rate in risk['asymptotic_zero_correlation_benchmark']['conditional_loss_rate'])
    return dict(product=data['product'],data_origin=data['data_origin'],version=data['version'],cases=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
