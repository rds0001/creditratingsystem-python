"""Alternate fitted macro laws applied to five actual specialized cash-flow contracts."""
from importlib.resources import files
from pathlib import Path
from copy import deepcopy
import json,math,runpy
import numpy as np
from scipy.stats import norm
import creditratingsystem as cr


def _fits(data):
    methods=dict(vecm=cr.fit_vecm,regime=cr.fit_regime_switching,factor=cr.fit_dynamic_factor,copula=cr.fit_copula)
    fitted={key:method(**data['fit_arguments'][key])for key,method in methods.items()}
    assert abs(fitted['vecm'].model['parameters']['beta'][1][0]+1)<.025
    assert fitted['regime'].diagnostics['converged'] and fitted['factor'].diagnostics['converged']
    means=np.asarray(fitted['regime'].model['parameters']['coefficients'])[:,0,0]
    assert np.allclose(means,[-2,2],atol=.03)
    assert fitted['factor'].diagnostics['log_likelihood']>fitted['factor'].diagnostics['initial_log_likelihood']+10
    # Three discordant pairs among28: Kendall tau=(25-3)/28=11/14.
    assert abs(fitted['copula'].diagnostics['kendall_tau'][0][1]-11/14)<1e-12
    return fitted


def run(data=None,verify_reference=True,artifact_paths=None,return_audit_artifacts=False,scenario_methods=None):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_alternate_scenarios.json').read_text())
    local=_fits(data);models={}
    for key,name in data['model_resources'].items():models[key]=cr.load_fitted_model(artifact_paths[name]if artifact_paths else files('creditratingsystem.datasets').joinpath(name)).value
    base=cr.fit_var(data['baseline_macro_history'],lags=1,training=data['baseline_macro_training'],target=data['target'],model_id='specialized-macro',model_version='1',covariance_estimator='mle')
    scenarios=data['scenario_contract'];ids=[s['scenario_id']for s in scenarios];weights=[s['weight']for s in scenarios];n=len(ids);z=np.asarray([s['innovations']for s in scenarios])
    def draws(k,factor_ids):return dict(kind='STANDARD_NORMAL',values=np.repeat(z,k,axis=2).tolist(),factor_ids=factor_ids,scenario_ids=ids,time_grid=[1,2],weights=weights)
    paths={};laws={};macro_audits={}
    for key in ['vecm','regime','factor']:
        model=models[key];p=model['parameters'];factors=p['state_ids']if key=='factor'else model['feature_order'];extra={}
        if key=='regime':extra['regime_uniforms']=[[.05 if i%2==0 else .95]*3 for i in range(n)]
        if key=='factor':extra.update(sampling_contract=dict(law='DECLARED_FIXED_STATE',assumption='Conditional sensitivity at the fitted filtered-state mean; posterior initial uncertainty is not integrated'),initial_state=p['filtered_state_end'])
        cube=cr.simulate_macro_scenarios(model,draws(len(factors),factors),**extra).value
        x=np.asarray(cube['values'])[:,:,0];anchor=data['fit_arguments']['vecm']['level_history']['values'][-1][0]if key=='vecm'else 0
        paths[key]=x-anchor;laws[key]=cube['sampling_contract'];macro_audits[key]=cube
    baseline=cr.simulate_macro_scenarios(base,draws(1,['market'])).value
    conditional=cr.condition_macro_paths(base,draws(1,['market']),[dict(time_index=1,factor_id='market',value=-1)]).value
    paths['conditional']=np.asarray(conditional['values'])[:,:,0];laws['conditional']=conditional['sampling_contract'];macro_audits['conditional']=conditional
    a=base.model['parameters']['lag_coefficients'][0][0][0];b=base.model['parameters']['intercept'][0];q=base.model['parameters']['covariance'][0][0];initial=base.model['parameters']['initial_history'][-1][0];m1=b+a*initial;m2=b+a*m1
    if verify_reference:
        raw=np.asarray(baseline['values'])[:,:,0];expected=raw[:,0]+a/(1+a*a)*(-1-raw[:,1]);assert np.allclose(paths['conditional'][:,0],expected,atol=1e-12);assert np.allclose(paths['conditional'][:,1],-1,atol=1e-12)
    reverse=cr.reverse_stress_scenario(base,lambda x:100*(1-math.exp(.2*x[1,0])),horizon=2,target_loss=30,shock_metric=[[1,0],[0,1]],tolerance=1e-8).value
    target=math.log(.7)/.2;v=np.array([a,1])*math.sqrt(q);optimal=(target-m2)*v/(v@v);reverse_path=np.array([m1+math.sqrt(q)*optimal[0],target])
    assert reverse['feasible'] and np.allclose(np.array(reverse['shocks']).reshape(-1),optimal,atol=2e-5)
    assert abs(reverse['objective']-.5*float(optimal@optimal))<2e-5
    paths['reverse']=np.repeat(reverse_path[None,:],n,axis=0);laws['reverse']=dict(law='CONSTRAINED_PRICE_MARKDOWN',conditioning_scope='CONDITIONAL_STRESS',constraint='30percent second-year common collateral price decline',analytic_solution_used_after_solver_verification=True)
    residual=[[-.6,-.3],[.2,.4],[.8,.6],[-.4,-.7]];uniform=norm.cdf(np.repeat(z,2,axis=2))
    bootstrap=cr.block_bootstrap_scenarios(residual,uniform.tolist(),factor_ids=['market','capacity'],scenario_ids=ids,time_grid=[1,2],frequency='annual',block_length=2,scheme='moving',weights=weights)
    if verify_reference:
        for i in range(n):
            start=min(int(3*uniform[i,0,0]),2)
            assert bootstrap.diagnostics['selected_indices'][i]==[start,start+1]
            assert np.allclose(np.asarray(bootstrap.value['values'])[i],np.asarray(residual)[start:start+2],atol=1e-12)
    paths['bootstrap']=np.asarray(bootstrap.value['values'])[:,:,0];laws['bootstrap']=dict(law='JOINT_MOVING_BLOCK_BOOTSTRAP',conditioning_scope='PIT_EMPIRICAL_STRESS',block_length=2,cycle_equilibrium_claimed=False);macro_audits['bootstrap']=bootstrap.to_dict()
    u=norm.cdf(np.c_[z[:,0,0],z[:,1,0]]);dependent=cr.simulate_copula(models['copula'],u.tolist(),variable_ids=['market','capacity']).value;normal=norm.ppf(dependent)
    rho=math.sin(math.pi*(11/14)/2)
    if verify_reference:assert np.allclose(normal[:,1],rho*z[:,0,0]+math.sqrt(1-rho*rho)*z[:,1,0],atol=1e-11)
    paths['copula']=normal;laws['copula']=dict(law='FITTED_GAUSSIAN_COPULA_TWO_DATE_MARKET',conditioning_scope='PIT',time_dependence='Fitted joint two-date ranks, no temporal Markov claim')
    ledger=runpy.run_path(str(Path(__file__).with_name('specialized_asset_finance.py')))['_ledger_oracle'];reports={};executions={}
    selected=list(paths)if scenario_methods is None else list(scenario_methods)
    if not selected or len(set(selected))!=len(selected)or set(selected)-set(paths):raise ValueError('Select distinct supported scenario methods')
    for law in selected:
        path=paths[law]
        price=np.exp(.2*path)
        cube=dict(schema_version='1.0',kind='ScenarioCube',axes=['scenario','time','factor'],scenario_ids=ids,time_grid=[1,2],weights=weights,origin='2023-01-01',frequency='annual',probability_measure='PHYSICAL',sampling_contract=dict(law='FORECAST_CONDITIONAL',conditioning_scope=laws[law].get('conditioning_scope','PIT'),source_sampling_contract=laws[law],ensemble='Original18explicit weighted innovation/counterparty nodes; no IID confidence interval or unconditional frequency estimate'),factor_ids=[],units=[],values=[[[]for _ in range(2)]for _ in ids])
        for case in data['cases']:
            key=case['case_id'];cube['factor_ids'] += [key+'-revenue',key+'-asset'];cube['units']+=['relative_index']*2
            for s in range(n):
                for t in range(2):cube['values'][s][t]+=[round(1-case['floating_share'][t]+case['floating_share'][t]*price[s,t],12),round(price[s,t]*case.get('completion_multiplier',[1,1])[t],12)]
        params=data['parameter_snapshot'];recipes=[dict(transaction_id=c['case_id'],version='1',subject_id=c['case_id'],parameter_id=c['case_id'],calendar=data['calendar'],parameter_hash=cr.content_fingerprint(params['parameters'][c['case_id']]).value,cube_hash=cr.content_fingerprint(cube).value)for c in data['cases']]
        execution=cr.simulate_transactions(recipes,cube,data['entity_innovations'],params).value;records=[]
        for case,tx in zip(data['cases'],execution['transactions']):
            # Execution order is canonical; join by identity rather than input order.
            tx=next(t for t in execution['transactions']if t['transaction_id']==case['case_id'])
            checks=[ledger(case,sc,price[s].tolist(),tx)for s,sc in enumerate(scenarios)]if verify_reference else[]
            if verify_reference:
                assert math.isclose(tx['risk']['pd'],math.fsum(w*c['defaulted']for w,c in zip(weights,checks)),abs_tol=1e-12)
                assert math.isclose(tx['risk']['expected_loss'],math.fsum(w*c['loss']for w,c in zip(weights,checks)),abs_tol=1e-10)
            records.append(dict(product=case['product'],case_id=case['case_id'],risk=tx['risk'],ledger_oracles=checks,transaction_hash=cr.content_fingerprint(tx).value))
        reports[law]=dict(sampling_contract=cube['sampling_contract'],scenario_cube_hash=execution['scenario_cube_hash'],transactions=records);executions[law]=dict(cube=cube,execution=execution)
    report=dict(data_origin='SYNTHETIC',reporting_contract=data['reporting_contract'],fit_checks=dict(vecm_cointegrating_relation=True,two_distinct_hidden_regimes=True,missing_observation_integrated=True,kalman_likelihood_improved=True,copula_kendall_tau=11/14),reverse_oracle=dict(price_markdown=.3,shocks=optimal.tolist(),objective=.5*float(optimal@optimal),path=reverse_path.tolist()),alternatives=reports)
    return dict(report=report,audits=dict(local_fits={k:v.to_dict()for k,v in local.items()},macro_paths=macro_audits,reverse_solver=reverse,executions=executions))if return_audit_artifacts else report

if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
