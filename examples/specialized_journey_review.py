"""Source-bound specialist governance and genuine collateral-kernel convergence."""
from importlib.resources import files
from pathlib import Path
from copy import deepcopy
from datetime import date,timedelta
import json,math,runpy
import creditratingsystem as cr


def _convergence(case,contract):
    params=deepcopy(case['asset_parameters']);params['idiosyncratic_volatility']=contract['volatility'];origin=date.fromisoformat(params['valuation_date']);sigma=params['idiosyncratic_volatility'];basis=params['year_basis_days'];fine=contract['fine_days'];T=contract['horizon_days']/basis
    age=params['initial_age_years'];life=params['economic_life_years'];remaining=max(life-age-T,0);ageing=remaining/(life-age)if params['depreciation_model']=='straight_line'else math.exp(-params['annual_depreciation_rate']*T);baseline=params['salvage_value']+(params['initial_value']-params['salvage_value'])*ageing
    runs=[]
    for rep,innovation in contract['innovations'].items():
        for size in contract['sample_sizes']:
            ids=innovation['path_ids'][:size];zs=innovation['values'][:size]
            for step in [fine,2*fine]:
                rows=[]
                for sid,z in zip(ids,zs):
                    shocks=z if step==fine else [(z[0]+z[1])/math.sqrt(2),(z[2]+z[3])/math.sqrt(2)]
                    rows.extend(dict(scenario_id=sid,date=(origin+timedelta(days=step*(i+1))).isoformat(),market_index=1,innovation=x)for i,x in enumerate(shocks))
                result=cr.simulate_asset_values(params,rows).value;terminal={r['scenario_id']:r['asset_value']for r in result['paths']if r['date']==(origin+timedelta(days=contract['horizon_days'])).isoformat()};values=[]
                for sid,z in zip(ids,zs):
                    oracle=baseline*math.exp(-.5*sigma*sigma*T+sigma*math.sqrt(fine/basis)*math.fsum(z));assert math.isclose(terminal[sid],oracle,rel_tol=1e-12,abs_tol=1e-11)
                    values.append(round(terminal[sid],2))
                runs.append(dict(replicate_id=rep,time_step=step,sample_size=size,path_ids=ids,values=values,innovation_fingerprint=cr.content_fingerprint(zs).value))
    target=dict(event='default',horizon=contract['horizon_days'],time_unit='days',probability_measure='PHYSICAL',conditioning='PIT')
    design=dict(model_id=case['case_id']+'-collateral-kernel',model_version='1',parameter_version='1',data_fingerprint=cr.content_fingerprint(params).value,population_id=case['case_id'],split_id='prospective-kernel-check',prediction_origin='EXTERNAL',evaluation_cutoff=params['valuation_date'],target=target,currency=params['currency'])
    sampling=dict(sampling_unit='iid_paths_with_independent_replicates',metric='mean',finest_time_step=fine,time_unit='days',reference=contract['scope'],analytic_reference=dict(value=baseline,source='Unit-mean lognormal multiplier under explicitly independent standard normals, exact deterministic ageing; cent-rounding expectation error bounded by 0.005 EUR'))
    result=cr.simulation_convergence(runs,contract['innovations'],sampling_contract=sampling,precision=dict(confidence_level=.95,mc_half_width=.5,grid_difference_tolerance=.005),design=design).value
    assert all(x['estimate']==0 for x in result['metrics']['paired_grid_differences'])
    return dict(result=result,parameters=params,analytic_unrounded_mean=baseline,reporting_convention=contract['reporting_convention'],stream=contract['stream'],scope=contract['scope'])


def run(data=None,verify_reference=True,accounts_path=None,releases_path=None,governance_data=None,model_path=None,return_artifacts=False):
    resources=files('creditratingsystem.datasets');here=Path(__file__).parent
    if data is None:data=json.loads(resources.joinpath('specialized_journey_review.json').read_text())
    if governance_data is None:governance_data=json.loads(resources.joinpath(data['governance_resource']).read_text())
    governance_data=deepcopy(governance_data);selected={c['case_id']for c in data['cases']};governance_data['journeys']=[j for j in governance_data['journeys']if j['case_id']in selected]
    lineage=runpy.run_path(str(here/'input_lineage.py'))['prepare_snapshot'](accounts_path or resources.joinpath('specialized_positions.csv'),releases_path or resources.joinpath('specialized_releases.csv'),data['lineage_config'],data['lineage_config']['cutoffs'][0]);rows={x['entity_id']:x for x in lineage['rows']}
    for j in governance_data['journeys']:
        case=next(c for c in data['cases']if c['case_id']==j['case_id']);p=j['frozen_arguments']['parameter_snapshot']['parameters'][j['case_id']];source=rows[j['case_id']]
        if source['product_id']!=j['product']or source['balance']!=sum(l['terms']['principal']for l in p['loans'])or source['x']!=p['operating_base'][0]['base_unit_price']or case['asset_parameters']!=p['assets'][0]['parameters']:raise ValueError('Source must bind actual loan principal, operating input and frozen asset profile')
    governance=runpy.run_path(str(here/'specialized_governance.py'))['run'](governance_data,verify_reference,model_path,return_audit_artifacts=True);tools=runpy.run_path(str(here/'_wholesale_journey_tools.py'));out=[];artifacts=[]
    for case in data['cases']:
        sid=case['case_id'];audit=governance['journeys'][sid];decision=audit['results'][0]['value']['decision'];reason={r['reason_code']:'Declared policy rule '+r['rule_id']for r in decision['rule_trace']};explanation=cr.decompose_policy_decision(decision,reason).value;replay=tools['_replay'](audit,rows[sid]);convergence=_convergence(case,data['convergence'])
        if verify_reference:assert lineage['audit']['decision']['status']=='PASS'
        out.append(dict(case_id=sid,product=case['product'],claim_ids=case['claim_ids'],source_snapshot_hash=lineage['selected_snapshot_hash'],governance=audit['report'],policy_explanation=explanation,replay=replay['result'],negative_replay=replay['negative'],collateral_convergence=convergence))
        artifacts.append(dict(case_id=sid,governance=audit,replay_inputs=replay['inputs']))
    report=dict(data_origin='SYNTHETIC',lineage=lineage['audit'],archive_sha256=lineage['archive_sha256'],cases=out)
    return dict(report=report,artifacts=artifacts)if return_artifacts else report

if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
