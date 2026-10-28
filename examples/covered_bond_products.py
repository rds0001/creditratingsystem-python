"""Mortgage/public-sector cover and conditional maturity extension examples."""
from copy import deepcopy
from datetime import date
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def _account_case(template, pool_paths):
    """Stress a remittance of cash actually collected in the first modeled period."""
    args=deepcopy(template);first=pool_paths['cashflows'][0];origin=first['date'];end=pool_paths['dates'][-1]
    amount=first['principal_available']+first['interest_available']+first['recovery_gross']
    args['as_of']=origin;args['scenario_contract']['dates']=[end]
    args['baseline_cashflows']=[dict(scenario_id='s',cashflow_id='cash',date=end,amount=amount)]
    args['exposure_paths']=[dict(scenario_id='s',exposure_id='account',date=end,signed_replacement_mark=0)]
    args['collateral_paths']=[dict(scenario_id='s',netting_set_id='net',date=end,known_at=origin,collateral_value=0,haircut=0)]
    args['settlement_terms'][0].update(estate_delay_days=0,collateral_delay_days=0,replacement_cost=0,replacement_delay_days=0)
    args['party_default_paths']=[dict(scenario_id='s',default_group_id='bank',date=end,conditional_pd=1,default_channel='bank')]
    args['shared_innovations']=[dict(scenario_id='s',date=end,channel_id='bank',uniform=.5)]
    return cr.stress_pool_counterparties(**args).value


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('covered_bond_products.json').read_text())
    data=deepcopy(data);out={};errors={}
    for name,case in data['variants'].items():
        subjects=cr.build_rating_subjects(**case['subject_input']).value
        pool=cr.build_credit_asset_pool(**case['pool_input']).value
        paths=cr.simulate_asset_pool(pool,**case['path_input']).value
        p=case['issuer_portfolio'];graph=cr.build_entity_graph(**p['graph_input'])
        prepared=cr.prepare_credit_portfolio(p['positions'],graph,p['default_group_policy'],p['contract'])
        issuer=cr.simulate_credit_portfolio(prepared,p['group_pds'],p['dependence_contract'],p['shared_innovations'],p['conditional_loss_paths'],levels=[.5]).value
        issuer_rows=deepcopy(case['issuer_paths'])
        for i,row in enumerate(issuer_rows):
            failed=issuer['position_defaults'][i][0]
            row['issuer_default_date']=paths['scenario_contract']['dates'][0] if failed else None
            row['extension_triggered']=name=='maturity_extension' and failed
        bond=deepcopy(case['bond_contract']);bond['innovation_hash']=paths['innovation_hash']
        covered=cr.covered_bond_dual_recourse(pool,paths,bond,issuer_rows).value
        performing=deepcopy(paths);performing['cashflows']=[r for r in paths['cashflows'] if r['scenario_id']=='t']
        account=_account_case(data['account_stress_template'],performing)
        origin=paths['origin'];horizon=(date.fromisoformat(covered['horizon_end'])-date.fromisoformat(origin)).days
        sampling=dict(contract_id=name+'-risk',version='1',subject_id=name+'-bond',currency='EUR',probability_measure='PHYSICAL',horizon=horizon,time_unit='days',conditioning='initially_performing',sampling_design='FIXED_WEIGHTED_SCENARIOS',event='first_default',default_definition='ULTIMATE_CONTRACTUAL_SHORTFALL')
        records=[]
        for weight,e in zip(paths['scenario_weights'],covered['ending_states']):
            failed=e['ultimate_shortfall']>0
            records.append(dict(scenario_id=e['scenario_id'],subject_id=name+'-bond',initial_default=False,defaulted=failed,horizon_complete=True,weight=weight,ead_at_default=e['contractual_claim'] if failed else 0,economic_loss=e['ultimate_shortfall'] if failed else 0))
        risk=cr.estimate_simulation_risk(records,sampling).value
        target={k:sampling[k]for k in ['event','horizon','time_unit','probability_measure','conditioning','default_definition']}
        subject=dict(subject_id=name+'-bond',subject_type='issue')
        components=[dict(component_id=name+'-pd',quantity='PD',value=risk['pd'],**subject,variant='dual_recourse_ultimate',definition=target),dict(component_id=name+'-el',quantity='EL',value=risk['expected_loss'],currency='EUR',**subject,variant='dual_recourse_ultimate',definition=target)]
        binding=dict(profile_id=name,profile_version='1',model_id=bond['bond_id'],model_version=bond['version'],model_hash=covered['sha256'],input_hash=paths['sha256'])
        assessment=cr.compose_risk_assessment(name+'-assessment',subject,components,binding,origin,origin,origin).value
        try:
            if name=='maturity_extension':
                wrong=deepcopy(issuer_rows);wrong[1]['extension_triggered']=True
                cr.covered_bond_dual_recourse(pool,paths,bond,wrong)
            else:
                wrong=deepcopy(case['pool_input']);wrong['eligibility_profile']['allowed_asset_classes']=['AUTO_LOAN'];cr.build_credit_asset_pool(**wrong)
        except cr.CreditRiskError as exc:errors[name]=exc.code
        else:raise AssertionError('Invalid covered-bond contract accepted')
        out[name]=dict(subjects=subjects,pool=pool,paths=paths,issuer_portfolio=issuer,covered_recourse=covered,account_stress=account,risk=risk,assessment=assessment)
    if verify_reference:
        def same(a,b):
            if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10):raise AssertionError((a,b))
        for name,r in out.items():
            e=r['covered_recourse']['ending_states'];same(e[0]['cover_payment'],35);same(e[0]['ultimate_shortfall'],35 if name=='maturity_extension'else 32.5);same(e[1]['issuer_payment'],100);same(e[1]['ultimate_shortfall'],0)
            same(r['risk']['pd'],.5);same(r['risk']['expected_loss'],17.5 if name=='maturity_extension'else 16.25)
            same(r['issuer_portfolio']['distribution']['expected_loss'],25) #standalone issuer loss, never added to bond EL
            same(r['account_stress']['ending_states'][0]['credit_loss'],22.5)
            assert r['assessment']['composition_status']=='CONSISTENT'
            assert errors[name]==('STRUCTURED_EXTENSION'if name=='maturity_extension'else'STRUCTURED_ROUTE')
        ext=out['maturity_extension']['covered_recourse']['ending_states'][0]
        assert ext['extension_activated'] and ext['effective_maturity']=='2024-07-01'
        same(ext['original_date_principal_deferral_or_shortfall'],100)
    return dict(data_origin=data['data_origin'],version=data['version'],product=data['product'],variants=out,contract_error_codes=errors)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
