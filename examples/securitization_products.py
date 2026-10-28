"""Six distinct securitization structures with replaceable local contracts."""
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
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('securitization_products.json').read_text())
    data=deepcopy(data);out={};errors={}
    for name,case in data['variants'].items():
        pool=cr.build_credit_asset_pool(**case['pool_input']).value
        paths=cr.simulate_asset_pool(pool,**case['path_input']).value
        waterfall=cr.run_tranche_waterfall(paths,case['structure_profile']).value
        origin=paths['origin'];horizon=(date.fromisoformat(paths['dates'][-1])-date.fromisoformat(origin)).days
        target=dict(event='ULTIMATE_PRINCIPAL',horizon=horizon,time_unit='days',probability_measure='PHYSICAL',conditioning=paths['scenario_contract']['conditioning'])
        discount={d:1 for d in case['structure_profile']['payment_dates']}
        metrics=cr.estimate_tranche_credit_metrics(waterfall,target,discount).value
        stressed_paths=cr.simulate_asset_pool(pool,**case['credit_stress_input']).value
        stressed_waterfall=cr.run_tranche_waterfall(stressed_paths,case['structure_profile']).value
        stress=cr.stress_tranche_structure(waterfall,[dict(alternative_id='debtor_default',tranche_paths=stressed_waterfall,assumption='First debtor defaults on the first risk date under the same supplied innovations')],target,discount).value
        account=_account_case(data['account_stress_template'],paths)
        static=cr.tranche_loss_from_pool([.1,.15,.2],.1,.2,pool_notional=100).value
        senior=metrics['metrics'][0];subject=dict(subject_id=name+'-senior',subject_type='issue')
        components=[dict(component_id=name+'-pd',quantity='PD',value=senior['event_probability'],**subject,variant='ultimate_principal',definition=target),dict(component_id=name+'-el',quantity='EL',value=senior['expected_ultimate_principal_loss'],currency='EUR',**subject,variant='ultimate_principal',definition=target)]
        binding=dict(profile_id=name,profile_version='1',model_id=case['structure_profile']['structure_id'],model_version=case['structure_profile']['version'],model_hash=waterfall['structure_hash'],input_hash=paths['sha256'])
        assessment=cr.compose_risk_assessment(name+'-assessment',subject,components,binding,origin,origin,origin).value
        try:cr.build_credit_asset_pool(**case['invalid_pool_input'])
        except cr.CreditRiskError as exc:errors[name]=exc.code
        else:raise AssertionError('Invalid product contract accepted: '+name)
        out[name]=dict(pool=pool,paths=paths,waterfall=waterfall,metrics=metrics,credit_stress=stress,account_stress=account,static_boundary=static,assessment=assessment)
    if verify_reference:
        def same(a,b):
            if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10):raise AssertionError((a,b))
        first_principal=dict(ABS=100,RMBS=100,CMBS=20,CLO=0,revolving_pool=0,static_pool=40)
        coupons=dict(ABS=5,RMBS=5,CMBS=9,CLO=10,revolving_pool=10,static_pool=8)
        for name,r in out.items():
            same(r['paths']['cashflows'][0]['principal_available'],first_principal[name]);same(sum(x['interest_available']for x in r['paths']['cashflows']),coupons[name])
            same(sum(x['expected_ultimate_principal_loss']for x in r['metrics']['metrics']),0)
            same(sum(x['expected_ultimate_principal_loss']for x in r['credit_stress']['comparisons'][0]['metrics']['metrics']),75)
            a=r['account_stress']['ending_states'][0];same(a['credit_loss'],a['baseline_cash']/2)
            for a,b in zip(r['static_boundary']['tranche_loss_fraction'],[0,.5,1]):same(a,b)
            assert r['assessment']['composition_status']=='CONSISTENT' and errors[name]==data['variants'][name]['expected_error']
        for name in ('CLO','revolving_pool'):
            purchase=out[name]['paths']['acquisitions'][0];same(purchase['purchase_price'],50);same(purchase['acquired_principal'],50);assert purchase['acquired']
        assert not out['static_pool']['paths']['acquisitions']
    return dict(data_origin=data['data_origin'],version=data['version'],product=data['product'],variants=out,contract_error_codes=errors)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
