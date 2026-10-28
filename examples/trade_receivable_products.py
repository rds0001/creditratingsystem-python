"""Invoice finance with limited seller recourse and distinct dilution losses."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def _account_case(template,amount):
    args=deepcopy(template);origin='2023-04-01';end='2023-05-01'
    args['as_of']=origin;args['scenario_contract']['dates']=[end]
    args['baseline_cashflows']=[dict(scenario_id='s',cashflow_id='cash',date=end,amount=amount)]
    args['exposure_paths']=[dict(scenario_id='s',exposure_id='account',date=end,signed_replacement_mark=0)]
    args['collateral_paths']=[dict(scenario_id='s',netting_set_id='net',date=end,known_at=origin,collateral_value=0,haircut=0)]
    args['settlement_terms'][0].update(estate_delay_days=0,collateral_delay_days=0,replacement_cost=0,replacement_delay_days=0)
    args['party_default_paths']=[dict(scenario_id='s',default_group_id='bank',date=end,conditional_pd=1,default_channel='bank')]
    args['shared_innovations']=[dict(scenario_id='s',date=end,channel_id='bank',uniform=.5)]
    return cr.stress_pool_counterparties(**args).value


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('trade_receivable_products.json').read_text())
    data=deepcopy(data);out={};errors={}
    for name,case in data['variants'].items():
        pool=cr.build_credit_asset_pool(**case['pool_input']).value
        borrowing=cr.receivables_borrowing_base(**case['borrowing_input']).value
        paths=cr.simulate_trade_receivables(pool,**case['simulation_input']).value
        losses=[r['net_credit_loss']+r['net_dilution_loss']+r['commingling_loss']+r['workout_and_replacement_cost'] for r in paths['ending_states']]
        distribution=cr.credit_loss_distribution(losses,[.25]*4,scenario_ids=['s','t','u','v'],levels=[.5,.95],currency='EUR').value
        # A subsequent custody stress on the fully collected performing path;
        # not an additional origination loss or an independent default probability.
        account=_account_case(data['account_stress_template'],paths['ending_states'][3]['net_investor_cash'])
        target=dict(event='net_investor_shortfall',horizon=90,time_unit='days',probability_measure='PHYSICAL',conditioning='PIT')
        subject=dict(subject_id=name+'-pool',subject_type='portfolio')
        components=[dict(component_id=name+'-el',quantity='EL',value=distribution['expected_loss'],currency='EUR',**subject,variant='credit_dilution_and_operating_cost',definition=target)]
        binding=dict(profile_id=name,profile_version='1',model_id='invoice-finance',model_version='1',model_hash=paths['sha256'],input_hash=pool['sha256'])
        assessment=cr.compose_risk_assessment(name+'-assessment',subject,components,binding,pool['as_of'],pool['as_of'],pool['as_of']).value
        wrong=deepcopy(case['simulation_input'])
        if name=='recourse':wrong['seller_terms'][0]['settlement_date']='2023-01-01'
        elif name=='nonrecourse':wrong['invoice_scenarios'][0]['payment_date']='2024-01-01'
        else:wrong['party_mapping'].append(deepcopy(wrong['party_mapping'][0]))
        try:cr.simulate_trade_receivables(pool,**wrong)
        except cr.CreditRiskError as exc:errors[name]=exc.code
        else:raise AssertionError('Invalid invoice contract accepted')
        out[name]=dict(pool=pool,borrowing_base=borrowing,paths=paths,distribution=distribution,account_stress=account,assessment=assessment)
    if verify_reference:
        def same(a,b):
            if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10):raise AssertionError((a,b))
        expected={'recourse':([30,40,20,0],40),'nonrecourse':([60,60,20,0],35),'invoice_pool':([70,100,40,0],85)}
        for name,r in out.items():
            loss,bb=expected[name]
            for a,b in zip(r['distribution']['loss_paths'],loss):same(a,b)
            same(r['distribution']['expected_loss'],sum(loss)/4);same(r['borrowing_base']['borrowing_base'],bb)
            same(r['paths']['ending_states'][2]['net_credit_loss'],0)
            assert not any(i['debtor_credit_event'] for i in r['paths']['invoice_results'] if i['scenario_id']=='u')
            assert r['assessment']['composition_status']=='CONSISTENT'
            same(r['account_stress']['ending_states'][0]['credit_loss'],100 if name=='invoice_pool'else 50)
            assert errors[name]==('STRUCTURED_DEFAULT_GROUP'if name=='invoice_pool'else'STRUCTURED_CALENDAR')
        assert all(s['aggregate_payment']==0 for s in out['nonrecourse']['paths']['seller_results'])
    return dict(data_origin=data['data_origin'],version=data['version'],product=data['product'],variants=out,contract_error_codes=errors)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
