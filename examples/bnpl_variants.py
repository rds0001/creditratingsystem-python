"""Allocated purchase plans, effective refunds and short observation windows."""
from importlib.resources import files
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('bnpl_variants.json').read_text())
    cases={}
    for name,args in data['cases'].items():
        plans=cr.build_bnpl_obligation_features(**args).value
        # Amounts are already net of individually allocated payments/refunds;
        # do not apply those cash events a second time in the DPD snapshot.
        obligations=[dict(obligation_id=x['installment_id'],account_id='customer-plan-pool',due_at=x['due_at'],known_at=args['information_cutoff'],amount=x['outstanding'],currency=args['currency'])for x in plans['installments']]
        dpd=cr.compute_days_past_due(obligations,[],as_of=args['as_of'],known_at=args['information_cutoff'],currency=args['currency']).value
        cases[name]=dict(plans=plans,net_obligation_dpd=dpd)
    short=cr.build_observation_windows(**data['observation'],horizon_days=20).value
    annual=cr.build_observation_windows(**data['observation'],horizon_days=365).value
    if verify_reference:
        assert cases['pay_in_installments']['plans']['total_outstanding']==75
        assert cases['pay_in_installments']['net_obligation_dpd']['past_due_amount']==25
        assert cases['pay_in_installments']['net_obligation_dpd']['dpd']==2
        assert cases['pay_later']['plans']['total_outstanding']==100 and cases['pay_later']['net_obligation_dpd']['dpd']==0
        repeat=cases['repeat_purchases']['plans']
        assert repeat['total_outstanding']==140 and repeat['known_near_term_due']==65
        assert repeat['active_plan_count']==2 and repeat['repeat_purchase_count']==1
        assert cases['effective_return']['plans']['total_outstanding']==0 and cases['effective_return']['net_obligation_dpd']['dpd']==0
        assert cases['completed_short_plan']['plans']['active_plan_count']==0 and cases['completed_short_plan']['plans']['total_outstanding']==0
        assert short[0]['target']==0 and short[0]['observed_until']=='2025-01-21'
        assert annual[0]['target'] is None and annual[0]['observed_until']=='2025-01-22' and annual[0]['label_status']=='exit'
    return dict(data_origin=data['data_origin'],product=data['product'],cases=cases,twenty_day_observation=short,annual_observation=annual)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
