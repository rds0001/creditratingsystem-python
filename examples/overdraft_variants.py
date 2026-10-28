"""Observed overdraft cash bridges and explicitly contracted demand dates."""
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('overdraft_variants.json').read_text())
    cases={}
    for name,item in data['cases'].items():
        args=item['transaction_features']
        cash=cr.build_transaction_features(**args).value
        closing=item['opening_signed_balance']+cash['known_external_net_cashflow']
        # These single-account examples contain no internal transfers, FX or
        # noncash postings; the fully captured cash bridge must reconcile.
        snapshots=args['daily_balances']
        if not snapshots or not math.isclose(closing,snapshots[-1]['closing_balance'],abs_tol=1e-10):
            raise ValueError('Supplied closing balance does not reconcile with booked cash')
        balances=[dict(snapshot_id=row['balance_id'],release_seq=row['release_seq'],account_id=row['account_id'],observed_at=row['observed_at'],known_at=row['known_at'],balance=max(-row['closing_balance'],0),credit_limit=item['credit_limit'],activity_status='ACTIVE',currency=row['currency'])for row in snapshots]
        revolving=cr.build_revolving_features(balances,[],args['coverage'],account_id='account',as_of=args['as_of'],information_cutoff=args['information_cutoff'],lookback_days=args['lookback_days'],currency=args['currency']).value
        payments=cr.compute_days_past_due(**item['payment_contract']).value
        cases[name]=dict(cash=cash,revolving=revolving,payments=payments,closing_signed_balance=closing,drawn_exposure=max(-closing,0),contractual_demand_reference=item['contractual_demand_reference'])
    ccf={name:cr.realized_ccf(**args).value for name,args in data['ccf_episodes'].items()}
    if verify_reference:
        assert cases['authorized_line']['drawn_exposure']==900 and cases['authorized_line']['payments']['dpd']==0
        assert cases['unauthorized_excess']['drawn_exposure']==1200
        assert cases['unauthorized_excess']['payments']['past_due_amount']==200 and cases['unauthorized_excess']['payments']['dpd']==1
        assert cases['salary_linked_received']['drawn_exposure']==200 and cases['salary_linked_missing']['drawn_exposure']==1200
        assert cases['salary_linked_received']['cash']['classified_income']==1000 and cases['salary_linked_missing']['cash']['classified_income']==0
        assert cases['salary_linked_missing']['payments']['dpd']==0
        assert ccf['no_available_undrawn']['ccf'] is None and ccf['no_available_undrawn']['denominator_status']=='NO_POSITIVE_UNDRAWN'
        assert math.isclose(ccf['positive_undrawn']['ccf'],4/7,abs_tol=1e-14)
    return dict(data_origin=data['data_origin'],product=data['product'],cases=cases,separate_historical_ccf_episodes=ccf)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
