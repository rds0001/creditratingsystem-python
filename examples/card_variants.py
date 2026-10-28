"""Credit-card minimum dues and charge-card full/deferred payment contracts."""
from importlib.resources import files
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('card_variants.json').read_text())
    revolving={name:cr.build_revolving_features(**args).value for name,args in data['revolving'].items()}
    payments={name:cr.compute_days_past_due(**args).value for name,args in data['payments'].items()}
    exposure={name:cr.exposure_schedule(**args).value for name,args in data['exposure'].items()}
    history=cr.build_payment_history_features(**data['new_history']).value
    authorizations={name:cr.reconcile_card_authorizations(**args).value for name,args in data['authorizations'].items()}
    if verify_reference:
        assert payments['transactor']['dpd']==payments['revolver']['dpd']==0
        assert revolving['transactor']['full_payer_count']==1
        assert revolving['revolver']['minimum_without_full_count']==1
        assert payments['monthly_full_payment']['unpaid_due_amount']==950
        assert payments['monthly_full_payment']['dpd']==2
        assert payments['deferred_charge']['dpd']==0 and payments['deferred_charge']['unpaid_due_amount']==0
        assert history['features']['max_dpd'] is None
        assert revolving['new_account']['latest_snapshot']['undrawn_active']
        assert revolving['new_account']['missing_days']==3
        cut=revolving['limit_change']['balance_changes'][0]
        assert cut['balance_change']==0 and cut['limit_change']==-1000
        assert cut['balance_effect_at_new_limit']==0 and cut['limit_effect_on_previous_balance']==.5
        for name,expected in [('transactor',0),('revolver',950),('new_account',200),('monthly_full_payment',950),('deferred_charge',1000)]:
            assert exposure[name]['schedule'][0]['conditional_ead']==expected
        assert authorizations['pending']['accounts'][0]['scenario_exposure']==200
        assert authorizations['settled']['accounts'][0]['scenario_exposure']==200
        assert authorizations['settled']['accounts'][0]['pending_authorization']==0
        assert authorizations['expired']['accounts'][0]['scenario_exposure']==0
    return dict(data_origin=data['data_origin'],products=data['products'],revolving=revolving,payments=payments,exposure=exposure,new_account_history=history,authorizations=authorizations)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
