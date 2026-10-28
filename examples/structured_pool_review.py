"""Forecast collateral cash and independently observed servicing reconcile separately."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('structured_pool_review.json').read_text())
    output={}
    for product,raw in data['cases'].items():
        case=deepcopy(raw);collateral=case['retail_collateral'];model=None
        if collateral is not None:
            model=cr.fit_retail_risk_pools(collateral['training_rows'],collateral['pooling_spec'],target=collateral['target'],training=collateral['training'],model_id=case['design']['model_id'],model_version='1').model
            fitted_pd=model['parameters']['segments'][0]['pools'][0]['statistics']['means']['pd']
            case['path_input']['scenario_contract']['weights']=[fitted_pd,1-fitted_pd]
            for row in case['path_input']['obligor_paths']:
                row['conditional_pd']=fitted_pd if row['date']==case['path_input']['scenario_contract']['dates'][0]else 0
        else:fitted_pd=case['path_input']['scenario_contract']['weights'][0]
        pool=cr.build_credit_asset_pool(**case['pool_input']).value
        forecast_inputs=dict(asset_pool=pool,**case['path_input'])
        paths=cr.simulate_asset_pool(**forecast_inputs).value
        args=dict(asset_pool=pool,pool_inputs=case['pool_input'],forecast_paths=paths,forecast_inputs=forecast_inputs,servicing=case['servicing'],allocations=case['allocations'],pool_accounts=case['pool_accounts'],coverage=case['coverage'],design=case['design'],cash_contract=case['cash_contract'])
        audit=cr.asset_pool_backtest(**args).value
        if collateral is not None:
            asset=case['pool_input']['asset_contracts'][0];origin=case['pool_input']['as_of'];end=case['path_input']['scenario_contract']['dates'][-1]
            assignment_inputs=dict(observations=[collateral['observation']],as_of=origin,information_cutoff=origin,expected_target=collateral['target'],uncovered_policy='error')
            assignments=cr.assign_retail_risk_pool(fitted_model=model,**assignment_inputs).value
            fitted_means=model['parameters']['segments'][0]['pools'][0]['statistics']['means']
            prediction=dict(prediction_id=asset['asset_id'],observation_id=asset['asset_id'],entity_id=asset['asset_id'],cluster_id=asset['default_group_id'],segment=product,as_of=origin,forecast_known_at=origin,horizon_end=end,model_id=case['design']['model_id'],model_version='1',parameter_version='1',excluded_from_training=True,excluded_from_calibration=True,reference_exposure=100,pd=fitted_pd,lgd=fitted_means['lgd'],ead=fitted_means['ead'])
            risk_pool=cr.retail_pool_diagnostics(model,assignments,assignment_inputs,[prediction],collateral['outcomes'],collateral['workouts'],collateral['links'],collateral['estimation_profile'],design=case['design']).value
        else:
            risk_pool=dict(applicability='NOT_APPLICABLE',reason='Commercial invoices use debtor/invoice contracts; retail household pooling does not apply. Asset-pool servicing, allocation and forecast diagnostics are supplied instead.')
        duplicate=deepcopy(args);extra=deepcopy(duplicate['allocations'][0]);extra.update(allocation_id='double-claim',pool_id='another-pool');duplicate['allocations'].append(extra)
        duplicate_audit=cr.asset_pool_backtest(**duplicate).value
        incomplete=deepcopy(args);incomplete['coverage'][0]['complete']=False
        incomplete_audit=cr.asset_pool_backtest(**incomplete).value
        output[product]=dict(pool=pool,forecast_paths=paths,asset_pool_validation=audit,retail_pool_model=model,retail_pool_validation=risk_pool,duplicate_allocation_validation=duplicate_audit,incomplete_coverage_validation=incomplete_audit)
    if verify_reference:
        for product,row in output.items():
            audit=row['asset_pool_validation'];assert not audit['violations']
            expected_brier=.64 if product=='SECURITIZATION_TRANCHE'else .04
            assert math.isclose(audit['metrics']['default_probability']['brier'],expected_brier,abs_tol=1e-12)
            assert all(abs(r['residual'])<1e-12 for r in audit['cash_residuals'])
            assert row['duplicate_allocation_validation']['violation_counts']['CASH_OR_LOSS_DOUBLE_ALLOCATION']>=1
            assert row['incomplete_coverage_validation']['metrics']['default_probability']['brier']is None
        abs_case=output['SECURITIZATION_TRANCHE'];metrics=abs_case['asset_pool_validation']['metrics']
        assert math.isclose(metrics['pool_cash_errors']['recovery_gross']['bias'],-22,abs_tol=1e-12)
        assert metrics['observed_owned_flows']['recovery_gross']==30
        pool=abs_case['retail_pool_validation']['metrics']['pools'][0]
        assert math.isclose(pool['lgd_error']['bias'],-.1,abs_tol=1e-12)and pool['ead_error']['bias']==0
        assert pool['within_pool_discrimination']['auc']is None
        assert output['COVERED_BOND']['asset_pool_validation']['metrics']['prepayment_probability']['observed_rate']==1
        assert output['TRADE_RECEIVABLES']['asset_pool_validation']['metrics']['observed_owned_flows']['dilution']==10
    return dict(products=output,interpretation='Same collateral IDs connect prior frozen forecasts, observed servicing, ownership allocations and separate default-conditioned LGD/EAD diagnostics. A single realized pool supplies no confidence distribution or empirical homogeneity proof; contractual bond/tranche priority is tested in the product examples, not inferred from cash reconciliation.')


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
