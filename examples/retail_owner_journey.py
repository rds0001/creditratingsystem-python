"""A sole-trader application keeps owner cash transfers and borrower identities explicit."""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import importlib.util
import json
import creditratingsystem as cr


def application_helper():
    spec=importlib.util.spec_from_file_location('owner_application_helper',Path(__file__).with_name('retail_application_journey.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run(data=None,verify_reference=True,return_artifacts=False,applications_path=None,releases_path=None,include_lifecycle=True):
    resource=files('creditratingsystem.datasets')
    if data is None:data=json.loads(resource.joinpath('retail_owner_journey.json').read_text())
    data=deepcopy(data);product='RETAIL_SMALL_BUSINESS_FINANCE';case=data['products'][product];helper=application_helper();business=[]
    for row,h in zip(case['owner_business_inputs'],case['households']):
        graph=cr.build_customer_household_graph(**row['household_graph'])
        obligations=cr.aggregate_household_obligations(**row['household_obligations'],household_graph=graph)
        features=cr.build_retail_sme_features(**row['features'],household_obligations=obligations).value
        groups=cr.assign_default_groups(cr.build_entity_graph(**row['legal_graph']),row['default_policy'],row['facility_mappings']).value
        if features['owner_draw_business_outflows']+row['other_household_income']!=h['monthly_income']:raise ValueError('Owner transfer and separate household income must reconcile to applicant capacity')
        if verify_reference:
            assert features['owner_draw_business_outflows']==features['matched_owner_draw_household_inflows']==row['expected_owner_draw']
            assert features['cash_after_business_debt']==features['cash_after_business_and_household_debt']==row['expected_after_business_debt']
            assert len(groups['groups'])==1
        business.append(dict(application_id=row['application_id'],features=features,household=obligations.value,default_groups=groups))
    chain=helper.run(data,verify_reference=verify_reference,applications_path=applications_path or resource.joinpath('owner_applications.json'),releases_path=releases_path or resource.joinpath('owner_releases.json'),return_artifacts=True)
    presentation=helper.presentation(chain)
    presentation['business_cash_conservation']=[dict(application_id=r['application_id'],business_after_current_debt=r['features']['cash_after_business_debt'],owner_transfer_out=r['features']['owner_draw_business_outflows'],owner_transfer_in=r['features']['matched_owner_draw_household_inflows'],household_debt_service=r['household']['households'][0]['total_budget_debt_service'],legal_default_groups=len(r['default_groups']['groups']))for r in business]
    lifecycle=None
    if include_lifecycle:
        spec=importlib.util.spec_from_file_location('owner_lifecycle_helper',Path(__file__).with_name('retail_owner_lifecycle.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        lifecycle=module.run(case,chain['products'][product],data['evaluation_cutoff'],verify_reference)
        presentation['lifecycle']=dict(account_id=lifecycle['account_id'],application_id=lifecycle['application_id'],proposal_status=case['lifecycle']['proposal_status'],proposal_remaining_claim=lifecycle['proposal']['candidate']['remaining_claim'],actual_default_ead=lifecycle['ead']['conditional_ead'][0],realized_principal_loss=lifecycle['workout']['economic_loss'],predicted_lgd=round(lifecycle['predicted_lgd'],8),portfolio_expected_loss=round(lifecycle['portfolio']['distribution']['expected_loss'],2),paired_brier_difference=round(lifecycle['benchmark']['result']['metrics']['paired_differences']['candidate']['brier']['difference'],8),override_rate=lifecycle['benchmark']['overrides']['metrics']['override_rate'],terminal_state=lifecycle['terminal_state'],replay_status=lifecycle['replay']['decision']['status'],first30day_expected_loss=round(lifecycle['forecast_horizon_loss']['lifetime_expected_loss'],2),prior_event_survival=lifecycle['prior_event_curve']['curves'][0]['survival'],phase_status=dict(ORIGINATION='ACTUAL_APPLICATION_AND_FUNDING',PERFORMING='DATED_ACCOUNT_PREDICTIONS',REVIEW_EARLY_WARNING='OBSERVED_ARREARS_AND_PAIRED_REVIEW',DELINQUENCY='UNPAID100_DPD5',RESTRUCTURING='OFFERED_NOT_ACCEPTED',DEFAULT_WORKOUT='ACTUAL600_PRINCIPAL_RECOVERY',EXIT='DOCUMENTED_FULL_AND_FINAL_SETTLEMENT',PORTFOLIO_MONITORING='ORIGINAL_TWO_LOAN_BOOK_AND_HELDOUT_VALIDATORS'))
    return dict(presentation=presentation,application_chain=chain,business=business,lifecycle=lifecycle,source_case=case,evaluation_cutoff=data['evaluation_cutoff'])if return_artifacts else presentation


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
