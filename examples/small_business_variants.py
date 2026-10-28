"""Business/household cash conservation, legal default units and working lines."""
from importlib.resources import files
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('small_business_variants.json').read_text())
    household=cr.aggregate_household_obligations(**data['household_obligations'],household_graph=cr.build_customer_household_graph(**data['household_graph']))
    cases={}
    for name,args in data['cases'].items():
        features=cr.build_retail_sme_features(**args['features'],household_obligations=household).value
        groups=cr.assign_default_groups(cr.build_entity_graph(**args['legal_graph']),data['default_policy'],data['facility_mappings']).value
        cases[name]=dict(features=features,default_groups=groups)
    line=cr.exposure_schedule(**data['working_capital']).value
    guarantee=cr.allocate_guarantee_coverage(**data['guarantee']).value
    if verify_reference:
        assert len(cases['sole_trader']['default_groups']['groups'])==1
        assert len(cases['micro_company']['default_groups']['groups'])==2
        for result in cases.values():
            f=result['features']
            assert f['known_operating_cash_before_capex_and_debt']==130 and f['known_cash_before_business_debt']==110
            assert f['cash_after_business_debt']==80 and f['cash_after_business_and_household_debt']==70
            assert f['owner_draw_business_outflows']==f['matched_owner_draw_household_inflows']==40
        assert line['schedule'][0]['conditional_ead']==40 and line['schedule'][0]['credit_limit']==120 and line['schedule'][0]['undrawn_commitment']==80
        assert [x['covered_amount']for x in guarantee['allocations']]==[100,50]
        assert sum(x['covered_amount']for x in guarantee['allocations'])==150
    return dict(data_origin=data['data_origin'],product=data['product'],household=household.value,cases=cases,subsequent_working_capital_scenario=line,shared_owner_guarantee=guarantee)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
