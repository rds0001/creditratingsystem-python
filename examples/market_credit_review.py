"""Local quote inversion, identification limits and an observed Q-to-P bridge."""
from copy import deepcopy
from importlib.resources import files
import math,json
import creditratingsystem as cr


def run(data=None,verify_reference=True,return_artifacts=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('market_credit_review.json').read_text())
    snapshot=cr.build_credit_quote_snapshot(data['quotes'],data['instruments'],data['discount'],data['snapshot_contract']).value
    cds_rows=[q for q in snapshot['quotes']if q['eligible']and q['quote_type']=='CDS_SPREAD']
    bond_rows=[dict(q,price_standard_error=data['price_standard_error'])for q in snapshot['quotes']if q['eligible']and q['quote_type']=='BOND_CLEAN_PRICE']
    cf=cr.bootstrap_cds_hazard_curve(cds_rows,data['discount'],data['cds_calibration']).value
    bf=cr.calibrate_bond_hazard_curve(bond_rows,data['discount'],data['bond_calibration']).value
    priced={};reference={i['instrument_id']:i for i in data['instruments']}
    for key,ins in reference.items():
        fn=cr.cds_expected_cashflow_legs if ins['instrument_type']=='CDS' else cr.risky_bond_expected_cashflows
        priced[key]=fn(cf['hazard_curve'] if ins['instrument_type']=='CDS' else bf['hazard_curve'],data['discount'],ins).value
    alternatives=[dict(scenario_id='recovery20',recovery=.2),dict(scenario_id='recovery60',recovery=.6)]
    ident_contract=dict(equivalence_tolerance=1e-8,quote_standard_errors=[.0001]*len(cds_rows),recovery_bounds=[0,.9])
    unidentified=cr.diagnose_market_credit_identifiability(cf,alternatives,ident_contract).value
    identified=cr.diagnose_market_credit_identifiability(cf,alternatives,dict(ident_contract,independent_recovery_observation=dict(value=.4,standard_error=.01,evidence_id='separate-synthetic-recovery-study',known_at='2025-01-01'))).value
    basis=cr.compare_cds_bond_credit_basis(cf,bf,dict(recovery=.4,bond_liquidity_spread=0.,funding_treatment='NO_ADDITIONAL_FUNDING_WEDGE',maximum_quote_age_days=5,comparison_cds_contract=reference['cds-3'])).value
    fitted=cr.fit_market_to_physical_pd_bridge(data['bridge_observations'],['q_pd'],data['bridge_contract'],data['folds'],model_id='observed-market-bridge',model_version='1')
    q=-math.expm1(-cf['hazard_curve']['rates'][0])
    physical=float(cr.predict_pd(fitted,[[q]],feature_order=['q_pd'],expected_target=data['bridge_contract']['target']).value[0])
    subject=dict(subject_id='REFERENCE-ISSUER',subject_type='obligor')
    q_target=dict(data['bridge_contract']['target'],probability_measure='RISK_NEUTRAL')
    components=[dict(component_id='market-Q',quantity='PD',value=q,**subject,variant='market-pricing',definition=q_target),dict(component_id='historical-P',quantity='PD',value=physical,**subject,variant='observed-issuer-bridge',definition=data['bridge_contract']['target'])]
    binding=dict(profile_id='market-review',profile_version='1',model_id=fitted.model['model_id'],model_version='1',model_hash=cr.content_fingerprint(fitted.model).value,input_hash=cr.content_fingerprint(dict(snapshot=snapshot['snapshot_hash'],q_pd=q)).value)
    assessment=cr.compose_risk_assessment('market-review',subject,components,binding,'2025-01-01','2025-01-01','2025-01-01').value
    coeff=fitted.model['parameters']['coefficients']
    # The ordinary report exposes financial quantities. Full backend-specific
    # artifacts and their local content seals remain available on explicit return.
    output=dict(snapshot=snapshot,pricing=priced,cds_hazards=cf['hazard_curve']['rates'],bond_hazards=bf['hazard_curve']['rates'],bond_objective=bf['objective'],identification=dict(rank_without_recovery=unidentified['joint_identifiability']['rank'],rank_with_recovery=identified['joint_identifiability']['rank'],equivalent_without_recovery=unidentified['equivalent_scenario_ids'],equivalent_with_recovery=identified['equivalent_scenario_ids'],alternative_hazards=[r['hazards']for r in unidentified['profiles']]),basis=dict(comparable=basis['comparable'],basis=basis['basis']),bridge=dict(coefficients=coeff,physical_pd=physical,q_pd=q,observed_evaluation_count=fitted.diagnostics['observed_evaluation_count'],out_of_time_brier=fitted.diagnostics['out_of_time_brier'],warmup_count=len(fitted.diagnostics['warmup_indices'])),assessment_components=assessment['components'],interpretation='Q quote-implied probabilities and historically estimated P issuer probabilities retain distinct measures; the bridge is a selected-population conditional association, not a universal measure conversion. Financial report parity does not imply bitwise-identical fresh optimization artifacts.')
    if verify_reference:
        assert all(math.isclose(v,.02,abs_tol=1e-9)for v in cf['hazard_curve']['rates']+bf['hazard_curve']['rates'])
        assert not next(r for r in snapshot['quotes']if r['quote_id']=='future-spread')['eligible']
        for t in [1,3]:
            expected=105*math.exp(-.05*t)+.8*(-math.expm1(-.05*t))/.05
            assert math.isclose(priced['bond-'+str(t)]['clean_price_per_100'],expected,abs_tol=1e-8)
            assert math.isclose(priced['cds-'+str(t)]['protection_pv'],.012*(-math.expm1(-.05*t))/.05,abs_tol=1e-10)
        assert unidentified['joint_identifiability']['rank']==2 and identified['joint_identifiability']['full_column_rank']
        assert len(unidentified['equivalent_scenario_ids'])==2 and not identified['equivalent_scenario_ids']
        assert basis['comparable']and abs(basis['basis'])<1e-9
        b=(math.log(.1/.9)-math.log(.02/.98))/.04;a=math.log(.02/.98)-.01*b
        assert math.isclose(coeff[0],a,abs_tol=1e-8)and math.isclose(coeff[1],b,abs_tol=1e-7)
        assert math.isclose(physical,1/(1+math.exp(-(a+b*q))),abs_tol=1e-9)
        assert math.isclose(fitted.diagnostics['out_of_time_brier'],(.02*.98+.1*.9)/2,abs_tol=1e-12)
        assert fitted.diagnostics['observed_evaluation_count']==200
    if return_artifacts:output['artifacts']=dict(cds=cf,bond=bf,bridge=fitted.model,assessment=assessment)
    return output


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
