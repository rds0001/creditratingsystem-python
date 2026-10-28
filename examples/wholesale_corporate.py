"""Synthetic corporate fundamentals, market inversion and issue recovery variants."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_corporate.json').read_text())
    target=data['target'];training=data['score_training']
    ratios=cr.build_financial_ratios(data['statements'],data['ratio_spec']).value
    score=cr.fit_scorecard(training['X'],training['y'],feature_order=training['feature_order'],fit_spec=training['fit_spec'],target=target,training=training['training'],model_id='corporate-fundamental',model_version='1',weights=training['weights'])
    design=[[row['ratios']['net_debt_ebitda']['value']]for row in ratios]
    fundamental=cr.score_scorecard(score,design,feature_order=training['feature_order'],expected_target=target).to_dict()['value']
    calibration=data['calibration'];calibrator=cr.calibrate_logistic(calibration['values'],calibration['outcomes'],calibration['weights'],metadata=calibration['metadata'])
    calibrated=cr.apply_calibrator(calibrator,fundamental['pd'],target=target,base_model_version='1').to_dict()['value']
    routes={};applicability={};negative={}
    for segment in ['SME','large_corporate']:
        facts=dict(size=segment,financial_statements=True,audited=segment=='large_corporate',listed=segment=='large_corporate')
        snapshot=dict(subject_id='SME'if segment=='SME'else'LARGE',subject_type='obligor',as_of=data['as_of'],known_at=data['known_at'],attributes={k:dict(value=v,observed_at='2023-12-31',known_at='2024-01-01')for k,v in facts.items()})
        routes[segment]=cr.route_rating_profile(snapshot,data['routing_rules'],data['as_of'],data['known_at']).value
        applicability[segment]=cr.assess_rating_applicability(snapshot,data['applicability_contracts'][segment],data['as_of'],data['known_at']).value
        bad=deepcopy(snapshot);bad['attributes']['financial_statements'if segment=='SME'else'audited']['value']=None
        negative[segment]=cr.assess_rating_applicability(bad,data['applicability_contracts'][segment],data['as_of'],data['known_at']).value['status']
    inversion=cr.infer_merton_assets(**data['market_inputs']).value
    a=inversion['asset_value'];sigma=inversion['asset_volatility'];barrier=data['market_inputs']['debt_face'];mu=data['physical_drift']
    distance=cr.distance_to_default(a,barrier,sigma,mu,1).value
    terminal=cr.merton_default_probability(a,barrier,sigma,mu,1,measure='PHYSICAL').value
    q=cr.merton_default_probability(a,barrier,sigma,data['market_inputs']['risk_free_rate'],1,measure='RISK_NEUTRAL').value
    passage=cr.first_passage_pd(a,dict(kind='constant',initial_level=barrier),sigma,.5*sigma*sigma,1,measure='PHYSICAL').value
    mapping=cr.calibrate_distance_pd_mapping(**data['distance_calibration'],evaluation_distances=[distance['distance_to_default'],1.]).to_dict()['value']
    signals=[dict(signal_id=name,model_id=name,model_version='1',row_ids=['LARGE'],pd=[p],target=target,as_of=data['as_of'],known_at=data['known_at'])for name,p in [('fundamental',calibrated[1]),('observed_distance',mapping['pd'][0])]]
    pooled={}
    for scale in ['linear','log_odds','mixture']:
        contract=dict(interpretation={'linear':'estimator_pool','log_odds':'log_odds_pool','mixture':'latent_regime_mixture'}[scale],reference='Synthetic declared signal pooling; no TTC assertion')
        if scale=='mixture':contract.update(regime_ids=['fundamental','observed_distance'],conditional_regime_probabilities=True)
        pooled[scale]=cr.combine_pd_signals(signals,[[.25,.75]]if scale=='mixture'else[.7,.3],combination_scale=scale,combination_contract=contract).to_dict()['value']
    standalone=pooled['linear']['pd'][0]
    risk=dict(entity_id='LARGE',pd=standalone,stage='STANDALONE',target=target,applied_effects=[])
    supporter=dict(entity_id='SPONSOR',pd=.02,stage='STANDALONE',target=target,applied_effects=[])
    supported=cr.apply_parent_support(risk,supporter,data['support'],joint_default_probability=.01).value
    ranked=cr.simulate_recovery_waterfall(**data['recovery']).value
    secured=cr.simulate_recovery_waterfall(**data['secured_recovery']).value
    variants={seg:dict(evidence_id='corporate-'+seg,profile_id=routes[seg]['profile_id'],standalone_pd=calibrated[0]if seg=='SME'else standalone,market_component='NOT_APPLICABLE_UNLISTED'if seg=='SME'else'EMPIRICAL_DISTANCE_MAPPING',negative_status=negative[seg])for seg in ['SME','large_corporate']}
    for name,recovery in [('senior',ranked['claim_results'][0]),('subordinated',ranked['claim_results'][1]),('secured',secured['claim_results'][0])]:
        issue_snapshot=dict(subject_id=name,subject_type='issue',as_of=data['as_of'],known_at=data['known_at'],attributes={'rank_security':dict(value=name,observed_at=data['as_of'],known_at=data['known_at']),'legal_verified':dict(value=True,observed_at=data['as_of'],known_at=data['known_at'])})
        issue_rule=dict(rule_id=name,profile_id='issue-'+name,profile_version='1',priority=1,effective_from='2023-01-01',effective_to=None,known_at='2022-12-01',conditions=[dict(field='rank_security',operator='eq',value=name),dict(field='legal_verified',operator='eq',value=True)])
        issue_route=cr.route_rating_profile(issue_snapshot,[issue_rule],data['as_of'],data['known_at']).value
        issue_snapshot['attributes']['legal_verified']['value']=False
        issue_negative=cr.route_rating_profile(issue_snapshot,[issue_rule],data['as_of'],data['known_at']).value
        variants[name]=dict(profile_id=issue_route['profile_id'],negative_status=issue_negative['status'],evidence_id='corporate-'+name,standalone_pd=standalone,lgd=recovery['lgd'],ead=recovery['ead_at_default'],economic_loss_at_default=recovery['economic_loss'],expected_loss=cr.expected_loss(standalone,recovery['lgd'],recovery['ead_at_default'],context={'currency':'EUR'}).value)
    # Raw financial missingness stays visible; it is not zero leverage.
    missing=deepcopy(data['statements']);missing[1]['values']['ebitda']=None
    missing_ratio=cr.build_financial_ratios(missing,data['ratio_spec']).value[1]['ratios']['net_debt_ebitda']
    if verify_reference:
        close=lambda a,b,tol=1e-7: math.isclose(a,b,rel_tol=tol,abs_tol=tol)
        assert [r['ratios']['net_debt_ebitda']['value']for r in ratios]==[3,3]
        expected=1/(1+math.exp(-.5*(math.log(.1/.9)+math.log(.3/.7))))
        assert all(close(x,expected)for x in fundamental['pd']) and all(close(x,expected)for x in calibrated)
        assert close(a,100)and close(sigma,.2)
        assert close(terminal,.5*math.erfc(distance['distance_to_default']/math.sqrt(2)))and terminal<q
        assert close(passage,math.erfc(math.log(100/80)/(.2*math.sqrt(2))))
        assert close(mapping['pd'][1],.2)
        p1,p2=signals[0]['pd'][0],signals[1]['pd'][0]
        assert close(pooled['linear']['pd'][0],.7*p1+.3*p2)
        assert close(pooled['mixture']['pd'][0],.25*p1+.75*p2)
        assert close(pooled['log_odds']['pd'][0],1/(1+math.exp(-(.7*math.log(p1/(1-p1))+.3*math.log(p2/(1-p2))))))
        assert close(supported['pd'],standalone-(standalone-.01)*.4)
        assert [variants[k]['lgd']for k in ['senior','subordinated','secured']]==[0,.5,.125]
        assert all(variants[k]['standalone_pd']==standalone for k in ['senior','subordinated','secured'])
        assert missing_ratio['value']is None and missing_ratio['status']=='MISSING_INPUT'
        assert all(x=='REFER'for x in negative.values())
    return dict(product=data['product'],data_origin='SYNTHETIC',version=data['version'],target=target,ratios=ratios,score_parameters=score.model['parameters'],calibration_parameters=calibrator.value['parameters'],fundamental=fundamental,calibrated=calibrated,routing=routes,applicability=applicability,inversion=dict(asset_value=a,asset_volatility=sigma),distance=distance,terminal_physical_pd=terminal,terminal_risk_neutral_pd=q,first_passage_pd=passage,empirical_distance_pd=mapping['pd'],pooled=pooled,supported=supported,grade=cr.map_pd_to_grade(supported['pd'],data['rating_scale']).value,ranked_recovery=ranked,secured_recovery=secured,variants=variants,missing_ratio=missing_ratio)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
