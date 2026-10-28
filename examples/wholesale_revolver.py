"""Undrawn corporate commitments: estimated conditional EAD and legal cancellation."""
from importlib.resources import files
from copy import deepcopy
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_revolver.json').read_text())
    t=data['target'];tr=data['training'];asof=data['as_of'];known=data['known_at']
    model=cr.fit_binary_glm(tr['X'],tr['y'],weights=tr['weights'],link='logit',feature_order=tr['feature_order'],fit_intercept=True,target=t,training=tr['metadata'],model_id='wholesale-revolver',model_version='1')
    calibrator=cr.fit_hierarchical_calibration(**data['calibration'])
    prior=cr.fit_elicited_beta_prior(**data['elicitation']).value
    rubric=cr.compile_qualitative_rubric(**data['rubric'])
    evidence=cr.evaluate_qualitative_evidence(rubric,data['observations'],data['evidence'],as_of=asof,known_at=known).value
    missing=cr.evaluate_qualitative_evidence(rubric,data['observations'],[dict(e,status='UNVERIFIED')for e in data['evidence']],as_of=asof,known_at=known).value
    lgdfit=cr.fit_fractional_lgd([[],[],[],[]],[.1,.3,.5,.7],feature_order=[],link='logit',covariance='HC0',target=data['lgd_target'],training=data['lgd_training'],model_id='revolver-lgd',model_version='1')
    lgd=cr.predict_lgd(lgdfit,[[]],feature_order=[],expected_target=data['lgd_target'],default_condition='given_default').value['mean_lgd'][0]
    eadfit=cr.fit_ead_model(data['exposure_episodes'],[[],[],[]],feature_order=[],target_mode='ccf',model_spec=dict(mean_model='identity',covariance='HC0'),target=data['ead_target'],training=data['ead_training'],model_id='revolver-ead',model_version='1',currency='EUR')
    policy=cr.compile_credit_decision_policy(data['decision_policy'])
    variants={}
    for v in data['variants']:
        identity=v['id'];facts=dict(product=identity,legal_cancellation=v['cancellable'],financial_statements=True,active=True)
        dated={k:dict(value=x,observed_at=asof,known_at=known)for k,x in facts.items()}
        subject=dict(subject_id=identity,version='1',attributes=dated,effective_from=asof,effective_to=None,known_at=known)
        subjects=cr.build_rating_subjects({'facility':[subject]},[],asof,known).value
        snap=dict(subject_id=identity,subject_type='facility',as_of=asof,known_at=known,attributes=dated)
        rule=dict(rule_id=identity,profile_id='wholesale-'+identity,profile_version='1',priority=1,effective_from=asof,effective_to=None,known_at=known,conditions=[dict(field='product',operator='eq',value=identity),dict(field='legal_cancellation',operator='eq',value=v['cancellable'])])
        route=cr.route_rating_profile(snap,[rule],asof,known).value
        applicability_contract=dict(contract_id=identity,version='1',required_features=['financial_statements','active','legal_cancellation'],allowed_values={'financial_statements':[True],'active':[True]},support_bounds={},maximum_age_days={})
        applicability=cr.assess_rating_applicability(snap,applicability_contract,asof,known).value
        bad=deepcopy(snap);bad['attributes']['financial_statements']['value']=None
        negative=cr.assess_rating_applicability(bad,applicability_contract,asof,known).value
        statement=dict(entity_id=identity,currency='EUR',as_of='2023-12-31',period_start='2023-01-01',period_end='2023-12-31',values=dict(debt=50,cash=10,ebitda=20),units=dict(debt='EUR',cash='EUR',ebitda='EUR'))
        leverage=cr.build_financial_ratios([statement],data['ratio_spec']).value[0]['ratios']['net_debt_ebitda']['value']
        prediction=cr.predict_pd(model,[[leverage]],feature_order=tr['feature_order'],expected_target=t)
        pd=cr.apply_calibrator(calibrator,prediction.value,groups=[v['group']],target=t,base_model_version='1').value[0]
        exposure=cr.predict_ead(eadfit,[[]],feature_order=[],expected_target=data['ead_target'],reference_dates=[asof],default_dates=['2024-01-31'],reference_drawn=[0],reference_limit=[v['limit']],currency='EUR').value['conditional_ead'][0]
        loss=cr.expected_loss(pd,lgd,exposure,context={'currency':'EUR'}).value
        contract=dict(facility_id=identity,currency='EUR',as_of=asof,drawn_principal=0,credit_limit=v['limit'],accrued_interest=0)
        event=dict(event_id='conditional-draw',event_date='2024-01-16',known_at=known,sequence=0,kind='DRAW',amount=v['planned_draw'])
        opts=dict(annual_interest_rates=[0],period_end_dates=['2024-01-31'],day_count='ACT/365F',default_timing='AFTER_PERIOD_END_EVENTS',allow_overlimit=False,currency='EUR')
        schedule=cr.exposure_schedule(contract,[event],**opts).value
        cancellation=None
        if v['cancellable']:
            cancel=dict(event_id='executed-cancellation',event_date='2024-01-10',known_at=known,sequence=0,kind='LIMIT_CHANGE',amount=0)
            cancellation=cr.exposure_schedule(contract,[cancel],**opts).value
        components=[dict(component_id=q,quantity=q,value=value,subject_id=identity,subject_type='facility',variant=variant,definition=definition,**({'currency':'EUR'}if q in ['EAD','EL']else{}))for q,value,variant,definition in [('PD',pd,'unconditional-default',t),('LGD',lgd,'conditional-recovery',data['lgd_target']),('EAD',exposure,'conditional-exposure',data['ead_target']),('EL',loss,'unconditional-loss',dict(t,event='expected_credit_loss'))]]
        binding=dict(profile_id=route['profile_id'],profile_version='1',model_id='wholesale-revolver',model_version='1',model_hash=prediction.diagnostics['model_hash'],input_hash=prediction.diagnostics['prediction_input_hash'])
        assessment=cr.compose_risk_assessment('assessment-'+identity,dict(subject_id=identity,subject_type='facility'),components,binding,asof,known,known).value
        timing=dict(as_of=asof,known_at=known)
        args=dict(risk=dict(timing,pd=pd,target=t,freshness='CURRENT'),affordability=dict(timing,status='AFFORDABLE'if data['credit_gate_inputs']['available_cashflow']>=data['credit_gate_inputs']['debt_service_due']else'NOT_AFFORDABLE'),eligibility=dict(timing,status='ELIGIBLE'if evidence['status']=='READY'else'INELIGIBLE'),operations=dict(timing,status=data['credit_gate_inputs']['operations_status']),as_of=asof,known_at=known)
        decision=cr.apply_credit_decision_policy(policy,**args).value
        args['eligibility']['status']='INELIGIBLE'
        blocked=cr.apply_credit_decision_policy(policy,**args).value
        variants[identity]=dict(profile_id=route['profile_id'],applicability=applicability['status'],negative_status=negative['status'],subject_count=len(subjects['subjects']),reference_drawn=0,reference_limit=v['limit'],pd=pd,lgd=lgd,conditional_ead=exposure,expected_loss=loss,conditional_draw_schedule=schedule,contractual_cancellation=cancellation,assessment_components=assessment['components'],assessment_conflicts=assessment['conflicts'],decision=decision['action'],missing_evidence_decision=blocked['action'])
        if verify_reference:
            assert math.isclose(pd,.1,abs_tol=1e-6) and math.isclose(lgd,.4,abs_tol=1e-10)
            assert exposure==v['limit']*.5 and math.isclose(loss,.02*v['limit'],abs_tol=1e-5)
            assert schedule['schedule'][0]['conditional_ead']==v['planned_draw']
            assert negative['status']=='REFER' and decision['action']=='APPROVE' and blocked['action']!='APPROVE'
            assert not assessment['conflicts']
    if verify_reference:
        assert prior['alpha']==2 and prior['beta']==18 and evidence['score']==20 and missing['status']=='REFER'
        assert variants['cancellable']['contractual_cancellation']['schedule'][0]['conditional_ead']==0
    return dict(product=data['product'],data_origin='SYNTHETIC',target=t,credit_gate_inputs=data['credit_gate_inputs'],expert_prior=dict(alpha=prior['alpha'],beta=prior['beta']),qualitative_score=evidence['score'],missing_evidence_status=missing['status'],hierarchical_variance=calibrator.value['parameters']['variance'],variants=variants)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
