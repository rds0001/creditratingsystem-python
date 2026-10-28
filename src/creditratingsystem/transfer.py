"""Explicit transaction-level transfer events and exclusive default states."""
from copy import deepcopy
import math
from .core import CalcResult, _numeric, _probability, _fail, _result, _fingerprint, _context_value
from .data import _rows, _required, _unique, _date
from ._loss_paths import event_conditioned_loss


def _object(value,name):
    value=value.value if isinstance(value,CalcResult) else value
    if not isinstance(value,dict):_fail('TRANSFER_CONTRACT',name+' must be a declared object')
    return value


def _string(value,name):
    if not isinstance(value,str) or not value:_fail('TRANSFER_CONTRACT',name+' must be a nonempty string')
    return value


def _names(value,name,empty=False):
    if not isinstance(value,list) or (not empty and not value) or any(not isinstance(x,str) or not x for x in value) or len(set(value))!=len(value):
        _fail('TRANSFER_CONTRACT',name+' must contain distinct named values')
    return value


def _p(value,name):
    value=_probability(value,name=name)
    if value.ndim:_fail('TRANSFER_PROBABILITY',name+' must be scalar')
    return float(value)


def _target(target,context):
    target=_object(target,'target')
    _required(target,['exposure_id','horizon','time_unit','probability_measure','conditioning'])
    for field in ('exposure_id','conditioning'):_string(target[field],field)
    h=_numeric(target['horizon'],'horizon')
    if h.ndim or h<=0 or target['time_unit'] not in ('days','months','years') or target['probability_measure'] not in ('PHYSICAL','RISK_NEUTRAL'):
        _fail('TRANSFER_TARGET','Explicit positive horizon, time unit and probability measure required')
    ctx=_context_value(context)
    for field in ('horizon','time_unit','probability_measure','conditioning'):
        if ctx.get(field) is not None and ctx[field]!=target[field]:_fail('TARGET_MISMATCH','Context differs from transfer target')
    return deepcopy(target),ctx


def assess_transfer_applicability(payments,event_policy,*,as_of,known_at,horizon_end,context=None):
    """Apply dated route/currency rules without inferring law from nationality."""
    start=_date(as_of,'as_of');known=_date(known_at,'known_at');end=_date(horizon_end,'horizon_end')
    if known<start or end<start:_fail('TRANSFER_DATE','Require as_of<=known_at and as_of<=horizon_end')
    rows=_rows(payments,'payments');_unique(rows,'payment_id','payments');policy=_object(event_policy,'event_policy')
    _required(policy,['policy_id','version','events']);_string(policy['policy_id'],'policy_id');_string(policy['version'],'version')
    events=_rows(policy['events'],'events');_unique(events,'event_id','events');findings=[]
    for event in events:
        _required(event,['event_type','jurisdictions','currencies','effective_from','known_at','exemptions'])
        if event['event_type'] not in ('CONVERSION_RESTRICTION','PAYMENT_FREEZE','BORDER_TRANSFER_BLOCK'):_fail('TRANSFER_EVENT','Unsupported explicit event type')
        _names(event['jurisdictions'],'jurisdictions');_names(event['currencies'],'currencies')
        begin=_date(event['effective_from'],'effective_from');_date(event['known_at'],'known_at')
        finish=None if event.get('effective_until') is None else _date(event['effective_until'],'effective_until')
        if finish is not None and finish<=begin:_fail('TRANSFER_DATE','Empty event applicability interval')
        exemptions=_rows(event['exemptions'],'exemptions');_unique(exemptions,'payment_id','exemptions')
        for exemption in exemptions:
            _required(exemption,['reason','evidence_reference','known_at']);_string(exemption['reason'],'reason');_string(exemption['evidence_reference'],'evidence_reference');_date(exemption['known_at'],'known_at')
    for payment in rows:
        _required(payment,['due_at','known_at']);due=_date(payment['due_at'],'due_at');pk=_date(payment['known_at'],'known_at')
        if pk>known or not start<=due<=end:
            findings.append(dict(payment_id=payment['payment_id'],status='OUTSIDE_KNOWN_WINDOW',events=[]));continue
        fields=('currency','source_currency','payer_jurisdiction','payee_jurisdiction','payment_jurisdictions')
        missing=[f for f in fields if payment.get(f) is None]
        if missing:findings.append(dict(payment_id=payment['payment_id'],status='INSUFFICIENT_INFORMATION',missing=missing,events=[]));continue
        for field in fields[:-1]:_string(payment[field],field)
        routes=_names(payment['payment_jurisdictions'],'payment_jurisdictions')
        jurisdictions=set(routes+[payment['payer_jurisdiction'],payment['payee_jurisdiction']]);applicable=[]
        for event in events:
            begin=_date(event['effective_from'],'effective_from');finish=None if event.get('effective_until') is None else _date(event['effective_until'],'effective_until')
            if _date(event['known_at'],'known_at')>known or due<begin or (finish is not None and due>=finish):continue
            if payment['currency'] not in event['currencies'] or not jurisdictions.intersection(event['jurisdictions']):continue
            if event['event_type']=='CONVERSION_RESTRICTION' and payment['currency']==payment['source_currency']:continue
            if event['event_type']=='BORDER_TRANSFER_BLOCK' and len(jurisdictions)==1:continue
            exemptions=[x for x in event['exemptions'] if x['payment_id']==payment['payment_id'] and _date(x['known_at'],'known_at')<=known]
            applicable.append(dict(event_id=event['event_id'],event_type=event['event_type'],status='EXEMPT' if exemptions else 'APPLICABLE',
                exemption=deepcopy(exemptions[0]) if exemptions else None))
        findings.append(dict(payment_id=payment['payment_id'],status='APPLICABLE' if any(x['status']=='APPLICABLE' for x in applicable) else 'EXEMPT' if applicable else 'NO_MATCHING_EVENT',events=applicable))
    return _result(dict(policy_id=policy['policy_id'],policy_version=policy['version'],payments=findings),
        'transfer.assess_transfer_applicability','applicability',context,
        diagnostics=dict(as_of=as_of,known_at=known_at,horizon_end=horizon_end,policy_hash=_fingerprint(policy)),
        limitations=('Rules are supplied economic/legal applicability assumptions, not legal determinations. Equal currencies do not exempt a payment freeze.',
                    'Event effective intervals describe policy applicability, not observed transfer-event occurrence or probability.'))


def transfer_event_default_probability(transfer_event_probability,event_to_default_probability,mitigation_contract,
        *,event_id,target,context=None):
    """Map an event probability through explicitly conditional failure/mitigation."""
    target,ctx=_target(target,context);event_id=_string(event_id,'event_id')
    p=_p(transfer_event_probability,'transfer_event_probability');failure=_p(event_to_default_probability,'event_to_default_probability')
    mitigation=_object(mitigation_contract,'mitigation_contract')
    _required(mitigation,['effect_id','success_probability_given_event_and_unmitigated_failure','conditioning','assumption_reference'])
    _string(mitigation['effect_id'],'effect_id');_string(mitigation['assumption_reference'],'assumption_reference')
    if mitigation['conditioning']!='event_and_unmitigated_failure':_fail('TRANSFER_CONDITIONING','Mitigation probability must condition on the event and would-be failure')
    success=_p(mitigation['success_probability_given_event_and_unmitigated_failure'],'mitigation_success')
    gross=p*failure;avoided=gross*success;probability=gross*(1-success)
    return _result(dict(stage='TRANSFER_DEFAULT',probability=probability,target=target,event_id=event_id,
        event_probability=p,unmitigated_default_probability=gross,mitigated_default_mass=avoided,
        applied_effects=[mitigation['effect_id']],assumption_reference=mitigation['assumption_reference']),
        'transfer.transfer_event_default_probability','probability',ctx,
        limitations=('Event-to-default failure and mitigation are declared conditional probabilities. Country ratings or arbitrary liquidity ratios are not silently converted into them.',))


def combine_default_transfer(pre_transfer_risk,transfer_risk,*,dependence,joint_probability=None,context=None):
    """Form a valid event union with explicit Fréchet bounds and stage checks."""
    base=_object(pre_transfer_risk,'pre_transfer_risk');transfer=_object(transfer_risk,'transfer_risk')
    _required(base,['stage','pd','target','applied_effects']);_required(transfer,['stage','probability','target','event_id','applied_effects'])
    if base['stage']!='PRE_TRANSFER' or transfer['stage']!='TRANSFER_DEFAULT':_fail('TRANSFER_DOUBLE_COUNT','Require unadjusted base PD and separate transfer-default risk')
    target,ctx=_target(base['target'],context)
    if _fingerprint(target)!=_fingerprint(transfer['target']):_fail('TARGET_MISMATCH','Exposure, horizon and conditioning must agree')
    base_effects=_names(base['applied_effects'],'base applied_effects',True);transfer_effects=_names(transfer['applied_effects'],'transfer applied_effects',True)
    if set(base_effects)&set(transfer_effects):_fail('TRANSFER_DOUBLE_COUNT','An effect cannot be applied at both risk levels')
    p=_p(base['pd'],'pre_transfer_pd');q=_p(transfer['probability'],'transfer_default_probability')
    lower=max(0.,math.fsum((p,q,-1.)));upper=min(p,q)
    if dependence=='explicit':
        if joint_probability is None:_fail('TRANSFER_DEPENDENCE','Explicit intersection probability required')
        joint=_p(joint_probability,'joint_probability')
    elif dependence=='independent':
        if joint_probability is not None:_fail('TRANSFER_DEPENDENCE','Do not supply an intersection with an independence assumption')
        joint=p*q
    else:_fail('TRANSFER_DEPENDENCE','Declare explicit or independent dependence')
    if joint<lower or joint>upper:_fail('INVALID_JOINT_PROBABILITY','Joint probability outside Fréchet bounds')
    neither=math.fsum((1.,-p,-q,joint))
    residual=0.
    if neither<0 and neither>=-8*math.ulp(1.):residual=neither;neither=0.
    if neither<0:_fail('NUMERICAL_FAILURE','Exclusive-state mass is negative')
    states=dict(neither=neither,default_only=p-joint,transfer_only=q-joint,joint=joint)
    return _result(dict(stage='TRANSFER_ADJUSTED',combined_pd=min(1.,math.fsum((p,q,-joint))),target=target,
        pre_transfer_pd=p,transfer_default_probability=q,joint_probability=joint,exclusive_probabilities=states,
        event_id=transfer['event_id'],applied_effects=base_effects+transfer_effects),
        'transfer.combine_default_transfer','probability',ctx,
        diagnostics=dict(dependence=dependence,frechet_lower=lower,frechet_upper=upper,neither_roundoff_adjustment=residual),
        limitations=('Independence is an explicit assumption, not an estimate. An already transfer-adjusted PD is rejected; the union is not a rating-notch adjustment.',))


def transfer_loss_components(combined_risk,conditional_losses,*,currency,applied_loss_effects,context=None):
    """Integrate separately specified monetary losses for all four event states."""
    combined=_object(combined_risk,'combined_risk');losses=_object(conditional_losses,'conditional_losses')
    _required(combined,['stage','exclusive_probabilities','target','applied_effects'])
    if combined['stage']!='TRANSFER_ADJUSTED':_fail('TRANSFER_CONTRACT','Combined event-union result required')
    target,ctx=_target(combined['target'],context);effects=_names(applied_loss_effects,'applied_loss_effects',True)
    previous=_names(combined['applied_effects'],'applied_effects',True)
    if set(effects)&set(previous):_fail('TRANSFER_DOUBLE_COUNT','Risk-level mitigation must not be applied again to state losses')
    names=['neither','transfer_only','default_only','joint']
    if set(losses)!=set(names) or set(combined['exclusive_probabilities'])!=set(names):_fail('TRANSFER_STATES','Explicit losses and probabilities for all four states required')
    states=[dict(state_id=name,events=dict(default=i>=2,transfer=i%2==1),probability=combined['exclusive_probabilities'][name],conditional_loss=losses[name]) for i,name in enumerate(names)]
    integrated=event_conditioned_loss(states,event_names=['default','transfer'],currency=currency,probability_measure=target['probability_measure'],context=ctx)
    return _result(dict(expected_loss=integrated.value['expected_loss'],contributions={row['state_id']:row['contribution'] for row in integrated.value['states']},target=target,
        applied_effects=previous+effects),'transfer.transfer_loss_components',currency,ctx,
        limitations=('Conditional losses are joint monetary expectations for each exclusive state. Joint-event loss is never inferred as the average of standalone losses.',
                    'Effect IDs must identify economic mitigants consistently across PD and recovery layers; unreported duplication cannot be inferred.'))
