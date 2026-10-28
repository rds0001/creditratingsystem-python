"""Dated payment-route assessment, exclusive default union and monetary loss."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def assess(data, known_at=None):
    known_at = known_at or data['known_at']
    applicability = cr.assess_transfer_applicability(data['payments'], data['event_policy'],
        as_of=data['as_of'], known_at=known_at, horizon_end=data['horizon_end']).value
    statuses = [row['status'] for row in applicability['payments']]
    if 'INSUFFICIENT_INFORMATION' in statuses:
        return dict(applicability=applicability, scenarios=None, status='INSUFFICIENT_INFORMATION')
    event_p = data['event_probability'] if 'APPLICABLE' in statuses else 0.
    transfer = cr.transfer_event_default_probability(event_p, data['event_to_default_probability'],
        data['mitigation'], event_id=data['event_policy']['events'][0]['event_id'], target=data['target']).value
    base = dict(stage='PRE_TRANSFER', pd=data['pre_transfer_pd'], target=data['target'], applied_effects=[])
    p, q = base['pd'], transfer['probability']
    scenarios = {}
    for case in data['scenarios']:
        rule = case['joint_rule']
        if rule not in ('independent', 'upper', 'lower'):
            raise ValueError('Declare independent, upper or lower joint-event assumption')
        j = p*q if rule == 'independent' else min(p,q) if rule == 'upper' else max(0.,math.fsum((p,q,-1.)))
        joint = cr.joint_default_probability(p,q,event_order=['obligor-default','transfer-default'],joint_probability=j).value
        combined = cr.combine_default_transfer(base,transfer,dependence='explicit',joint_probability=joint).value
        loss = cr.transfer_loss_components(combined,data['conditional_losses'],currency=data['currency'],applied_loss_effects=[]).value
        grade = cr.map_pd_to_grade(combined['combined_pd'],data['rating_scale']).value
        scenarios[case['case_id']] = dict(assumption=rule,risk=combined,loss=loss,grade=grade)
    return dict(applicability=applicability,transfer=transfer,base=base,scenarios=scenarios,status='ASSESSED')


def run(data=None, verify_reference=True):
    if data is None:
        data=json.loads(files('creditratingsystem.datasets').joinpath('transfer_review.json').read_text())
    original=assess(data)
    later=assess(data, '2025-03-01')
    missing=deepcopy(data);missing['payments'][0].pop('payment_jurisdictions')
    unknown=assess(missing)
    replay=assess(data)
    output=dict(original=original, later_knowledge=later, missing_route=unknown,
        original_hash=cr.content_fingerprint(original).value, replay_hash=cr.content_fingerprint(replay).value,
        interpretation='Conditional synthetic transfer-default probabilities, explicit dependence sensitivity and four disjoint monetary losses. Route exemptions follow knowledge time. No country-rating notch or legal determination is inferred.')
    if verify_reference:
        for case,pd,loss,grade in [('independent',.172,7.52,'B'),('nested',.1,6.8,'A'),('disjoint',.18,7.6,'B')]:
            row=original['scenarios'][case]
            assert math.isclose(row['risk']['combined_pd'],pd,abs_tol=1e-14)
            assert math.isclose(row['loss']['expected_loss'],loss,abs_tol=1e-12)
            assert row['grade']==grade
        assert original['applicability']['payments'][0]['status']=='APPLICABLE'
        assert later['applicability']['payments'][0]['status']=='EXEMPT'
        assert all(math.isclose(v['loss']['expected_loss'],6.,abs_tol=1e-14)for v in later['scenarios'].values())
        assert unknown['scenarios'] is None and output['original_hash']==output['replay_hash']
    return output


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
