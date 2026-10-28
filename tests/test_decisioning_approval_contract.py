"""Versioned approval applicability: independent gates cannot be bypassed."""
from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError,compile_credit_decision_policy,apply_credit_decision_policy
from test_decisioning import policy_spec,inputs


def specification(domain='INVESTMENT'):
    s=policy_spec();s['schema_version']='1.1';s['policy_id']='investment';s['fields'].pop('affordability.status');s['rules']=[r for r in s['rules']if r['rule_id']!='affordability']
    s['fields']['operations.legal_status']=dict(type='category',values=['VERIFIED','UNVERIFIED'])
    s['fields']['operations.limit_status']=dict(type='category',values=['WITHIN_LIMIT','EXCEEDED'])
    s['approval_contract']=dict(domain=domain,affordability=dict(mode='NOT_APPLICABLE',reason='Investment/CSA exposure; household budget is not the target',reference='scope-v1',maximum_age_days=30),required_gates=[dict(field='operations.legal_status',value='VERIFIED'),dict(field='operations.limit_status',value='WITHIN_LIMIT')])
    return s


def data(domain='INVESTMENT'):
    d=inputs();d['affordability'].update(status='NOT_APPLICABLE',domain=domain,applicability_reference='scope-v1',valid_until='2024-07-01');d['operations'].update(legal_status='VERIFIED',limit_status='WITHIN_LIMIT');return d


def run(s=None,d=None):return apply_credit_decision_policy(compile_credit_decision_policy(s or specification()),**(d or data())).value


def test_legacy_exact_artifact_hashes_and_no_silently_ignored_waiver():
    p=compile_credit_decision_policy(policy_spec()).value;r=apply_credit_decision_policy(p,**inputs()).value
    assert p['sha256']=='8e6c728e84e847fc1579e7d6720cd1f7748bc927be5bb81334f5200863ccf092'
    assert r['sha256']=='250aa1df53b889b608f9a6d739061b6216167f4d54a4b66d8e9548af70c9fdbd'
    s=policy_spec();s['approval_contract']=None
    with pytest.raises(CreditRiskError,match='INVALID_POLICY'):compile_credit_decision_policy(s)


@pytest.mark.parametrize('domain',['INVESTMENT','COUNTERPARTY_LIMIT'])
def test_dated_scoped_applicability_and_all_categorical_gates(domain):
    r=run(specification(domain),data(domain));assert r['action']=='APPROVE'and all(x['passed']for x in r['approval_gate_trace'])
    assert r['approval_domain']==domain and len(r['approval_gate_trace'])==5
    d=data(domain);d['operations']['limit_status']='EXCEEDED';r=run(specification(domain),d)
    assert r['action']=='REFER'and 'operations.limit_status.APPROVAL_GATE'in r['findings']


@pytest.mark.parametrize('source,field,value,expected',[
    ('eligibility','status','INELIGIBLE','DECLINE'),('operations','status','PENDING','REFER'),
    ('operations','legal_status','UNVERIFIED','REFER'),('affordability','status','AFFORDABLE','REFER'),
    ('risk','freshness','STALE','BLOCKED'),('affordability','known_at','2024-06-03','BLOCKED'),
    ('affordability','applicability_reference','other','BLOCKED'),('affordability','domain','RETAIL_CREDIT','BLOCKED'),
    ('affordability','valid_until','2024-06-02','BLOCKED'),('affordability','as_of','2024-05-02','BLOCKED')])
def test_no_priority_or_stale_applicability_bypass(source,field,value,expected):
    d=data();d[source][field]=value;r=run(d=d);assert r['action']==expected
    # A highest-priority unconditional approve still cannot bypass universal gates.
    if source in ('eligibility','operations'):
        s=specification();s['rules']=[dict(rule_id='bypass',priority=0,action='APPROVE',reason_code='TEST',when=[])];r=run(s,d);assert r['action']=='REFER'


def test_expiry_and_age_exact_boundaries_missing_evidence():
    d=data();d['affordability']['as_of']='2024-05-03';assert run(d=d)['action']=='APPROVE'
    d['affordability'].pop('valid_until');assert 'affordability.MISSING_VALIDITY'in run(d=d)['findings']
    d=data();d['affordability'].pop('applicability_reference');assert 'affordability.APPLICABILITY_BINDING'in run(d=d)['findings']
    d=data();d['operations'].pop('legal_status');assert run(d=d)['action']=='BLOCKED'


@pytest.mark.parametrize('mutation',[
    lambda s:s['approval_contract'].update(domain='RETAIL_CREDIT'),
    lambda s:s['approval_contract'].update(domain='WHOLESALE_CREDIT'),
    lambda s:s['approval_contract']['affordability'].update(maximum_age_days=-1),
    lambda s:s['approval_contract']['affordability'].update(reason=''),
    lambda s:s['approval_contract']['required_gates'].append(deepcopy(s['approval_contract']['required_gates'][0])),
    lambda s:s['approval_contract']['required_gates'][0].update(field='risk.pd'),
    lambda s:s['approval_contract']['required_gates'][0].update(value='UNKNOWN'),
    lambda s:s['approval_contract'].update(unknown='ignored')])
def test_malformed_or_waived_credit_contract_rejected(mutation):
    s=specification();mutation(s)
    with pytest.raises(CreditRiskError,match='INVALID_POLICY'):compile_credit_decision_policy(s)


def test_required_credit_affordability_and_fingerprint_integrity():
    s=specification('RETAIL_CREDIT');s['approval_contract']['affordability']=dict(mode='REQUIRED');d=data('RETAIL_CREDIT')
    assert run(s,d)['action']=='REFER';d['affordability']['status']='AFFORDABLE';assert run(s,d)['action']=='APPROVE'
    p=compile_credit_decision_policy(s).value;p['specification']['approval_contract']['required_gates']=[]
    with pytest.raises(CreditRiskError,match='ARTIFACT_HASH_MISMATCH'):apply_credit_decision_policy(p,**d)


def test_native_json_integral_priority_and_fractional_rejection():
    s=specification();expected=compile_credit_decision_policy(s).value['sha256']
    for rule in s['rules']:rule['priority']=float(rule['priority'])
    assert compile_credit_decision_policy(s).value['sha256']==expected
    s['rules'][0]['priority']=.5
    with pytest.raises(CreditRiskError,match='INVALID_POLICY'):compile_credit_decision_policy(s)
    s['rules'][0]['priority']=True
    with pytest.raises(CreditRiskError,match='INVALID_POLICY'):compile_credit_decision_policy(s)
