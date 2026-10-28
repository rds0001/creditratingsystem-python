"""Independent dated bureau and same-income-component oracles."""
from copy import deepcopy
import pytest
from creditratingsystem.retail_features import build_bureau_features,build_thin_file_features,reconcile_income_evidence
from creditratingsystem.core import CreditRiskError
DATE='2024-01-31'
def snapshot():return dict(snapshot_id='s',release_seq=1,customer_id='c',provider_id='p',observed_at='2024-01-20',reported_at='2024-01-21',known_at='2024-01-22',coverage_status='COMPLETE',currency='EUR')
def line():return dict(snapshot_id='s',snapshot_release_seq=1,tradeline_id='loan',opened_at='2023-01-01',closed_at=None,status='OPEN',balance=100,limit=200,max_dpd=0)
def bureau(snaps=None,lines=None,queries=None,**kwargs):
    args=dict(customer_id='c',provider_id='p',as_of=DATE,information_cutoff=DATE,lookback_days=30,max_age_days=30,currency='EUR');args.update(kwargs)
    return build_bureau_features([snapshot()]if snaps is None else snaps,[line()]if lines is None else lines,[]if queries is None else queries,dict(mapping_id='local',version='1',known_at='2023-01-01',snapshots={},tradelines={},inquiries={}),**args)
def profile():return dict(profile_id='e',version='1',known_at='2023-01-01',minimum_tradelines=2,minimum_history_days=365,maximum_evidence_age_days=30,allow_partial_evidence=False,feature_requirements={'bureau_utilization':['BUREAU'],'cashflow':['TRANSACTIONS']})
def thin(b=None,e=None,p=None):return build_thin_file_features(bureau()if b is None else b,[]if e is None else e,profile()if p is None else p,customer_id='c',as_of=DATE,information_cutoff=DATE)
def income_row(id,amount,component='salary',kind='SALARY'):return dict(evidence_id=id,release_seq=1,known_at='2024-01-30',customer_id='c',component_id=component,component_type=kind,observed_from='2024-01-01',observed_until='2024-01-29',amount=amount,basis='NET',period='month',currency='EUR',evidence_reference='local document')
def policy():return dict(policy_id='p',version='1',known_at='2023-01-01',source_priority=['DOCUMENT','TRANSACTION','DECLARED'],included_component_types=['SALARY','BONUS'],expected_component_ids=['salary'],coverage_status='COMPLETE',maximum_age_days=30,allow_stale=False,relative_discrepancy_threshold=.05)
def income(declared=None,docs=None,transactions=None,p=None):return reconcile_income_evidence([income_row('d',100)]if declared is None else declared,[income_row('v',90)]if docs is None else docs,[income_row('t',95)]if transactions is None else transactions,policy()if p is None else p,customer_id='c',as_of=DATE,information_cutoff=DATE,currency='EUR',period='month',basis='NET')
def test_bureau_revision_binding_and_closed_balances():
    revised=dict(snapshot(),release_seq=2,known_at='2024-02-01');new=dict(line(),snapshot_release_seq=2,balance=999)
    closed=dict(line(),tradeline_id='old',closed_at='2024-01-10',status='CLOSED',balance=7)
    r=bureau([snapshot(),revised],[line(),new,closed]);f=r.value['features']
    assert f['aggregate_open_utilization']==.5 and f['total_open_balance']==100
    assert f['known_closed_balance']==7 and f['reported_tradeline_count']==2
    assert r.diagnostics['snapshot_release_seq']==1
    assert bureau([snapshot(),revised],[line(),new],as_of='2024-02-02',information_cutoff='2024-02-02').value['features']['total_open_balance']==999

def test_bureau_queries_interval_and_receipt_cutoff():
    def query(id,date,kind='HARD',known=None):return dict(inquiry_id=id,release_seq=1,customer_id='c',provider_id='p',occurred_at=date,known_at=date if known is None else known,kind=kind)
    f=bureau(queries=[query('excluded','2024-01-01'),query('hard','2024-01-02'),query('soft',DATE,'SOFT'),query('future','2024-01-03',known='2024-02-01')]).value['features']
    assert f['hard_inquiries']==1 and f['soft_inquiries']==1
    assert bureau(lookback_days=10**20).value['features']['hard_inquiries']==0

def test_absent_nohit_stale_are_not_zero_debt():
    absent=bureau([],[]);assert absent.value['features']['reported_tradeline_count']is None
    nohit=bureau([dict(snapshot(),coverage_status='NO_HIT')],[]);assert nohit.value['features']['reported_tradeline_count']==0
    assert nohit.value['features']['total_open_balance']is None
    assert thin(absent).value['thin_file_stratum']=='BUREAU_ABSENT'
    assert thin(nohit).value['thin_file_stratum']=='BUREAU_NO_HIT'
    assert thin(bureau(max_age_days=1)).value['thin_file_stratum']=='BUREAU_STALE'
    assert thin().value['thin_file_stratum']=='THIN'

def test_thin_evidence_eligibility_and_outcome_leakage():
    e=dict(evidence_id='tx',release_seq=1,known_at='2024-01-30',customer_id='c',evidence_type='TRANSACTIONS',observed_from='2023-01-01',observed_until='2024-01-29',coverage_status='COMPLETE')
    assert thin(e=[e]).value['supported_features']==['bureau_utilization','cashflow']
    assert thin(e=[dict(e,coverage_status='UNKNOWN')]).value['unsupported_features'][0]['feature']=='cashflow'
    p=profile();p['feature_requirements']={'future_default':['BUREAU']}
    with pytest.raises(CreditRiskError):thin(p=p)
    raw=deepcopy(bureau().value);raw['features']['history_days']='many'
    with pytest.raises(CreditRiskError):thin(raw)

def test_income_selects_same_component_once_and_keeps_bonus_separate():
    r=income(docs=[income_row('v',90),income_row('bonus',10,'bonus','BONUS')]).value
    assert r['total_selected_income']==100 and r['document_supported_current_income']==100
    salary=next(x for x in r['components']if x['component_id']=='salary')
    assert salary['source_amounts']=={'DOCUMENT':90,'TRANSACTION':95,'DECLARED':100}
    assert salary['declared_difference']==-10 and salary['declared_relative_difference']==-.1 and salary['discrepancy_flag']

def test_income_basis_stale_coverage_and_revenue_do_not_create_income():
    r=income(docs=[dict(income_row('v',900),basis='GROSS'),income_row('sales',10000,'sales','BUSINESS_REVENUE')]).value
    assert r['total_selected_income']==95 and r['document_supported_current_income']==0
    stale=dict(income_row('v',90),observed_from='2023-10-01',observed_until='2023-10-31',known_at='2023-11-01')
    r=income([], [stale],[]).value
    assert r['total_selected_income']is None and r['known_selected_income']==0 and r['missing_component_ids']==['salary']
    p=policy();p['coverage_status']='PARTIAL'
    assert income(p=p).value['total_selected_income']is None

def test_income_late_revision_not_visible_and_missing_component_not_zero():
    revised=dict(income_row('v',1),release_seq=2,known_at='2024-02-01')
    assert income(docs=[income_row('v',90),revised]).value['total_selected_income']==90
    p=policy();p['expected_component_ids'].append('rent')
    r=income(p=p).value;assert r['known_selected_income']==90 and r['total_selected_income']is None
