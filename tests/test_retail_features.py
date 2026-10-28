"""Independent relational and household obligation oracles."""
from copy import deepcopy
import pytest
from creditratingsystem.retail_features import validate_retail_dataset,build_customer_household_graph,aggregate_household_obligations
from creditratingsystem.core import CreditRiskError
ASOF='2024-01-31'
def entities():return [dict(entity_id=i,entity_type=t,known_at='2024-01-01')for i,t in [('a','CUSTOMER'),('b','CUSTOMER'),('g','CUSTOMER'),('h','HOUSEHOLD'),('h2','HOUSEHOLD'),('f','FACILITY'),('f2','FACILITY'),('app','APPLICATION')]]
def edge(id,source,target,role,share=None,legal=None):return dict(relationship_id=id,release_seq=1,source_id=source,target_id=target,role=role,effective_from='2024-01-01',effective_until=None,known_at='2024-01-01',liability_fraction=legal,budget_share=share,evidence_reference='explicit signed account relation')
def relationships():return [edge('ah','a','h','MEMBER_OF'),edge('bh','b','h','MEMBER_OF'),edge('gh','g','h2','MEMBER_OF'),edge('af','a','f','BORROWER_ON',.5,1),edge('bf','b','f','BORROWER_ON',.5,1),edge('gf','g','f','GUARANTOR_ON',0,.25),edge('aa','a','app','COAPPLICANT_ON'),edge('ba','b','app','COAPPLICANT_ON')]
def graph(rows=None,asof=ASOF):return build_customer_household_graph(entities(),relationships()if rows is None else rows,as_of=asof,information_cutoff=asof)
def obligation(id='f',amount=100):return dict(facility_id=id,release_seq=1,known_at='2024-01-01',observed_at='2024-01-01',source_id='bank',debt_stock=amount,scheduled_debt_service=amount/10,currency='EUR',period='month',status='ACTIVE')
def coverage(status='COMPLETE'):return dict(status=status,sources=['bank'],known_at='2024-01-01')
def aggregate(rows=None,g=None,status='COMPLETE',policy='exclude'):return aggregate_household_obligations([obligation()]if rows is None else rows,graph()if g is None else g,coverage(status),currency='EUR',period='month',guarantee_budget_policy=policy)
def test_joint_debt_once_and_legal_liability_separate():
    r=aggregate().value;h=r['households'][0]
    assert h['facility_ids']==['f'];assert h['known_linked_debt_stock']==100;assert h['total_budget_debt_service']==10
    assert [x['legal_debt_stock']for x in r['person_legal_liabilities']]==[100,100,25]
    assert r['households'][1]['facility_ids']==[]
    assert aggregate(policy='include_declared_share').value['households'][1]['known_linked_debt_stock']==100
    assert aggregate(policy='include_declared_share').value['households'][1]['total_budget_debt_stock']==0

def test_household_release_revisions_do_not_rewrite_earlier_cutoffs():
    rows=relationships();revision=deepcopy(rows[0]);revision.update(release_seq=2,effective_until='2024-01-15',known_at='2024-01-20');new=edge('ah2','a','h2','MEMBER_OF');new.update(effective_from='2024-01-15',known_at='2024-01-20');rows +=[revision,new]
    assert graph(rows,'2024-01-16').value['memberships'][0]['household_id']=='h'
    assert graph(rows,'2024-01-21').value['memberships'][0]['household_id']=='h2'
    at_boundary=deepcopy(rows);at_boundary[-1]['known_at']='2024-01-15';at_boundary[-2]['known_at']='2024-01-15'
    assert graph(at_boundary,'2024-01-15').value['memberships'][0]['household_id']=='h2'

def test_unknown_coverage_missing_amount_and_separate_creditors():
    h=aggregate(status='UNKNOWN').value['households'][0];assert h['known_linked_debt_stock']==100;assert h['total_linked_debt_stock']is None
    h=aggregate(rows=[]).value['households'][0];assert h['known_linked_debt_stock']==0;assert h['uncertain_facility_ids']==['f'];assert h['total_linked_debt_stock']is None
    rows=relationships()+[edge('af2','a','f2','BORROWER_ON',1,1)]
    h=aggregate([obligation(),obligation('f2',30)],graph(rows)).value['households'][0];assert h['total_linked_debt_stock']==130
    r=obligation();r['scheduled_debt_service']=None
    assert aggregate([r]).value['households'][0]['total_linked_debt_service']is None
    assert aggregate([r]).value['households'][0]['total_linked_debt_stock']==100

def test_graph_duplicate_identity_and_budget_integrity():
    rows=relationships()+[dict(edge('duplicate','a','h','MEMBER_OF'))]
    with pytest.raises(CreditRiskError,match='DOUBLE_COUNTING'):graph(rows)
    rows=relationships();rows[3]['budget_share']=.8
    with pytest.raises(CreditRiskError,match='RETAIL_LIABILITY'):graph(rows)
    g=graph().value;g['memberships'][0]['household_id']='h2'
    with pytest.raises(CreditRiskError,match='INTEGRITY_ERROR'):aggregate(g=g)
    future=obligation();future.update(release_seq=2,debt_stock=0,known_at='2024-02-01')
    assert aggregate([obligation(),future]).value['households'][0]['known_linked_debt_stock']==100

def schema():
    def field(name,type='string',nullable=False):return dict(name=name,type=type,nullable=nullable)
    def spec(name,key,extra):return dict(table=name,role='IDENTITY'if name=='customers'else 'FEATURE',primary_key=[key],fields=[field(key),field('observed_at','date'),field('known_at','date')]+extra,foreign_keys=[],date_order=[['observed_at','known_at']],event_field='observed_at',known_field='known_at',feature_fields=[],allow_extra_fields=False)
    customer=spec('customers','customer_id',[]);application=spec('applications','application_id',[field('customer_id'),field('income','number',True)])
    application.update(foreign_keys=[dict(field='customer_id',table='customers',target_field='customer_id')],feature_fields=['income'])
    return dict(schema_version='1.0',contract_id='bank',version='1',table_specs=[customer,application],prohibited_feature_fields=['booked_amount'],source_coverage=[dict(source_id='bureau',status='UNKNOWN',known_at='2024-01-01')])
def tables():return dict(customers=[dict(customer_id='a',observed_at='2024-01-01',known_at='2024-01-01')],applications=[dict(application_id='app'+str(i),customer_id='a',observed_at='2024-01-02',known_at='2024-01-02',income=None)for i in (1,2)])
def test_validation_repeated_applications_and_null_coverage():
    r=validate_retail_dataset(tables(),schema(),as_of=ASOF,information_cutoff=ASOF).value
    assert r['valid'];assert r['row_counts']['applications']==2;assert r['source_coverage'][0]['status']=='UNKNOWN'
    rows=tables();rows['customers'][0]['known_at']='2024-02-01';r=validate_retail_dataset(rows,schema(),as_of=ASOF,information_cutoff=ASOF).value
    assert [v['code']for v in r['violations']]==['PARENT_NOT_KNOWN_AT_CUTOFF']*2

def test_validation_labels_dates_and_duplicate_application_keys():
    contract=schema();spec=contract['table_specs'][1];spec['fields'].append(dict(name='future_default',type='boolean',nullable=True));spec['feature_fields'].append('future_default')
    r=validate_retail_dataset(tables(),contract,as_of=ASOF,information_cutoff=ASOF).value
    assert not r['valid'];assert r['violations'][0]['code']=='OUTCOME_AS_FEATURE'
    rows=tables();rows['applications'][1]['application_id']='app1';rows['applications'][0]['known_at']='2023-12-31'
    r=validate_retail_dataset(rows,schema(),as_of=ASOF,information_cutoff=ASOF).value
    assert {v['code']for v in r['violations']}=={'DATE_ORDER','DUPLICATE_KEY'}

def test_reversal_identity_must_reference_visible_payment():
    contract=schema();spec=deepcopy(contract['table_specs'][1]);spec.update(table='payments',primary_key=['payment_id'],fields=[dict(name=k,type=t,nullable=n)for k,t,n in [('payment_id','string',False),('reverses_payment_id','string',True),('observed_at','date',False),('known_at','date',False)]],foreign_keys=[dict(field='reverses_payment_id',table='payments',target_field='payment_id')],feature_fields=[])
    contract['table_specs']=[spec]
    payment=dict(payment_id='p',reverses_payment_id=None,observed_at='2024-01-10',known_at='2024-02-01');reversal=dict(payment_id='r',reverses_payment_id='p',observed_at='2024-01-20',known_at='2024-01-20')
    v=validate_retail_dataset(dict(payments=[payment,reversal]),contract,as_of=ASOF,information_cutoff=ASOF).value
    assert v['violations']==[dict(table='payments',row_index=1,code='PARENT_NOT_KNOWN_AT_CUTOFF',field='reverses_payment_id')]
    payment['known_at']='2024-01-10';assert validate_retail_dataset(dict(payments=[payment,reversal]),contract,as_of=ASOF,information_cutoff=ASOF).value['valid']
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):build_customer_household_graph(entities(),relationships(),as_of=ASOF,information_cutoff=ASOF,context=dict(known_at='2024-01-30'))
