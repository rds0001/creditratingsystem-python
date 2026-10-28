from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._structured_pool import build_credit_asset_pool,tranche_loss_from_pool,receivables_borrowing_base
D='2023-01-01'
def profile(kind='ABS',cls='AUTO_LOAN',mode='STATIC'):return dict(profile_id='p',version='1',known_at=D,currency='EUR',product_kind=kind,pool_mode=mode,allowed_asset_classes=[cls],maximum_dpd=30,exclude_disputed=True,eligible_recourse=['NONE','SELLER'],revolving_until=None if mode=='STATIC'else'2024-01-01')
def asset(id='a',cls='AUTO_LOAN'):
 return dict(asset_id=id,asset_class=cls,invoice_id='invoice-'+id if cls=='INVOICE'else None,obligor_id='o'+id,default_group_id='g'+id,seller_id='seller',servicer_id='service',currency='EUR',observed_at=D,known_at=D,origination_date='2022-01-01',balance_date=D,principal_balance=100,schedule=[dict(date='2023-07-01',principal_due=40,interest_due=5),dict(date='2024-01-01',principal_due=60,interest_due=3)],overdue_days=0,disputed=False,dilution_reserve=0,recourse='NONE',collateral_allocations=[])
def assignment(id='m',aid='a',pool='pool',share=1,start=D,end=None):return dict(assignment_id=id,pool_id=pool,asset_id=aid,known_at=D,effective_from=start,effective_until=end,share=share,purchase_price=share*100,purchase_priority=1)
def build(assets=None,assignments=None,p=None):return build_credit_asset_pool(assets or [asset()],assignments or [assignment()],p or profile(),pool_id='pool',as_of=D,information_cutoff=D).value

def test_fractional_pool_ownership_and_half_open_transfer():
 a=[assignment(share=.6),assignment('other',pool='other',share=.4)]
 r=build(assignments=a);assert r['reference_principal']==60 and len(r['active_assignments'])==1
 a[1]['share']=.5
 with pytest.raises(CreditRiskError,match='STRUCTURED_ASSIGNMENT'):build(assignments=a)
 old=assignment('old',pool='old',start='2022-01-01',end=D)
 assert build(assignments=[old,assignment()])['reference_principal']==100

def test_revolving_existing_inventory_and_static_exclusion():
 a=[asset(),asset('b')];rows=[assignment(),assignment('next','b',start='2023-07-01')]
 r=build(a,rows,profile(mode='REVOLVING'));assert r['reference_principal']==100 and len(r['reinvestment_candidates'])==1
 r=build(a,rows);assert r['reinvestment_candidates']==[] and r['excluded_assignments'][0]['reasons']==['STATIC_POOL_ADDITION']
 rows[1]['effective_from']='2024-01-01'
 assert build(a,rows,profile(mode='REVOLVING'))['excluded_assignments'][0]['reasons']==['OUTSIDE_REINVESTMENT_WINDOW']

@pytest.mark.parametrize('kind,cls',[('ABS','CONSUMER_LOAN'),('RMBS','RESIDENTIAL_MORTGAGE'),('CMBS','COMMERCIAL_MORTGAGE'),('CLO','CORPORATE_LOAN'),('MORTGAGE_COVER','RESIDENTIAL_MORTGAGE'),('PUBLIC_SECTOR_COVER','PUBLIC_SECTOR_CLAIM'),('TRADE_RECEIVABLES','INVOICE')])
def test_product_route_positive_and_incompatible(kind,cls):
 a=asset(cls=cls);assert build([a],p=profile(kind,cls))['reference_principal']==100
 p=profile(kind,cls);p['allowed_asset_classes']=['AUTO_LOAN'if cls!='AUTO_LOAN'else'INVOICE']
 if kind=='ABS':p['allowed_asset_classes']=['INVOICE']
 with pytest.raises(CreditRiskError,match='STRUCTURED_ROUTE'):build([a],p=p)

def test_future_knowledge_and_stale_balance_not_current_pool():
 a=asset();a['known_at']='2023-02-01';r=build([a]);assert r['reference_principal']==0 and r['excluded_assets'][0]['reasons']==['NOT_YET_KNOWN_OR_OBSERVED']
 a=asset();a['balance_date']='2022-12-31';assert build([a])['excluded_assignments'][0]['reasons']==['STALE_BALANCE']
 p=profile();p['revolving_until']='2024-01-01'
 with pytest.raises(CreditRiskError,match='STRUCTURED_ROUTE'):build(p=p)

def test_duplicate_invoice_default_group_collateral_and_schedule():
 a,b=asset(cls='INVOICE'),asset('b',cls='INVOICE');b['invoice_id']=a['invoice_id']
 with pytest.raises(CreditRiskError,match='STRUCTURED_DUPLICATE'):build([a,b],p=profile('TRADE_RECEIVABLES','INVOICE'))
 a,b=asset(),asset('b');b['obligor_id']=a['obligor_id']
 with pytest.raises(CreditRiskError,match='STRUCTURED_DEFAULT_GROUP'):build([a,b])
 b['default_group_id']=a['default_group_id'];a['collateral_allocations']=[dict(collateral_id='house',allocation_fraction=.6)];b['collateral_allocations']=[dict(collateral_id='house',allocation_fraction=.5)]
 with pytest.raises(CreditRiskError,match='STRUCTURED_COLLATERAL'):build([a,b])
 a=asset();a['schedule'][1]['principal_due']=50
 with pytest.raises(CreditRiskError,match='STRUCTURED_SCHEDULE'):build([a])

def test_static_tranche_loss_normalization_and_boundaries():
 r=tranche_loss_from_pool([0,.1,.15,.2,1,2],.1,.2,pool_notional=100).value
 assert r['tranche_notional']==10
 assert r['tranche_loss_fraction']==pytest.approx([0,0,.5,1,1,1])
 assert r['tranche_loss_amount']==pytest.approx([0,0,5,10,10,10])
 assert tranche_loss_from_pool(15,.1,.2,pool_notional=100,loss_unit='amount').value['tranche_loss_fraction']==pytest.approx([.5])
 with pytest.raises(CreditRiskError,match='STRUCTURED_TRANCHE'):tranche_loss_from_pool([.1],.2,.2,pool_notional=100)
 with pytest.raises(CreditRiskError,match='NUMERICAL_FAILURE'):tranche_loss_from_pool([1e308],.1,.2,pool_notional=1e-308,loss_unit='amount')

def borrowing_profile():return dict(profile_id='b',version='1',known_at=D,currency='EUR',maximum_overdue_days=30,exclude_disputed=True,allowed_recourse=['NONE','SELLER'],advance_rates={'NONE':.8,'SELLER':.9},default_group_cap=.5,group_caps={},additional_reserve=5,reserve_scope='ADDITIONAL_NOT_IN_INVOICE_DILUTION',coverage_status='COMPLETE')
def invoice(id,group,amount):return dict(invoice_id=id,debtor_id='debtor'+group,debtor_group_id=group,seller_id='s',servicer_id='v',currency='EUR',issue_date='2022-12-01',due_date='2023-02-01',balance_date=D,known_at=D,outstanding=amount,disputed=False,dilution_reserve=0,recourse='NONE')
def bb(rows,p=None):return receivables_borrowing_base(rows,p or borrowing_profile(),as_of=D,information_cutoff=D).value

def test_borrowing_base_shared_debtor_cap_and_dilution():
 rows=[invoice('one','g',100),invoice('two','g',100),invoice('three','h',100)];rows[0]['dilution_reserve']=20
 r=bb(rows)
 # Denominator280; groupg180 capped140, h100 retained. Advances192-reserve5=187.
 assert r['concentration_denominator']==280 and r['group_retained']=={'g':140,'h':100}
 assert r['gross_advance_value']==pytest.approx(192) and r['borrowing_base']==pytest.approx(187)
 assert r['invoice_results'][0]['eligible_after_concentration']==pytest.approx(80*140/180)
 assert r['invoice_results'][2]['concentration_binding']

def test_invoice_dispute_aging_recourse_and_coverage():
 rows=[invoice('late','g',100),invoice('dispute','h',100),invoice('ok','i',100)]
 rows[0]['due_date']='2022-12-01';rows[1]['disputed']=True;rows[2]['recourse']='SELLER'
 r=bb(rows);assert r['invoice_results'][0]['exclusion_reasons']==['DISPUTED']
 assert r['invoice_results'][1]['exclusion_reasons']==['AGING']
 assert r['borrowing_base']==40 # retained50*.9 -5
 p=borrowing_profile();p['coverage_status']='PARTIAL';r=bb(rows,p);assert r['borrowing_base']is None and r['known_borrowing_base']==40
 p=borrowing_profile();p['allowed_recourse']=['NONE'];assert bb(rows,p)['borrowing_base']==0

def test_borrowing_duplicate_and_group_evasion():
 with pytest.raises(CreditRiskError,match='STRUCTURED_DUPLICATE'):bb([invoice('a','g',100)]*2)
 a,b=invoice('a','g',100),invoice('b','h',100);b['debtor_id']=a['debtor_id']
 with pytest.raises(CreditRiskError,match='STRUCTURED_DEFAULT_GROUP'):bb([a,b])

def test_empty_static_loss_rejected():
 with pytest.raises(CreditRiskError,match='EMPTY_INPUT'):tranche_loss_from_pool([],.1,.2,pool_notional=100)
