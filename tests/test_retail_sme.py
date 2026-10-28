"""Business/household reconciliation with independently calculated cash waterfall."""
from copy import deepcopy
import pytest
from creditratingsystem._retail_sme import build_retail_sme_features
from creditratingsystem.retail_features import build_customer_household_graph,aggregate_household_obligations
from creditratingsystem.core import CreditRiskError
D='2024-01-31'
def households(facility='private'):
    entities=[dict(entity_id=id,entity_type=kind,known_at='2024-01-01')for id,kind in [('owner','CUSTOMER'),('house','HOUSEHOLD'),(facility,'FACILITY')]]
    def edge(id,target,role,share,legal):return dict(relationship_id=id,release_seq=1,source_id='owner',target_id=target,role=role,effective_from='2024-01-01',effective_until=None,known_at='2024-01-01',liability_fraction=legal,budget_share=share,evidence_reference='local signed link')
    graph=build_customer_household_graph(entities,[edge('member','house','MEMBER_OF',None,None),edge('debt',facility,'BORROWER_ON',1,1)],as_of=D,information_cutoff=D)
    obligations=[dict(facility_id=facility,release_seq=1,known_at=D,observed_at=D,source_id='bank',debt_stock=100,scheduled_debt_service=10,currency='EUR',period='month',status='ACTIVE')]
    return aggregate_household_obligations(obligations,graph,dict(status='COMPLETE',sources=['bank'],known_at=D),currency='EUR',period='month',guarantee_budget_policy='exclude')
def flow(id,amount,kind,**kw):return dict(transaction_id=id,release_seq=1,business_id='biz',posted_at='2024-01-15',known_at='2024-01-15',amount=amount,category=kind,currency='EUR',**kw)
def flows():return [flow('s1',120,'SALES',counterparty_id='buyer1'),flow('s2',80,'SALES',counterparty_id='buyer2'),flow('cost',-60,'OPERATING_COST'),flow('tax',-10,'TAX'),flow('capex',-20,'CAPEX'),flow('debt',-30,'DEBT_SERVICE',facility_id='bizloan'),flow('draw',-40,'OWNER_DRAW'),flow('finance',100,'FINANCING')]
def drawing():return dict(drawing_id='draw',release_seq=1,known_at='2024-01-15',business_transaction_id='draw',household_transaction_id='privatecredit',owner_customer_id='owner',business_posted_at='2024-01-15',household_posted_at='2024-01-15',business_amount=-40,household_amount=40,currency='EUR')
def liability():return dict(liability_id='link',release_seq=1,known_at='2024-01-01',business_id='biz',customer_id='owner',liability_type='LIMITED_OWNERSHIP',effective_from='2024-01-01',effective_until=None,liability_fraction=0,maximum_amount=0,evidence_reference='limited company ownership')
def calculate(rows=None,draws=None,h=None,status='COMPLETE',links=None):return build_retail_sme_features(flows()if rows is None else rows,[drawing()]if draws is None else draws,households()if h is None else h,[liability()]if links is None else links,dict(status=status,known_at=D,observed_from='2024-01-01',complete_through=D),business_id='biz',owner_customer_ids=['owner'],business_debt_facility_ids=['bizloan'],household_debt_service_window={'from':'2024-01-01','through':D,'period':'month'},as_of=D,information_cutoff=D,lookback_days=31,currency='EUR')
def test_operating_waterfall_not_sales_or_owner_draw_doublecount():
    r=calculate().value
    assert r['known_operating_cash_before_capex_and_debt']==130
    assert r['known_cash_before_business_debt']==110 and r['cash_after_business_debt']==80
    assert r['owner_draw_business_outflows']==40 and r['matched_owner_draw_household_inflows']==40
    assert r['cash_after_business_and_household_debt']==70
    assert r['customer_sales_hhi']==pytest.approx(.52) and r['monthly_sales']=={'2024-01':200}
    assert r['legal_liability_links'][0]['liability_type']=='LIMITED_OWNERSHIP'
def test_overlap_or_missing_coverage_suppresses_combined_subtraction():
    r=calculate(h=households('bizloan')).value;assert r['cash_after_business_and_household_debt']is None and r['overlapping_business_household_facility_ids']==['bizloan']
    r=calculate(status='UNKNOWN').value;assert r['cash_after_business_debt']is None and r['cash_after_business_and_household_debt']is None and r['known_cash_after_business_debt']==80

def test_unmatched_transfer_not_income_and_duplicate_assignment_rejected():
    r=calculate(draws=[]).value;assert r['unmatched_business_drawing_ids']==['draw'] and r['cash_after_business_debt']==80
    with pytest.raises(CreditRiskError):calculate(draws=[drawing(),dict(drawing(),drawing_id='duplicate')])
    with pytest.raises(CreditRiskError):calculate(draws=[dict(drawing(),household_amount=41)])

def test_late_business_revision_and_liability_end_boundary():
    rows=flows();rows.append(dict(rows[0],release_seq=2,known_at='2024-02-01',amount=999))
    link=liability();link['effective_until']=D
    r=calculate(rows=rows,links=[link]).value;assert r['cash_after_business_debt']==80 and r['legal_liability_links']==[]

def test_household_context_and_period_are_binding():
    h=households().to_dict();h['context']['as_of']='2024-01-30'
    with pytest.raises(CreditRiskError):calculate(h=h)
    h=households().to_dict();h['value']['period']='year'
    with pytest.raises(CreditRiskError):calculate(h=h)

def test_negative_net_customer_sales_do_not_produce_false_concentration():
    r=calculate(rows=[flow('refund',-10,'SALES',counterparty_id='buyer')],draws=[]).value
    assert r['customer_sales_hhi']is None and r['observed_peak_month_sales_share']is None
