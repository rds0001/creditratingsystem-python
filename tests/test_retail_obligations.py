"""Overlapping purchase plans, allocated refund and knowledge-cutoff oracles."""
from copy import deepcopy
import pytest
from creditratingsystem._retail_obligations import build_bnpl_obligation_features
from creditratingsystem.core import CreditRiskError
D='2024-01-05'
def plan(id,purchased='2024-01-01',**kw):return dict(plan_id=id,release_seq=1,customer_id='c',provider_id='p',purchased_at=purchased,known_at=purchased,currency='EUR',closed_at=None,refinances_plan_id=None,schedule_complete=True,**kw)
def installment(id,pid,due,amount=25):return dict(installment_id=id,plan_id=pid,release_seq=1,known_at='2024-01-01',due_at=due,amount=amount,currency='EUR')
def event(id,pid,i,kind,amount,date='2024-01-03',related=None):return dict(event_id=id,release_seq=1,known_at=date,occurred_at=date,plan_id=pid,installment_id=i,kind=kind,amount=amount,currency='EUR',related_event_id=related)
def calculate(plans,installments,events=None,status='COMPLETE'):return build_bnpl_obligation_features(plans,installments,[]if events is None else events,dict(status=status,sources=['p'],known_at=D),customer_id='c',as_of=D,information_cutoff=D,lookback_days=5,near_term_days=10,currency='EUR')
def test_two_pay_in_four_plans_and_partial_refund():
    p=[plan('a'),plan('b','2024-01-03')];i=[installment(f'{pid}{n}',pid,f'2024-01-{day:02}')for pid,dates in [('a',[1,8,15,22]),('b',[3,10,17,24])]for n,day in enumerate(dates)]
    e=[event('paid_a','a','a0','PAYMENT',25),event('paid_b','b','b0','PAYMENT',25),event('return','a','a1','MERCHANT_REFUND',10)]
    r=calculate(p,i,e).value
    assert r['total_outstanding']==140 and r['known_near_term_due']==65
    assert r['maximum_known_concurrent_plans']==2 and r['active_plan_count']==2
    assert [v['known_active_plan_count']for v in r['daily_known_concurrency']]==[1,1,2,2,2]
    assert r['known_overdue_amount']==0 and r['explicit_refinancing_count']==0 and r['repeat_purchase_count']==1

def test_disputes_do_not_cancel_debt_and_refund_reversal_restores_it():
    p=[plan('a')];i=[installment('i','a','2024-01-06',100)];e=[event('open','a',None,'DISPUTE_OPEN',0),event('refund','a','i','MERCHANT_REFUND',30),event('reverse','a','i','REFUND_REVERSAL',10,'2024-01-04','refund')]
    r=calculate(p,i,e).value;assert r['total_outstanding']==80 and r['open_dispute_count']==1
    e.append(event('close','a',None,'DISPUTE_CLOSE',0,'2024-01-04','open'));assert calculate(p,i,e).value['open_dispute_count']==0

def test_late_refund_excluded_and_incomplete_schedule_not_zero_debt():
    p=[plan('a')];i=[installment('i','a','2024-01-06',100)];e=[dict(event('refund','a','i','MERCHANT_REFUND',30),known_at='2024-01-06')]
    assert calculate(p,i,e).value['total_outstanding']==100
    assert calculate(p,i,status='UNKNOWN').value['total_outstanding']is None
    assert calculate(p,[]).value['total_outstanding']is None
    p[0]['schedule_complete']=False;assert calculate(p,i).value['total_outstanding']is None

def test_actual_closure_boundary_and_explicit_refinance():
    old=plan('a');old.update(closed_at='2024-01-03',known_at='2024-01-03');new=plan('b','2024-01-03');new['refinances_plan_id']='a';i=[installment('i','a','2024-01-01',25),installment('j','b','2024-01-10',25)]
    r=calculate([old,new],i,[event('paid','a','i','PAYMENT',25)]).value
    assert r['maximum_known_concurrent_plans']==1 and r['active_plan_count']==1 and r['explicit_refinancing_count']==1
    with pytest.raises(CreditRiskError):calculate([old,new],i)

def test_allocations_exceeding_obligation_or_refund_fail():
    p=[plan('a')];i=[installment('i','a','2024-01-06',100)]
    with pytest.raises(CreditRiskError):calculate(p,i,[event('paid','a','i','PAYMENT',80),event('refund','a','i','MERCHANT_REFUND',30)])
    with pytest.raises(CreditRiskError):calculate(p,i,[event('refund','a','i','MERCHANT_REFUND',30),event('reverse','a','i','REFUND_REVERSAL',31,related='refund')])
