"""Cash movements, category uncertainty and own-transfer hand oracles."""
import math
import pytest
from creditratingsystem._retail_cashflows import build_transaction_features
from creditratingsystem.core import CreditRiskError
D='2024-01-03'
def coverage(status='COMPLETE'):return dict(status=status,known_at=D,observed_from='2024-01-01',complete_through=D)
def tx(id,amount,code,date='2024-01-02',account='a',**kw):return dict(transaction_id=id,release_seq=1,account_id=account,posted_at=date,value_at=date,known_at=date,amount=amount,currency='EUR',kind='ORIGINAL',category_code=code,**kw)
def balance(id,account,date,amount):return dict(balance_id=id,release_seq=1,account_id=account,observed_at=date,known_at=date,closing_balance=amount,currency='EUR')
def calculate(rows,balances=None,status='COMPLETE'):return build_transaction_features(rows,[]if balances is None else balances,dict(mapping_id='local',version='1',known_at='2024-01-01',categories={'wage':'SALARY','food':'CONSUMPTION','loan':'BORROWING','refund':'REFUND','transfer':'OWN_TRANSFER','business':'BUSINESS_REVENUE','draw':'OWNER_DRAW'}),coverage(status),customer_id='c',own_account_ids=['a','b'],as_of=D,information_cutoff=D,lookback_days=2,currency='EUR')
def test_own_transfer_never_income_and_external_cashflow_matches():
    rows=[tx('wage',100,'wage'),tx('out',-60,'transfer',transfer_id='pair',counterparty_own_account_id='b'),tx('in',60,'transfer',account='b',transfer_id='pair',counterparty_own_account_id='a'),tx('food',-20,'food',date=D)]
    r=calculate(rows).value
    assert r['known_classified_income']==100 and r['known_classified_expense']==20
    assert r['known_gross_inflows']==160 and r['known_gross_outflows']==80 and r['known_external_net_cashflow']==80
    assert r['daily_external_net_cashflow']==[100,-20] and r['daily_net_population_sd']==pytest.approx(60)
    assert r['unmatched_own_transfer_ids']==[]
def test_reversal_neutralizes_original_and_inherits_classification():
    original=tx('food',-20,'food');reverse=dict(tx('reversal',20,'nonsense',date=D),kind='REVERSAL',reverses_transaction_id='food')
    r=calculate([original,reverse]).value;assert r['known_classified_expense']==0
    assert r['category_totals']['CONSUMPTION']['net']==0 and r['unknown_category_absolute_volume']==0
    assert r['daily_external_net_cashflow']==[-20,20]
    original['posted_at']=original['value_at']=original['known_at']='2024-01-01'
    r=calculate([original,reverse]).value;assert r['known_net_cashflow']==20 and r['known_classified_expense']==-20

def test_unknown_borrowing_sales_refund_and_draw_not_income():
    rows=[tx('x',50,'missing'),tx('loan',1000,'loan'),tx('sale',500,'business'),tx('refund',20,'refund'),tx('draw',100,'draw')]
    r=calculate(rows).value
    assert r['classified_income']==0 and r['classified_expense']==0 and r['unknown_category_absolute_volume']==50
    assert r['unknown_category_fraction']==pytest.approx(50/1670)
    assert r['known_external_net_cashflow']==1570

def test_late_correction_and_receipt_do_not_enter_cutoff():
    original=tx('x',100,'wage');revision=dict(original,release_seq=2,known_at='2024-01-04',amount=500)
    assert calculate([original,revision]).value['classified_income']==100
    assert calculate([dict(original,known_at='2024-01-04')]).value['known_gross_inflows']==0

def test_balances_negative_account_days_and_missing_not_zero():
    balances=[balance('a2','a','2024-01-02',-10),balance('b2','b','2024-01-02',20),balance('a3','a',D,-5),balance('b3','b',D,5)]
    r=calculate([],balances).value
    assert r['overdraft_account_days']==2 and r['complete_balance_days']==2 and r['minimum_observed_complete_household_balance']==0 and r['mean_observed_complete_household_balance']==5
    r=calculate([],balances[:-1]).value;assert r['overdraft_account_days']is None and r['missing_balance_account_days']==1
    assert calculate([],status='UNKNOWN').value['daily_net_population_sd']is None

def test_bad_reversal_and_mismatched_transfer_fail():
    with pytest.raises(CreditRiskError):calculate([tx('food',-20,'food'),dict(tx('r',21,'food',date=D),kind='REVERSAL',reverses_transaction_id='food')])
    with pytest.raises(CreditRiskError):calculate([tx('out',-60,'transfer',transfer_id='p',counterparty_own_account_id='b'),tx('in',59,'transfer',account='b',transfer_id='p',counterparty_own_account_id='a')])
    assert calculate([tx('out',-60,'transfer',transfer_id='p',counterparty_own_account_id='b')]).value['unmatched_own_transfer_ids']==['p']
