"""Hand ledgers and limit-cut decompositions, independent of helper internals."""
import pytest
from creditratingsystem._retail_behavior import build_payment_history_features,build_revolving_features
from creditratingsystem.core import CreditRiskError
D='2024-01-05'
def coverage(status='COMPLETE'):return dict(status=status,known_at=D,observed_from='2024-01-01',complete_through=D)
def obligation(id='one',amount=100,due='2024-01-02'):return dict(obligation_id=id,account_id='a',due_at=due,known_at='2024-01-01',amount=amount,currency='EUR')
def payment(id,amount,date,kind='payment',**kw):return dict(payment_id=id,account_id='a',paid_at=date,known_at=date,amount=amount,currency='EUR',kind=kind,**kw)
def history(obs=None,pays=None,**kw):
    args=dict(account_id='a',as_of=D,information_cutoff=D,lookback_days=4,allocation_policy='fifo',currency='EUR',grace_days=0);args.update(kw)
    return build_payment_history_features([obligation()]if obs is None else obs,[]if pays is None else pays,coverage(),**args)
def balance(date,amount=100,limit=200,activity='ACTIVE'):return dict(snapshot_id=date,release_seq=1,account_id='a',observed_at=date,known_at=date,balance=amount,credit_limit=limit,activity_status=activity,currency='EUR')
def revolving(rows,statements=None,cov=None):return build_revolving_features(rows,[]if statements is None else statements,coverage()if cov is None else cov,account_id='a',as_of=D,information_cutoff=D,lookback_days=4,currency='EUR')
def statement(id,bal,minimum,paid):return dict(statement_id=id,release_seq=1,account_id='a',issued_at='2024-01-01',due_at='2024-01-03',observed_at='2024-01-04',known_at='2024-01-04',statement_balance=bal,minimum_due=minimum,paid_by_due=paid,currency='EUR')
def test_partial_due_day_then_late_known_reversal():
    pays=[payment('p',40,'2024-01-02'),dict(payment('r',40,'2024-01-03','reversal',reverses_payment_id='p'),known_at='2024-01-04')]
    r=history(pays=pays).value
    assert [x['unpaid_due_amount']for x in r['daily_history']]==[60,60,100,100]
    assert [x['dpd']for x in r['daily_history']]==[0,1,2,3]
    assert r['features']['mean_dpd']==1.5 and r['features']['arrears_days']==3
    assert r['assessments'][0]['payment_ratio']==.4 and r['features']['partial_payment_count']==1

def test_prepayment_and_due_date_full_payment():
    r=history(pays=[payment('pre',60,'2024-01-01'),payment('due',40,'2024-01-02')]).value
    assert r['features']['full_payment_count']==1 and r['features']['max_dpd']==0
    assert r['features']['assessment_payment_ratio']==1

def test_grace_does_not_change_due_date_or_dpd():
    r=history(pays=[payment('p',100,'2024-01-04')],grace_days=2).value
    assert [v['dpd']for v in r['daily_history']]==[0,1,0,0]
    assert r['assessments'][0]['due_at']=='2024-01-02' and r['assessments'][0]['assessment_at']=='2024-01-04'
    assert r['features']['full_payment_count']==1
    assert history(grace_days=10).value['not_assessable'][0]['reason']=='GRACE_NOT_ELAPSED'

def test_partial_coverage_never_establishes_clean_history():
    r=build_payment_history_features([],[],coverage('UNKNOWN'),account_id='a',as_of=D,information_cutoff=D,lookback_days=4,allocation_policy='fifo',currency='EUR',grace_days=0).value
    assert r['features']['max_dpd']is None and r['known_history_features']['max_dpd']==0

def test_limit_cut_explains_utilization_without_balance_draw():
    r=revolving([balance('2024-01-02'),balance('2024-01-03',limit=100),balance('2024-01-04',limit=80),balance(D,limit=80)]).value
    assert r['overlimit_days']==2 and r['missing_days']==0
    change=r['balance_changes'][0]
    assert change['balance_change']==0 and change['limit_change']==-100
    assert change['utilization_change']==.5 and change['balance_effect_at_new_limit']==0 and change['limit_effect_on_previous_balance']==.5
    assert r['utilization_endpoint_trend_per_day']==.25

def test_zero_limit_missing_days_and_explicit_activity():
    r=revolving([balance(D,0,0)]).value
    assert r['latest_snapshot']['utilization']is None and r['latest_snapshot']['undrawn_active']
    assert r['overlimit_days']is None and r['missing_days']==3
    assert not revolving([balance(D,0,0,'INACTIVE')]).value['latest_snapshot']['undrawn_active']

def test_minimum_full_and_below_minimum_payers_distinct():
    r=revolving([], [statement('min',100,10,10),statement('full',100,10,100),statement('below',100,10,5)]).value
    assert r['full_payer_count']==1 and r['minimum_without_full_count']==1 and r['below_minimum_count']==1
    ratios={x['statement_id']:x['full_repayment_ratio']for x in r['statement_assessments']}
    assert ratios=={'below':.05,'full':1,'min':.1}

def test_conflicting_daily_snapshots_and_receipt_time_rejected():
    rows=[balance(D),dict(balance(D),snapshot_id='other')]
    with pytest.raises(CreditRiskError):revolving(rows)
    with pytest.raises(CreditRiskError):revolving([dict(balance(D),known_at='2024-01-04')])
