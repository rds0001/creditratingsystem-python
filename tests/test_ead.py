"""Independent balance and first-default timing oracles."""
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.ead import exposure_schedule,realized_ccf


def schedule_case(timing='AFTER_PERIOD_END_EVENTS'):
    return dict(contract=dict(facility_id='F',currency='EUR',as_of='2024-01-01',drawn_principal=1000,credit_limit=2000,accrued_interest=0),
        events=[dict(event_id='draw',event_date='2024-01-16',known_at='2024-01-01',sequence=0,kind='DRAW',amount=500),
                dict(event_id='pay',event_date='2024-02-01',known_at='2024-01-01',sequence=0,kind='PRINCIPAL_REPAYMENT',amount=500)],
        annual_interest_rates=[.365,.365],period_end_dates=['2024-02-01','2024-03-01'],day_count='ACT/365F',default_timing=timing,allow_overlimit=False,currency='EUR')


def test_exact_between_event_interest_and_leap_day():
    v=exposure_schedule(**schedule_case()).value['schedule']
    # 15 days at principal1000, then16 at1500:15+24=39.
    assert v[0]['interest_accrued']==pytest.approx(39)
    assert v[0]['exposure_before_period_end_events']==pytest.approx(1539)
    assert v[0]['conditional_ead']==pytest.approx(1039)
    assert v[0]['undrawn_commitment']==1000
    # 2024 has29 days in February; interest remains noncapitalized.
    assert v[1]['interest_accrued']==pytest.approx(29)
    assert v[1]['conditional_ead']==pytest.approx(1068)
    assert exposure_schedule(**schedule_case('BEFORE_PERIOD_END_EVENTS')).value['schedule'][0]['conditional_ead']==pytest.approx(1539)


def test_bullet_repayment_before_vs_after_default_and_limit_cut():
    a=schedule_case();a['annual_interest_rates']=[0,0];a['events']=a['events'][1:];a['events'][0]['amount']=1000
    assert exposure_schedule(**a).value['schedule'][0]['conditional_ead']==0
    a['default_timing']='BEFORE_PERIOD_END_EVENTS'
    assert exposure_schedule(**a).value['schedule'][0]['conditional_ead']==1000
    a=schedule_case();a['events'].append(dict(event_id='limit',event_date='2024-02-10',known_at='2024-01-01',sequence=0,kind='LIMIT_CHANGE',amount=900))
    r=exposure_schedule(**a).value['schedule'][1]
    assert r['overlimit_amount']==100 and r['undrawn_commitment']==0


def test_explicit_draw_limits_knowledge_and_simultaneous_sequence():
    a=schedule_case();a['contract']['credit_limit']=1200
    with pytest.raises(CreditRiskError,match='EXPOSURE_LIMIT'):exposure_schedule(**a)
    a['allow_overlimit']=True;assert exposure_schedule(**a).value['event_ledger'][0]['principal_after']==1500
    a=schedule_case();a['events'][0]['known_at']='2024-01-02'
    with pytest.raises(CreditRiskError,match='EXPOSURE_DATE'):exposure_schedule(**a)
    a=schedule_case();a['events'].append(dict(a['events'][0],event_id='duplicate-time'))
    with pytest.raises(CreditRiskError,match='EXPOSURE_AMBIGUITY'):exposure_schedule(**a)


def ccf_dates():return dict(reference_date='2023-12-01',default_date='2024-01-01',reference_known_at='2023-12-01',default_known_at='2024-01-02',evaluation_cutoff='2024-01-31',currency='EUR')


def test_raw_ccf_and_undefined_or_small_denominator():
    assert realized_ccf(40,100,70,**ccf_dates()).value['ccf']==.5
    assert realized_ccf(40,100,10,**ccf_dates()).value['ccf']==-.5
    assert realized_ccf(40,100,130,**ccf_dates()).value['ccf']==1.5
    for drawn in [100,110]:
        v=realized_ccf(drawn,100,110,**ccf_dates()).value
        assert v['ccf'] is None and v['denominator_status']=='NO_POSITIVE_UNDRAWN'
    v=realized_ccf(99,100,110,minimum_undrawn=1,**ccf_dates()).value
    assert v['ccf'] is None and v['denominator_status']=='BELOW_MINIMUM_UNDRAWN'
    a=ccf_dates();a['reference_known_at']='2023-12-02'
    with pytest.raises(CreditRiskError,match='EXPOSURE_DATE'):realized_ccf(40,100,70,**a)
