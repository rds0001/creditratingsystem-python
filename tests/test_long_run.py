from copy import deepcopy
import pytest
from creditratingsystem._long_run import estimate_long_run_pd
from creditratingsystem.core import CreditRiskError

TARGET=dict(event='default',default_definition='90dpd_or_utp',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='observed_cohorts')
REP=dict(assessment='unknown',rationale='Synthetic test',reference_period='2020-2022')
COHORTS=[dict(cohort_id='a',period_start='2020-01-01',period_end='2021-01-01',known_at='2021-02-01',defaults=1,denominator=10,mature=True,default_definition=TARGET['default_definition']),dict(cohort_id='b',period_start='2021-01-01',period_end='2022-01-01',known_at='2022-02-01',defaults=2,denominator=100,mature=True,default_definition=TARGET['default_definition'])]
def estimate(rows=None,**kw):
    args=dict(selected_cohort_ids=['a','b'],knowledge_date='2023-01-01',weighting='annual_mean',uncertainty='none',dependence_assumption='unspecified',representativeness=REP,target=TARGET)
    args.update(kw)
    return estimate_long_run_pd(COHORTS if rows is None else rows,**args)

def test_distinct_estimands():
    assert estimate().value['pd']==pytest.approx(.06)
    assert estimate(weighting='pooled_obligor').value['pd']==pytest.approx(3/110)
    assert not estimate().diagnostics['automatically_ttc']
    assert estimate().value['lower'] is None

def test_zero_event_exact_interval_analytic_oracle():
    rows=deepcopy(COHORTS)
    for row in rows: row['defaults']=0
    result=estimate(rows,weighting='pooled_obligor',uncertainty='clopper_pearson',dependence_assumption='independent_homogeneous_obligors')
    assert result.value['lower']==0
    assert result.value['upper']==pytest.approx(1-.025**(1/110),abs=1e-14)

def test_year_t_cauchy_oracle():
    # df=1 Student quantile is tan(pi*(q-.5)); standard error=.04.
    import math
    result=estimate(uncertainty='year_t',dependence_assumption='independent_years')
    assert result.value['lower']==0
    assert result.value['upper']==pytest.approx(.06+.04*math.tan(math.pi*.475),rel=1e-10)

@pytest.mark.parametrize('field,value,code',[('mature',False,'IMMATURE_COHORT'),('known_at','2024-01-01','IMMATURE_COHORT'),('defaults',11,'INVALID_COUNTS'),('defaults',.5,'INVALID_COUNTS'),('default_definition','other','TARGET_MISMATCH'),('period_end','2021-01-02','INCOMPATIBLE_HORIZON')])
def test_invalid_selected_cohort(field,value,code):
    rows=deepcopy(COHORTS); rows[0][field]=value
    with pytest.raises(CreditRiskError) as caught: estimate(rows)
    assert caught.value.code==code

def test_explicit_selection_and_assumptions():
    with pytest.raises(CreditRiskError,match='INVALID_COHORT_SELECTION'): estimate(selected_cohort_ids=['a','a'])
    with pytest.raises(CreditRiskError,match='INCOMPATIBLE_UNCERTAINTY'): estimate(uncertainty='clopper_pearson')
    rows=deepcopy(COHORTS); rows[1].update(period_start='2020-06-01',period_end='2021-06-01')
    assert estimate(rows).diagnostics['overlapping_periods']
    with pytest.raises(CreditRiskError,match='INCOMPATIBLE_UNCERTAINTY'): estimate(rows,uncertainty='year_t',dependence_assumption='independent_years')
