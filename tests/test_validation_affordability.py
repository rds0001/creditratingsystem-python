from copy import deepcopy
import pytest
from creditratingsystem.affordability import stress_affordability
from creditratingsystem._validation_affordability import affordability_stress_backtest
from creditratingsystem.core import CreditRiskError


def fixture():
 b=dict(currency='EUR',as_of='2024-01-01',liquid_buffer=100,cltv_scope='not_applicable',incomes=[dict(economic_id='salary',currency='EUR',monthly_amount=200,basis='net')],expenses=[dict(economic_id='living',currency='EUR',monthly_amount=50)],contracts=[dict(economic_id='loan',currency='EUR',principal=1000,remaining_months=12,annual_rate=.12,repayment='interest_only',rate_type='reset',balloon_amount=1000,reset_dates=['2024-02-01'],rate_floor=0,rate_cap=1)],collateral=[],thresholds=dict(minimum_residual=0,maximum_dsti=.5,maximum_cltv=1))
 s=dict(schema_version='1.0',scenario_id='rate-shock',periods=[dict(date='2024-02-01',fx_to_reporting={'EUR':1},income_factors={'salary':1},expense_factors={'living':1},collateral_factors={},rate_shifts={'loan':.12})]);evidence=[];alloc=[]
 for group in ['incomes','expenses','contracts']:
  item=b[group][0];eid=item['economic_id'];terms={k:v for k,v in item.items()if k!='economic_id'};proof=dict(evidence_id=eid,economic_id=eid,kind=group,reference='verified-source',known_at='2023-12-01',observed_at='2023-12-01',verified=True,terms=terms)
  if group=='incomes':proof.update(source_basis='gross',monthly_gross_amount=250,monthly_tax_and_deductions=50,normalization_reference='verified-net-payroll')
  evidence.append(proof);alloc.append(dict(allocation_id=eid,economic_id=eid,household_id='h',as_of='2024-01-01',known_at='2023-12-01',share=1,mode='FULL_RECOURSE'if group=='contracts'else'PROPORTIONAL'))
 row=dict(prediction_id='f',entity_id='h',cluster_id='h',segment='retail',as_of='2024-01-01',forecast_known_at='2024-01-01',horizon_end='2024-02-01',model_id='budget',model_version='1',parameter_version='1',household_id='h',baseline=b,scenario=s,assessment=stress_affordability(b,s).value,evidence_bindings={e['economic_id']:e['evidence_id']for e in evidence},expected_economic_ids={g:[v['economic_id']for v in b[g]]for g in ['incomes','expenses','contracts','collateral']},scenario_known_at='2024-01-01',profile_known_at='2023-12-01',profile_version='1')
 d=dict(model_id='budget',model_version='1',parameter_version='1',data_fingerprint='verified-household',population_id='households',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2024-02-02',currency='EUR',target=dict(event='default',horizon=1,time_unit='months',probability_measure='PHYSICAL',conditioning='household_budget'))
 payments=[dict(prediction_id='f',period_date='2024-02-01',known_at='2024-02-02',observed_through='2024-02-01',complete=True,currency='EUR',scheduled_payment=10,paid_amount=8)]
 return dict(cases=[row],economic_evidence=evidence,allocations=alloc,design=d,payment_outcomes=payments)


def test_contract_reset_gross_net_and_factual_payment_separation():
 kw=fixture();r=affordability_stress_backtest(**kw).value
 assert r['decision']['status']=='PASS'and r['metrics']['verified_projection_cases']==1
 p=r['projection_traces'][0]['recomputed_assessment']['periods'][0]
 assert p['debt_service']==20 and p['residual']==130
 assert r['metrics']['factual_scheduled_minus_paid']['bias']==2
 assert not r['scenario_projection_is_default_probability']and not r['factual_payments_are_scenario_realization']
 kw['cases'][0]['baseline']['contracts'][0].update(rate_type='fixed',reset_dates=[]);kw['economic_evidence'][2]['terms'].update(rate_type='fixed',reset_dates=[]);kw['cases'][0]['assessment']=stress_affordability(kw['cases'][0]['baseline'],kw['cases'][0]['scenario']).value
 assert affordability_stress_backtest(**kw).value['projection_traces'][0]['recomputed_assessment']['periods'][0]['residual']==140


def test_shared_income_unknown_income_and_gross_net_never_silent_zero():
 kw=fixture();duplicate=deepcopy(kw['allocations'][0]);duplicate.update(allocation_id='other-household',household_id='other');kw['allocations'].append(duplicate);r=affordability_stress_backtest(**kw).value
 assert r['violation_counts']['SHARED_ECONOMIC_DOUBLE_COUNT']==1 and r['metrics']['verified_projection_cases']==0
 kw=fixture();kw['economic_evidence'][0]['verified']=False;r=affordability_stress_backtest(**kw).value
 assert r['decision']['status']=='BLOCKED'and r['violation_counts']['INCOMPLETE_ECONOMIC_EVIDENCE']==1
 kw=fixture();kw['economic_evidence'][0]['monthly_tax_and_deductions']=40;r=affordability_stress_backtest(**kw).value
 assert r['violation_counts']['GROSS_NET_RECONCILIATION']==1 and r['metrics']['verified_projection_cases']==0


def test_frozen_assessment_late_policy_and_incomplete_actual_payment():
 kw=fixture();kw['cases'][0]['assessment']['periods'][0]['residual']=131;kw['payment_outcomes'][0]['complete']=False;r=affordability_stress_backtest(**kw).value
 assert r['violation_counts']['FROZEN_ASSESSMENT_RECONCILIATION']==1
 assert r['metrics']['factual_scheduled_minus_paid']['applicability']=='NOT_APPLICABLE'
 kw=fixture();kw['cases'][0]['profile_known_at']='2024-02-01'
 with pytest.raises(CreditRiskError):affordability_stress_backtest(**kw)
