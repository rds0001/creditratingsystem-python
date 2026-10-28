from copy import deepcopy
import pytest
from creditratingsystem._validation_retail import validate_retail_cohort
from creditratingsystem.application import build_application_targets
from creditratingsystem.behavioral import build_account_month_panel,build_behavioral_targets


def cohort_fixture():
 applications=[dict(application_id='a'+str(i),customer_id='c'+str(i),product_id='revolving',population_id='through-door',requested_terms=dict(amount=100,currency='EUR'),application_at='2024-01-01',information_cutoff='2024-01-01')for i in range(3)]
 decisions=[dict(decision_id='d'+str(i),application_id='a'+str(i),status='approved'if i<2 else'rejected',decided_at='2024-01-02',known_at='2024-01-02',**(dict(accepted_at='2024-01-02',accepted_known_at='2024-01-02')if i<2 else{}))for i in range(3)]
 bookings=[dict(application_id='a0',facility_id='f0',booked_at='2024-01-03',known_at='2024-01-03',first_draw_at='2024-01-03',first_draw_known_at='2024-01-03',followup_end='2024-03-01',outcome_available_at='2024-03-01')]
 apps=build_application_targets(applications,decisions,bookings,[],[],30,'2024-04-01').value
 accounts=[dict(account_id='f0',customer_id='c0',product_id='revolving',opened_at='2024-01-03',known_at='2024-01-03',currency='EUR')];statements=[dict(statement_id='s',account_id='f0',observed_at='2024-01-31',known_at='2024-01-31',balance=0,credit_limit=100,dpd=0,past_due_amount=0,currency='EUR')];panel=build_account_month_panel(accounts,statements,[],observation_dates=['2024-01-31']).value['accounts'];targets=build_behavioral_targets(panel,[],horizon_months=1,target_event='DEFAULT',competing_events=['CLOSED','PREPAID'],exit_treatment='competing',evaluation_cutoff='2024-04-01',coverage=[dict(account_id='f0',observed_through='2024-03-31',known_at='2024-04-01')]).value
 profile=dict(profile_id='retail',version='1',reference='explicit-contract',known_at='2023-12-01',default_definition='contract-first-default',default_group_rule='facility-only-no-household-contagion',default_unit='facility',first_default=True,exit_treatment='competing',default_event='DEFAULT',chargeoff_event='CHARGEOFF',cure_event='CURED',competing_events=['CLOSED','PREPAID'],risk_set_by_product=dict(revolving=dict(active_states=['ACTIVE','CURED'],include_zero_balance=True)),split_isolation_keys=['customer_id','household_id'],application_horizon_days=30,behavioral_horizon_months=1)
 splits=[dict(customer_id='c'+str(i),household_id='h'+str(i),role='train'if i==0 else'test')for i in range(3)];sources=[dict(source_id='s'+str(i),prediction_key='application:a'+str(i),source_hash='frozen'+str(i),kind='BUREAU',observed_at='2023-12-01',available_at='2023-12-02')for i in range(3)]+[dict(source_id='statement',prediction_key='account:f0@2024-01-31',source_hash='statement',kind='STATEMENT',observed_at='2024-01-31',available_at='2024-01-31')];d=dict(model_id='frozen',model_version='1',parameter_version='1',data_fingerprint='synthetic-retail',population_id='through-door',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2024-04-01',target=dict(event='default',horizon=1,time_unit='months',probability_measure='PHYSICAL',conditioning='performing'))
 return apps,panel,targets,dict(target_profile=profile,split_memberships=splits,source_lineage=sources,design=d)


def test_actual_operational_panels_preserve_selection_and_zero_balance_risk():
 apps,panel,targets,kw=cohort_fixture();r=validate_retail_cohort(apps,panel,targets,**kw).value
 assert r['decision']['status']=='PASS'
 assert r['metrics']['population_flow']['applications']==3 and r['metrics']['population_flow']['accepted_not_funded']==1 and r['metrics']['population_flow']['funded']==1
 assert r['metrics']['maturity_and_risk_coverage']['active_zero_balance_at_risk']==1
 assert r['metrics']['maturity_and_risk_coverage']['behavioral_mature_observed']==1


def test_retail_reject_pseudolabel_household_leakage_and_late_information_reported():
 apps,panel,targets,kw=cohort_fixture();apps[1].update(target=0,label_observed=True,label_origin='NO_HIT');panel[0]['at_risk']=False;kw['split_memberships'][1]['household_id']='h0';kw['source_lineage'][0]['available_at']='2024-01-02'
 r=validate_retail_cohort(apps,panel,targets,**kw).value
 assert r['decision']['status']=='FAIL'
 assert {'UNFUNDED_OWN_CONTRACT_OUTCOME','NON_OUTCOME_LABEL_SOURCE','RISK_SET_CONFLICT','SPLIT_GROUP_OVERLAP','FEATURE_LOOKAHEAD'}<=set(r['violation_counts'])

from creditratingsystem._validation_retail import roll_rate_backtest


def roll_fixture():
 _,_,_,kw=cohort_fixture();d=kw['design'];d['target'].update(horizon=2,conditioning='origin_state');d['evaluation_cutoff']='2024-04-30';states=['A','D','C','CO','X'];P=[[.8,.1,0,.05,.05],[0,.5,.4,.1,0],[.7,.2,0,.05,.05],[0,0,0,1,0],[0,0,0,0,1]];contract=dict(scale_id='retail-state',scale_version='1',reference='explicit-state-episodes',known_at='2023-01-01',default_states=['D','CO'],chargeoff_states=['CO'],cure_states=['C'],exit_states=['X'],absorbing_states=['CO','X']);forecasts=[];panel=[]
 for i,state in enumerate(['D','A','CO']):
  default='default'+str(i)if state in ['D','CO']else None
  forecasts.append(dict(prediction_id='p'+str(i),account_id='a'+str(i),entity_id='c'+str(i),cluster_id='g'+str(i),segment='revolving',as_of='2024-01-31',forecast_known_at='2024-01-31',horizon_end='2024-03-31',model_id='frozen',model_version='1',parameter_version='1',risk_episode_id='risk0',default_episode_id=default,origin_state=state,opening_balance=0 if i==1 else 100,transition_matrix=P))
  panel.append(dict(account_id='a'+str(i),as_of='2024-01-31',known_at='2024-01-31',risk_episode_id='risk0',default_episode_id=default,state=state))
 panel.extend([dict(account_id='a0',as_of='2024-02-29',known_at='2024-02-29',risk_episode_id='risk1',default_episode_id=None,state='C'),dict(account_id='a0',as_of='2024-03-31',known_at='2024-03-31',risk_episode_id='risk1',default_episode_id='new-default',state='D'),dict(account_id='a1',as_of='2024-02-29',known_at='2024-02-29',risk_episode_id='risk0',default_episode_id=None,state='A')])
 return forecasts,panel,dict(states=states,state_contract=contract,design=d,weighting='count',uncertainty='none')


def test_roll_exact_endpoints_absorption_and_real_cure_redefault_episodes():
 forecasts,panel,kw=roll_fixture();r=roll_rate_backtest(forecasts,panel,**kw).value
 assert r['coverage']['included_count']==2 and r['diagnostics']['missing_successor_count']==1
 assert r['metrics']['overall']['brier']==pytest.approx(.21)
 assert r['diagnostics']['endpoint_traces'][0]['cure_transitions']==1 and r['diagnostics']['endpoint_traces'][0]['redefault_transitions']==1
 assert r['diagnostics']['zero_balance_origin_count']==1
 panel[4]['default_episode_id']='default0';r=roll_rate_backtest(forecasts,panel,**kw).value
 assert r['coverage']['included_count']==1
 assert r['diagnostics']['episode_violations'][0]['codes']==['REUSED_DEFAULT_EPISODE_AFTER_CURE']
