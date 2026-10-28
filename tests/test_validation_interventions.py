from copy import deepcopy
import pytest
from creditratingsystem._validation_interventions import override_monitoring
from creditratingsystem.core import CreditRiskError
from test_validation_benchmarks import fixture


def test_override_empty_identical_and_real_replayed_changes():
 entries,d=fixture(identical=True);policy=dict(approved_authorities=['committee'],reference='approved-policy',concentration_threshold=.6)
 r=override_monitoring(entries['reference'],entries['candidate'],[],policy=policy,design=d,metrics=['brier','auc'],uncertainty='none').value
 assert r['metrics']['override_rate']==0
 assert r['metrics']['paired_performance']['paired_differences']['candidate']['brier']['difference']==0
 entries,d=fixture();ledger=[dict(override_id='o'+str(i),case_id='c'+str(i),reason='evidence',authority='committee',applied_at='2019-12-30',known_at='2019-12-30',expires_at='2020-02-01',from_pd=p,to_pd=q)for i,(p,q)in enumerate(zip([.1,.8,.3,.6],[.2,.7,.4,.5]))]
 first=ledger[0];first['to_pd']=.15;ledger.append(dict(first,override_id='o4',applied_at='2019-12-31',known_at='2019-12-31',from_pd=.15,to_pd=.2));expired=dict(ledger[1],override_id='expired',applied_at='2019-01-01',known_at='2019-01-01',expires_at='2019-12-01');ledger.append(expired)
 r=override_monitoring(entries['reference'],entries['candidate'],ledger,policy=policy,design=d,metrics=['brier'],uncertainty='none').value
 assert r['metrics']['override_rate']==1 and r['metrics']['pd_increase_cases']==2 and r['metrics']['pd_decrease_cases']==2
 assert r['diagnostics']['multiple_active_case_ids']==['c0'] and r['diagnostics']['expired_override_ids']==['expired']
 assert r['metrics']['concentration']['authority']['flag']
 assert r['metrics']['paired_performance']['paired_differences']['candidate']['brier']['difference']==pytest.approx(.06)
 assert not r['diagnostics']['causal_effect_identified']
 ledger[0]['from_pd']=.7
 with pytest.raises(CreditRiskError):override_monitoring(entries['reference'],entries['candidate'],ledger,policy=policy,design=d,metrics=['brier'],uncertainty='none')

from creditratingsystem._validation_interventions import support_transfer_backtest


def support_fixture():
 _,d=fixture();d['currency']='EUR';names=['entity','transfer','joint','final','counterfactual','guarantee'];contract={s:dict(kind='amount'if s=='guarantee'else'binary',observable=s!='counterfactual',event_definition=s+'-definition',reference='staged-events')for s in names};contract['guarantee']['currency']='EUR';union=dict(component_a='entity',component_b='transfer',joint='joint',union='final')
 rows=[]
 for i in range(2):
  outcomes={s:dict(value=(20 if i==0 else 0)if s=='guarantee'else 1-i,known_at='2020-02-01',complete_through='2020-01-31',event_ids=['shared'+str(i)]if i==0 else[],event_times=['2020-01-15']if i==0 else[])for s in names if s!='counterfactual'}
  rows.append(dict(prediction_id='p'+str(i),entity_id='e'+str(i),cluster_id='g'+str(i),segment='all',as_of='2020-01-01',forecast_known_at='2020-01-01',horizon_end='2020-01-31',model_id='reference',model_version='1',parameter_version='1',stage_predictions=dict(entity=.2,transfer=.1,joint=.05,final=.25,counterfactual=.3,guarantee=30),stage_outcomes=outcomes))
 return rows,dict(stage_contract=contract,union_contract=union,design=d,uncertainty='none')


def test_support_union_deduplicated_and_counterfactual_not_inferred():
 rows,kw=support_fixture();r=support_transfer_backtest(rows,**kw).value
 assert r['metrics']['by_stage']['final']['brier']==.3125
 assert r['metrics']['by_stage']['guarantee']['bias']==20
 assert r['metrics']['by_stage']['counterfactual']['applicability']=='NOT_APPLICABLE'
 assert r['diagnostics']['event_reconciliation'][0]['distinct_component_event_count']==1
 assert r['diagnostics']['event_reconciliation'][1]['distinct_component_event_count']==0
 kw.update(uncertainty='cluster_bootstrap',cluster_resamples=[['g0','g0'],['g1','g1']]);r=support_transfer_backtest(rows,**kw).value
 assert r['uncertainty']['intervals']['guarantee/bias']['lower']==pytest.approx(10.5)
 rows[0]['stage_predictions']['final']=.3
 with pytest.raises(CreditRiskError):support_transfer_backtest(rows,**kw)
 rows[0]['stage_predictions']['final']=.25;rows[0]['stage_outcomes']['final']['value']=0;rows[0]['stage_outcomes']['final']['event_ids']=[];rows[0]['stage_outcomes']['final']['event_times']=[]
 with pytest.raises(CreditRiskError):support_transfer_backtest(rows,**kw)
