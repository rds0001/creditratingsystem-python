"""Dated same-account journey and independent economic identities."""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json
import math
import os
import runpy
import pytest
import creditratingsystem as cr
E=runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/revolving_lifecycle.py'))
CARD='RETAIL_CREDIT_CARD'
def data():return json.loads(files('creditratingsystem.datasets').joinpath('revolving_lifecycle.json').read_text())
@pytest.fixture(scope='module')
def result():return E['run'](product_ids=[CARD],return_artifacts=True)
def test_same_accounts_first_default_cure_redefault_and_economics(result):
 r=result['products'][CARD]
 assert r['application_pd']==[.03]*3 and r['decision_actions']==['APPROVE']*3
 assert [a['target']for a in r['application_targets']]==[0,1,None]
 assert r['ead_forecast']==pytest.approx([500+.5*(1000-500)])
 assert r['observed_ccf']['ccf']==pytest.approx((600-500)/(1000-500))
 assert [w['economic_lgd']for w in r['workout_losses']]==pytest.approx([0,(200-80+10)/200])
 assert r['cure_targets'][0]['cure_at']=='2024-05-20'and r['cure_targets'][0]['redefault_at']=='2024-06-20'
 assert r['episodes']['episodes'][1]['previous_episode_id']==r['episodes']['episodes'][0]['episode_id']
 assert r['validation_metrics']['vintage']['cells'][3]['cumulative']['mean_outcome']==pytest.approx(1/3)
 assert r['validation_metrics']['collections']['overall']['discounted_net']['bias']==pytest.approx((.6*600+.6*200-(600+80-10))/2)
 pool=r['validation_metrics']['retail_pool']['pools'][0]
 assert pool['pd_count_weighted']['observed_rate']==.5 and pool['pd_exposure_weighted']['observed_rate']==1
 assert all(not s['at_risk']for s in r['panel']if s['account_id']=='card-main'and s['as_of']>='2024-04-30')
 assert all(s['at_risk']for s in r['panel']if s['account_id']=='card-zero')
 assert r['phase_details']['delinquency']['dpd']==14 and not r['phase_details']['arrangement']['contract_changed']
 assert not r['phase_details']['action_history']['rows'][0]['accepted']
def test_local_original_seals_and_independent_frozen_reexecution(result):
 raw=result['raw_artifacts']['products'][CARD]
 assert all(g['replay']['status']=='REPRODUCED'for g in raw['governance']['rows'])
 assert cr.replay_consistency(**raw['replay']['input']).value==raw['replay']['validation']
 assert raw['replay']['known_limits']==[1000,600]
 assert cr.collections_backtest(**raw['workout']['validation_input']).value==raw['validation']['collections']
 model=raw['origin']['model'];cal=raw['origin']['calibrator'];snap=raw['origin']['snapshots'][1]
 assert cr.predict_application_risk(snap,json.loads(json.dumps(model)),model['target'],calibrator=json.loads(json.dumps(cal))).value==raw['origin']['risks'][1]
 wrong=deepcopy(model);wrong['application_contract']['product_ids']=['OTHER_PRODUCT']
 with pytest.raises(cr.CreditRiskError):cr.predict_application_risk(snap,wrong,model['target'],calibrator=cal)
 diagnostics=raw['linked_diagnostics'];chain=diagnostics['collections']
 assert chain['cure_targets'][0]['cure_at']=='2024-05-20'and chain['cure_targets'][0]['redefault_at']=='2024-06-20'
 assert chain['remaining_claim']==600 and chain['prior_discounted_recovery']==0
 assert chain['action_value']['expected_net_recovery']==pytest.approx(.25*(7/12*600-5)+.75*120-2)
 assert all(t['cure_label']is None for t in chain['missing_probation_targets'])
 assert diagnostics['lifetime_loss']['lifetime_expected_loss']==pytest.approx(.4*1000*(1-.9**8))
 assert diagnostics['path_loss']['expected_loss']==pytest.approx(diagnostics['lifetime_loss']['lifetime_expected_loss'])
 assert diagnostics['default_count']['overall']['observed_count']==5 and diagnostics['default_count']['overall']['observed_defaults']==1
 assert cr.validation_report(**diagnostics['report_inputs']).value==diagnostics['validation_report']
 assert raw['behavioral_and_loss_models']['model']['training']['known_at']<'2024-01-01'
 assert raw['pools']['model']['training']['known_at']<'2024-01-01'
 assert all(c['missing_evidence']['assessment']=='INSUFFICIENT_EVIDENCE'for c in raw['origin']['capacities'])
def test_raw_ledger_negative_controls_and_late_amendment():
 c=data()['cases'][CARD];c['statements'][0]['balance']+=1
 with pytest.raises(ValueError,match='principal ledger'):E['verify_input_ledgers'](c)
 c=data()['cases'][CARD];c['execution_inputs']['vintage']['account_outcomes'][0]['cumulative_default']=1
 with pytest.raises(ValueError,match='same first event'):E['verify_input_ledgers'](c)
 c=data()['cases'][CARD];c['new_draw_contract']['known_at']='2024-06-02'
 with pytest.raises(ValueError,match='prior known historical amendment'):E['verify_input_ledgers'](c)
@pytest.mark.skipif(os.environ.get('CREDITRATINGSYSTEM_EXTENDED_TESTS')!='true',reason='Full four-product journey opt-in; representative real fits remain in default tests')
def test_all_four_prior_fits_dated_regimes_and_heloc_household():
 r=E['run'](return_artifacts=True)
 expected={'RETAIL_CREDIT_CARD':(.03,750,.65),'RETAIL_CHARGE_CARD':(.02,math.sqrt(400*600)*math.exp((math.log(600/400)/2)**2/2),.525),'RETAIL_OVERDRAFT':(.04,850,.75),'RETAIL_HOME_EQUITY_LINE':(.015,62500,.55)}
 for p,(pd,ead,lgd)in expected.items():
  q=r['products'][p];assert q['application_pd']==[pd]*3;assert q['ead_forecast']==pytest.approx([ead]);assert q['workout_losses'][1]['economic_lgd']==pytest.approx(lgd)
  assert q['cohort_contract_status']=='PASS'and q['replay_status']=='PASS'
 h=r['products']['RETAIL_HOME_EQUITY_LINE']['phase_details']['household']
 assert h['joint']['monthly_income']==5000 and h['joint']['monthly_debt_service']==1000
 assert h['stress_projection']['periods'][0]['residual']==pytest.approx(-1500)
 a=r['raw_artifacts']['products']['RETAIL_HOME_EQUITY_LINE'];assert a['validation']['affordability']['decision']['status']=='PASS'
 assert cr.affordability_stress_backtest(**a['affordability_inputs']).value==a['validation']['affordability']

def test_heloc_closed_claim_lgd_validation():
 r=E['run'](product_ids=['RETAIL_HOME_EQUITY_LINE'],return_artifacts=True)
 a=r['raw_artifacts']['products']['RETAIL_HOME_EQUITY_LINE'];v=a['validation']['lgd']
 assert v['metrics']['overall']['count_weighted']['bias']==pytest.approx(.125)
 assert v['metrics']['overall']['ead_weighted']['bias']==pytest.approx(.2625)
 assert {x['cluster_id']for x in a['workout']['lgd_validation_input']['records']}=={'heloc-main-person-household'}
 assert cr.lgd_backtest(**a['workout']['lgd_validation_input']).value==v
