"""Local end-to-end content-bound assessment and historical replay oracles."""
from copy import deepcopy
import math
import pytest
from creditratingsystem import rating_system as rs
from creditratingsystem.core import _fingerprint


def ref(source,identity,*path):
    return {'$ref':{'source':source,'id':identity,'path':list(path)}}


def fixture():
    target={'event':'first_default','horizon':1,'time_unit':'years','probability_measure':'PHYSICAL','conditioning':'PIT'}
    model={'schema_version':'1.0','model_id':'M','model_version':'1','method_id':'classification.fit_binary_glm','feature_order':['x'],'parameters':{'link':'logit','fit_intercept':True,'coefficients':[0,1],'offset_required':False},'preprocessing':{'transformation':'identity'},'target':target,'training':{'data_fingerprint':'hand-oracle','split_id':'train','known_at':'2023-12-01'},'diagnostics':{},'provenance':{'data_origin':'SYNTHETIC','parameter_origin':'USER_SUPPLIED'}}
    rule={'rule_id':'all','profile_id':'consumer','profile_version':'1','priority':1,'effective_from':'2023-01-01','effective_to':None,'known_at':'2023-01-01','conditions':[{'field':'income','operator':'ge','value':1000}]}
    contract={'contract_id':'consumer','version':'1','required_features':['income'],'allowed_values':{},'support_bounds':{'income':[1000,10000]},'maximum_age_days':{'income':60}}
    pdstep={'step_id':'pd','method_id':'classification.predict_pd','arguments':{'fitted_model':ref('model','primary'),'X':ref('input','features','value'),'feature_order':['x'],'expected_target':target},'argument_types':{},'output_contract':{'shape':'vector','unit':'probability','target':target}}
    loss={'step_id':'el','method_id':'losses.expected_loss','arguments':{'pd':ref('step','pd','value',0),'lgd':.4,'ead':100,'context':{'currency':'EUR'}},'argument_types':{},'output_contract':{'shape':'scalar','unit':'EUR','target':target}}
    component={'component_id':'pd','quantity':'PD','value':ref('step','pd','value',0),'subject_id':'O','subject_type':'obligor','variant':'base','definition':target}
    recipe={'method':'assessment_pipeline','version':'2.0','parameters':{'routing_rules':[rule],'applicability_contract':contract,'steps':[loss,pdstep],'components':[component,dict(component,component_id='el',quantity='EL',value=ref('step','el','value'),currency='EUR')],'effects':[],'transitions':[{'from_state':'DRAFT','to_state':'REVIEWED','event_type':'review','required_gates':['reviewed']},{'from_state':'REVIEWED','to_state':'FINAL','event_type':'finalize','required_gates':['approved']}],'event_dependencies':[{'assessment_id':'A','dependency_type':'model','dependency_id':'M'}],'decision':None}}
    shock={'kind':'innovations','content':{'draw_ids':['d1'],'values':[0.]},'known_at':'2023-12-01'};shock['sha256']=_fingerprint(shock['content'])
    bundle={'model':model,'dependencies':{'artifacts':{'draws':shock}},'configuration':{'assessment_recipe':recipe}}
    registry=rs.register_rating_model_version(None,bundle,{'model_id':'M','model_version':'1','registered_at':'2023-12-15'}).value
    event={'event_id':'update','revision':'1','occurred_at':'2024-01-03','known_at':'2024-01-03','dependencies':[{'type':'model','id':'M'}],'payload':{'note':'review flag'}}
    correction=dict(event,revision='2',corrects_revision='1',occurred_at='2024-01-04',known_at='2024-01-04',payload={'note':'corrected review flag'})
    inputs={'snapshot':{'subject_id':'O','subject_type':'obligor','as_of':'2024-01-01','known_at':'2024-01-02','attributes':{'income':{'value':2000,'observed_at':'2023-12-31','known_at':'2024-01-01'}}},'datasets':{'features':{'value':[[0]],'as_of':'2024-01-01','known_at':'2024-01-02'}},'assessment_id':'A','assessment_known_at':'2024-01-02','calculated_at':'2024-01-02','lifecycle_events':[{'event_id':'review','type':'review','occurred_at':'2024-01-03','known_at':'2024-01-03','reason':'reviewed','gates':{'reviewed':True}},{'event_id':'final','type':'finalize','occurred_at':'2024-01-04','known_at':'2024-01-04','reason':'finalized locally','gates':{'approved':True}}],'events':[event,event,correction]}
    request={'row_id':'one','known_at':'2024-01-05','model_id':'M','model_version':'1','model_hash':registry['entries'][0]['model_hash'],'recipe':recipe,'input':inputs}
    return request,registry


def repin(request,registry):
    bundle=deepcopy(registry['entries'][0]['bundle']);bundle['configuration']['assessment_recipe']=request['recipe']
    registry=rs.register_rating_model_version(None,bundle,registry['entries'][0]['registration']).value
    request['model_hash']=registry['entries'][0]['model_hash']
    return request,registry


def run(request,registry):
    return rs.rate_subject_batch([request],registry,'batch','2024-01-06').value['rows'][0]


def test_pipeline_hand_oracle_dag_lifecycle_revision_idempotency_and_replay():
    request,registry=fixture();row=run(request,registry)
    assert row['status']=='OK',row
    value=row['value'];a=value['assessment']
    assert a['state']=='FINAL'
    assert [c['value'] for c in a['components']]==[20.,.5]
    assert [s['step_id'] for s in value['steps']]==['pd','el']
    assert [e['status'] for e in value['event_trace']]==['APPLIED','DUPLICATE','APPLIED']
    assert len(value['event_state']['processed'])==2
    assert rs.replay_assessment_history([request],registry,[row],'2024-01-06').value['status']=='REPRODUCED'
    later=deepcopy(registry['entries'][0]['bundle']);later['model']['model_version']='2';later['model']['parameters']['coefficients']=[4,2]
    registry2=rs.register_rating_model_version(registry,later,{'model_id':'M','model_version':'2','registered_at':'2024-06-01'}).value
    assert rs.replay_assessment_history([request],registry2,[row],'2025-01-01').value['status']=='REPRODUCED'
    tampered=deepcopy(row);tampered['event_state_hash']='not-recorded'
    assert rs.replay_assessment_history([request],registry,[tampered],'2024-01-06').value['findings'][0]['fields']==['event_state_hash']
    changed=deepcopy(request);changed['input']['datasets']['features']['value']=[[1]]
    assert rs.replay_assessment_history([changed],registry,[row],'2024-01-06').value['status']=='NOT_REPRODUCIBLE'


@pytest.mark.parametrize('mutation,code',[
 ('future_data','RATING_REQUEST_UNAVAILABLE'),('future_model','RATING_MISSING_MODEL'),('future_event','RATING_REQUEST_UNAVAILABLE'),('gate','RATING_GATE'),('cycle','RATING_RECIPE'),('unknown','RATING_RECIPE'),('target','RATING_OUTPUT'),('unit','RATING_OUTPUT'),('unbound_recipe','RATING_INTEGRITY'),('fit','RATING_RECIPE')])
def test_pipeline_explicit_rejections(mutation,code):
    request,registry=fixture()
    if mutation=='future_data':request['input']['datasets']['features']['known_at']='2024-01-03'
    if mutation=='future_event':request['input']['lifecycle_events'][1]['known_at']='2024-01-07'
    if mutation=='future_model':request['known_at']='2023-12-10'
    if mutation=='gate':request['input']['lifecycle_events'][1]['gates']['approved']=False
    if mutation=='cycle':request['recipe']['parameters']['steps'][1]['arguments']['X']=ref('step','el','value')
    if mutation=='unknown':request['recipe']['parameters']['steps'][0]['method_id']='os.system'
    if mutation=='target':request['recipe']['parameters']['steps'][1]['output_contract']['target']=dict(request['recipe']['parameters']['steps'][1]['output_contract']['target'],probability_measure='RISK_NEUTRAL')
    if mutation=='unit':request['recipe']['parameters']['steps'][1]['output_contract']['unit']='money'
    if mutation=='unbound_recipe':request['recipe']['parameters']['effects']=[{'bad':True}]
    if mutation=='fit':request['recipe']['parameters']['steps'][1]['method_id']='classification.fit_binary_glm'
    if mutation in ('cycle','unknown','target','unit','fit'):request,registry=repin(request,registry)
    assert run(request,registry)['error_code']==code


def test_pipeline_permutation_and_separate_row_errors():
    request,registry=fixture();second=deepcopy(request);second['row_id']='two';second['input']['datasets']['features']['value']=[[1]]
    bad=deepcopy(request);bad['row_id']='bad';bad['model_hash']='tampered'
    a=rs.rate_subject_batch([request,second,bad],registry,'batch','2024-01-06').value
    b=rs.rate_subject_batch([bad,second,request],registry,'batch','2024-01-06').value
    assert a==b and a['rows'][0]['status']=='ERROR'
    assert a['rows'][2]['value']['assessment']['components'][1]['value']==pytest.approx(1/(1+math.exp(-1)))


def test_pipeline_frozen_development_artifact_is_consumed_without_refitting():
    request,registry=fixture();step={'step_id':'fit','method_id':'classification.fit_binary_glm','arguments':{'artifact':ref('artifact','fitted')},'argument_types':{},'output_contract':{'shape':'object','unit':'artifact','target':registry['entries'][0]['bundle']['model']['target']}}
    request['recipe']['parameters']['steps'].append(step)
    model=registry['entries'][0]['bundle']['model'];bundle=deepcopy(registry['entries'][0]['bundle']);bundle['dependencies']['artifacts']['fitted']={'kind':'model','content':model,'sha256':_fingerprint(model),'known_at':'2023-12-01'}
    bundle['configuration']['assessment_recipe']=request['recipe'];registry=rs.register_rating_model_version(None,bundle,registry['entries'][0]['registration']).value;request['model_hash']=registry['entries'][0]['model_hash']
    row=run(request,registry);assert row['status']=='OK',row
    assert row['value']['steps'][0]['stage']=='PRE_FITTED'


def test_pipeline_structural_measure_cannot_be_relabelled_physical():
    request,registry=fixture();step=request['recipe']['parameters']['steps'][1]
    step.update(method_id='structural.merton_default_probability',arguments={'asset_value':100.,'debt_face':80.,'asset_volatility':.2,'drift_or_rate':.03,'horizon':1.,'measure':'RISK_NEUTRAL'})
    step['output_contract']['shape']='scalar'
    request,registry=repin(request,registry)
    assert run(request,registry)['error_code']=='RATING_OUTPUT'


def test_lifecycle_duplicate_idempotency_and_corrupt_registry_replay_findings():
    request,registry=fixture();original=run(request,registry)
    request['input']['lifecycle_events'].insert(1,deepcopy(request['input']['lifecycle_events'][0]))
    row=run(request,registry)
    assert row['status']=='OK' and len(row['value']['assessment']['history'])==2
    assert row['value']['assessment']['state']=='FINAL'
    registry['entries'][0]['bundle']['model']['parameters']['coefficients'][0]=9
    result=rs.replay_assessment_history([request],registry,[original],'2024-01-06').value
    assert result['status']=='NOT_REPRODUCIBLE' and result['findings'][0]['code']=='RATING_INTEGRITY'


def test_typed_matrix_arguments_and_no_mixed_type_coercion():
    request,registry=fixture();request['recipe']['parameters']['steps'][1]['argument_types']={'X':'numeric_matrix','feature_order':'character_vector'}
    request,registry=repin(request,registry);assert run(request,registry)['status']=='OK'
    request['input']['datasets']['features']['value']=[['category']]
    assert run(request,registry)['error_code']=='RATING_ARGUMENT'


def test_simulation_pipeline_preserves_shared_paths_and_actual_cashflow_loss():
    import runpy
    from pathlib import Path
    from creditratingsystem._simulation_transactions import simulate_transactions
    fixtures=runpy.run_path(str(Path(__file__).with_name('test_simulation_transactions.py')))
    args=fixtures['fixture']();simulation=simulate_transactions(*args).value
    target=simulation['transactions'][0]['risk']['target']
    request,registry=fixture();bundle=deepcopy(registry['entries'][0]['bundle'])
    names=['transaction_recipes','scenario_cube','entity_innovations','parameter_snapshot']
    for name,content in zip(names,args):bundle['dependencies']['artifacts'][name]={'kind':'innovations' if name=='entity_innovations' else 'parameters','content':content,'sha256':_fingerprint(content),'known_at':'2023-12-01'}
    path=['transactions',0,'risk','expected_loss']
    request['recipe']['parameters']['steps']=[{'step_id':'cashflow','method_id':'simulation.simulate_transactions','arguments':{name:ref('artifact',name) for name in names},'argument_types':{},'output_contract':{'shape':'object','unit':'transaction_paths','target':None,'fields':[{'path':path,'unit':'EUR','target':target,'target_path':['transactions',0,'risk','target']}]}}]
    request['recipe']['parameters']['components']=[{'component_id':'sim-el','quantity':'EL','value':ref('step','cashflow','value',*path),'subject_id':'O','subject_type':'obligor','variant':'simulated','definition':target,'currency':'EUR'}]
    bundle['configuration']['assessment_recipe']=request['recipe'];registry=rs.register_rating_model_version(None,bundle,registry['entries'][0]['registration']).value;request['model_hash']=registry['entries'][0]['model_hash']
    row=run(request,registry);assert row['status']=='OK',row
    assert row['value']['assessment']['components'][0]['value']==20
    risk=row['value']['steps'][0]['value']['transactions'][0]['risk'];assert risk['pd']==1 and risk['conditional_mean_lgd']==.2
    assert row['value']['steps'][0]['value']['sampling_contract']['conditioning_scope']=='PIT'
    assert rs.replay_assessment_history([request],registry,[row],'2024-01-06').value['status']=='REPRODUCED'


def test_separate_pinned_policy_decision_cannot_change_risk():
    import runpy
    from pathlib import Path
    from creditratingsystem.decisioning import compile_credit_decision_policy
    fixtures=runpy.run_path(str(Path(__file__).with_name('test_decisioning.py')))
    request,registry=fixture();target=registry['entries'][0]['bundle']['model']['target'];spec=fixtures['policy_spec']();spec['target']=target
    policy=compile_credit_decision_policy(spec).value
    bundle=deepcopy(registry['entries'][0]['bundle']);bundle['dependencies']['artifacts']['policy']={'kind':'decision_policy','content':policy,'sha256':_fingerprint(policy),'known_at':'2024-01-01'}
    timing={'as_of':'2024-01-01','known_at':'2024-01-02'}
    request['recipe']['parameters']['decision']={'policy':ref('artifact','policy'),'risk':dict(timing,pd=ref('step','pd','value',0),target=target,freshness='CURRENT'),'affordability':dict(timing,status='AFFORDABLE'),'eligibility':dict(timing,status='ELIGIBLE'),'operations':dict(timing,status='CLEAR'),'as_of':'2024-01-02','known_at':'2024-01-02'}
    bundle['configuration']['assessment_recipe']=request['recipe'];registry=rs.register_rating_model_version(None,bundle,registry['entries'][0]['registration']).value;request['model_hash']=registry['entries'][0]['model_hash']
    row=run(request,registry);assert row['status']=='OK',row
    assert row['value']['decision']['action']=='DECLINE'
    assert row['value']['assessment']['components'][1]['value']==.5
    assert row['decision_hash']==_fingerprint(row['value']['decision'])
    for mutation,code in [('unknown_keyword','RATING_ARGUMENT'),('missing_keyword','RATING_ARGUMENT'),('scalar_arguments','RATING_ARGUMENT'),('scalar_risk','RATING_DECISION')]:
        bad=deepcopy(request);args=bad['recipe']['parameters']['decision']
        if mutation=='unknown_keyword':args['compiled_policy']=args.pop('policy')
        elif mutation=='missing_keyword':args.pop('operations')
        elif mutation=='scalar_arguments':bad['recipe']['parameters']['decision']=42
        else:args['risk']=42
        bad,bad_registry=repin(bad,registry)
        rejected=run(bad,bad_registry)
        assert rejected['status']=='ERROR' and rejected['error_code']==code


def test_frozen_pd_shapley_adapter_uses_bound_model_without_user_callbacks():
    request,registry=fixture();model=registry['entries'][0]['bundle']['model']
    contract={key:model[key] for key in ('model_id','model_version','feature_order','target')};contract.update(output_scale='probability',deterministic_frozen=True)
    step={'step_id':'explanation','method_id':'explainability.estimate_credit_shapley_values','arguments':{'predictor':{'kind':'FROZEN_PD','model':ref('model','primary'),'feature_order':['x'],'expected_target':model['target']},'record':[1.], 'background':{'values':[[0.]],'weights':[1.]},'model_contract':contract,'feature_groups':{'income':['x']},'masking_contract':{'masking':'INTERVENTIONAL_EMPIRICAL','reference_id':'zero','reference_version':'1'}},'argument_types':{'record':'numeric_vector'},'output_contract':{'shape':'object','unit':'probability','target':model['target']}}
    request['recipe']['parameters']['steps'].append(step);request,registry=repin(request,registry)
    row=run(request,registry);assert row['status']=='OK',row
    explained=next(s['value'] for s in row['value']['steps'] if s['step_id']=='explanation')
    assert explained['baseline']==.5
    assert explained['values']==pytest.approx([1/(1+math.exp(-1))-.5])
    step=request['recipe']['parameters']['steps'][-1];step['arguments']['predictor']='eval untrusted code'
    request,registry=repin(request,registry);assert run(request,registry)['error_code']=='RATING_RECIPE'


def test_native_json_integral_reference_indices_preserve_historical_seals():
    request,registry=fixture();baseline=run(request,registry)
    def native_numbers(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key=='path':value[key]=[float(v) if isinstance(v,int) and not isinstance(v,bool) else v for v in item]
                else:native_numbers(item)
        elif isinstance(value,list):
            for item in value:native_numbers(item)
    native_numbers(request)
    request,registry=repin(request,registry)
    assert run(request,registry)==baseline
    from creditratingsystem._rating_execution import _path
    from creditratingsystem.core import CreditRiskError
    for key in [True,False,.5,-1.,float('nan'),float('inf'),'0']:
        with pytest.raises(CreditRiskError):_path([20],[key])
    assert _path({'0':20},['0'])==20
