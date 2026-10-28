from copy import deepcopy
import numpy as np
import pytest
from scipy.special import expit,logit
from creditratingsystem._hybrid_graph import evaluate_model_graph,decompose_hybrid_prediction
from creditratingsystem.core import build_calculation_context,CreditRiskError


def fixture():
    target=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
    ctx=build_calculation_context('2024-01-01','2024-01-01',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing',currency='EUR').value
    pd=dict(unit='probability',target=target,row_ids=['a','b'],as_of=ctx['as_of'],known_at=ctx['known_at']);lgd=deepcopy(pd);lgd['unit']='loss_fraction';lgd['target']['conditioning']='default';ead=deepcopy(lgd);ead['unit']='money';ead['currency']='EUR';el=deepcopy(pd);el['unit']='money';el['currency']='EUR'
    data=dict(p=dict(value=[.1,.2],contract=pd),l=dict(value=[.5,.4],contract=lgd),e=dict(value=[100,200],contract=ead))
    nodes=[dict(node_id=k,operation='input',inputs={},parameters=dict(input_id=k),output_contract=v['contract'])for k,v in data.items()]
    nodes+=[dict(node_id='p_again',operation='input',inputs={},parameters=dict(input_id='p'),output_contract=pd),dict(node_id='adjusted',operation='log_odds_shift',inputs=dict(pd='p'),parameters=dict(shift=[np.log(2),0],effect_id='adverse-case',reference='fixed stress odds ratio'),output_contract=pd),dict(node_id='loss',operation='expected_loss',inputs=dict(pd='adjusted',lgd='l',ead='e'),parameters=dict(loss_assumption='deterministic_conditional_amounts',reference='fixed scenario amounts'),output_contract=el)]
    return dict(graph_id='frozen-loss',graph_version='1',nodes=nodes,output_node='loss'),data,ctx


def test_graph_loss_hand_oracle_cache_and_immutable_inputs():
    graph,data,ctx=fixture();original=deepcopy((graph,data,ctx));r=evaluate_model_graph(graph,data,{},context=ctx)
    assert (graph,data,ctx)==original
    np.testing.assert_allclose(r.value['final']['value'],[100/11,16],atol=1e-14)
    assert r.diagnostics['cache_hits']==1
    assert r.value['node_results']['p']==r.value['node_results']['p_again']
    assert evaluate_model_graph(graph,data,{},context=ctx).value==r.value


def test_decomposition_product_interaction_oracle_and_frozen_replay():
    graph,data,ctx=fixture();graph['nodes'][-1]['inputs']['pd']='p';result=evaluate_model_graph(graph,data,{},context=ctx);base=deepcopy(data);base['p']['value']=[.05,.1];base['l']['value']=[.25,.2];base['e']['value']=[50,100]
    out=decompose_hybrid_prediction(result,base).value
    np.testing.assert_allclose(out['baseline'],[.625,2]);np.testing.assert_allclose(out['actual'],[5,16]);np.testing.assert_allclose([r['contribution']for r in out['input_contributions']],[[.625,2]]*3);np.testing.assert_allclose(out['interaction_remainder'],[2.5,8]);assert out['actual']==out['reconstructed'];assert all(not s['additive_global_contribution']for s in out['steps'])
    graph['output_node']='adjusted';result=evaluate_model_graph(graph,data,{},context=ctx);out=decompose_hybrid_prediction(result,base,decomposition_scale='log_odds').value;np.testing.assert_allclose(out['actual'],logit([.1,.2])+[np.log(2),0]);np.testing.assert_allclose(out['interaction_remainder'],0,atol=1e-15)


def test_graph_rejects_units_horizons_cycles_repeated_effect_and_lookahead():
    graph,data,ctx=fixture()
    bad=deepcopy(graph);bad['nodes'][-1]['inputs']['lgd']='p'
    with pytest.raises(CreditRiskError,match='requires PD'):evaluate_model_graph(bad,data,{},context=ctx)
    bad=deepcopy(data);bad['l']['contract']['target']['horizon']=2
    with pytest.raises(CreditRiskError,match='target differs'):evaluate_model_graph(graph,bad,{},context=ctx)
    bad=deepcopy(graph);bad['nodes'][0]['inputs']={'pd':'adjusted'}
    with pytest.raises(CreditRiskError,match='Acyclic'):evaluate_model_graph(bad,data,{},context=ctx)
    bad=deepcopy(data);bad['p']['effects']=['adverse-case']
    with pytest.raises(CreditRiskError,match='already applied'):evaluate_model_graph(graph,bad,{},context=ctx)
    bad=deepcopy(data);bad['l']['effects']=['adverse-case']
    with pytest.raises(CreditRiskError,match='several loss legs'):evaluate_model_graph(graph,bad,{},context=ctx)


def test_frozen_model_calibration_and_pool_nodes():
    from creditratingsystem import fit_penalized_glm,calibrate_intercept
    graph,data,ctx=fixture();pd=deepcopy(data['p']['contract']);design=deepcopy(pd);design['unit']='design';design['feature_order']=[]
    model=fit_penalized_glm(np.empty((8,0)),[0]*6+[1]*2,feature_order=[],penalty='ridge',lambda_path=[.1],selected_lambda=.1,target=pd['target'],training=dict(data_fingerprint='synthetic',split_id='train',known_at='2020-01-01'),model_id='constant',model_version='1').model
    metadata=dict(target=pd['target'],calibration=dict(start='2020-01-01',end='2020-12-31',known_at='2021-01-01',split_id='cal',prediction_origin='EXTERNAL_TARGET'),base_model_version='1',population_definition='synthetic',outcome_policy='fully observed',weight_type='frequency',data_origin='SYNTHETIC')
    cal=calibrate_intercept([.25,.25],.5,metadata=metadata).value
    nodes=[dict(node_id='design',operation='input',inputs={},parameters=dict(input_id='x'),output_contract=design),dict(node_id='pd',operation='predict_pd',inputs=dict(design='design'),parameters=dict(snapshot_id='model'),output_contract=pd),dict(node_id='cal',operation='apply_calibrator',inputs=dict(pd='pd'),parameters=dict(snapshot_id='cal',base_model_version='1'),output_contract=pd),dict(node_id='pool',operation='combine_pd_signals',inputs=dict(raw='pd',calibrated='cal'),parameters=dict(signal_order=['raw','calibrated'],weights=[1,1],combination_scale='linear',combination_contract=dict(interpretation='estimator_pool',reference='explicit average')),output_contract=pd)]
    graph=dict(graph_id='frozen-models',graph_version='1',nodes=nodes,output_node='pool');snap=dict(model=model,cal=cal);data=dict(x=dict(value=[[],[]],contract=design));result=evaluate_model_graph(graph,data,snap,context=ctx);np.testing.assert_allclose(result.value['final']['value'],[.375,.375],atol=1e-8)
    bad=deepcopy(snap);bad['model']['training']['known_at']='2025-01-01'
    with pytest.raises(CreditRiskError,match='unavailable'):evaluate_model_graph(graph,data,bad,context=ctx)
