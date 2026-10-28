import os
import numpy as np
import pytest
from creditratingsystem import predict_pd
from scipy.special import expit
from creditratingsystem._classification_neural import fit_neural_pd,fit_sequence_pd,_nn_mlp,_nn_recurrent,_neural_raw,_nn_loss
T=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing');M=dict(target=T,training=dict(data_fingerprint='neural-analytic',split_id='train',known_at='2025-01-01',data_origin='SYNTHETIC'),model_id='neural',model_version='1')
def optimizer(n,epochs=30):return dict(algorithm='adam',learning_rate=.03,beta1=.9,beta2=.999,epsilon=1e-8,l2=.001,batch_size=n,epoch_orders=[list(range(n)) for _ in range(epochs)])
def rw(variant,p=1,h=2):
 rng=np.random.default_rng(38);g=3 if variant=='gru' else 4
 return {k:rng.normal(scale=.2,size=s) for k,s in dict(input_weight=(p,g*h),hidden_weight=(h,g*h),input_bias=(g*h,),hidden_bias=(g*h,),output_weight=(h,1),output_bias=(1,)).items()}
@pytest.mark.parametrize('variant',['gru','lstm'])
@pytest.mark.skipif(os.environ.get('CREDITRATINGSYSTEM_EXTENDED_TESTS', '').lower() != 'true', reason='Extended independent autograd oracle')
def test_recurrent_full_gated_bptt_matches_optional_torch_autograd(variant):
 torch=pytest.importorskip("torch",reason="Independent optional autograd oracle; finite differences remain mandatory")
 torch.set_num_threads(1);w=rw(variant);arch=dict(kind='recurrent',variant=variant,hidden_size=2);x=np.array([[[.2],[.7]],[[-.3],[.5]]]);y=np.array([0.,1.]);eta,grad=_nn_recurrent(w,arch,list(x),y)
 net=(torch.nn.GRU if variant=='gru' else torch.nn.LSTM)(1,2,batch_first=True).double()
 with torch.no_grad():
  net.weight_ih_l0.copy_(torch.tensor(w['input_weight'].T));net.weight_hh_l0.copy_(torch.tensor(w['hidden_weight'].T));net.bias_ih_l0.copy_(torch.tensor(w['input_bias']));net.bias_hh_l0.copy_(torch.tensor(w['hidden_bias']))
 out,_=net(torch.tensor(x));ow=torch.tensor(w['output_weight'],requires_grad=True);ob=torch.tensor(w['output_bias'],requires_grad=True);score=(out[:,-1,:]@ow+ob)[:,0];loss=torch.nn.functional.binary_cross_entropy_with_logits(score,torch.tensor(y));loss.backward()
 assert eta==pytest.approx(score.detach().numpy(),abs=1e-14)
 expected=dict(input_weight=net.weight_ih_l0.grad.numpy().T,hidden_weight=net.weight_hh_l0.grad.numpy().T,input_bias=net.bias_ih_l0.grad.numpy(),hidden_bias=net.bias_hh_l0.grad.numpy(),output_weight=ow.grad.numpy(),output_bias=ob.grad.numpy())
 for key in w:
  assert grad[key]==pytest.approx(expected[key],abs=1e-13)

def test_mlp_fixed_forward_gradient_and_actual_training():
 w=dict(layer0_weight=np.array([[.2,-.1]]),layer0_bias=np.array([.3,-.2]),layer1_weight=np.array([[.4],[-.6]]),layer1_bias=np.array([.1]));arch=dict(kind='mlp',hidden_units=[2],activations=['tanh']);x=np.array([[-1.],[.3],[2.]]);y=np.array([0.,1.,1.])
 eta,g=_nn_mlp(w,arch,x,y);expected=np.tanh(x@w['layer0_weight']+w['layer0_bias'])@w['layer1_weight']+w['layer1_bias'];assert eta==pytest.approx(expected[:,0])
 for k in w:
  for index in np.ndindex(w[k].shape):
   old=w[k][index];w[k][index]=old+1e-5;up=_nn_loss(_nn_mlp(w,arch,x),y);w[k][index]=old-1e-5;down=_nn_loss(_nn_mlp(w,arch,x),y);w[k][index]=old;assert g[k][index]==pytest.approx((up-down)/2e-5,abs=1e-10)
 m=fit_neural_pd(x,y,feature_order=['x'],architecture=arch,initial_weights=w,optimizer_spec=optimizer(3),train_row_ids=['a','b','c'],**M)
 assert m.diagnostics['loss_path'][-1]['train_log_loss']<m.diagnostics['initial_train_log_loss']-.15
 assert not m.diagnostics['global_optimum_claimed']
 assert predict_pd(m,x,feature_order=['x'],expected_target=T).value==pytest.approx(expit(_neural_raw(m.model,x,['x'])))
 for kwargs,code in [(dict(offset=[0]*3),'OFFSET_MISMATCH'),(dict(context={'horizon':2}),'TARGET_MISMATCH')]:
  with pytest.raises(ValueError,match=code):predict_pd(m,x,feature_order=['x'],expected_target=T,**kwargs)

def records():
 return [dict(values=[[a],[a+.2]],observation_times=[0.,1.],known_times=[0.,1.],mask=[1,1],forecast_time=1.5) for a in [-1.,-.5,.5,1.]]
@pytest.mark.parametrize('variant',['gru','lstm'])
def test_sequence_actual_training_padding_future_causality_and_label_maturity(variant):
 rows=records();w=rw(variant,p=2);m=fit_sequence_pd(rows,[0,0,1,1],feature_order=['x'],variant=variant,hidden_size=2,initial_weights=w,optimizer_spec=optimizer(4,50),train_row_ids=list('abcd'),label_known_times=[3]*4,training_as_of_time=3,**M)
 assert m.diagnostics['loss_path'][-1]['train_log_loss']<m.diagnostics['initial_train_log_loss']-.1
 initial=_neural_raw(m.model,rows,['x'])
 assert predict_pd(m,rows,feature_order=['x'],expected_target=T).value==pytest.approx(expit(initial))
 for kwargs,code in [(dict(offset=[0]*4),'OFFSET_MISMATCH'),(dict(context={'horizon':2}),'TARGET_MISMATCH')]:
  with pytest.raises(ValueError,match=code):predict_pd(m,rows,feature_order=['x'],expected_target=T,**kwargs)
 import copy
 altered=copy.deepcopy(rows)
 for row in altered:
  row['values']+= [[999],[-999]];row['observation_times'] += [2,3];row['known_times'] += [2,3];row['mask'] += [1,0]
 assert _neural_raw(m.model,altered,['x'])==pytest.approx(initial,abs=0)
 assert _neural_raw(m.model,[rows[3],rows[0]],['x'])==pytest.approx(initial[[3,0]],abs=0)
 with pytest.raises(ValueError,match='LOOKAHEAD'):fit_sequence_pd(rows,[0,0,1,1],feature_order=['x'],variant=variant,hidden_size=2,initial_weights=w,optimizer_spec=optimizer(4,2),train_row_ids=list('abcd'),label_known_times=[2]*4,training_as_of_time=3,**M)

@pytest.mark.parametrize('activation',['relu','sigmoid'])
def test_mlp_other_activation_gradients(activation):
 w=dict(layer0_weight=np.array([[.2,-.1]]),layer0_bias=np.array([.8,-.6]),layer1_weight=np.array([[.4],[-.6]]),layer1_bias=np.array([.1]));arch=dict(kind='mlp',hidden_units=[2],activations=[activation]);x=np.array([[-1.],[.3],[2.]]);y=np.array([0.,1.,1.]);_,g=_nn_mlp(w,arch,x,y)
 for k in w:
  for index in np.ndindex(w[k].shape):
   old=w[k][index];w[k][index]=old+1e-5;up=_nn_loss(_nn_mlp(w,arch,x),y);w[k][index]=old-1e-5;down=_nn_loss(_nn_mlp(w,arch,x),y);w[k][index]=old;assert g[k][index]==pytest.approx((up-down)/2e-5,abs=1e-10)

def test_mlp_explicit_inner_early_stopping_and_no_scaling_leakage():
 x=np.array([[-2.],[-1.],[1.],[2.]]);y=np.array([0,0,1,1]);w=dict(layer0_weight=np.array([[.1]]),layer0_bias=np.array([0.]));arch=dict(hidden_units=[],activations=[])
 inner=dict(X=x,y=1-y,row_ids=list('wxyz'),role='inner_validation',split_id='inner',known_at='2024-12-31');opt=optimizer(4,20);opt['early_stopping_patience']=3
 m=fit_neural_pd(x,y,feature_order=['x'],architecture=arch,initial_weights=w,optimizer_spec=opt,train_row_ids=list('abcd'),inner_validation=inner,**M)
 assert m.diagnostics['selected_epoch']==0 and m.diagnostics['executed_epochs']==3
 assert m.model['preprocessing']['center']==[0]
 assert _neural_raw(m.model,[[1]],['x'])[0]==_neural_raw(m.model,[[1],[999]],['x'])[0]
 inner['row_ids'][0]='a'
 with pytest.raises(ValueError,match='INVALID_SPLIT'):fit_neural_pd(x,y,feature_order=['x'],architecture=arch,initial_weights=w,optimizer_spec=opt,train_row_ids=list('abcd'),inner_validation=inner,**M)

@pytest.mark.parametrize('variant',['gru','lstm'])
def test_recurrent_all_parameters_finite_difference_mandatory(variant):
 w=rw(variant);arch=dict(kind='recurrent',variant=variant,hidden_size=2);x=np.array([[[.2],[.7]],[[-.3],[.5]]]);y=np.array([0.,1.]);_,grad=_nn_recurrent(w,arch,list(x),y)
 for key in w:
  for index in np.ndindex(w[key].shape):
   old=w[key][index];w[key][index]=old+1e-5;up=_nn_loss(_nn_recurrent(w,arch,list(x)),y);w[key][index]=old-1e-5;down=_nn_loss(_nn_recurrent(w,arch,list(x)),y);w[key][index]=old
   assert grad[key][index]==pytest.approx((up-down)/2e-5,abs=1e-10)

@pytest.mark.parametrize('variant',['gru','lstm'])
def test_two_step_gate_hand_oracle(variant):
 gates=3 if variant=='gru' else 4
 w=dict(input_weight=np.zeros((1,gates)),hidden_weight=np.zeros((1,gates)),input_bias=np.zeros(gates),hidden_bias=np.zeros(gates),output_weight=np.ones((1,1)),output_bias=np.zeros(1));w['input_weight'][0,2]=1
 actual=_nn_recurrent(w,dict(kind='recurrent',variant=variant,hidden_size=1),[np.array([[1.],[2.]])])[0]
 state=.25*np.tanh(1)+.5*np.tanh(2)
 expected=state if variant=='gru' else .5*np.tanh(state)
 assert actual==pytest.approx(expected,abs=1e-15)
