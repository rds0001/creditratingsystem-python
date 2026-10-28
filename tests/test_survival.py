import math
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import weibull_min,lognorm,fisk
from creditratingsystem.survival import *
TARGET={'event':'default','horizon':5,'time_unit':'years','probability_measure':'PHYSICAL','conditioning':'performing'}
TRAIN={'data_fingerprint':'analytic-survival','split_id':'train','known_at':'2025-01-01','data_origin':'SYNTHETIC'}
META=dict(target=TARGET,training=TRAIN,model_id='survival-oracle',model_version='1')
def path(x=(),stop=10,stratum='all'):
 return [{'path_id':'case','start':0,'stop':stop,'covariates':list(x),'stratum':stratum,'covariate_time':0}]
def panel(times,events):
 return [{'entity_id':str(i),'start':0,'stop':t,'event':e,'covariate_time':0} for i,(t,e) in enumerate(zip(times,events))]
def pred(model,grid,features=(),x=(),**kw):
 return predict_event_curves(model,path(x),grid,feature_order=list(features),target=TARGET,**kw).value['curves'][0]

def test_km_delayed_entry_frequency_and_greenwood():
 m=estimate_survival_curve([0,0,1],[1,2,3],[1,0,1],weight_policy='frequency',**META)
 c=m.model['parameters']['curve'];assert [r['at_risk'] for r in c]==[2,2,1];assert [r['survival'] for r in c]==[.5,.5,0];assert c[0]['greenwood_sum']==.5
 v=predict_event_curves(m,[],[0,1,2,3],feature_order=[],target=TARGET).value['curves'][0];assert v['survival']==[1,.5,.5,0]
 weighted=estimate_survival_curve([0,0],[1,2],[1,1],weight_policy='frequency',weights=[2,1],**META);assert weighted.model['parameters']['curve'][0]['survival']==pytest.approx(1/3)
 with pytest.raises(ValueError,match='INVALID_WEIGHTS'):estimate_survival_curve([0,0],[1,2],[1,1],weight_policy='frequency',weights=[.5,1],**META)

def history():
 return [{'entity_id':'a','start':0,'stop':1,'from_state':'performing','to_state':'default'},{'entity_id':'a','start':1,'stop':2,'from_state':'default','to_state':'cured'},{'entity_id':'a','start':2,'stop':3,'from_state':'cured','to_state':'default'},{'entity_id':'b','start':0,'stop':4,'from_state':'performing','to_state':None}]
def multistate():
 return estimate_multistate_transitions(history(),states=['performing','default','cured'],allowed_transitions=[['performing','default'],['default','cured'],['cured','default']],initial_state='performing',**META)

def test_aalen_johansen_repeated_transition_cure_never_reverses_first_default():
 m=multistate();v=predict_event_curves(m,[],[0,1,2,3],feature_order=[],target=TARGET,event_states=['default']).value['curves'][0]
 assert v['first_hit_probability']==[0,.5,.5,.5];assert np.array(v['state_occupancy'])[:,1].tolist()==[0,.5,0,.5]
 v=predict_event_curves(m,[],[0,1],feature_order=[],target=TARGET,initial_state='cured',event_states=['default'],start_time=2).value['curves'][0];assert v['first_hit_probability']==[0,1]
 bad=history();bad[1]['from_state']='performing'
 with pytest.raises(ValueError,match='FORBIDDEN_TRANSITION|INVALID_HISTORY'):estimate_multistate_transitions(bad,states=['performing','default','cured'],allowed_transitions=[['performing','default'],['default','cured'],['cured','default']],initial_state='performing',**META)

def test_aalen_single_cause_matches_km():
 h=[{'entity_id':str(i),'start':0,'stop':t,'from_state':'performing','to_state':'default' if e else None} for i,(t,e) in enumerate(zip([1,2,3,4],[1,0,1,0]))]
 m=estimate_multistate_transitions(h,states=['performing','default'],allowed_transitions=[['performing','default']],initial_state='performing',**META)
 v=predict_event_curves(m,[],[0,1,2,3,4],feature_order=[],target=TARGET,event_states=['default']).value['curves'][0]
 assert v['first_hit_probability']==pytest.approx([0,.25,.25,.625,.625])

@pytest.mark.parametrize('ties',['breslow','efron'])
def test_cox_ties_strata_strict_entry_and_ph_score(ties):
 times=np.repeat([1.,2.,3.,4.],2);events=np.repeat([1,1,0,1],2);x=np.tile([[-1.],[1.]],(4,1));rows=panel(times,events)
 m=fit_cox_ph(rows,x,feature_order=['x'],ties=ties,**META);assert m.model['parameters']['coefficients']==pytest.approx([0],abs=1e-9)
 increments=[r['increment'] for r in m.model['parameters']['baseline']];assert increments==pytest.approx([.25,1/3,1]);assert m.diagnostics['ph_diagnostic']['global_statistic']==pytest.approx(0,abs=1e-12)
 v=pred(m,[0,1,2,4],['x'],[0]);assert v['survival']==pytest.approx(np.exp(-np.array([0,.25,.25+1/3,.25+1/3+1])))
 split=[];xx=[]
 for i,(t,e) in enumerate(zip(times,events)):
  split.append(dict(entity_id=str(i),start=0,stop=1,event=int(e) if t==1 else 0,covariate_time=0));xx.append(x[i])
  if t>1:split.append(dict(entity_id=str(i),start=1,stop=float(t),event=int(e),covariate_time=1));xx.append(-x[i])
 ms=fit_cox_ph(split,np.array(xx),feature_order=['x'],ties=ties,**META);assert [r['at_risk'] for r in ms.model['parameters']['baseline']]==[8,6,2];assert ms.model['parameters']['coefficients']==pytest.approx([0],abs=1e-9)
 with pytest.raises(ValueError,match='LOOKAHEAD'):
  split[0]['covariate_time']=.5;fit_cox_ph(split,np.array(xx),feature_order=['x'],ties=ties,**META)

@pytest.mark.parametrize('distribution',['weibull','lognormal','loglogistic'])
def test_aft_independent_uncensored_profile_likelihood_oracles(distribution):
 z=np.array([-1.5,-1.,-.5,.5,1.,1.5]);time=np.exp(z);m=fit_aft_model(time,np.ones(6),np.empty((6,0)),feature_order=[],distribution=distribution,**META);p=m.model['parameters'];location=p['coefficients'][0];scale=p['log_time_scale']
 if distribution=='weibull':
  shape=brentq(lambda k:1/k+z.mean()-np.sum(np.exp(k*z)*z)/np.sum(np.exp(k*z)),.1,10);expected_location=np.log(np.mean(time**shape))/shape;expected_scale=1/shape;dist=weibull_min(c=1/scale,scale=np.exp(location))
 elif distribution=='lognormal':expected_location=0.;expected_scale=np.sqrt(np.mean(z*z));dist=lognorm(s=scale,scale=np.exp(location))
 else:expected_location=0.;expected_scale=brentq(lambda s:np.mean((z/s)*np.tanh(z/(2*s)))-1,.1,10);dist=fisk(c=1/scale,scale=np.exp(location))
 assert location==pytest.approx(expected_location,abs=2e-5);assert scale==pytest.approx(expected_scale,abs=2e-5)
 assert m.diagnostics['row_log_likelihood']==pytest.approx(dist.logpdf(time),abs=1e-10)
 v=pred(m,[0,.5,1,2]);assert v['survival']==pytest.approx(dist.sf([0,.5,1,2]),abs=1e-10)
 events=np.array([1,0,1,1,0,1]);mc=fit_aft_model(time,events,np.empty((6,0)),feature_order=[],distribution=distribution,**META);pc=mc.model['parameters'];s=pc['log_time_scale'];loc=pc['coefficients'][0];d={'weibull':lambda:weibull_min(c=1/s,scale=np.exp(loc)),'lognormal':lambda:lognorm(s=s,scale=np.exp(loc)),'loglogistic':lambda:fisk(c=1/s,scale=np.exp(loc))}[distribution]();expected=np.where(events,d.logpdf(time),d.logsf(time));assert mc.diagnostics['row_log_likelihood']==pytest.approx(expected,abs=1e-10)

@pytest.mark.parametrize('link',['logit','cloglog'])
def test_discrete_period_hazard_and_partial_period_contract(link):
 rows=[dict(entity_id=str(i),start=0,stop=1,event=int(i<2),period=0,interval_end_type='scheduled') for i in range(10)];m=fit_discrete_hazard(rows,np.empty((10,0)),feature_order=[],period_grid=[0,1],link=link,partial_period_policy='exclude_censored',**META);v=pred(m,[0,1]);assert v['survival']==pytest.approx([1,.8],abs=1e-8)
 rows.append(dict(entity_id='partial',start=0,stop=.5,event=0,period=0,interval_end_type='censor'));m=fit_discrete_hazard(rows,np.empty((11,0)),feature_order=[],period_grid=[0,1],link=link,partial_period_policy='exclude_censored',**META);assert m.diagnostics['excluded_partial_periods']==1
 if link=='cloglog':
  v=pred(m,[0,.5,1]);assert v['survival']==pytest.approx([1,math.sqrt(.8),.8],abs=1e-8)
 else:
  with pytest.raises(ValueError,match='INVALID_PERIOD_CONTRACT'):pred(m,[0,.5,1])


def test_cause_specific_cif_joint_integration_and_balance():
 rows=panel([1,1,2,2],[1,1,0,0]);m=fit_cause_specific_hazards(rows,np.empty((4,0)),['default','prepay',None,None],feature_order=[],causes=['default','prepay'],ties='breslow',**META);v=pred(m,[0,1,2]);f=v['cumulative_incidence'];assert f['default'][1]==pytest.approx((1-math.exp(-.5))/2);assert f['default'][1]!=pytest.approx(1-math.exp(-.25));assert np.array(v['survival'])+np.array(f['default'])+np.array(f['prepay'])==pytest.approx(np.ones(3))

def test_fine_gray_reverse_km_retained_risk_and_no_competitor_limit():
 spec={'estimator':'kaplan_meier','independence_assumption':'independent_within_censoring_group'}
 m=fit_fine_gray([1,2,3,4,4],['prepay',None,'default','default',None],np.empty((5,0)),cause_of_interest='default',feature_order=[],censoring_spec=spec,**META);baseline=m.model['parameters']['baseline'];assert [r['increment'] for r in baseline]==pytest.approx([1/3.75,1/2.75]);assert .75 in m.diagnostics['expanded_weights'];v=pred(m,[0,3,4]);assert v['cumulative_incidence']['default']==pytest.approx([0,1-math.exp(-1/3.75),1-math.exp(-1/3.75-1/2.75)])
 times=np.repeat([1.,2.,3.,4.],2);events=np.repeat([1,1,0,1],2);x=np.tile([[-1.],[1.]],(4,1));fg=fit_fine_gray(times,['default' if e else None for e in events],x,cause_of_interest='default',feature_order=['x'],censoring_spec=spec,**META);cox=fit_cox_ph(panel(times,events),x,feature_order=['x'],ties='efron',**META);assert fg.model['parameters']['coefficients']==pytest.approx(cox.model['parameters']['coefficients'],abs=1e-7);assert [r['increment'] for r in fg.model['parameters']['baseline']]==pytest.approx([r['increment'] for r in cox.model['parameters']['baseline']],abs=1e-7)

def test_frozen_target_support_and_zero_survival_conditioning_errors():
 m=estimate_survival_curve([0,0],[1,2],[1,1],weight_policy='frequency',**META)
 with pytest.raises(ValueError,match='MODEL_CONTRACT_MISMATCH'):predict_event_curves(m,[],[0,1],feature_order=[],target=dict(TARGET,horizon=1))
 with pytest.raises(ValueError,match='OUTSIDE_SUPPORT'):predict_event_curves(m,[],[0,3],feature_order=[],target=TARGET)
 with pytest.raises(ValueError,match='IMPOSSIBLE_CONDITIONING'):predict_event_curves(m,[],[0,1],feature_order=[],target=TARGET,start_time=2,extrapolation='flat')

@pytest.mark.parametrize('distribution',['weibull','lognormal','loglogistic'])
def test_aft_conditional_quantiles_and_explicit_flat_tail(distribution):
 m=fit_aft_model(np.exp([-1.5,-1.,-.5,.5,1.,1.5]),np.ones(6),np.empty((6,0)),feature_order=[],distribution=distribution,**META)
 result=predict_event_curves(m,path(stop=100),[0,1,10],feature_order=[],target=TARGET,start_time=.5,extrapolation='parametric',quantile_levels=[.1,.5,.9]).value
 quantiles=result['curves'][0]['conditional_event_quantiles'];assert np.all(np.diff(quantiles)>0)
 v=predict_event_curves(m,path(stop=100),quantiles,feature_order=[],target=TARGET,start_time=.5,extrapolation='parametric').value['curves'][0]
 assert v['cumulative_incidence']['default']==pytest.approx([.1,.5,.9],abs=2e-14)
 flat=pred(m,[5,10],extrapolation='flat');assert flat['survival'][0]==flat['survival'][1]

def test_empty_risk_interval_is_not_silent_identified_survival():
 m=estimate_survival_curve([0,3],[1,4],[0,1],weight_policy='frequency',**META)
 assert m.model['parameters']['risk_support_gaps']==[{'start':1.,'stop':3.}]
 with pytest.raises(ValueError,match='INSUFFICIENT_RISK_SUPPORT'):predict_event_curves(m,[],[0,4],feature_order=[],target=TARGET)
 r=predict_event_curves(m,[],[0,4],feature_order=[],target=TARGET,extrapolation='flat');assert r.diagnostics['unsupported_time_intervals']==[{'start':1.,'stop':3.}]
