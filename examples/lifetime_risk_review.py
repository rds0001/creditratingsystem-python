"""Competing account exits, conditional loss paths and independent held-out outcomes."""
from copy import deepcopy
from datetime import date,timedelta
from importlib.resources import files
import json
import math
import numpy as np
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('lifetime_risk_review.json').read_text())
    d=deepcopy(data);target=d['target'];grid=d['horizons'];train=d['training'];origin=d['origin']
    if (date.fromisoformat(d['training_origin'])+timedelta(days=max(d['training_times']))).isoformat()>train['known_at'] or train['known_at']>origin:
        raise ValueError('Training event support must precede model availability and forecast origin')
    rows=[dict(entity_id='train-'+str(i),start=0,stop=t,event=int(e is not None),covariate_time=0)for i,(t,e)in enumerate(zip(d['training_times'],d['training_events']))]
    meta=dict(feature_order=[],target=target,training=train,model_version='1')
    cs=cr.fit_cause_specific_hazards(rows,np.empty((len(rows),0)),d['training_events'],causes=['default','prepay'],ties='breslow',model_id='lifetime-cause',**meta)
    fg=cr.fit_fine_gray(d['training_times'],d['training_events'],np.empty((len(rows),0)),cause_of_interest='default',censoring_spec=dict(estimator='kaplan_meier',independence_assumption='independent_within_censoring_group'),model_id='lifetime-fine-gray',**meta)
    path=[dict(path_id='account',start=0,stop=60,covariates=[],stratum='all',covariate_time=0)]
    curves=cr.predict_event_curves(cs,path,grid,feature_order=[],target=target).value['curves'][0]
    alternate=cr.predict_event_curves(fg,path,grid,feature_order=[],target=target).value['curves'][0]
    cumulative=curves['cumulative_incidence']['default'];marginal=np.diff([0]+cumulative).tolist()
    events=[dict(event_id='repay-'+str(i),event_date=t,known_at=origin,sequence=0,kind='PRINCIPAL_REPAYMENT',amount=a)for i,(t,a)in enumerate(zip(d['dates'],[10,20,30]))]
    schedule=cr.exposure_schedule(dict(facility_id='amortizing',currency='EUR',as_of=origin,drawn_principal=100,credit_limit=100,accrued_interest=0),events,[0,0,0],period_end_dates=d['dates'],day_count='ACT/365F',default_timing='BEFORE_PERIOD_END_EVENTS',allow_overlimit=False,currency='EUR')
    ead=cr.predict_ead(schedule,None,feature_order=[],expected_target=None,reference_dates=[origin]*3,default_dates=d['dates'],currency='EUR').value['conditional_ead']
    lt=dict(event='economic_loss',horizon=60,time_unit='days',probability_measure='PHYSICAL',conditioning='given_default')
    lf=cr.fit_fractional_lgd(np.empty((len(d['lgd_training_targets']),0)),d['lgd_training_targets'],feature_order=[],link='logit',covariance='HC0',target=lt,training=dict(train,workout_population='closed_defaulted_episodes'),model_id='lifetime-lgd',model_version='1')
    lp=cr.predict_lgd(lf,np.empty((3,0)),feature_order=[],expected_target=lt,default_condition='given_default').value
    lgd=[round(v,12)for v in lp['mean_lgd']];conditional=[e*l for e,l in zip(ead,lgd)]
    loss=cr.lifetime_expected_loss(marginal,conditional,d['discount_factors'],[t/365 for t in grid],currency='EUR',probability_measure='PHYSICAL').value
    design=dict(model_id='lifetime-cause',model_version='1',parameter_version='1',data_fingerprint='independent-heldout-account-events',population_id='first-default-before-exit',split_id='holdout-2024',prediction_origin='HOLDOUT',evaluation_cutoff='2024-06-01',days_per_time_unit=1,target=target)
    obs=[]
    for r in d['holdout']:
        if r.get('terminal_exit_date') is not None and (r['event'] or r.get('coverage_end')!='2024-03-02'):
            raise ValueError('Terminal prepaid noncase needs explicit observed first-default-before-exit target coverage')
        obs.append(dict(prediction_id=r['id'],entity_id=r['id'],cluster_id=r['id'],segment='all',as_of=origin,forecast_known_at=origin,horizon_end=d['dates'][-1],model_id='lifetime-cause',model_version='1',parameter_version='1',outcome_known_at='2024-03-03',observed_time=r['time'],event=r['event']))
    # Frozen independent administrative censor DGP: two of eight censor at30, six follow to90.
    censor=dict(method='frozen_ipcw',reference='Independent administrative censor cohort: 2/8 at day30, remaining6 supported through90',independence_assumption='censoring_independent_of_event_features_and_across_clusters',tie_rule='event_before_censoring',model_fingerprint='independent-administrative-censor-eight',model_known_at='2022-04-03',times=[30],survival=[.75],support_end=90,minimum_survival=.1)
    groups=[r['id']for r in d['holdout']];draws=[groups,list(reversed(groups)),[groups[0]]*6,[groups[-1]]*6]
    survival=cr.survival_backtest([cumulative]*len(obs),obs,horizons=grid,censoring=censor,design=design,uncertainty='cluster_bootstrap',cluster_resamples=draws).value
    default_rows=[]
    for r,o in zip(d['holdout'],obs):
        if not r['event']:continue
        j=grid.index(r['time']);closed=r['workout_status']=='CLOSED'
        default_rows.append(dict(o,default_date=r['default_date'],outcome_known_at='2024-05-01',predicted_ead=ead[j],realized_ead=r['ead'],realized_ead_date=r['default_date'],drawn_at_reference=100,limit_at_reference=100,currency='EUR',predicted_lgd=lgd[j],realized_lgd=r['lgd'],ead_at_default=r['ead'],workout_status='CLOSED'if closed else'OPEN',workout_closed_at='2024-05-01'if closed else None))
    dd=dict(design,target=dict(target,conditioning='default'),currency='EUR')
    ev=cr.ead_backtest(default_rows,design=dd,weighting_policy=dict(method='equal',reference='Matched observed default balances'),uncertainty='none').value
    lv=cr.lgd_backtest(default_rows,design=dd,workout_policy=dict(method='closed_only',reference='Actual finalized claims only; excludes unresolved claim'),uncertainty='none').value
    w=d['open_workout'];oa=dict(episodes=[w['episode']],cashflows=w['cashflows'],fitted_recovery_model=d['residual_model'],X=[[]],valuation_contracts=[w['valuation']],future_cost_cashflows=w['future_costs'],feature_order=[],expected_target=d['recovery_target'],stage_spec=d['stage_spec'],as_of='2024-03-29',currency='EUR',extrapolation='parametric')
    open_value=cr.estimate_open_workout_lgd(**oa).value
    unknown=deepcopy(oa);unknown['valuation_contracts'][0].update(tail_discount_factor=None,tail_assumption_reference=None)
    unidentified=cr.estimate_open_workout_lgd(**unknown).value
    out=dict(data_origin=d['data_origin'],target_definition=d['target_definition'],cause_specific_curves=curves,fine_gray_curves=alternate,fine_gray_baseline=fg.model['parameters']['baseline'],marginal_first_default=marginal,conditional_ead=ead,conditional_lgd=lgd,lifetime_loss=loss,survival_validation=survival,ead_validation=ev,lgd_validation=lv,open_workout=open_value,unidentified_tail=unidentified,ignored_prepayment_probability=1-math.exp(-5/6),incorrect_cumulative_mass_loss=sum(p*l*v for p,l,v in zip(cumulative,conditional,d['discount_factors'])))
    if verify_reference:
        expected=[0,math.exp(-.2)*(1-math.exp(-1/3)),math.exp(-.2)*(1-math.exp(-5/6))]
        assert np.allclose(cumulative,expected,rtol=0,atol=1e-12)
        assert np.allclose(np.array(curves['survival'])+np.array(cumulative)+np.array(curves['cumulative_incidence']['prepay']),1)
        assert np.allclose(ead,[100,90,70])and np.allclose(lgd,.4)
        assert np.allclose([r['increment']for r in out['fine_gray_baseline']],[1/3.75,1/2.75])
        assert math.isclose(loss['lifetime_expected_loss'],expected[1]*90*.4*.97+(expected[2]-expected[1])*70*.4*.95,abs_tol=1e-10)
        assert out['incorrect_cumulative_mass_loss']>loss['lifetime_expected_loss']
        b=[0,(2*(1-expected[1])**2+3*expected[1]**2)/4.5,(3*(1-expected[2])**2+2*expected[2]**2)/4.5]
        assert np.allclose([r['brier']for r in survival['metrics']['by_horizon']],b)
        assert math.isclose(ev['metrics']['ead']['overall']['selected_weighting']['bias'],-10)
        assert math.isclose(lv['metrics']['overall']['ead_weighted']['bias'],4/180,abs_tol=1e-10)
        assert lv['coverage']['exclusion_counts']=={'OPEN_WORKOUT':1}
        assert unidentified['episodes'][0]['expected_final_lgd']is None
    return out


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
