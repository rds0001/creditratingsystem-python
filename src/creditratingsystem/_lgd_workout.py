"""Landmark terminal-payout models with explicit censored resolution timing."""
from copy import deepcopy
import math
import numpy as np
from .core import _fail,_numeric,_model_artifact,_fingerprint,_fit_result,_result
from .data import _rows,_required,_unique,_date
from .classification import _design
from ._loss_paths import _multiply,_sum
from .lgd import _scalar,_currency,_lgd_target_context


def _stage(stage):
    fixed=dict(payout_definition='RESIDUAL_TERMINAL_GROSS_RECOVERY_OR_CURE_VALUE',cashflow_pattern='SINGLE_TERMINAL_PAYOUT_AFTER_REFERENCE',timing_outcome_independence='CONDITIONAL_ON_REFERENCE_COVARIATES',positive_distribution='lognormal',cashflow_completeness='COMPLETE_THROUGH_OBSERVATION_END',payout_horizon='ULTIMATE')
    if not isinstance(stage,dict) or any(stage.get(k)!=v for k,v in fixed.items()) or stage.get('timing_distribution') not in ('weibull','lognormal','loglogistic'):_fail('RECOVERY_STAGE','Explicit residual terminal-payout stage and conditional timing/outcome independence required')
    _scalar(stage.get('cure_residual_value_ratio'),'cure_residual_value_ratio',0)
    if not isinstance(stage.get('landmark_definition'),str) or not stage['landmark_definition'].strip():_fail('RECOVERY_STAGE','Document the common training and prediction landmark selection rule')
    return deepcopy(stage)


def _episodes(episodes,x,cutoff,currency,training=False):
    rows=_rows(episodes,'episodes');_unique(rows,'episode_id','episodes')
    if len(rows)!=len(x):_fail('INCOMPATIBLE_SHAPE','One episode per covariate row required')
    parsed=[]
    for row in rows:
        _required(row,['default_date','default_known_at','reference_date','reference_known_at','features_observed_at','features_known_at','currency','ead_at_default','residual_exposure_at_reference'])
        if 'resolution_date' not in row or 'resolution_known_at' not in row:_fail('DATA_SCHEMA','Explicit nullable resolution dates required')
        default,dk,ref,rk,fo,fk=(_date(row[k],k)for k in ('default_date','default_known_at','reference_date','reference_known_at','features_observed_at','features_known_at'))
        end=_date(row.get('observation_end'),'observation_end') if training else cutoff
        if not default<=dk<=ref<=end<=cutoff or not ref<=rk<=ref or not fo<=fk<=ref:_fail('RECOVERY_DATE','Default, landmark features and observation dates are inconsistent or unavailable at landmark')
        if row['currency']!=currency:_fail('TARGET_MISMATCH','Episode reporting currency differs')
        ead=_scalar(row['ead_at_default'],'ead_at_default',0);residual=_scalar(row['residual_exposure_at_reference'],'residual_exposure_at_reference',0)
        if training and residual<=0:_fail('RECOVERY_STAGE','Positive residual exposure needed to fit payout ratios')
        rd=row['resolution_date'];known=row['resolution_known_at'];closed=False
        if (rd is None)!=(known is None):_fail('RECOVERY_DATE','Resolution date and known_at must both be present or both null')
        if rd is not None:
            rd=_date(rd,'resolution_date');known=_date(known,'resolution_known_at')
            if rd<=ref or known<rd:_fail('RECOVERY_DATE','Resolution must follow the landmark and precede its knowledge date')
            closed=rd<=end and known<=cutoff
        duration=(rd-ref).days if closed else (end-ref).days
        if training and duration<=0:_fail('RECOVERY_DATE','Positive observed/censored follow-up required')
        cured=0;value=0.
        if closed:
            _required(row,['cured','cure_value'])
            if isinstance(row['cured'],bool) or row['cured'] not in (0,1):_fail('RECOVERY_STAGE','Observed cure outcome must be binary zero/one')
            cured=int(row['cured']);value=_scalar(row['cure_value'],'cure_value',0)
            if not cured and value!=0:_fail('RECOVERY_STAGE','Noncured resolutions cannot carry cure asset value')
        parsed.append(dict(row=row,reference=ref,end=end,closed=closed,duration=duration,cured=cured,cure_value=value,ead=ead,residual=residual,resolution=rd))
    return rows,parsed


def _flows(cashflows,episodes,cutoff,currency):
    flows=_rows(cashflows,'cashflows');_unique(flows,'cashflow_id','cashflows');out={r['episode_id']:[] for r in episodes}
    for row in flows:
        _required(row,['episode_id','cashflow_date','known_at','currency','kind','amount'])
        if row['episode_id'] not in out:_fail('RECOVERY_STAGE','Cashflow has no matching episode')
        date=_date(row['cashflow_date'],'cashflow_date');known=_date(row['known_at'],'known_at');_scalar(row['amount'],'amount',0)
        if known<date or row['kind'] not in ('RECOVERY','DIRECT_COST','INDIRECT_COST'):_fail('RECOVERY_DATE','Realized cashflow kind or knowledge date invalid')
        if row['currency']!=currency:_fail('TARGET_MISMATCH','Training/payout stage requires already converted reporting-currency flows')
        if known<=cutoff:out[row['episode_id']].append(row)
    return flows,out


def _terminal_amount(ep,flows,stage):
    terminal=[]
    for flow in flows:
        date=_date(flow['cashflow_date'],'cashflow_date')
        if date<_date(ep['row']['default_date'],'default_date'):_fail('RECOVERY_DATE','Cashflow precedes default')
        if ep['reference']<date<=ep['end'] and flow['kind']=='RECOVERY' and flow['amount']>0:
            if not ep['closed'] or date!=ep['resolution']:_fail('RECOVERY_STAGE','Intermediate post-landmark recoveries violate the single terminal-payout model')
            terminal.append(flow['amount'])
    amount=_sum(terminal)
    if ep['closed'] and ep['cured']:
        expected=float(_multiply(ep['residual'],stage['cure_residual_value_ratio']))
        if amount!=0 or not math.isclose(ep['cure_value'],expected,rel_tol=1e-12,abs_tol=0):_fail('RECOVERY_STAGE','Cure value must equal declared residual ratio and cannot duplicate terminal cash recovery')
    return amount


def fit_recovery_hurdle(episodes,cashflows,X_train,*,feature_order,stage_spec,target,training,model_id,model_version,currency,fit_intercept=True,context=None):
    """Fit cure/positive-payout gates, lognormal amount and censored AFT resolution."""
    from .lgd import fit_fractional_lgd
    from .ead import _ead_regression
    from .survival import fit_aft_model
    stage=_stage(stage_spec)
    if not isinstance(target,dict):_fail('TARGET_MISMATCH','Explicit frozen target object required')
    ctx=_currency(currency,context);ctx=_lgd_target_context(target,ctx,model_version)
    if target.get('horizon_role')!='TIMING_REPORTING_REFERENCE' or target.get('event')!='residual_terminal_loss_fraction' or target.get('probability_measure')!='PHYSICAL' or target.get('conditioning')!='given_default' or target.get('time_unit')!='days' or target.get('currency')!=currency:_fail('TARGET_MISMATCH','Declare ultimate residual_terminal_loss_fraction with horizon_role=TIMING_REPORTING_REFERENCE, PHYSICAL, given_default, days and currency')
    if not isinstance(training,dict) or training.get('workout_population')!='open_and_closed_defaulted_landmarks':_fail('LGD_POPULATION','Declare open_and_closed_defaulted_landmarks training population')
    cutoff=_date(training['known_at'],'training.known_at');x,d=_design(X_train,feature_order,fit_intercept);rows,eps=_episodes(episodes,x,cutoff,currency,True);flows,byid=_flows(cashflows,rows,cutoff,currency)
    amounts=np.asarray([_terminal_amount(e,byid[e['row']['episode_id']],stage)for e in eps]);closed=np.array([e['closed']for e in eps]);cured=np.array([e['cured']for e in eps]);noncure=closed&(cured==0);positive=noncure&(amounts>0)
    if not np.any(closed&(cured==1)) or not np.any(noncure&~positive) or np.count_nonzero(positive)<2:_fail('NON_IDENTIFIABLE','Observed cure, zero noncure payout and at least two positive noncure payouts required')
    nested_training=dict(training,workout_population='closed_defaulted_episodes');options=dict(feature_order=feature_order,link='logit',covariance='HC0',target=target,training=nested_training,model_version=model_version,fit_intercept=fit_intercept)
    cure=fit_fractional_lgd(x[closed],cured[closed],model_id=model_id+'.cure',**options).model
    gate=fit_fractional_lgd(x[noncure],(amounts[noncure]>0).astype(float),model_id=model_id+'.positive',**options).model
    residual=np.array([e['residual']for e in eps]);logratio=np.log(amounts[positive])-np.log(residual[positive]);amount=_ead_regression(d[positive],logratio,np.ones(np.count_nonzero(positive)))
    if amount['residual_variance']<=1e-24:_fail('NON_IDENTIFIABLE','Positive terminal payout needs residual lognormal dispersion')
    timing_target=dict(target,event='workout_resolution');timing=fit_aft_model([e['duration']for e in eps],closed.astype(float),x,feature_order=feature_order,distribution=stage['timing_distribution'],target=timing_target,training=training,model_id=model_id+'.timing',model_version=model_version).model
    diagnostics=dict(observed_count=int(closed.sum()),censored_count=int((~closed).sum()),cure_count=int(cured.sum()),positive_noncure_count=int(positive.sum()),zero_noncure_count=int((noncure&~positive).sum()),identification='Outcome and payout independent of resolution time and independent right censoring, conditional on landmark covariates')
    parameters=dict(fit_intercept=fit_intercept,default_condition='given_default',currency=currency,stage_spec=stage,feature_min=x.min(axis=0).tolist(),feature_max=x.max(axis=0).tolist(),positive_feature_min=x[positive].min(axis=0).tolist(),positive_feature_max=x[positive].max(axis=0).tolist(),cure_gate=cure,positive_gate=gate,amount_regression=amount,timing_model=timing)
    model=_model_artifact(dict(schema_version='1.0',model_id=model_id,model_version=model_version,method_id='lgd.fit_recovery_hurdle',feature_order=list(feature_order),parameters=parameters,preprocessing=dict(transformation='identity'),target=target,training=training,diagnostics=diagnostics,provenance=dict(data_origin=training.get('data_origin','USER_SUPPLIED'),parameter_origin='ESTIMATED')))
    model['training_input_fingerprint']=_fingerprint(dict(episodes=rows,cashflows=flows,X=x.tolist()))
    return _fit_result(model,'lgd.fit_recovery_hurdle',diagnostics=diagnostics,context=ctx)


def _hurdle_prediction(model,X,feature_order,expected_target,default_condition,levels,context):
    from .lgd import predict_lgd
    from scipy.stats import norm
    p=model['parameters'];stage=_stage(p.get('stage_spec'))
    target=model['target']
    if target.get('horizon_role')!='TIMING_REPORTING_REFERENCE' or target.get('event')!='residual_terminal_loss_fraction' or target.get('probability_measure')!='PHYSICAL' or target.get('conditioning')!='given_default' or target.get('time_unit')!='days' or target.get('currency')!=p.get('currency'):_fail('TARGET_MISMATCH','Frozen ultimate residual payout target or horizon role is malformed')
    x,d=_design(X,feature_order,p.get('fit_intercept'))
    if list(feature_order)!=model['feature_order'] or expected_target!=model['target'] or default_condition!='given_default' or p.get('default_condition')!=default_condition:_fail('TARGET_MISMATCH','Frozen residual target, feature order and conditioning must match')
    q=None if levels is None else _numeric(levels,'quantile_levels',vector_only=True)
    if q is not None and (np.any((q<=0)|(q>=1)) or np.any(np.diff(q)<=0)):_fail('INVALID_QUANTILE','Strictly increasing interior quantile levels required')
    gates=[];outside=np.zeros(len(x),dtype=bool)
    for key in ('cure_gate','positive_gate'):
        component=_model_artifact(p.get(key))
        if component['training']['known_at']>model['training']['known_at']:_fail('INVALID_MODEL','Component model was fitted after the composite model knowledge date')
        if component['method_id']!='lgd.fit_fractional_lgd':_fail('INVALID_MODEL','Recovery gate must be a frozen fractional-logit model')
        r=predict_lgd(component,X,feature_order=feature_order,expected_target=expected_target,default_condition=default_condition);gates.append(np.array(r.value['mean_lgd']));outside|=np.array(r.diagnostics['outside_marginal_training_range'])
    for prefix in ('','positive_'):
        lo=_numeric(p.get(prefix+'feature_min'),'feature_min',allow_empty=True,vector_only=True);hi=_numeric(p.get(prefix+'feature_max'),'feature_max',allow_empty=True,vector_only=True)
        if lo.shape!=(x.shape[1],) or hi.shape!=lo.shape or np.any(lo>hi):_fail('INVALID_MODEL','Recovery marginal support dimensions invalid')
        outside|=np.any((x<lo)|(x>hi),axis=1)
    amount=p.get('amount_regression',{})
    if not isinstance(amount,dict):_fail('INVALID_MODEL','Amount regression must be a frozen object')
    beta=_numeric(amount.get('coefficients'),'amount coefficients',vector_only=True);variance=_scalar(amount.get('residual_variance'),'residual_variance',0)
    if beta.shape!=(d.shape[1],) or variance<=0:_fail('INVALID_MODEL','Positive lognormal coefficients or variance invalid')
    mu=d@beta
    if not np.all(np.isfinite(mu)):_fail('NUMERICAL_FAILURE','Positive payout linear predictor outside finite range')
    pc,pg=gates;mass=(1-pc)*pg;c=stage['cure_residual_value_ratio']
    with np.errstate(over='ignore',invalid='ignore'):positive_mean=np.exp(mu+variance/2)
    if not np.all(np.isfinite(positive_mean)):_fail('NUMERICAL_FAILURE','Conditional terminal payout mean outside finite range')
    recovery=np.array([_sum((float(_multiply(a,c)),float(_multiply(b,m))))for a,b,m in zip(pc,mass,positive_mean)])
    quantiles=None
    if q is not None:
        quantiles=[]
        for a,b,m in zip(pc,mass,mu):
            before=0. if c==0 else b*norm.sf((math.log(c)-m)/math.sqrt(variance));row=[]
            for u in q:
                if u>=b+(a if c>0 else 0):loss=1.
                elif c>0 and before<=u<=before+a:loss=1-c
                else:
                    tail=(u-(a if c>0 and u>before+a else 0))/b
                    exponent=m+math.sqrt(variance)*norm.isf(tail)
                    if exponent>math.log(np.finfo(float).max):_fail('NUMERICAL_FAILURE','Terminal loss quantile outside finite range')
                    loss=1-math.exp(exponent)
                row.append(loss)
            quantiles.append(row)
    return _result(dict(mean_lgd=(1-recovery).tolist(),distribution='CURE_RECOVERY_HURDLE',payout_horizon='ULTIMATE',expected_terminal_value_ratio=recovery.tolist(),cure_probability=pc.tolist(),positive_noncure_probability=mass.tolist(),conditional_positive_log_mean=mu.tolist(),conditional_positive_log_variance=variance,quantile_levels=None if q is None else q.tolist(),quantiles=quantiles),'lgd.predict_lgd','loss_fraction',context,provenance=model['provenance'],diagnostics=dict(model_id=model['model_id'],model_version=model['model_version'],model_fingerprint=_fingerprint(model),default_condition=default_condition,outside_marginal_training_range=outside.tolist()),limitations=('Residual terminal loss fraction at the declared landmark; this is not whole-workout discounted LGD. Cure value is an asset value, distinct from cash recovery. Timing/outcome independence conditional on reference covariates is a substantive identifying assumption.',))


def estimate_open_workout_lgd(episodes,cashflows,fitted_recovery_model,X,valuation_contracts,future_cost_cashflows,*,feature_order,expected_target,stage_spec,as_of,currency,extrapolation,context=None):
    """Value realized net flows once and conditional remaining terminal payouts.

    Resolution mass is valued at supplied interval endpoints. Unresolved tail
    value is left unknown unless the user supplies its conditional average factor.
    """
    from .lgd import predict_lgd,discount_recoveries
    from .survival import predict_event_curves
    model=_model_artifact(fitted_recovery_model);p=model['parameters'];stage=_stage(stage_spec);cutoff=_date(as_of,'as_of');ctx=_lgd_target_context(model['target'],_currency(currency,context),model['model_version'])
    if model['method_id']!='lgd.fit_recovery_hurdle' or stage!=p.get('stage_spec') or p.get('currency')!=currency:_fail('RECOVERY_STAGE','Frozen terminal-payout family, stage and currency must match')
    if _date(model['training']['known_at'],'training.known_at')>cutoff or (ctx.get('as_of') is not None and ctx['as_of']!=as_of):_fail('RECOVERY_DATE','Model must be available at the declared evaluation date')
    if extrapolation not in ('error','flat','parametric'):_fail('INVALID_EXTRAPOLATION','Declare error, flat or parametric timing extrapolation')
    x,_=_design(X,feature_order,p.get('fit_intercept'));rows,eps=_episodes(episodes,x,cutoff,currency);flows,byid=_flows(cashflows,rows,cutoff,currency)
    predictions=predict_lgd(model,X,feature_order=feature_order,expected_target=expected_target,default_condition='given_default',context=context)
    valuations=_rows(valuation_contracts,'valuation_contracts');_unique(valuations,'episode_id','valuation_contracts');lookup={r['episode_id']:r for r in valuations}
    if set(lookup)!=set(byid):_fail('RECOVERY_CONTRACT','Exactly one valuation contract per episode required')
    costs=_rows(future_cost_cashflows,'future_cost_cashflows');_unique(costs,'cashflow_id','future costs');costby={k:[]for k in byid}
    pastids={r['cashflow_id']for r in flows}
    for cost in costs:
        _required(cost,['episode_id','cashflow_date','known_at','currency','kind','amount'])
        if cost['episode_id']not in costby or cost['cashflow_id']in pastids:_fail('RECOVERY_CONTRACT','Future cost has unknown episode or duplicates realized flow ID')
        if cost['currency']!=currency or cost['kind']not in ('DIRECT_COST','INDIRECT_COST') or _date(cost['cashflow_date'],'cashflow_date')<=cutoff or _date(cost['known_at'],'known_at')>cutoff:_fail('RECOVERY_CONTRACT','Future costs require reporting currency, future payment date and current knowledge')
        _scalar(cost['amount'],'future cost',0);costby[cost['episode_id']].append(cost)
    outputs=[]
    timing=_model_artifact(p.get('timing_model'))
    if timing['training']['known_at']>model['training']['known_at']:_fail('INVALID_MODEL','Resolution model was fitted after composite model knowledge date')
    if timing['method_id']!='survival.fit_aft_model' or timing['feature_order']!=model['feature_order'] or timing['target']!=dict(model['target'],event='workout_resolution') or timing['parameters'].get('distribution')!=stage['timing_distribution']:_fail('INVALID_MODEL','Frozen resolution timing model does not match terminal-payout stage')
    for index,ep in enumerate(eps):
        row=ep['row'];eid=row['episode_id'];valuation=lookup[eid];_required(valuation,['discount_curve','fx_contract','future_resolution_dates','resolution_time_convention','unmodeled_future_costs_assumption'])
        if any(k not in valuation for k in ('tail_discount_factor','tail_assumption_reference')):_fail('DATA_SCHEMA','Explicit nullable tail valuation fields required')
        if valuation['resolution_time_convention']!='RIGHT_ENDPOINT' or valuation['unmodeled_future_costs_assumption']!='NONE_BEYOND_SUPPLIED':_fail('RECOVERY_CONTRACT','Declare right-endpoint resolution valuation and no costs beyond supplied schedule')
        _terminal_amount(ep,byid[eid],stage)
        known=discount_recoveries(byid[eid],valuation['discount_curve'],valuation['fx_contract'],default_date=row['default_date'],evaluation_cutoff=as_of,reporting_currency=currency).value
        df={v['cashflow_date']:v['discount_factor']for v in valuation['discount_curve']['points']}
        def factor(date):
            if date not in df:_fail('RECOVERY_CONTRACT','Missing exact-date discount factor for terminal value or future cost')
            return df[date]
        future_cost=_sum([float(_multiply(c['amount'],factor(c['cashflow_date'])))for c in costby[eid]])
        past=known['discounted_net_recoveries'];future_grid=0.;future_tail=None;tail_mass=0.;undiscounted_tail=0.;resolution_prob=[];grid=[];dates=[];cure_realized=0.
        if ep['closed']:
            if costby[eid]:_fail('RECOVERY_STAGE','Closed final workout cannot retain unpaid future costs')
            cure_realized=float(_multiply(ep['cure_value'],factor(ep['resolution'].isoformat()))) if ep['cured']else 0.
            future_tail=0.;status='OBSERVED_CLOSED'
        else:
            dates=valuation['future_resolution_dates']
            if not isinstance(dates,list) or not dates:_fail('RECOVERY_CONTRACT','Open workouts require a future resolution grid')
            parsed=[_date(v,'future_resolution_dates')for v in dates]
            if parsed[0]<=cutoff or any(a>=b for a,b in zip(parsed,parsed[1:])):_fail('RECOVERY_DATE','Resolution grid must increase strictly after evaluation cutoff')
            grid=[(v-cutoff).days for v in parsed];age=(cutoff-ep['reference']).days
            path=[dict(path_id=eid,start=age,stop=age+grid[-1],covariates=x[index].tolist(),stratum='all',covariate_time=0)]
            curve=predict_event_curves(timing,path,grid,feature_order=feature_order,target=timing['target'],start_time=age,extrapolation=extrapolation).value['curves'][0]
            resolution_prob=curve['marginal_probabilities']['workout_resolution'];tail_mass=curve['survival'][-1]
            expected=float(_multiply(ep['residual'],predictions.value['expected_terminal_value_ratio'][index]))
            future_grid=_sum([float(_multiply(expected,prob,factor(date)))for prob,date in zip(resolution_prob,dates)]);undiscounted_tail=float(_multiply(expected,tail_mass));tail_factor=valuation['tail_discount_factor']
            if tail_factor is not None:
                tail_factor=_scalar(tail_factor,'tail_discount_factor',0)
                if not isinstance(valuation['tail_assumption_reference'],str) or not valuation['tail_assumption_reference'].strip():_fail('RECOVERY_CONTRACT','Supplied tail factor requires explicit economic assumption reference')
                future_tail=float(_multiply(undiscounted_tail,tail_factor));status='ESTIMATED_WITH_SUPPLIED_TAIL_FACTOR'
            elif tail_mass==0 or expected==0:future_tail=0.;status='ESTIMATED_NO_REMAINING_VALUE'
            else:status='UNRESOLVED_TAIL_VALUE'
        net_grid=_sum((past,cure_realized,future_grid,-future_cost));loss_grid=_sum((ep['ead'],-net_grid));loss_final=None if future_tail is None else _sum((loss_grid,-future_tail))
        lgd_grid=None if ep['ead']==0 else loss_grid/ep['ead'];lgd_final=None if ep['ead']==0 or loss_final is None else loss_final/ep['ead']
        if any(v is not None and not math.isfinite(v)for v in (lgd_grid,lgd_final)):_fail('NUMERICAL_FAILURE','Workout LGD ratio outside finite range')
        outputs.append(dict(episode_id=eid,status=status,ead_at_default=ep['ead'],known_discounted_net_cashflows=past,realized_discounted_cure_value=cure_realized,expected_discounted_future_terminal_value_within_grid=future_grid,discounted_supplied_future_costs=future_cost,unresolved_probability=tail_mass,expected_undiscounted_terminal_value_beyond_grid=undiscounted_tail,expected_discounted_tail_value=future_tail,expected_loss_within_grid=loss_grid,expected_final_loss=loss_final,expected_lgd_within_grid=lgd_grid,expected_final_lgd=lgd_final,ratio_status='ZERO_EAD_UNDEFINED'if ep['ead']==0 else 'DEFINED',future_resolution_dates=dates,resolution_probabilities=resolution_prob,excluded_cashflows=[r['cashflow_id']for r in flows if r['episode_id']==eid and (_date(r['known_at'],'known_at')>cutoff or _date(r['cashflow_date'],'cashflow_date')>cutoff)]))
    return _result(dict(episodes=outputs,as_of=as_of,currency=currency,extrapolation=extrapolation),'lgd.estimate_open_workout_lgd','loss_fraction',ctx,provenance=model['provenance'],diagnostics=dict(model_fingerprint=_fingerprint(model),stage_fingerprint=_fingerprint(stage),outside_marginal_training_range=predictions.diagnostics['outside_marginal_training_range'],input_fingerprint=_fingerprint(dict(episodes=rows,cashflows=flows,valuation_contracts=valuations,future_cost_cashflows=costs,X=x.tolist()))),limitations=('Single residual terminal gross payout or cure asset value, conditional on the declared landmark. Payout outcome/amount and resolution timing are conditionally independent; censoring is independent given covariates. Resolution within each interval is valued at its right endpoint. Supplied future costs are unconditional dated amounts and no additional costs are inferred. Unknown tail discount value prevents a final LGD estimate; supplied tail factors are user assumptions. No automatic floors, regulatory calibration or payment-path model is implied.',))
