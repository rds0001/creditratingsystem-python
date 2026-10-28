"""Frozen survival and event-history models with explicit risk-set semantics."""
from collections.abc import Mapping
import math
import warnings
import numpy as np
from scipy.special import expit,log_ndtr,ndtri_exp
from scipy.stats import norm,chi2
from statsmodels.duration.hazard_regression import PHReg
from .core import _fail,_numeric,_model_artifact,_fit_result,_result,_fingerprint


def _scalar(x,name,low=None,high=None):
    v=_numeric(x,name)
    if v.ndim or (low is not None and v<low) or (high is not None and v>high):_fail('INVALID_INPUT',name+' outside scalar domain')
    return float(v)


def _strings(x,name,empty=False):
    if not isinstance(x,(list,tuple)) or (not x and not empty) or any(not isinstance(v,str) or not v for v in x) or len(set(x))!=len(x):_fail('INVALID_INPUT',name+' requires distinct nonempty names')
    return list(x)


def _rows(x):
    if not isinstance(x,(list,tuple)) or not x or any(not isinstance(r,Mapping) for r in x):_fail('INVALID_TABLE','Nonempty records required')
    return x


def _model(method,features,parameters,target,training,model_id,model_version,diagnostics):
    if not isinstance(target,Mapping) or target.get('probability_measure')!='PHYSICAL':_fail('INCOMPATIBLE_TARGET','Physical event target required')
    return _model_artifact(dict(schema_version='1.0',model_id=model_id,model_version=model_version,method_id='survival.'+method,feature_order=list(features),parameters=parameters,preprocessing={'transformation':'identity'},target=dict(target),training=training,diagnostics=diagnostics,provenance={'data_origin':training.get('data_origin','USER_SUPPLIED'),'parameter_origin':'ESTIMATED'}))


def _times(entry_time,exit_time,event_indicator):
    stop=_numeric(exit_time,'exit_time',vector_only=True);start=np.zeros(len(stop)) if entry_time is None else _numeric(entry_time,'entry_time',vector_only=True);event=_numeric(event_indicator,'event_indicator',vector_only=True)
    if len(start)!=len(stop) or len(event)!=len(stop) or np.any(start<0) or np.any(stop<=start) or np.any(~np.isin(event,[0,1])):_fail('INVALID_RISK_INTERVAL','Require 0 <= entry < exit and binary event')
    return start,stop,event


def estimate_survival_curve(entry_time,exit_time,event_indicator,*,weight_policy,weights=None,confidence_level=.95,target,training,model_id,model_version,context=None):
    """Delayed-entry Kaplan–Meier with Greenwood log-log intervals."""
    start,stop,event=_times(entry_time,exit_time,event_indicator);w=np.ones(len(stop)) if weights is None else _numeric(weights,'weights',vector_only=True)
    if weight_policy!='frequency' or len(w)!=len(stop) or np.any(w<0) or np.any(w!=np.floor(w)) or not 0<w.sum()<2**53:_fail('INVALID_WEIGHTS','Nonnegative integer frequency weights with positive total required')
    c=_scalar(confidence_level,'confidence_level',0,1)
    if c in (0,1):_fail('INVALID_INPUT','Interior confidence level required')
    times=np.unique(stop[w>0]);survival=1.;greenwood=0.;curve=[];z=norm.ppf((1+c)/2)
    for t in times:
        risk=float(w[(start<t)&(stop>=t)].sum());d=float(w[(stop==t)&(event==1)].sum());censored=float(w[(stop==t)&(event==0)].sum())
        if risk<=0:_fail('EMPTY_RISK_SET','Observed exit has no active risk set')
        if d==risk:survival=0.;lower=upper=0.
        else:
            survival*=1-d/risk;greenwood+=d/(risk*(risk-d))
            if survival in (0,1):lower=upper=survival
            else:
                se=math.sqrt(greenwood)/abs(math.log(survival));eta=math.log(-math.log(survival));lower=math.exp(-math.exp(min(eta+z*se,709)));upper=math.exp(-math.exp(eta-z*se))
        curve.append({'time':float(t),'at_risk':risk,'events':d,'censored':censored,'survival':survival,'lower':lower,'upper':upper,'greenwood_sum':greenwood})
    gaps=[];right=None
    for a,b in sorted(zip(start[w>0],stop[w>0])):
        if right is not None and a>right:gaps.append({'start':float(right),'stop':float(a)})
        right=b if right is None else max(right,b)
    params={'kind':'kaplan_meier','risk_support_gaps':gaps,'curve':curve,'support_end':float(max(stop[w>0])),'support_start':float(min(start[w>0])),'entry_time':start.tolist(),'exit_time':stop.tolist(),'frequency_weights':w.tolist()}
    diag={'risk_interval':'(entry,exit]','confidence_level':c,'confidence_method':'greenwood_log_log','weight_policy':weight_policy,'independence_assumption':'independent_subjects_noninformative_censoring_and_truncation','no_events':not bool(np.any(event[w>0])),'boundary_intervals':'asymptotic_degenerate_not_exact_small_sample_bounds'}
    return _fit_result(_model('estimate_survival_curve',[],params,target,training,model_id,model_version,diag),'survival.estimate_survival_curve',diagnostics=diag,context=context)


def estimate_multistate_transitions(event_history,*,states,allowed_transitions,initial_state,target,training,model_id,model_version,context=None):
    """Counting-process Aalen–Johansen matrices, retaining repeated transitions."""
    rows=_rows(event_history);states=_strings(states,'states');index={s:i for i,s in enumerate(states)}
    if initial_state not in states:_fail('INVALID_STATE','Initial state is unknown')
    if not isinstance(allowed_transitions,(list,tuple)) or any(not isinstance(e,(list,tuple)) or len(e)!=2 or e[0] not in index or e[1] not in index or e[0]==e[1] for e in allowed_transitions):_fail('INVALID_GRAPH','Allowed directed non-self state pairs required')
    graph={tuple(e) for e in allowed_transitions}
    if len(graph)!=len(allowed_transitions):_fail('INVALID_GRAPH','Duplicate graph edge')
    data=[];entities={}
    for row in rows:
        entity=row.get('entity_id');origin=row.get('from_state');destination=row.get('to_state');start=_scalar(row.get('start'),'start',0);stop=_scalar(row.get('stop'),'stop',0)
        if not isinstance(entity,str) or not entity or stop<=start or origin not in index or (destination is not None and destination not in index):_fail('INVALID_RISK_INTERVAL','Named entity/state and positive at-risk interval required')
        if destination is not None and destination!=origin and (origin,destination) not in graph:_fail('FORBIDDEN_TRANSITION','Observed transition absent from graph')
        r={'entity_id':entity,'start':start,'stop':stop,'from_state':origin,'to_state':destination};data.append(r);entities.setdefault(entity,[]).append(r)
    for history in entities.values():
        history.sort(key=lambda r:r['start'])
        for previous,current in zip(history,history[1:]):
            expected=previous['to_state'] or previous['from_state']
            if current['start']!=previous['stop'] or current['from_state']!=expected:_fail('INVALID_HISTORY','Within-entity intervals must be contiguous and state-consistent')
    times=sorted({r['stop'] for r in data if r['to_state'] is not None and r['to_state']!=r['from_state']});increments=[];risk_sets=[];products=[];counts=[];P=np.eye(len(states))
    for t in times:
        risk=np.array([sum(r['from_state']==s and r['start']<t<=r['stop'] for r in data) for s in states]);count=np.zeros_like(P)
        for r in data:
            if r['stop']==t and r['to_state'] is not None and r['to_state']!=r['from_state']:count[index[r['from_state']],index[r['to_state']]]+=1
        if np.any((risk==0)&(count.sum(axis=1)>0)) or np.any(count.sum(axis=1)>risk):_fail('EMPTY_RISK_SET','Event counts inconsistent with state risk sets')
        A=np.divide(count,risk[:,None],out=np.zeros_like(count),where=risk[:,None]>0);np.fill_diagonal(A,-A.sum(axis=1));P=P@(np.eye(len(states))+A)
        increments.append(A.tolist());risk_sets.append(risk.tolist());counts.append(count.tolist());products.append(P.tolist())
    params={'kind':'aalen_johansen','states':states,'allowed_transitions':[list(x) for x in sorted(graph)],'initial_state':initial_state,'event_times':times,'hazard_increment_matrices':increments,'risk_sets':risk_sets,'transition_counts':counts,'transition_matrices':products,'state_occupancy':[v[index[initial_state]] for v in products],'support_start':float(min(r['start'] for r in data)),'support_end':float(max(r['stop'] for r in data)),'risk_intervals':data}
    diag={'estimator':'Aalen-Johansen','risk_interval':'(start,stop]','entity_count':len(entities),'interval_count':len(data),'transition_interpretation':'time_inhomogeneous_Markov_given_state','unobserved_risk_rows':'identity_increment_with_unidentified_transition_intensity','first_default':'separate_absorbing_first_hit_projection_not_current_state_occupancy','independence_assumption':'independent_entities_noninformative_censoring_and_truncation'}
    return _fit_result(_model('estimate_multistate_transitions',[],params,target,training,model_id,model_version,diag),'survival.estimate_multistate_transitions',diagnostics=diag,context=context)


def _panel(panel,X,feature_order,strata=None):
    rows=_rows(panel);features=_strings(feature_order,'feature_order',True);x=_numeric(X,'X',allow_empty=True)
    if x.ndim!=2 or x.shape!=(len(rows),len(features)):_fail('FEATURE_MISMATCH','X must match panel rows and declared columns')
    start,stop,event=_times([r.get('start') for r in rows],[r.get('stop') for r in rows],[r.get('event') for r in rows]);ids=[];histories={}
    groups=['all']*len(rows) if strata is None else list(strata)
    if len(groups)!=len(rows) or any(not isinstance(s,str) or not s for s in groups):_fail('INVALID_STRATA','One nonempty stratum label per row required')
    for i,r in enumerate(rows):
        entity=r.get('entity_id')
        if not isinstance(entity,str) or not entity:_fail('INVALID_ENTITY','Named entity required')
        if features and _scalar(r.get('covariate_time'),'covariate_time')>start[i]:_fail('LOOKAHEAD','Covariates must be available by interval start')
        ids.append(entity);histories.setdefault(entity,[]).append(i)
    for history in histories.values():
        history.sort(key=lambda i:start[i])
        for previous,current in zip(history,history[1:]):
            if event[previous]==1 or start[current]!=stop[previous]:_fail('INVALID_HISTORY','At-risk intervals must be contiguous and terminate at event')
    return rows,x,features,start,stop,event,np.array(ids),np.array(groups)


def _cox_statistics(start,stop,event,x,strata,beta,ties,weights=None):
    """Uncentered Breslow baseline and exact partial-likelihood time-interaction score."""
    w=np.ones(len(start)) if weights is None else weights;eta=x@beta;risk_score=np.exp(eta)
    if not np.all(np.isfinite(risk_score)):_fail('NUMERICAL_RANGE','Cox risk score overflow')
    baseline=[];p=x.shape[1];score=np.zeros(2*p);information=np.zeros((2*p,2*p));event_times=stop[event==1];center=float(event_times.mean()) if len(event_times) else 0.
    residuals=[]
    for s in sorted(set(strata)):
        for t in sorted(set(stop[(event==1)&(strata==s)])):
            risk=(start<t)&(stop>=t)&(strata==s);deaths=(stop==t)&(event==1)&(strata==s);d=int(deaths.sum());rr=risk_score[risk]*w[risk];dd=risk_score[deaths]*w[deaths];R=float(rr.sum());D=float(dd.sum());event_weight=float(w[deaths].sum())
            if R<=0:_fail('EMPTY_RISK_SET','Positive event without risk support')
            baseline.append({'stratum':s,'time':float(t),'increment':event_weight/R,'at_risk':int(risk.sum()),'events':d,'weighted_risk':R})
            if p and weights is None:
                z=np.column_stack([x,x*(t-center)]);zr=z[risk];zd=z[deaths];R1=rr@zr;R2=zr.T@(rr[:,None]*zr);D1=dd@zd;D2=zd.T@(dd[:,None]*zd);score+=zd.sum(axis=0);expected=np.zeros(p)
                for j in range(d):
                    fraction=j/d if ties=='efron' else 0.;den=R-fraction*D;mean=(R1-fraction*D1)/den;score-=mean;information+=(R2-fraction*D2)/den-np.outer(mean,mean);expected+=mean[:p]
                residuals.append({'stratum':s,'time':float(t),'events':d,'summed_schoenfeld':(x[deaths].sum(axis=0)-expected).tolist()})
    ph={'method':'efficient_score_for_identity_time_interactions','time_center':center,'applicability':'NOT_APPLICABLE','reason':'NO_COVARIATES_OR_WEIGHTED_IPCW','global_statistic':None,'global_p_value':None,'per_feature':[],'schoenfeld_event_sums':residuals}
    if p and weights is None:
        try:
            eff=information[p:,p:]-information[p:,:p]@np.linalg.solve(information[:p,:p],information[:p,p:]);u=score[p:]-information[p:,:p]@np.linalg.solve(information[:p,:p],score[:p]);eigen=np.linalg.eigvalsh(eff)
            if min(eigen)>max(eigen)*1e-10:
                stat=float(u@np.linalg.solve(eff,u));ph.update(applicability='APPLICABLE',reason=None,global_statistic=stat,global_p_value=float(chi2.sf(stat,p)),per_feature=[{'score':float(u[j]),'information':float(eff[j,j]),'statistic':float(u[j]**2/eff[j,j]),'p_value':float(chi2.sf(u[j]**2/eff[j,j],1))} for j in range(p)])
            else:ph['reason']='TIME_INTERACTION_NOT_IDENTIFIABLE'
        except np.linalg.LinAlgError:ph['reason']='TIME_INTERACTION_NOT_IDENTIFIABLE'
    return baseline,ph


def fit_cox_ph(start_stop_panel,X,*,feature_order,ties,strata=None,covariance='model_based',target,training,model_id,model_version,max_iter=200,tolerance=1e-9,context=None):
    """Counting-process Cox using established PHReg likelihood estimation."""
    rows,x,features,start,stop,event,ids,groups=_panel(start_stop_panel,X,feature_order,strata)
    if ties not in ('efron','breslow') or covariance not in ('model_based','cluster_robust'):_fail('INVALID_ESTIMATOR','Declare tie rule and covariance policy')
    if not np.any(event):_fail('INSUFFICIENT_EVENTS','Cox estimation requires an event')
    p=len(features);tol=_scalar(tolerance,'tolerance',0,1);iterations=_scalar(max_iter,'max_iter',1)
    if tol==0 or iterations!=int(iterations):_fail('INVALID_INPUT','Positive tolerance and integer max_iter required')
    if p:
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(x)),x]))!=p+1:_fail('NON_IDENTIFIABLE','Cox design contains an implicit intercept or collinearity')
        # PHReg includes entry ties; successor floats enforce the (start,stop] contract.
        entry=np.nextafter(start,np.inf)
        if np.any(entry>stop):_fail('INVALID_RISK_INTERVAL','Interval narrower than floating-point resolution')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                backend=PHReg(stop,x,status=event,entry=entry,strata=groups,ties=ties,missing='raise');fit=backend.fit(groups=ids if covariance=='cluster_robust' else None,maxiter=int(iterations),tol=tol,disp=False);beta=np.asarray(fit.params);info=-backend.hessian(beta);gradient=backend.score(beta);cov=np.asarray(fit.cov_params())
            except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:_fail('NUMERICAL_FAILURE','Cox likelihood fit failed',{'reason':str(exc)})
        if any('converge' in str(w.message).lower() for w in caught):_fail('NON_CONVERGENCE','Cox backend did not converge')
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(info)) or not np.all(np.isfinite(cov)):_fail('NON_CONVERGENCE','Finite identified Cox fit required')
        eigen=np.linalg.eigvalsh(info)
        if min(eigen)<=0:_fail('NON_IDENTIFIABLE','Cox observed information not positive definite')
        scaled=float(max(abs(gradient)/np.sqrt(np.diag(info))))
        if scaled>max(math.sqrt(tol),1e-6):_fail('NON_CONVERGENCE','Cox likelihood score above tolerance')
        ll=float(backend.loglike(beta));warnings_out=[str(w.message) for w in caught]
    else:beta=np.array([]);cov=np.empty((0,0));ll=None;scaled=0.;warnings_out=[]
    baseline,ph=_cox_statistics(start,stop,event,x,groups,beta,ties)
    params={'kind':'cox_ph','coefficients':beta.tolist(),'covariance':cov.tolist(),'ties':ties,'baseline_method':'breslow_uncentered','baseline':baseline,'strata':sorted(set(groups)),'support_start':float(min(start)),'support_end':float(max(stop)),'risk_intervals':[{'start':float(a),'stop':float(b),'stratum':s} for a,b,s in zip(start,stop,groups)]}
    diag={'backend':'statsmodels.PHReg' if p else 'closed_form_risk_sets','log_partial_likelihood':ll,'scaled_score':scaled,'converged':True,'covariance_type':covariance,'ph_diagnostic':ph,'ph_assumption':'independent_entities; pointwise score diagnostics are not proof of proportionality','warnings':warnings_out,'risk_interval':'(start,stop]','baseline_prediction':'exp(-Breslow_cumulative_hazard)','entry_boundary_adapter':'nextafter(start,+inf) only in backend risk-set lookup'}
    return _fit_result(_model('fit_cox_ph',features,params,target,training,model_id,model_version,diag),'survival.fit_cox_ph',diagnostics=diag,context=context)


def _aft_loglik(time,event,x,beta,scale,distribution):
    z=(np.log(time)-np.column_stack([np.ones(len(x)),x])@beta)/scale
    if distribution=='weibull':
        with np.errstate(over='ignore'):ez=np.exp(z)
        logS=-ez;logf=z-ez-np.log(time)-math.log(scale)
    elif distribution=='lognormal':logS=log_ndtr(-z);logf=-.5*z*z-.5*math.log(2*math.pi)-np.log(time)-math.log(scale)
    else:logS=-np.logaddexp(0,z);logf=-np.logaddexp(0,z)-np.logaddexp(0,-z)-np.log(time)-math.log(scale)
    return np.where(event==1,logf,logS)


def fit_aft_model(durations,events,X,*,feature_order,distribution,target,training,model_id,model_version,max_iter=500,tolerance=1e-9,context=None):
    """Three full censored AFT likelihoods via lifelines fitters; common log-time scale."""
    time=_numeric(durations,'durations',vector_only=True);event=_numeric(events,'events',vector_only=True);features=_strings(feature_order,'feature_order',True);x=_numeric(X,'X',allow_empty=True)
    if x.ndim!=2 or x.shape!=(len(time),len(features)) or len(event)!=len(time) or np.any(time<=0) or np.any(~np.isin(event,[0,1])):_fail('INVALID_SURVIVAL_DATA','Positive durations, binary events and aligned X required')
    if distribution not in ('weibull','lognormal','loglogistic'):_fail('INVALID_DISTRIBUTION','Declare Weibull, lognormal or loglogistic')
    if not np.any(event) or np.linalg.matrix_rank(np.column_stack([np.ones(len(x)),x]))!=len(features)+1:_fail('NON_IDENTIFIABLE','Events and full-rank AFT design required')
    from lifelines import WeibullAFTFitter,LogNormalAFTFitter,LogLogisticAFTFitter
    import pandas as pd
    names=['x'+str(i) for i in range(len(features))];frame=pd.DataFrame(x,columns=names);frame['duration']=time;frame['event']=event;fitter={'weibull':WeibullAFTFitter,'lognormal':LogNormalAFTFitter,'loglogistic':LogLogisticAFTFitter}[distribution]()
    iterations=_scalar(max_iter,'max_iter',1);tol=_scalar(tolerance,'tolerance',0,1)
    if iterations!=int(iterations) or tol==0:_fail('INVALID_INPUT','Positive integer max_iter and interior tolerance required')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:fitter.fit(frame,'duration','event',ancillary=False,fit_options={'maxiter':int(iterations),'ftol':tol})
        except Exception as exc:_fail('NUMERICAL_FAILURE','AFT likelihood estimation failed',{'reason':str(exc)})
    location={'weibull':'lambda_','lognormal':'mu_','loglogistic':'alpha_'}[distribution];ancillary={'weibull':'rho_','lognormal':'sigma_','loglogistic':'beta_'}[distribution];beta=np.array([fitter.params_.loc[(location,k)] for k in ['Intercept']+names]);a=float(fitter.params_.loc[(ancillary,'Intercept')]);scale=math.exp(a if distribution=='lognormal' else -a);cov=np.asarray(fitter.variance_matrix_);ll=_aft_loglik(time,event,x,beta,scale,distribution)
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(cov)) or min(np.linalg.eigvalsh(cov))<=0 or not np.all(np.isfinite(ll)):_fail('NON_IDENTIFIABLE','Finite positive-information AFT fit required')
    if not np.isclose(ll.sum(),fitter.log_likelihood_,rtol=1e-7,atol=1e-7):_fail('NUMERICAL_FAILURE','AFT parametrization disagrees with backend likelihood')
    order=[(location,k) for k in ['Intercept']+names]+[(ancillary,'Intercept')];cov=fitter.variance_matrix_.loc[order,order].to_numpy();sign=np.ones(len(order));sign[-1]=1 if distribution=='lognormal' else -1;cov=cov*np.outer(sign,sign)
    params={'kind':'aft','distribution':distribution,'coefficients':beta.tolist(),'log_time_scale':scale,'covariance_beta_log_scale':cov.tolist(),'support_start':0.,'support_end':float(max(time))}
    diag={'backend':'lifelines.'+type(fitter).__name__,'parameterization':'log(T)=intercept+X beta+scale*epsilon','log_likelihood':float(ll.sum()),'row_log_likelihood':ll.tolist(),'converged':True,'warnings':[str(w.message) for w in caught],'censoring':'independent_right_censoring','extrapolation':'parametric_tail_requires_explicit_prediction_policy'}
    return _fit_result(_model('fit_aft_model',features,params,target,training,model_id,model_version,diag),'survival.fit_aft_model',diagnostics=diag,context=context)


def fit_discrete_hazard(person_period_panel,X,*,feature_order,period_grid,link,partial_period_policy,target,training,model_id,model_version,max_iter=200,tolerance=1e-9,context=None):
    """Grouped-time logit/cloglog with explicit partial-period observation policy."""
    rows,x,features,start,stop,event,ids,groups=_panel(person_period_panel,X,feature_order);grid=_numeric(period_grid,'period_grid',vector_only=True)
    if len(grid)<2 or grid[0]<0 or np.any(np.diff(grid)<=0) or link not in ('logit','cloglog') or partial_period_policy not in ('exclude_censored','cloglog_exposure'):_fail('INVALID_PERIOD_CONTRACT','Increasing grid and explicit supported link/partial-period policy required')
    if partial_period_policy=='cloglog_exposure' and link!='cloglog':_fail('INVALID_PERIOD_CONTRACT','Exposure scaling requires cloglog')
    periods=[];keep=[];offset=[]
    for i,r in enumerate(rows):
        k=_scalar(r.get('period'),'period',0,len(grid)-2)
        if k!=int(k):_fail('INVALID_PERIOD_CONTRACT','Zero-based integer period index required')
        k=int(k)
        if not grid[k]<=start[i]<stop[i]<=grid[k+1]:_fail('INVALID_PERIOD_CONTRACT','Risk interval crosses its declared period')
        if r.get('interval_end_type') not in ('scheduled','censor') or (event[i] and r['interval_end_type']!='scheduled'):_fail('INVALID_PERIOD_CONTRACT','Binary interval outcomes require scheduled event windows; exact failure times need continuous likelihood')
        if partial_period_policy=='exclude_censored':
            if start[i]!=grid[k]:_fail('INVALID_PERIOD_CONTRACT','Grouped logit requires entry at the period boundary')
            complete=stop[i]==grid[k+1]
            if not complete and event[i]:_fail('INVALID_PERIOD_CONTRACT','Partial scheduled event windows require cloglog exposure policy')
            active=complete
        else:active=True
        periods.append(k);keep.append(active);offset.append(math.log((stop[i]-start[i])/(grid[k+1]-grid[k])) if partial_period_policy=='cloglog_exposure' else 0.)
    keep=np.array(keep);periods=np.array(periods);dummies=np.eye(len(grid)-1)[periods];design=np.column_stack([dummies,x]);names=['period_'+str(i) for i in range(len(grid)-1)]+features
    if len(set(names))!=len(names):_fail('FEATURE_MISMATCH','Feature collides with internal period effect name')
    from .classification import fit_binary_glm
    fitted=fit_binary_glm(design[keep],event[keep],feature_order=names,link=link,fit_intercept=False,target=target,training=training,model_id=model_id,model_version=model_version,offset=np.array(offset)[keep],max_iter=max_iter,tolerance=tolerance)
    coef=fitted.model['parameters']['coefficients'];params={'kind':'discrete_hazard','link':link,'period_grid':grid.tolist(),'period_intercepts':coef[:len(grid)-1],'coefficients':coef[len(grid)-1:],'covariance':fitted.model['parameters']['covariance'],'partial_period_policy':partial_period_policy,'support_start':float(grid[0]),'support_end':float(grid[-1])}
    diag={**fitted.diagnostics,'excluded_partial_periods':int(sum(~keep)),'risk_interval':'(start,stop]','event_time_semantics':'binary_event_within_prespecified_window','partial_period_assumption':'cloglog constant intensity within period; censor end is noninformative' if partial_period_policy=='cloglog_exposure' else 'incomplete periods excluded, no identification beyond observed complete-window population'}
    return _fit_result(_model('fit_discrete_hazard',features,params,target,training,model_id,model_version,diag),'survival.fit_discrete_hazard',diagnostics=diag,context=context)


def fit_cause_specific_hazards(start_stop_panel,X,event_types,*,feature_order,causes,ties,strata=None,target,training,model_id,model_version,covariance='model_based',max_iter=200,tolerance=1e-9,context=None):
    """Joint cause-specific Cox bundle; competing terminal events censor each cause fit."""
    rows=_rows(start_stop_panel);causes=_strings(causes,'causes')
    if len(event_types)!=len(rows) or any(e is not None and e not in causes for e in event_types):_fail('INVALID_CAUSE','Null censoring or declared cause required for each row')
    terminal=[dict(r,event=int(e is not None)) for r,e in zip(rows,event_types)];_panel(terminal,X,feature_order,strata);models=[]
    for cause in causes:
        specific=[dict(r,event=int(e==cause)) for r,e in zip(rows,event_types)];fit=fit_cox_ph(specific,X,feature_order=feature_order,ties=ties,strata=strata,covariance=covariance,target=target,training=training,model_id=model_id+':'+cause,model_version=model_version,max_iter=max_iter,tolerance=tolerance)
        models.append({'cause':cause,'parameters':fit.model['parameters'],'diagnostics':fit.diagnostics})
    params={'kind':'cause_specific','causes':causes,'cause_models':models,'support_start':min(m['parameters']['support_start'] for m in models),'support_end':min(m['parameters']['support_end'] for m in models),'joint_integration':'exponential_total_increment_with_proportional_cause_allocation'}
    diag={'ties':ties,'competing_event_fit_policy':'censor_at_competing_terminal_event','prediction':'all_cause_hazards_integrated_jointly; cause_CIF_is_not_1_minus_cause_specific_survival'}
    return _fit_result(_model('fit_cause_specific_hazards',feature_order,params,target,training,model_id,model_version,diag),'survival.fit_cause_specific_hazards',diagnostics=diag,context=context)


def _fine_gray_expansion(time,events,x,ids,cause,censor_groups):
    target_times=sorted(set(time[np.array([e==cause for e in events])]))
    censor_curves={};expanded=[];rows_x=[];weights=[]
    for group in sorted(set(censor_groups)):
        active=censor_groups==group;curve=[];G=1.
        for t in sorted(set(time[active])):
            censored=active&(time==t)&np.array([e is None for e in events]);risk=int(sum(active&(time>t)))+int(sum(censored));d=int(sum(censored))
            # Terminal events occur before censoring at tied times (reverse KM).
            if d:G*=1-d/risk;curve.append({'time':float(t),'survival':G,'risk':risk,'censored':d})
        censor_curves[group]=curve
    def Gleft(t,group):
        before=[r['survival'] for r in censor_curves[group] if r['time']<t]
        return before[-1] if before else 1.
    previous=0.
    for t in target_times:
        for i,T in enumerate(time):
            if T>=t:w=1.
            elif events[i] is not None and events[i]!=cause:
                denominator=Gleft(T,censor_groups[i])
                if denominator<=0:_fail('INSUFFICIENT_CENSORING_SUPPORT','Positive censoring survival required for retained competing event')
                w=Gleft(t,censor_groups[i])/denominator
            else:continue
            if w<=0:continue
            expanded.append({'entity_id':ids[i],'start':previous,'stop':float(t),'event':int(T==t and events[i]==cause),'covariate_time':0.});rows_x.append(x[i]);weights.append(w)
        previous=t
    return expanded,np.asarray(rows_x).reshape(len(rows_x),x.shape[1]),np.array(weights),censor_curves


def fit_fine_gray(event_times,event_types,X,*,cause_of_interest,feature_order,censoring_spec,target,training,model_id,model_version,max_iter=200,tolerance=1e-9,context=None):
    """Fine–Gray IPCW subdistribution regression with explicit censoring strata."""
    time=_numeric(event_times,'event_times',vector_only=True);features=_strings(feature_order,'feature_order',True);x=_numeric(X,'X',allow_empty=True)
    if x.ndim!=2 or x.shape!=(len(time),len(features)) or len(event_types)!=len(time) or np.any(time<=0) or any(e is not None and (not isinstance(e,str) or not e) for e in event_types):_fail('INVALID_SURVIVAL_DATA','Positive event/censor times and aligned baseline covariates required')
    if not isinstance(cause_of_interest,str) or not cause_of_interest or cause_of_interest not in event_types:_fail('INSUFFICIENT_EVENTS','Observed cause of interest required')
    if not isinstance(censoring_spec,Mapping) or censoring_spec.get('estimator')!='kaplan_meier' or censoring_spec.get('independence_assumption')!='independent_within_censoring_group':_fail('INVALID_CENSORING_MODEL','Declare within-group independent reverse-KM censoring model')
    groups=np.array(censoring_spec.get('groups',['all']*len(time)))
    if groups.shape!=(len(time),) or any(not isinstance(s,str) or not s for s in groups):_fail('INVALID_CENSORING_MODEL','One censoring group per subject required')
    iterations=_scalar(max_iter,'max_iter',1);tol=_scalar(tolerance,'tolerance',0,1)
    if iterations!=int(iterations) or tol==0:_fail('INVALID_INPUT','Positive integer max_iter and tolerance required')
    ids=['subject_'+str(i) for i in range(len(time))];rows,xx,w,curves=_fine_gray_expansion(time,list(event_types),x,ids,cause_of_interest,groups);start=np.array([r['start'] for r in rows]);stop=np.array([r['stop'] for r in rows]);event=np.array([r['event'] for r in rows]);p=len(features)
    if p:
        from lifelines import CoxPHFitter
        import pandas as pd
        names=['x'+str(i) for i in range(p)];frame=pd.DataFrame(xx,columns=names);frame['start']=start;frame['stop']=stop;frame['event']=event;frame['weight']=w
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:fit=CoxPHFitter().fit(frame,'stop','event',entry_col='start',weights_col='weight',robust=False,fit_options={'max_steps':int(max_iter),'precision':tolerance});beta=fit.params_.to_numpy();cov=fit.variance_matrix_.to_numpy()
            except Exception as exc:_fail('NUMERICAL_FAILURE','Fine-Gray weighted Cox fit failed',{'reason':str(exc)})
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(cov)) or min(np.linalg.eigvalsh(cov))<=0:_fail('NON_IDENTIFIABLE','Finite identified Fine-Gray fit required')
        warning_strings=[str(v.message) for v in caught];ll=float(fit.log_likelihood_)
        if any('converge' in s.lower() for s in warning_strings):_fail('NON_CONVERGENCE','Weighted Cox convergence warning',{'warnings':warning_strings})
    else:beta=np.array([]);cov=np.empty((0,0));ll=None;warning_strings=[]
    baseline,_=_cox_statistics(start,stop,event,xx,np.array(['all']*len(rows)),beta,'efron',weights=w)
    params={'kind':'fine_gray','cause':cause_of_interest,'coefficients':beta.tolist(),'conditional_ipcw_covariance':cov.tolist(),'ties':'efron','baseline_method':'breslow_uncentered','baseline':baseline,'strata':['all'],'censoring_curves':curves,'support_start':0.,'support_end':float(max(time)),'risk_intervals':[{'start':0.,'stop':float(max(time)),'stratum':'all'}]}
    diag={'backend':'lifelines.CoxPHFitter_weighted_delayed_entry' if p else 'closed_form_IPCW_risk_sets','log_partial_likelihood':ll,'expanded_rows':len(rows),'expanded_risk_rows':rows,'expanded_weights':w.tolist(),'censoring_spec':dict(censoring_spec),'censor_tie_order':'terminal_events_before_censoring','weight_formula':'G(t-)/G(T-) for retained competing failures, otherwise one while observed at risk','covariance_limitation':'conditional_on_estimated_IPCW; not full sampling covariance, resample original subjects and refit censoring for inference','warnings':warning_strings,'prediction':'selected-cause CIF, complement is not all-event survival','covariates':'baseline fixed; delayed entry and future time-varying competing-subject covariates not inferred'}
    return _fit_result(_model('fit_fine_gray',features,params,target,training,model_id,model_version,diag),'survival.fit_fine_gray',diagnostics=diag,context=context)


def _prediction_paths(paths,features,start,end):
    rows=_rows(paths);grouped={}
    for r in rows:
        key=r.get('path_id');a=_scalar(r.get('start'),'path.start',0);b=_scalar(r.get('stop'),'path.stop',0);x=_numeric(r.get('covariates'),'covariates',allow_empty=True,vector_only=True);s=r.get('stratum','all')
        if not isinstance(key,str) or not key or b<=a or len(x)!=len(features) or not isinstance(s,str) or not s:_fail('INVALID_COVARIATE_PATH','Named paths, positive intervals and aligned covariates required')
        if _scalar(r.get('covariate_time'),'covariate_time')>a:_fail('LOOKAHEAD','Prediction covariates must be known by interval start')
        grouped.setdefault(key,[]).append({'start':a,'stop':b,'x':x,'stratum':s})
    for part in grouped.values():
        part.sort(key=lambda r:r['start'])
        if part[0]['start']>start or part[-1]['stop']<end or any(a['stop']!=b['start'] for a,b in zip(part,part[1:])):_fail('INCOMPLETE_PATH','Prediction path must cover the complete requested horizon contiguously')
    return grouped


def predict_event_curves(fitted_model,covariate_paths,horizon_grid,*,feature_order,target,initial_state=None,event_states=None,start_time=0,extrapolation='error',quantile_levels=None,context=None):
    """Apply data-only frozen event models; horizons are elapsed from the landmark."""
    model=_model_artifact(fitted_model);features=_strings(feature_order,'feature_order',True);grid=_numeric(horizon_grid,'horizon_grid',vector_only=True);start=_scalar(start_time,'start_time',0);params=model['parameters'];kind=params.get('kind')
    if features!=model['feature_order'] or _fingerprint(target)!=_fingerprint(model['target']):_fail('MODEL_CONTRACT_MISMATCH','Feature order and full event target must match artifact')
    if not model['method_id'].startswith('survival.') or kind not in ('kaplan_meier','aalen_johansen','cox_ph','aft','discrete_hazard','cause_specific','fine_gray'):_fail('UNSUPPORTED_MODEL','Supported frozen survival artifact required')
    if np.any(grid<0) or np.any(np.diff(grid)<=0):_fail('INVALID_HORIZON','Nonnegative strictly increasing elapsed horizons required')
    if extrapolation not in ('error','flat','parametric') or (extrapolation=='parametric' and kind!='aft'):_fail('INVALID_EXTRAPOLATION','Parametric tails require AFT; other models support error or explicit flat continuation')
    end=start+float(grid[-1]);outside=end>params['support_end'] or start<params['support_start']
    if outside and extrapolation=='error':_fail('OUTSIDE_SUPPORT','Requested landmark/horizon outside observed support')
    levels=None if quantile_levels is None else _numeric(quantile_levels,'quantile_levels',vector_only=True)
    if levels is not None and (kind!='aft' or extrapolation!='parametric' or np.any(levels<=0) or np.any(levels>=1) or np.any(np.diff(levels)<=0)):_fail('INVALID_QUANTILES','Ordered interior AFT quantiles require explicit parametric-tail policy')
    curves=[];unsupported=[];unsupported_times=[]
    if kind in ('kaplan_meier','aalen_johansen'):
        if covariate_paths not in ([],()):_fail('UNSUPPORTED_COVARIATES','Nonparametric population artifact requires empty covariate paths')
        if kind=='kaplan_meier':
            entry=np.array(params['entry_time']);exit=np.array(params['exit_time']);w=np.array(params['frequency_weights'])
            if not np.any((entry<=start)&(exit>start)&(w>0)) and extrapolation=='error':_fail('INSUFFICIENT_RISK_SUPPORT','No observed risk support at landmark')
            unsupported_times=[g for g in params.get('risk_support_gaps',[]) if max(start,g['start'])<min(end,g['stop'])]
            if unsupported_times and extrapolation=='error':_fail('INSUFFICIENT_RISK_SUPPORT','Prediction crosses an empty observation risk interval')
            def S(t):
                v=[r['survival'] for r in params['curve'] if r['time']<=t];return v[-1] if v else 1.
            base=S(start)
            if base<=0:_fail('IMPOSSIBLE_CONDITIONING','Cannot condition on zero surviving mass')
            surv=np.array([S(start+h)/base for h in grid]);F=1-surv;curves=[{'path_id':'population','survival':surv.tolist(),'cumulative_incidence':{target['event']:F.tolist()},'marginal_probabilities':{target['event']:np.diff(np.r_[0,F]).tolist()}}]
        else:
            states=params['states'];initial=params['initial_state'] if initial_state is None else initial_state;marked=_strings(event_states if event_states is not None else [target['event']],'event_states');n=len(states)
            if initial not in states or any(s not in states for s in marked):_fail('INVALID_STATE','Initial and marked states must be known')
            origin=states.index(initial);mark=[states.index(s) for s in marked];outgoing={e[0] for e in params['allowed_transitions']};P=np.eye(n);Q=np.eye(n);products=[];occupancy=[];hit=[];position=0;times=params['event_times']
            while position<len(times) and times[position]<=start:position+=1
            if not any(r['from_state']==initial and r['start']<=start<r['stop'] for r in params['risk_intervals']) and extrapolation=='error':_fail('INSUFFICIENT_RISK_SUPPORT','Initial-state landmark risk support absent')
            for h in grid:
                while position<len(times) and times[position]<=start+h:
                    A=np.array(params['hazard_increment_matrices'][position]);risk=np.array(params['risk_sets'][position]);missing=[states[i] for i in range(n) if P[origin,i]>1e-12 and risk[i]==0 and states[i] in outgoing]
                    if missing:
                        unsupported.append({'time':times[position],'states':missing})
                        if extrapolation=='error':_fail('INSUFFICIENT_RISK_SUPPORT','Positive state mass has an unobserved outgoing hazard',unsupported[-1])
                    M=np.eye(n)+A;P=P@M;Mhit=M.copy();Mhit[mark]=np.eye(n)[mark];Q=Q@Mhit;position+=1
                products.append(P.tolist());occupancy.append(P[origin].tolist());hit.append(float(Q[origin,mark].sum()))
            F=np.array(hit);base=1. if initial in marked else 0.;curves=[{'path_id':'population','states':states,'state_occupancy':occupancy,'transition_matrices':products,'first_hit_probability':hit,'survival':(1-F).tolist(),'cumulative_incidence':{'first_hit':hit},'marginal_probabilities':{'first_hit':np.diff(np.r_[base,F]).tolist()}}]
    else:
        paths=_prediction_paths(covariate_paths,features,start,end)
        for key,part in sorted(paths.items()):
            def at(t):
                candidates=[r for r in part if r['start']<t<=r['stop']]
                if not candidates and t==start:candidates=[r for r in part if r['start']<=t<r['stop']]
                if not candidates:_fail('INCOMPLETE_PATH','Covariate path lacks a required risk instant')
                return candidates[-1]
            quantiles=None
            if kind=='aft':
                if any(not np.array_equal(r['x'],part[0]['x']) for r in part):_fail('UNSUPPORTED_COVARIATES','AFT requires a fixed profile, not an inferred time-varying acceleration law')
                eta=float(np.r_[1.,part[0]['x']]@np.array(params['coefficients']));scale=params['log_time_scale'];distribution=params['distribution']
                def logS(t):
                    if extrapolation=='flat':t=min(t,params['support_end'])
                    if t==0:return 0.
                    z=(math.log(t)-eta)/scale
                    if distribution=='weibull':return -math.exp(z) if z<709 else -math.inf
                    if distribution=='lognormal':return float(log_ndtr(-z))
                    return -float(np.logaddexp(0,z))
                base=logS(start)
                if not math.isfinite(base):_fail('IMPOSSIBLE_CONDITIONING','Landmark survival underflows')
                surv=np.array([math.exp(min(0.,logS(start+h)-base)) for h in grid]);cifs={target['event']:1-surv}
                if levels is not None:
                    logtail=base+np.log1p(-levels)
                    zq=np.log(-logtail) if distribution=='weibull' else -ndtri_exp(logtail) if distribution=='lognormal' else np.log(-np.expm1(logtail))-logtail
                    logq=eta+scale*zq
                    if np.any(logq>709):_fail('NUMERICAL_RANGE','AFT quantile exceeds finite time range')
                    quantiles=(np.exp(logq)-start).tolist()
            elif kind=='discrete_hazard':
                boundaries=np.array(params['period_grid']);beta=np.array(params['coefficients']);surv=[];S=1.;previous=start
                for h in grid:
                    endpoint=start+h;cuts=sorted({previous,endpoint}|{float(t) for t in boundaries if previous<t<endpoint}|{r['stop'] for r in part if previous<r['stop']<endpoint})
                    for a,b in zip(cuts,cuts[1:]):
                        k=int(np.searchsorted(boundaries,a,side='right')-1)
                        if k<0 or k>=len(boundaries)-1:
                            if extrapolation!='flat':_fail('OUTSIDE_SUPPORT','Discrete period not fitted')
                            continue
                        r=at(b);eta=float(params['period_intercepts'][k]+r['x']@beta);fraction=(b-a)/(boundaries[k+1]-boundaries[k])
                        if params['link']=='logit':
                            if a!=boundaries[k] or b!=boundaries[k+1]:_fail('INVALID_PERIOD_CONTRACT','Logit forecasts require full fitted periods and boundary-aligned covariates')
                            S*=float(expit(-eta))
                        else:S*=math.exp(-math.exp(min(eta,709))*fraction)
                    surv.append(S);previous=endpoint
                surv=np.array(surv);cifs={target['event']:1-surv}
            else:
                causes=params['causes'] if kind=='cause_specific' else [params['cause'] if kind=='fine_gray' else target['event']]
                components=[m['parameters'] for m in params['cause_models']] if kind=='cause_specific' else [params]
                for r in part:
                    if any(r['stratum'] not in m['strata'] for m in components):_fail('UNKNOWN_STRATUM','Prediction stratum not fitted')
                times=sorted({b['time'] for m in components for b in m['baseline'] if b['time']>start});position=0;S=1.;F=np.zeros(len(causes));surv=[];allF=[]
                for h in grid:
                    while position<len(times) and times[position]<=start+h:
                        t=times[position];r=at(t);deltas=[]
                        for m in components:
                            base=sum(b['increment'] for b in m['baseline'] if b['stratum']==r['stratum'] and b['time']==t);eta=float(r['x']@np.array(m['coefficients']))
                            if eta>709:_fail('NUMERICAL_RANGE','Predicted hazard overflows')
                            deltas.append(base*math.exp(eta))
                        total=sum(deltas)
                        if total:F+=S*(-math.expm1(-total))*np.array(deltas)/total;S*=math.exp(-total)
                        position+=1
                    surv.append(S);allF.append(F.copy())
                surv=np.array(surv);cifs={cause:np.array(allF)[:,i] for i,cause in enumerate(causes)}
            curves.append({'path_id':key,'survival':surv.tolist(),'cumulative_incidence':{k:v.tolist() for k,v in cifs.items()},'marginal_probabilities':{k:np.diff(np.r_[0.,v]).tolist() for k,v in cifs.items()}})
            if quantiles is not None:curves[-1]['conditional_event_quantiles']=quantiles
    value={'horizon_grid':grid.tolist(),'start_time':start,'curves':curves}
    if levels is not None:value['quantile_levels']=levels.tolist()
    return _result(value,'survival.predict_event_curves','probability',context,diagnostics={'model_id':model['model_id'],'model_version':model['model_version'],'kind':kind,'extrapolation':extrapolation,'outside_observed_time_support':outside,'unsupported_state_times':unsupported,'unsupported_time_intervals':unsupported_times,'survival_meaning':'complement_of_selected_cause_CIF_not_all_event_survival' if kind=='fine_gray' else 'no_first_hit_in_marked_states' if kind=='aalen_johansen' else 'event_free_conditional_on_landmark','curve_integration':'joint_exponential_hazard_increments' if kind in ('cox_ph','cause_specific','fine_gray') else 'frozen_model','state_transition_assumption':'time_inhomogeneous_Markov_given_landmark_state' if kind=='aalen_johansen' else None},limitations=('Frozen forecasts preserve their model target and observation clock. Empirical flat continuation is an explicit extrapolation assumption. First-hit incidence is distinct from reversible-state occupancy.',),provenance=model['provenance'])
