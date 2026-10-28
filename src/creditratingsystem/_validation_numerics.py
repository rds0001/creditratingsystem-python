"""Sampling error, grid differences and evaluated frozen-model sensitivity."""
from copy import deepcopy
import numpy as np
from scipy.stats import norm
from .core import _fail,_fingerprint,_numeric,_result
from ._expert_common import _ex_object,_ex_ids
from ._validation_common import _vb_design,_vb_string,_vb_number


def simulation_convergence(runs,shared_innovations,*,sampling_contract,precision,design,context=None):
    """Separate iid Monte Carlo uncertainty from paired resolution differences."""
    d=_vb_design(design,context,False);contract=deepcopy(_ex_object(sampling_contract,'sampling_contract'));p=deepcopy(_ex_object(precision,'precision'));innovations=_ex_object(shared_innovations,'shared_innovations')
    if contract.get('sampling_unit')!='iid_paths_with_independent_replicates' or contract.get('metric')not in ('mean','bernoulli_probability'):_fail('CONVERGENCE_SAMPLING','Declare iid path sampling, independent replicates and metric')
    _vb_string(contract.get('reference'),'sampling reference');finest=_vb_number(contract.get('finest_time_step'),'finest_time_step',0)
    if finest==0:_fail('CONVERGENCE_GRID','Finest time step must be positive')
    if contract.get('time_unit')!=d['target']['time_unit']:_fail('CONVERGENCE_GRID','Time step units must match target units')
    level=_vb_number(p.get('confidence_level'),'confidence_level');tol=_vb_number(p.get('mc_half_width'),'mc_half_width',0);gridtol=_vb_number(p.get('grid_difference_tolerance'),'grid_difference_tolerance',0)
    if not 0<level<1:_fail('CONVERGENCE_PRECISION','Interior confidence level required')
    z=float(norm.ppf((1+level)/2));matrices={};path_ids={}
    for rep,entry in innovations.items():
        _vb_string(rep,'replicate_id');entry=_ex_object(entry,'shared innovation entry');v=_numeric(entry.get('values'),'shared innovations');ids=_ex_ids(entry.get('path_ids'),'path_ids')
        if v.ndim!=2 or v.shape[0]!=len(ids)or v.shape[1]<1:_fail('CONVERGENCE_INNOVATIONS','Finite path-by-innovation matrix and aligned unique IDs required')
        matrices[rep]=v;path_ids[rep]=ids
    if not isinstance(runs,(list,tuple))or not runs:_fail('CONVERGENCE_RUNS','Nonempty run array required')
    indexed={};trace=[]
    for run in runs:
        run=_ex_object(run,'run');rep=run.get('replicate_id');dt=_vb_number(run.get('time_step'),'time_step',0);N=_vb_number(run.get('sample_size'),'sample_size',2);values=_numeric(run.get('values'),'run values')
        if rep not in matrices or N!=int(N)or dt<=0 or abs(dt/finest-round(dt/finest))>1e-10:_fail('CONVERGENCE_RUN','Known replicate, integer path count and nested time-step multiple required')
        N=int(N);ids=_ex_ids(run.get('path_ids'),'run path IDs')
        if values.ndim!=1 or len(values)!=N or ids!=path_ids[rep][:N]or N>len(path_ids[rep]):_fail('CONVERGENCE_NESTING','Sample values and IDs must be the declared path prefix')
        expected=_fingerprint(matrices[rep][:N].tolist())
        if run.get('innovation_fingerprint')!=expected:_fail('CONVERGENCE_NESTING','Run innovation fingerprint differs from shared path prefix')
        if contract['metric']=='bernoulli_probability'and np.any((values!=0)&(values!=1)):_fail('CONVERGENCE_BERNOULLI','Bernoulli path values must be zero or one')
        key=(dt,N,rep)
        if key in indexed:_fail('CONVERGENCE_DUPLICATE','Duplicate resolution/sample/replicate run')
        indexed[key]=values;trace.append(dict(replicate_id=rep,time_step=dt,sample_size=N,path_ids=ids,innovation_fingerprint=expected))
    levels=sorted({(dt,N)for dt,N,_ in indexed});reps=sorted(matrices)
    if any(set(rep for t,n,rep in indexed if(t,n)==key)!=set(reps)for key in levels):_fail('CONVERGENCE_REPLICATES','Every reported sample/resolution must contain every declared replicate')
    def summary(v,bernoulli=False):
        N=len(v);mean=float(np.mean(v));se=float(np.std(v,ddof=1)/np.sqrt(N));half=z*se;lower=mean-half;upper=mean+half;method='IID_NORMAL_APPROXIMATION'
        if bernoulli:
            den=1+z*z/N;center=(mean+z*z/(2*N))/den;half=z*np.sqrt(mean*(1-mean)/N+z*z/(4*N*N))/den;lower=max(0.,center-half);upper=min(1.,center+half);method='IID_WILSON_SCORE';half=max(mean-lower,upper-mean)
        if not np.all(np.isfinite([mean,se,lower,upper,half])):_fail('CONVERGENCE_NUMERICAL','Sampling statistic exceeds finite range')
        return dict(n=N,estimate=mean,mc_standard_error=se,lower=float(lower),upper=float(upper),maximum_interval_radius=float(half),interval_method=method,mc_precision_met=bool(half<=tol))
    estimates=[]
    for dt,N in levels:
        values=np.concatenate([indexed[(dt,N,rep)]for rep in reps]);s=summary(values,contract['metric']=='bernoulli_probability');s.update(time_step=dt,paths_per_replicate=N,replicate_estimates={rep:float(np.mean(indexed[(dt,N,rep)]))for rep in reps});estimates.append(s)
    comparisons=[]
    for N in sorted({N for _,N in levels}):
        grids=sorted(t for t,n in levels if n==N)
        for fine,coarse in zip(grids[:-1],grids[1:]):
            difference=np.concatenate([indexed[(coarse,N,rep)]-indexed[(fine,N,rep)]for rep in reps]);s=summary(difference);s.pop('mc_precision_met');s.update(coarse_time_step=coarse,fine_time_step=fine,paths_per_replicate=N,orientation='COARSE_MINUS_FINE',absolute_difference_upper_bound=max(abs(s['lower']),abs(s['upper'])),observed_grid_tolerance_met=max(abs(s['lower']),abs(s['upper']))<=gridtol);comparisons.append(s)
    reference=contract.get('analytic_reference');analytic=None
    if reference is not None:
        reference=_ex_object(reference,'analytic_reference');value=_vb_number(reference.get('value'),'analytic reference');_vb_string(reference.get('source'),'analytic source');analytic=[dict(time_step=s['time_step'],paths_per_replicate=s['paths_per_replicate'],absolute_error=abs(s['estimate']-value),reference_in_interval=s['lower']<=value<=s['upper'])for s in estimates]
    out=dict(artifact_type='ValidationResult',metrics=dict(sampling=estimates,paired_grid_differences=comparisons,analytic_reference_comparisons=analytic),design=d,sampling_contract=contract,precision=p,innovation_nesting=trace,uncertainty=dict(independence='DECLARED_NOT_AUTHENTICATED',confidence_level=level,multiplicity='UNADJUSTED_DIAGNOSTIC_INTERVALS'),discretization_bias_identified=False,decision=dict(status='DIAGNOSTIC_ONLY',reason='Successive grid differences do not bound error to continuous-time limit'),applicability='APPLICABLE',input_fingerprint=_fingerprint(runs),shared_innovation_fingerprint=_fingerprint(innovations))
    return _result(out,'validation.simulation_convergence','simulation_precision',context,limitations=('More paths reduce sampling error, not discretization bias. Fingerprints verify declared shared inputs and path nesting, not execution by an external simulator. Normal intervals for general means are asymptotic; Wilson intervals avoid zero-width zero-default inference.',))


def sensitivity_analysis(model,baseline,perturbations,*,method,shared_samples,design,context=None):
    """Evaluate frozen PD models or typed graphs under explicit numeric interventions."""
    from .classification import predict_pd
    from ._hybrid_graph import evaluate_model_graph
    from .core import _model_artifact,_json_data
    from .data import _date
    d=_vb_design(design,context,False);descriptor=deepcopy(_ex_object(model,'evaluation model'));x=_numeric(baseline,'baseline',vector_only=True);specs=deepcopy(perturbations)
    if not isinstance(specs,(list,tuple))or len(specs)!=len(x)or not len(x):_fail('SENSITIVITY_DOMAIN','One perturbation specification per nonempty baseline coordinate required')
    names=_ex_ids([_ex_object(s,'perturbation').get('name')for s in specs],'coordinate names');lower=[];upper=[];steps=[];locations=[];kind=descriptor.get('kind')
    if kind=='pd_model':
        fitted=_model_artifact(descriptor.get('fitted_model'));descriptor['fitted_model']=fitted
        if fitted['target']!=d['target']or _date(fitted['training']['known_at'],'model known_at')>_date(d['evaluation_cutoff'],'evaluation_cutoff'):_fail('SENSITIVITY_MODEL','Frozen model target/date differs from validation design')
        if names!=fitted['feature_order']:_fail('SENSITIVITY_FEATURES','Coordinate names must equal complete trained feature order')
    elif kind=='model_graph':
        for key in ('model_graph','input_data','parameter_snapshots','execution_context'):_ex_object(descriptor.get(key),key)
    else:_fail('SENSITIVITY_MODEL','Use frozen pd_model or model_graph evaluator')
    def get_path(obj,path):
        try:
            for key in path:obj=obj[key]
        except (KeyError,IndexError,TypeError):_fail('SENSITIVITY_PATH','Unknown numeric intervention path')
        return obj
    for i,spec in enumerate(specs):
        a=_vb_number(spec.get('lower'),'lower');b=_vb_number(spec.get('upper'),'upper');h=_vb_number(spec.get('step'),'step',0)
        if not a<=x[i]<=b or a>=b or h<=0:_fail('SENSITIVITY_DOMAIN','Baseline must lie in nondegenerate bounds and step be positive')
        lower.append(a);upper.append(b);steps.append(h)
        if kind=='pd_model':
            if spec.get('location')!='feature':_fail('SENSITIVITY_PATH','PD model coordinates modify features; use graph parameters for parameter interventions')
            locations.append(None)
        else:
            path=spec.get('path');location=spec.get('location')
            if not isinstance(path,list)or not path or any(not((type(k)is int and k>=0)or(isinstance(k,str)and k))for k in path):_fail('SENSITIVITY_PATH','Explicit nonnegative zero-based numeric indices and nonempty field names required')
            if location=='input':
                if len(path)<3 or path[1]!='value':_fail('SENSITIVITY_PATH','Input interventions may change value leaves only')
                root='input_data'
            elif location=='parameter':
                if len(path)<4 or path[0]!='nodes'or type(path[1])is not int or path[2]!='parameters':_fail('SENSITIVITY_PATH','Parameter interventions may change declared node parameter leaves only')
                root='model_graph'
            else:_fail('SENSITIVITY_PATH','Graph coordinates must identify input or parameter leaves')
            value=_vb_number(get_path(descriptor[root],path),'baseline graph leaf')
            if value!=x[i]:_fail('SENSITIVITY_BASELINE','Baseline must equal frozen graph numeric leaf')
            locations.append((root,path))
    if kind=='model_graph'and len({_fingerprint(v)for v in locations})!=len(locations):_fail('SENSITIVITY_PATH','Duplicate intervention paths')
    lower=np.array(lower);upper=np.array(upper);cache={};trace=[]
    def evaluate(v):
        v=np.asarray(v,float)
        if v.shape!=x.shape or np.any(~np.isfinite(v))or np.any((v<lower)|(v>upper)):_fail('SENSITIVITY_DOMAIN','Intervention outside declared coordinate domain')
        key=_fingerprint(v.tolist())
        if key in cache:return cache[key]
        if kind=='pd_model':result=float(predict_pd(descriptor['fitted_model'],[v.tolist()],feature_order=names,expected_target=d['target'],context=context).value[0])
        else:
            altered=deepcopy(descriptor)
            for value,(root,path)in zip(v,locations):
                parent=get_path(altered[root],path[:-1]);parent[path[-1]]=float(value)
            replay=evaluate_model_graph(altered['model_graph'],altered['input_data'],altered['parameter_snapshots'],context=altered['execution_context']).value;final=replay['final'];row=descriptor.get('output_row',0)
            if type(row)is not int or row<0 or row>=len(final['value'])or final['contract']['unit']=='design':_fail('SENSITIVITY_OUTPUT','Scalar final output row required')
            if final['contract']['target']!=d['target']or replay['context']['known_at']!=d['evaluation_cutoff']:_fail('SENSITIVITY_TARGET','Graph output target/cutoff differs from declared design')
            result=_vb_number(final['value'][row],'graph output')
        cache[key]=result;trace.append(dict(coordinates=v.tolist(),value=result));return result
    base=evaluate(x);result={};samples_info=None
    if method=='finite_difference':
        if shared_samples is not None:_fail('SENSITIVITY_SAMPLES','Local deterministic differentiation does not consume random samples')
        derivatives=[]
        for j,(name,h)in enumerate(zip(names,steps)):
            left=x[j]-lower[j];right=upper[j]-x[j]
            if min(left,right)>0:scheme='central';h=min(h,left,right)
            elif right>0:scheme='forward';h=min(h,right/2)
            else:scheme='backward';h=min(h,left/2)
            def derivative(step):
                direction=np.zeros(len(x));direction[j]=step
                if scheme=='central':return(evaluate(x+direction)-evaluate(x-direction))/(2*step)
                if scheme=='forward':return(-3*base+4*evaluate(x+direction)-evaluate(x+2*direction))/(2*step)
                return(3*base-4*evaluate(x-direction)+evaluate(x-2*direction))/(2*step)
            if h/2==0 or (x[j]+h/2==x[j] and x[j]-h/2==x[j]):_fail('SENSITIVITY_NUMERICAL','Step is not representable at baseline scale')
            full=derivative(h);half=derivative(h/2)
            if not np.all(np.isfinite([full,half])):_fail('SENSITIVITY_NUMERICAL','Derivative exceeds numerical range')
            derivatives.append(dict(coordinate=name,scheme=scheme,step=h,derivative=half,coarse_derivative=full,step_halving_difference=abs(half-full),elasticity=None if base==0 else half*x[j]/base))
        result=dict(baseline_value=base,local_derivatives=derivatives)
    elif method=='variance_based':
        sample=_ex_object(shared_samples,'shared_samples');distribution=_ex_object(sample.get('distribution_contract'),'distribution_contract')
        if distribution.get('independent_inputs')is not True:_fail('SENSITIVITY_DEPENDENCE','Sobol variance indices require independent coordinate distributions; correlated inputs need another estimand')
        _vb_string(distribution.get('reference'),'distribution reference');A=_numeric(sample.get('A'),'A');B=_numeric(sample.get('B'),'B')
        if A.ndim!=2 or A.shape!=B.shape or A.shape[1]!=len(x)or A.shape[0]<2:_fail('SENSITIVITY_SAMPLES','Equal nontrivial independent A/B matrices required')
        ya=np.array([evaluate(v)for v in A]);yb=np.array([evaluate(v)for v in B]);variance=float(np.var(np.r_[ya,yb]));first=[];total=[];hybrids=[]
        for j in range(len(x)):
            C=A.copy();C[:,j]=B[:,j];yc=np.array([evaluate(v)for v in C]);hybrids.append(yc);first.append(None if variance==0 else float(1-np.mean((yb-yc)**2)/(2*variance)));total.append(None if variance==0 else float(np.mean((ya-yc)**2)/(2*variance)))
        interactions=[]
        for j in range(len(x)):
            for k in range(j+1,len(x)):
                C=A.copy();C[:,[j,k]]=B[:,[j,k]];yc=np.array([evaluate(v)for v in C]);closed=None if variance==0 else float(1-np.mean((yb-yc)**2)/(2*variance));interactions.append(dict(coordinates=[names[j],names[k]],second_order=None if variance==0 else closed-first[j]-first[k]))
        if not np.isfinite(variance)or any(v is not None and not np.isfinite(v)for v in first+total+[r['second_order']for r in interactions]):_fail('SENSITIVITY_NUMERICAL','Variance estimator exceeds finite range')
        result=dict(baseline_value=base,variance=variance,first_order=dict(zip(names,first)),total_order=dict(zip(names,total)),pairwise_interactions=interactions,applicability='NOT_APPLICABLE'if variance==0 else'APPLICABLE',reason='ZERO_OUTPUT_VARIANCE'if variance==0 else None,finite_sample_indices_clipped=False)
        samples_info=dict(sample_fingerprint=_fingerprint(sample),rows=len(A),distribution_contract=distribution,estimator='JANSEN_SHARED_A_B_HYBRIDS_POPULATION_VARIANCE',uncertainty='NOT_ESTIMATED',independent_sampling_assumption='DECLARED_NOT_AUTHENTICATED')
    else:_fail('SENSITIVITY_METHOD','Use finite_difference or variance_based')
    out=dict(artifact_type='ValidationResult',metrics=result,method=method,design=d,baseline=x.tolist(),perturbations=specs,evaluation_count=len(trace),evaluation_trace=trace,shared_samples=samples_info,model_fingerprint=_fingerprint(_json_data(descriptor)),causal_effect_identified=False,decision=dict(status='DIAGNOSTIC_ONLY'),randomness='FROZEN_DETERMINISTIC_EVALUATION_WITH_SHARED_SAMPLE_DESIGN')
    return _result(out,'validation.sensitivity_analysis','sensitivity',context,limitations=('Coordinate interventions preserve frozen model parameters except explicit graph-node parameter leaves. Local derivatives are numerical and nonsmooth models need careful interpretation. Variance decomposition assumes independent inputs and is not causal; finite sample indices may fall outside [0,1] and are not clipped.',))
