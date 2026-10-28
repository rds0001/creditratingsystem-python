"""Absorbing first-default curves; conditional and marginal PD are distinct."""
from ._ttc import pit_ttc_transform
import numpy as np

from .core import _probability, _result


def cumulative_pd_from_conditional(conditional_pd, *, context=None):
    """Cumulative first-default probability, stable for tiny hazards.

    Ordered disjoint periods are supplied by the caller. Empty input returns an
    empty vector; a unit hazard absorbs all remaining survivor probability.
    """
    q = _probability(conditional_pd, name="conditional_pd", vector_only=True)
    with np.errstate(divide="ignore"):
        value = -np.expm1(np.cumsum(np.log1p(-q)))
    value[value == 0] = 0.0
    return _result(value, "pd_term_structure.cumulative_pd_from_conditional", "probability", context,
                   diagnostics={"curve_type": "cumulative", "default_absorbing": True})


def marginal_pd_from_conditional(conditional_pd, *, context=None):
    """Unconditional first-default mass in each period, not conditional PD."""
    q = _probability(conditional_pd, name="conditional_pd", vector_only=True)
    with np.errstate(divide="ignore"):
        prior = np.concatenate(([0.0], np.cumsum(np.log1p(-q))))[:-1]
    value = np.exp(prior) * q
    return _result(value, "pd_term_structure.marginal_pd_from_conditional", "probability", context,
                   diagnostics={"curve_type": "marginal", "default_absorbing": True})


from collections.abc import Mapping
from datetime import date
import math
from scipy.linalg import expm
from .core import _numeric, _fail, _context_value


def conditional_pd_from_cumulative(cumulative_pd, *, context=None):
    """Invert an ordered cumulative curve; periods after full default are undefined."""
    f = _probability(cumulative_pd, name='cumulative_pd', vector_only=True)
    if np.any(np.diff(f)<0): _fail('NONMONOTONE_CURVE','Cumulative default probabilities must not decrease')
    previous = np.r_[0., f[:-1]] if len(f) else np.array([])
    undefined = previous == 1
    values = [None if missing else float((current-prior)/(1-prior)) for current,prior,missing in zip(f,previous,undefined)]
    return _result({'conditional_pd':values,'undefined_after_full_default':undefined.tolist()},'pd_term_structure.conditional_pd_from_cumulative','probability',context,
        diagnostics={'conditioning':'survival_to_period_start','default_absorbing':True})


def _term_context(context, horizon=None):
    ctx=_context_value(context)
    if ctx.get('time_unit') is not None and ctx['time_unit']!='years': _fail('INCOMPATIBLE_HORIZON','Rates are per year and horizons in years')
    if horizon is not None and ctx.get('horizon') is not None and ctx['horizon']!=horizon: _fail('TARGET_MISMATCH','Context horizon differs')
    return ctx


def hazard_to_pd(hazard, interval_lengths, *, context=None):
    """Piecewise-constant annual intensities integrated over positive year fractions."""
    h=_numeric(hazard,'hazard',vector_only=True);dt=_numeric(interval_lengths,'interval_lengths',vector_only=True)
    if len(h)!=len(dt): _fail('INCOMPATIBLE_SHAPE','One intensity and interval length per period required')
    if np.any(h<0) or np.any(dt<=0): _fail('INVALID_HAZARD','Nonnegative rates and positive interval lengths required')
    with np.errstate(over='ignore'):
        integrated=h*dt; cumulative=np.cumsum(integrated)
    if np.any(~np.isfinite(cumulative)): _fail('NUMERICAL_RANGE','Integrated hazard exceeds finite range')
    q=-np.expm1(-integrated);f=-np.expm1(-cumulative);survival=np.exp(-cumulative)
    prior=np.r_[1.,survival[:-1]] if len(h) else np.array([])
    return _result({'conditional_pd':q,'cumulative_pd':f,'survival':survival,'marginal_pd':prior*q},'pd_term_structure.hazard_to_pd','probability',_term_context(context,float(dt.sum())),
        diagnostics={'time_unit':'years','rate_unit':'per_year','integrated_hazard':integrated.tolist(),'default_absorbing':True})


def _states(states,default_state):
    if not isinstance(states,(list,tuple)) or not states or any(not isinstance(s,str) or not s.strip() for s in states) or len(set(states))!=len(states):
        _fail('INVALID_STATES','Unique nonempty ordered state labels required')
    if default_state not in states: _fail('INVALID_STATES','default_state must be a declared state')
    return list(states),states.index(default_state)


def project_transition_matrix(matrix,horizon,*,matrix_kind,states,default_state,step_length=None,time_unit='years',context=None):
    """Project a homogeneous row-oriented Markov matrix or annual CTMC generator."""
    states,default=_states(states,default_state);m=_numeric(matrix,'matrix');t=_numeric(horizon,'horizon')
    if t.ndim or t<0 or time_unit!='years': _fail('INCOMPATIBLE_HORIZON','Nonnegative scalar horizon in years required')
    if m.ndim!=2 or m.shape!=(len(states),len(states)): _fail('INCOMPATIBLE_SHAPE','Square matrix matching ordered states required')
    tol=1e-12
    if matrix_kind=='transition':
        step=_numeric(step_length,'step_length')
        if step.ndim or step<=0: _fail('INCOMPATIBLE_HORIZON','Positive matrix step length in years required')
        if np.any(m<0) or np.any(m>1) or np.any(abs(m.sum(axis=1)-1)>tol): _fail('INVALID_TRANSITION_MATRIX','Nonnegative stochastic rows required')
        if not np.array_equal(m[default],np.eye(len(states))[default]): _fail('DEFAULT_NOT_ABSORBING','Default row must be exactly absorbing')
        periods=float(t/step)
        if not math.isfinite(periods) or abs(periods-round(periods))>1e-9 or periods>2**31-1:
            _fail('INCOMPATIBLE_HORIZON','Transition projection requires an integer number of declared periods')
        projected=np.linalg.matrix_power(m,int(round(periods)));method='integer_matrix_power'
    elif matrix_kind=='generator':
        if step_length is not None: _fail('INCOMPATIBLE_HORIZON','Generator is per year; do not supply a discrete step')
        off=m.copy();np.fill_diagonal(off,0)
        if np.any(off<0) or np.any(np.diag(m)>0) or np.any(abs(m.sum(axis=1))>tol*np.maximum(1,np.sum(abs(m),axis=1))): _fail('INVALID_GENERATOR','Generator requires nonnegative off-diagonals and zero row sums')
        if np.any(m[default]!=0): _fail('DEFAULT_NOT_ABSORBING','Default generator row must be zero')
        with np.errstate(over='ignore'): scaled=m*float(t)
        if np.any(~np.isfinite(scaled)): _fail('NUMERICAL_RANGE','Generator horizon product overflow')
        projected=expm(scaled);method='matrix_exponential'
    else: _fail('INVALID_INPUT','matrix_kind must be transition or generator')
    if np.any(~np.isfinite(projected)) or np.min(projected)<-tol or np.max(abs(projected.sum(axis=1)-1))>1e-10:
        _fail('NUMERICAL_PROJECTION_FAILURE','Projected matrix failed stochastic checks')
    # Only remove numerical noise after explicit checks; no input repair.
    correction=float(np.max(abs(np.minimum(projected,0))))
    projected=np.maximum(projected,0);projected=projected/projected.sum(axis=1)[:,None]
    return _result({'matrix':projected,'pd_by_origin':projected[:,default],'survival_by_origin':1-projected[:,default]},'pd_term_structure.project_transition_matrix','transition_probabilities',_term_context(context,float(t)),
        diagnostics={'states':states,'default_state':default_state,'horizon':float(t),'time_unit':'years','matrix_kind':matrix_kind,'algorithm':method,'roundoff_negative_correction':correction,'orientation':'origin_rows_destination_columns','default_absorbing':True},
        limitations=('Projection assumes time-homogeneous Markov dynamics. State occupation of absorbing default equals cumulative default probability.',))


def _term_date(value,name):
    try:
        d=date.fromisoformat(value)
        if d.isoformat()!=value: raise ValueError
        return d
    except (TypeError,ValueError): _fail('INVALID_DATE',f'{name} must be an ISO date')


def estimate_transition_matrix(ratings,observation_windows,*,states,default_state,estimator,knowledge_date,observation_scheme,exits=None,cohort_period_days=None,days_per_year=365.25,empty_row_policy='error',context=None):
    """Cohort transition proportions or exact-event homogeneous CTMC intensity MLE.

    Ratings: entity_id, at, known_at, state. Windows: entity_id, start, end,
    complete_until, known_at. Exits: entity_id, at, known_at. All dates ISO.
    Cohort exits before endpoint are excluded unless default was already observed.
    """
    states,default=_states(states,default_state);n=len(states);lookup={s:i for i,s in enumerate(states)}
    cutoff=_term_date(knowledge_date,'knowledge_date');basis=_numeric(days_per_year,'days_per_year')
    if basis.ndim or basis<=0: _fail('INVALID_INPUT','Positive days_per_year required')
    if estimator not in ('cohort','duration') or empty_row_policy not in ('error','identity'):
        _fail('INVALID_INPUT','Declare cohort/duration and error/identity empty-row policy')
    if observation_scheme not in ('exact_event_times','state_carried_until_change') or (estimator=='duration' and observation_scheme!='exact_event_times'):
        _fail('INVALID_OBSERVATION_SCHEME','Duration estimation requires all state transitions observed at exact event times')
    period=None
    if estimator=='cohort':
        period=_numeric(cohort_period_days,'cohort_period_days')
        if period.ndim or period<=0 or period!=np.floor(period): _fail('INCOMPATIBLE_HORIZON','Positive integer cohort_period_days required')
    if not isinstance(ratings,(list,tuple)) or not isinstance(observation_windows,(list,tuple)) or not observation_windows:
        _fail('INVALID_INPUT','Rating records and nonempty observation windows required')
    windows={};groups={};exit_map={};ignored=0
    def entity(row):
        if not isinstance(row,Mapping) or not isinstance(row.get('entity_id'),str) or not row['entity_id']: _fail('INVALID_ENTITY','Nonempty entity_id required')
        return row['entity_id']
    for row in observation_windows:
        eid=entity(row)
        if eid in windows: _fail('DUPLICATE_WINDOW','One window per entity per estimation call required')
        start,end,complete,known=(_term_date(row.get(k),k) for k in ('start','end','complete_until','known_at'))
        if start>=end or complete<start or known<complete or known>cutoff: _fail('INVALID_WINDOW','Window completeness must be known by knowledge date and follow start')
        if estimator=='cohort' and (end-start).days!=period: _fail('INCOMPATIBLE_HORIZON','All cohort windows require declared period days')
        windows[eid]=(start,end,min(end,complete,cutoff));groups[eid]=[]
    for row in ratings:
        eid=entity(row);at=_term_date(row.get('at'),'at');known=_term_date(row.get('known_at'),'known_at')
        if known<at: _fail('INVALID_CHRONOLOGY','Rating cannot be known before occurrence')
        if row.get('state') not in lookup: _fail('INVALID_STATES','Unknown rating state')
        if eid not in windows: _fail('UNKNOWN_ENTITY','Rating entity lacks observation window')
        if known>cutoff: ignored+=1;continue
        groups[eid].append((at,lookup[row['state']]))
    if exits is not None:
        if not isinstance(exits,(list,tuple)): _fail('INVALID_INPUT','Exit records required')
        for row in exits:
            eid=entity(row);at=_term_date(row.get('at'),'at');known=_term_date(row.get('known_at'),'known_at')
            if known<at: _fail('INVALID_CHRONOLOGY','Exit cannot be known before occurrence')
            if eid not in windows: _fail('UNKNOWN_ENTITY','Exit entity lacks observation window')
            if known>cutoff: ignored+=1;continue
            if eid in exit_map: _fail('DUPLICATE_EXIT','One known exit per entity required')
            exit_map[eid]=at
    counts=np.zeros((n,n));exposure=np.zeros(n);excluded=[];used=[]
    for eid,(start,end,complete) in windows.items():
        events=sorted(groups[eid])
        if len({at for at,_ in events})!=len(events): _fail('AMBIGUOUS_EVENT','Multiple ratings at the same date require upstream resolution')
        seen_default=False
        for at,state in events:
            if seen_default and state!=default: _fail('DEFAULT_NOT_ABSORBING','Observed rating leaves absorbing default')
            if state==default: seen_default=True
        before=[event for event in events if event[0]<=start]
        if not before: excluded.append({'entity_id':eid,'reason':'missing_initial_state'});continue
        origin=before[-1][1];stop=min(complete,exit_map.get(eid,complete))
        if stop<start: excluded.append({'entity_id':eid,'reason':'exited_before_start'});continue
        transitions=[event for event in events if start<event[0]<=stop]
        if estimator=='cohort':
            terminal=transitions[-1][1] if transitions else origin
            if stop<end and terminal!=default: excluded.append({'entity_id':eid,'reason':'censored_before_endpoint'});continue
            counts[origin,terminal]+=1;used.append(eid)
        else:
            current=origin;previous=start
            for at,new in transitions:
                exposure[current]+=(at-previous).days/float(basis)
                if new!=current: counts[current,new]+=1
                previous=at;current=new
            exposure[current]+=(stop-previous).days/float(basis)
            used.append(eid)
    denominator=counts.sum(axis=1) if estimator=='cohort' else exposure
    empty=[states[i] for i in range(n) if i!=default and denominator[i]==0]
    if empty and empty_row_policy=='error': _fail('EMPTY_TRANSITION_ROW','No observed denominator for some nondefault origins',{'states':empty})
    matrix=np.zeros((n,n))
    for i in range(n):
        if i==default:
            if estimator=='cohort': matrix[i,i]=1
        elif denominator[i]>0:
            matrix[i]=counts[i]/denominator[i]
            if estimator=='duration': matrix[i,i]=-matrix[i].sum()
        elif estimator=='cohort': matrix[i,i]=1
    return _result({'matrix':matrix,'matrix_kind':'transition' if estimator=='cohort' else 'generator','counts':counts,'exposure_years':exposure},'pd_term_structure.estimate_transition_matrix','transition_probabilities' if estimator=='cohort' else 'per_year',_term_context(context),
        diagnostics={'states':states,'default_state':default_state,'estimator':estimator,'knowledge_date':knowledge_date,'observation_scheme':observation_scheme,'days_per_year':float(basis),'step_length_years':float(period/basis) if period is not None else None,'empty_row_policy':empty_row_policy,'unidentified_rows':empty,'excluded_entities':excluded,'used_entities':used,'ignored_not_yet_known_records':ignored,'default_absorbing':True,'orientation':'origin_rows_destination_columns'},
        limitations=('Cohort estimates use complete cases and can be selection-biased by informative censoring. Duration MLE assumes all transition times observed and homogeneous Markov intensities.',
                     'Identity/zero-generator completion of unidentified rows is an explicit assumption, not an estimated zero migration rate. Default has no cure transition in this contract.'))
