"""Dated collections evidence; operational cure rules are supplied explicitly."""
from datetime import timedelta
from copy import deepcopy
import math
import numpy as np
from .core import CalcResult, FitResult, _result, _fail, _fingerprint, _numeric, _probability, _context_value, _model_artifact
from .data import _rows, _required, _unique, _date, _days


def _scalar(value,name,minimum=0):
    x=_numeric(value,name)
    if x.ndim or float(x)<minimum:_fail('COLLECTIONS_AMOUNT',name+' must be a finite scalar in its permitted range')
    return float(x)


def _finite(value):
    if not math.isfinite(value):_fail('NUMERICAL_FAILURE','Collections calculation outside finite range')
    return value


def _currency_context(currency,context):
    if not isinstance(currency,str) or not currency:_fail('COLLECTIONS_CURRENCY','Currency required')
    ctx=_context_value(context)
    if ctx.get('currency') is not None and ctx['currency']!=currency:
        _fail('TARGET_MISMATCH','Context currency differs from cashflow currency')
    return ctx


def _rule(rule):
    if not isinstance(rule,dict): _fail("COLLECTIONS_RULE","Explicit episode rule required")
    _required(rule,["rule_id","version","problem_states","clean_states","terminal_states","default_states","probation_days","separation_days"])
    groups=[]
    for field in ("problem_states","clean_states","terminal_states","default_states"):
        values=rule[field]
        if not isinstance(values,list) or any(not isinstance(v,str) or not v for v in values) or len(set(values))!=len(values):
            _fail("COLLECTIONS_RULE","State lists require distinct strings")
        groups.append(set(values))
    if not groups[0] or not groups[1] or any(groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2))) or not groups[3]<=groups[0]:
        _fail("COLLECTIONS_RULE","Problem, clean and terminal states must partition declared states; defaults are problem states")
    _days(rule['probation_days'],'probation_days');_days(rule['separation_days'],'separation_days')
    return deepcopy(rule)


def _history(states,coverage,cutoff,rule):
    rows=_rows(states,'states');_unique(rows,'state_event_id','states')
    cov=_rows(coverage,'coverage');_unique(cov,'account_id','coverage');ends={}
    for row in cov:
        _required(row,['observed_from','observed_through','known_at'])
        start,end,known=(_date(row[k],k) for k in ('observed_from','observed_through','known_at'))
        if start>end or known<end:_fail('COLLECTIONS_DATE','Invalid state-history coverage')
        if known<=cutoff:ends[row['account_id']]=(start,min(end,cutoff))
    by_account={};allowed=set(rule['problem_states']+rule['clean_states']+rule['terminal_states'])
    for row in rows:
        _required(row,['account_id','state','occurred_at','known_at'])
        date,known=(_date(row[k],k) for k in ('occurred_at','known_at'))
        if known<date or row['state'] not in allowed:_fail('COLLECTIONS_STATE','Invalid dated state event')
        if date<=cutoff and known<=cutoff:by_account.setdefault(row['account_id'],[]).append(deepcopy(row))
    for aid,history in by_account.items():
        history.sort(key=lambda r:(r['occurred_at'],r['state_event_id']))
        if len({r['occurred_at'] for r in history})!=len(history):_fail('COLLECTIONS_AMBIGUITY','Resolve simultaneous states explicitly')
        if aid in ends:
            first,last=ends[aid]
            # Complete coverage asserts all changes, with an initial state at
            # or before its start. Missing baseline cannot imply clean history.
            if not history or _date(history[0]['occurred_at'],'date')>first:
                _fail('COLLECTIONS_COVERAGE','Coverage requires an initial state at or before observed_from')
    return by_account,ends


def _actions(actions,cutoff):
    rows=_rows(actions,'actions');_unique(rows,'action_id','actions');output=[]
    for row in rows:
        _required(row,['account_id','action_type','stage','occurred_at','known_at'])
        if row['stage'] not in ('ASSIGNED','OFFERED','ACCEPTED','PERFORMED','CANCELLED'):
            _fail('COLLECTIONS_ACTION','Explicit action stage required')
        d,k=(_date(row[f],f) for f in ('occurred_at','known_at'))
        if k<d:_fail('COLLECTIONS_DATE','Action known before occurrence')
        if d<=cutoff and k<=cutoff:output.append(deepcopy(row))
    return output


def build_collections_episodes(states,actions,coverage,episode_rule,*,evaluation_cutoff,context=None):
    """Link problem spells only after a documented continuous clean interval.

    States are complete changes, not sparse samples. Coverage is a separate
    assertion of complete state-change capture; an open endpoint never cures.
    """
    rule=_rule(episode_rule);cutoff=_date(evaluation_cutoff,'evaluation_cutoff')
    histories,ends=_history(states,coverage,cutoff,rule);actions=_actions(actions,cutoff)
    quiet=max(rule['probation_days'],rule['separation_days']);episodes=[]
    for aid,history in sorted(histories.items()):
        current=None;candidate=None;ever_default=False;previous_id=None
        coverage_start,coverage_end=ends.get(aid,(None,None))
        def close(end,status):
            nonlocal current,candidate,previous_id
            current.update(ended_at=end.isoformat(),status=status)
            episodes.append(current);previous_id=current['episode_id'];current=None;candidate=None
        for row in history:
            when=_date(row['occurred_at'],'occurred_at');state=row['state']
            # A return at exactly the quiet boundary interrupts that interval.
            if current is not None and candidate is not None:
                end=candidate+timedelta(days=quiet)
                if when>end and coverage_start is not None and coverage_start<=candidate and coverage_end>=end:
                    close(end,'CURED')
            if state in rule['default_states']:ever_default=True
            if current is None and state in rule['problem_states']:
                current=dict(episode_id=aid+':'+row['state_event_id'],account_id=aid,
                    started_at=row['occurred_at'],ended_at=None,status='OPEN',previous_episode_id=previous_id,
                    timeline=[],actions=[],ever_default=ever_default,observation_cutoff=evaluation_cutoff)
            if current is not None:
                current['timeline'].append(row);current['ever_default']=ever_default
                if state in rule['terminal_states']:close(when,'TERMINATED')
                elif state in rule['problem_states']:candidate=None
                elif state in rule['clean_states'] and candidate is None:candidate=when
        if current is not None:
            end=None if candidate is None else candidate+timedelta(days=quiet)
            if end is not None and coverage_start is not None and coverage_start<=candidate and coverage_end>=end:
                close(end,'CURED')
            else:
                current['cure_candidate_at']=None if candidate is None else candidate.isoformat()
                episodes.append(current)
    assigned=set()
    for ep in episodes:
        ep.setdefault('cure_candidate_at',None)
        ep['actions']=[row for row in actions if row['account_id']==ep['account_id'] and row['occurred_at']>=ep['started_at'] and (ep['ended_at'] is None or row['occurred_at']<=ep['ended_at'])]
        assigned.update(row['action_id'] for row in ep['actions'])
        ep['episode_hash']=_fingerprint(ep)
    value=dict(episodes=episodes,state_history=[r for aid in sorted(histories) for r in histories[aid]],
        unlinked_actions=[r for r in actions if r['action_id'] not in assigned],
        rule=rule,coverage=_rows(coverage,'coverage'),evaluation_cutoff=evaluation_cutoff)
    value['history_hash']=_fingerprint(value)
    return _result(value,
        'collections.build_collections_episodes','episodes',context,
        limitations=('Caller supplies complete dated state changes and independent coverage evidence. No regulatory cure or automatic legal contract change is inferred.',
                    'Quiet interval is max(probation_days,separation_days); a problem event exactly at its end interrupts cure.'))


def build_cure_targets(episodes,*,horizon_days,redefault_horizon_days,evaluation_cutoff,context=None):
    """Observe sustained cure and a separate post-cure first re-default window."""
    source=episodes.value if isinstance(episodes,CalcResult) else episodes
    if not isinstance(source,dict):_fail('COLLECTIONS_INPUT','Episode result required')
    _required(source,['episodes','state_history','rule','coverage','evaluation_cutoff'])
    sealed=dict(source);digest=sealed.pop('history_hash',None)
    if digest!=_fingerprint(sealed):_fail('INTEGRITY_ERROR','Episode history or rules changed since construction')
    cutoff=_date(evaluation_cutoff,'evaluation_cutoff')
    if cutoff!=_date(source['evaluation_cutoff'],'episode cutoff'):
        _fail('COLLECTIONS_CUTOFF','Rebuild episodes at the requested evaluation cutoff')
    horizon=_days(horizon_days,'horizon_days',1);red_horizon=_days(redefault_horizon_days,'redefault_horizon_days',1)
    rule=_rule(source['rule']);by_account={};cov={}
    for row in source['state_history']:
        by_account.setdefault(row['account_id'],[]).append(row)
    for row in _rows(source['coverage'],'coverage'):
        if _date(row['known_at'],'known_at')<=cutoff:cov[row['account_id']]=(_date(row['observed_from'],'observed_from'),_date(row['observed_through'],'observed_through'))
    for ep in source['episodes']:
        payload=dict(ep);digest=payload.pop('episode_hash',None)
        if digest!=_fingerprint(payload):_fail('INTEGRITY_ERROR','Episode changed since construction')
    output=[]
    for ep in source['episodes']:
        start=_date(ep['started_at'],'started_at');end=start+timedelta(days=horizon)
        history=sorted({r['state_event_id']:r for r in by_account[ep['account_id']]}.values(),key=lambda r:r['occurred_at'])
        coverage=cov.get(ep['account_id']);candidate=None;cure=None;exit_at=None
        for row in history:
            when=_date(row['occurred_at'],'occurred_at')
            if when<start:continue
            if candidate is not None:
                proposed=candidate+timedelta(days=rule['probation_days'])
                if when>proposed and coverage is not None and coverage[0]<=candidate and coverage[1]>=proposed:
                    cure=proposed;break
            if row['state'] in rule['problem_states']:candidate=None
            elif row['state'] in rule['clean_states'] and candidate is None:candidate=when
            elif row['state'] in rule['terminal_states']:exit_at=when;candidate=None;break
        if cure is None and candidate is not None and coverage is not None:
            proposed=candidate+timedelta(days=rule['probation_days'])
            if coverage[0]<=candidate and coverage[1]>=proposed and proposed<=cutoff:cure=proposed
        observed_cure=cure is not None and cure<=min(end,cutoff)
        complete=coverage is not None and coverage[0]<=start and coverage[1]>=end and cutoff>=end
        exited=exit_at is not None and exit_at<=min(end,cutoff)
        label=1 if observed_cure else 0 if complete and not exited else None
        status='SUSTAINED_CURE' if observed_cure else 'CENSORED_EXIT' if exited else 'MATURE_NO_CURE' if complete else 'CENSORED'
        redate=None;relabel=None;restatus='NOT_IN_POST_CURE_RISK_SET'
        if observed_cure:
            rend=cure+timedelta(days=red_horizon)
            later=[r for r in history if cure<_date(r['occurred_at'],'date')<=min(rend,cutoff) and r['state'] in rule['default_states']+rule['terminal_states']]
            if later and later[0]['state'] in rule['default_states']:redate=later[0]['occurred_at'];relabel=1;restatus='REDEFAULT'
            elif later:restatus='CENSORED_EXIT'
            elif coverage is not None and coverage[0]<=cure and coverage[1]>=rend and cutoff>=rend:relabel=0;restatus='MATURE_NO_REDEFAULT'
            else:restatus='CENSORED'
        output.append(dict(episode_id=ep['episode_id'],account_id=ep['account_id'],horizon_end=end.isoformat(),
            cure_label=label,cure_status=status,cure_at=cure.isoformat() if observed_cure else None,
            redefault_label=relabel,redefault_status=restatus,redefault_at=redate,
            ever_default=ep['ever_default'],evaluation_cutoff=evaluation_cutoff))
    return _result(output,'collections.build_cure_targets','labels',context,
        limitations=('Operational sustained cure follows the supplied probation rule; it never erases historical default. Termination censors rather than fabricates cure.',))


def build_collections_action_history(actions,features,outcomes,coverage,*,index_stage,
        required_features,horizon_days,maximum_feature_age_days,evaluation_cutoff,context=None):
    """Keep pre-action information distinct from subsequent acceptance and outcomes.

    `treatment_id` links stage records; `action_id` identifies each stage event.
    Only the declared assignment or offer is an index. It is never inferred from
    eventual acceptance, performance, payment, or successful recovery.
    """
    cutoff=_date(evaluation_cutoff,'evaluation_cutoff');actions=_actions(actions,cutoff)
    if index_stage not in ('ASSIGNED','OFFERED'):_fail('COLLECTIONS_ACTION','Index must be assignment or offer')
    if not isinstance(required_features,list) or not required_features or any(not isinstance(f,str) or not f for f in required_features) or len(set(required_features))!=len(required_features):
        _fail('COLLECTIONS_FEATURES','Distinct required feature names needed')
    horizon=_days(horizon_days,'horizon_days',1);age=_days(maximum_feature_age_days,'maximum_feature_age_days')
    features=_rows(features,'features');outcomes=_rows(outcomes,'outcomes');cov=_rows(coverage,'coverage')
    _unique(features,'feature_record_id','features');_unique(outcomes,'outcome_id','outcomes');_unique(cov,'account_id','coverage')
    for f in features:
        _required(f,['account_id','feature','observed_at','known_at'])
        if 'value' not in f:_fail('COLLECTIONS_FEATURES','Feature value key required, null allowed')
        if _date(f['known_at'],'known_at')<_date(f['observed_at'],'observed_at'):_fail('COLLECTIONS_DATE','Feature known before observation')
    for o in outcomes:
        _required(o,['account_id','event_type','occurred_at','known_at'])
        if _date(o['known_at'],'known_at')<_date(o['occurred_at'],'occurred_at'):_fail('COLLECTIONS_DATE','Outcome known before occurrence')
    ends={}
    for c in cov:
        _required(c,['observed_from','observed_through','known_at'])
        start,end,known=(_date(c[k],k) for k in ('observed_from','observed_through','known_at'))
        if start>end or known<end:_fail('COLLECTIONS_DATE','Invalid outcome coverage')
        if known<=cutoff:ends[c['account_id']]=(start,end)
    groups={}
    for action in actions:
        _required(action,['treatment_id'])
        if not isinstance(action['treatment_id'],str) or not action['treatment_id']:
            _fail('COLLECTIONS_ACTION','Treatment ID must be a nonempty string')
        groups.setdefault(action['treatment_id'],[]).append(action)
    rows=[];unindexed=[]
    for tid,history in sorted(groups.items()):
        if len({(a['account_id'],a['action_type']) for a in history})!=1:
            _fail('COLLECTIONS_ACTION','Treatment stages must retain account and action type')
        stages=[a['stage'] for a in history]
        if len(set(stages))!=len(stages):_fail('COLLECTIONS_ACTION','One record per stage; use separate treatment IDs for repeated offers')
        indexed=[a for a in history if a['stage']==index_stage]
        if not indexed:unindexed.append(tid);continue
        index=indexed[0];date=_date(index['occurred_at'],'index_at');aid=index['account_id'];end=date+timedelta(days=horizon)
        stage_dates={a['stage']:a['occurred_at'] for a in history}
        rank=['ASSIGNED','OFFERED','ACCEPTED','PERFORMED']
        dated=[stage_dates[s] for s in rank if s in stage_dates]
        invalid_order=dated!=sorted(dated)
        selected={};source_ids={};missing=[];stale=[]
        for name in required_features:
            available=[f for f in features if f['account_id']==aid and f['feature']==name and _date(f['observed_at'],'observed_at')<date and _date(f['known_at'],'known_at')<date]
            available.sort(key=lambda f:(f['observed_at'],f['known_at'],f['feature_record_id']))
            if len(available)>1 and (available[-1]['observed_at'],available[-1]['known_at'])==(available[-2]['observed_at'],available[-2]['known_at']):
                _fail('COLLECTIONS_AMBIGUITY','Resolve tied pre-action feature revisions')
            record=available[-1] if available else None
            selected[name]=None if record is None else record['value'];source_ids[name]=None if record is None else record['feature_record_id']
            if record is None or record['value'] is None:missing.append(name)
            elif (date-_date(record['observed_at'],'date')).days>age:stale.append(name)
        known_outcomes=[o for o in outcomes if o['account_id']==aid and _date(o['known_at'],'known_at')<=cutoff and _date(o['occurred_at'],'occurred_at')<=cutoff]
        prior=[o['outcome_id'] for o in known_outcomes if _date(o['occurred_at'],'occurred_at')<=date]
        subsequent=[o for o in known_outcomes if date<_date(o['occurred_at'],'occurred_at')<=end]
        complete=aid in ends and ends[aid][0]<=date and ends[aid][1]>=end and cutoff>=end
        row=dict(treatment_id=tid,account_id=aid,action_type=index['action_type'],index_stage=index_stage,
            index_at=index['occurred_at'],index_recorded_at=index['known_at'],preaction_features=selected,
            feature_sources=source_ids,missing_features=missing,stale_features=stale,
            descriptive_eligible=not missing and not stale and not invalid_order,
            stage_dates=stage_dates,invalid_stage_order=invalid_order,
            accepted='ACCEPTED' in stage_dates,performed='PERFORMED' in stage_dates,
            prior_or_same_day_outcome_ids=prior,outcomes=subsequent,horizon_end=end.isoformat(),
            outcome_complete=complete,evaluation_cutoff=evaluation_cutoff,
            causal_effect_identified=False)
        row['row_hash']=_fingerprint(row);rows.append(row)
    return _result(dict(rows=rows,unindexed_treatments=unindexed),
        'collections.build_collections_action_history','panel',context,
        limitations=('Post-index stages and outcomes are separate from pre-action features. An offered plan is not a performed treatment.',
                    'Same-day ordering is unresolved at date resolution and such outcomes are flagged as prior or same-day, never prospective.',
                    'Factual action groups may be selected. No causal effect or treatment eligibility decision is inferred.'))


def simulate_repayment_arrangement(contract,baseline,candidate,affordability_paths,discount_factors,
        *,period_end_dates,currency,context=None):
    """Compare explicit repayment plans using one reconciled claim ledger each.

    Rates are effective rates for each supplied period, not annual rates.
    Arrears principal is already part of outstanding principal. Cash pays fees,
    interest, then principal; capitalization of unpaid interest is explicit.
    """
    context=_currency_context(currency,context)
    if not isinstance(contract,dict):_fail('COLLECTIONS_INPUT','Contract object required')
    _required(contract,['currency','principal','principal_arrears','accrued_interest','fees','as_of'])
    if contract['currency']!=currency:_fail('COLLECTIONS_CURRENCY','Contract currency mismatch')
    start=_date(contract['as_of'],'as_of')
    p0=_scalar(contract['principal'],'principal');arrears=_scalar(contract['principal_arrears'],'principal_arrears')
    i0=_scalar(contract['accrued_interest'],'accrued_interest');f0=_scalar(contract['fees'],'fees')
    if arrears>p0:_fail('COLLECTIONS_AMOUNT','Arrears principal must be a subset of outstanding principal')
    if not isinstance(period_end_dates,list) or not period_end_dates:_fail('COLLECTIONS_DATE','Explicit period dates required')
    dates=[_date(d,'period_end_date') for d in period_end_dates];n=len(dates)
    if dates[0]<=start or any(a>=b for a,b in zip(dates,dates[1:])):_fail('COLLECTIONS_DATE','Increasing future period ends required')
    discounts=_numeric(discount_factors,'discount_factors',vector_only=True)
    if len(discounts)!=n or np.any(discounts<0):_fail('COLLECTIONS_DISCOUNT','Nonnegative factor per period required')
    budgets=_rows(affordability_paths,'affordability_paths');_unique(budgets,'path_id','affordability_paths')
    for row in budgets:
        _required(row,['currency','available_cash'])
        cash=_numeric(row['available_cash'],'available_cash',vector_only=True)
        if row['currency']!=currency or len(cash)!=n:_fail('COLLECTIONS_AFFORDABILITY','Budget path currency/period mismatch')
    def project(spec):
        if not isinstance(spec,dict):_fail('COLLECTIONS_INPUT','Schedule specification required')
        _required(spec,['plan_id','payments','period_interest_rates','new_fees','servicing_costs','receipt_fractions','capitalize_unpaid_interest'])
        if type(spec['capitalize_unpaid_interest']) is not bool:_fail('COLLECTIONS_INPUT','Explicit Boolean capitalization choice required')
        arrays={}
        for name in ('payments','period_interest_rates','new_fees','servicing_costs','receipt_fractions'):
            x=_numeric(spec[name],name,vector_only=True)
            if len(x)!=n or np.any(x<0) or (name=='receipt_fractions' and np.any(x>1)):_fail('COLLECTIONS_SCHEDULE','Invalid schedule amount/rate/fraction vector')
            arrays[name]=x
        principal,interest,fees=p0,i0,f0;rows=[];pv=[];households=[]
        for j,date in enumerate(period_end_dates):
            opening=principal;accrued=_finite(principal*arrays['period_interest_rates'][j]);interest=_finite(interest+accrued);fees=_finite(fees+arrays['new_fees'][j])
            due=_finite(principal+interest+fees)
            scheduled=min(float(arrays['payments'][j]),due)
            received=_finite(scheduled*arrays['receipt_fractions'][j]);remaining=received
            paid_fees=min(remaining,fees);remaining-=paid_fees;fees-=paid_fees
            paid_interest=min(remaining,interest);remaining-=paid_interest;interest-=paid_interest
            paid_principal=min(remaining,principal);principal-=paid_principal
            capitalized=interest if spec['capitalize_unpaid_interest'] else 0.
            principal=_finite(principal+capitalized);interest-=capitalized
            net=_finite(received-arrays['servicing_costs'][j]);discounted=_finite(net*discounts[j]);pv.append(discounted)
            rows.append(dict(period_end=date,opening_principal=opening,interest_accrued=accrued,new_fees=float(arrays['new_fees'][j]),
                scheduled_payment=scheduled,received_payment=received,unused_payment_capacity=float(arrays['payments'][j])-scheduled,
                paid_fees=paid_fees,paid_interest=paid_interest,paid_principal=paid_principal,capitalized_interest=capitalized,
                closing_principal=principal,unpaid_interest=interest,unpaid_fees=fees,
                servicing_cost=float(arrays['servicing_costs'][j]),net_cashflow=net,discounted_net_cashflow=discounted))
        for row in budgets:
            residual=[_finite(float(c)-r['scheduled_payment']) for c,r in zip(row['available_cash'],rows)]
            households.append(dict(path_id=row['path_id'],residual_cash=residual,shortfall_periods=[period_end_dates[j] for j,v in enumerate(residual) if v<0]))
        try:npv=math.fsum(pv)
        except OverflowError:_fail('NUMERICAL_FAILURE','Collections NPV outside finite range')
        return dict(plan_id=spec['plan_id'],schedule=rows,npv=_finite(npv),remaining_claim=_finite(principal+interest+fees),affordability=households)
    base=project(baseline);new=project(candidate)
    return _result(dict(baseline=base,candidate=new,npv_change=_finite(new['npv']-base['npv']),
        principal_arrears_included=arrears,contract_changed=False,default_status_changed=False),
        'collections.simulate_repayment_arrangement',currency,context,
        diagnostics=dict(payment_waterfall=['fees','interest','principal'],interest_convention='effective_period_rate_on_opening_principal'),
        limitations=('Explicit receipt fractions are deterministic scenario assumptions, not estimated cure probabilities. Remaining claims carry no assumed terminal recovery value.',
                    'Affordability subtracts scheduled obligations, not reduced receipts. This local calculation does not implement the proposal or change legal/default status.'))


def expected_collections_value(actions,discount_factors,*,ead,currency,period_end_dates,context=None):
    """Integrate uptake and exclusive conditional recovery scenarios by action."""
    context=_currency_context(currency,context)
    exposure=_scalar(ead,'ead');rows=_rows(actions,'actions');_unique(rows,'action_id','actions')
    if not rows:_fail('COLLECTIONS_INPUT','At least the baseline action is required')
    if not isinstance(period_end_dates,list) or not period_end_dates:_fail('COLLECTIONS_DATE','Period dates required')
    dates=[_date(d,'period_end_date') for d in period_end_dates]
    if any(a>=b for a,b in zip(dates,dates[1:])):_fail('COLLECTIONS_DATE','Increasing period dates required')
    df=_numeric(discount_factors,'discount_factors',vector_only=True)
    if len(df)!=len(dates) or np.any(df<0):_fail('COLLECTIONS_DISCOUNT','Nonnegative factor per period required')
    output=[]
    for action in rows:
        _required(action,['currency','uptake_probability','accepted_scenarios','declined_scenarios','offer_cost','support_status','assumption_reference'])
        if action['currency']!=currency:_fail('COLLECTIONS_CURRENCY','Action currency mismatch')
        uptake=_probability(action['uptake_probability'])
        if uptake.ndim:_fail('COLLECTIONS_INPUT','Scalar uptake required')
        offer_cost=_scalar(action['offer_cost'],'offer_cost');scenarios=[]
        for branch,mass in [('accepted',float(uptake)),('declined',1-float(uptake))]:
            paths=_rows(action[branch+'_scenarios'],branch+'_scenarios');_unique(paths,'scenario_id','scenarios')
            if not paths:_fail('COLLECTIONS_SCENARIOS','Supply complete conditional scenarios for both uptake branches')
            probabilities=[]
            for path in paths:
                _required(path,['probability','cashflows'])
                probability=_probability(path['probability'])
                if probability.ndim:_fail('COLLECTIONS_SCENARIOS','Scalar conditional probability required')
                probabilities.append(float(probability));cashflows=_rows(path['cashflows'],'cashflows');_unique(cashflows,'cashflow_id','cashflows')
                recovery=[];cost=[]
                for cash in cashflows:
                    _required(cash,['kind','period_end','amount'])
                    if cash['kind'] not in ('RECOVERY','COST') or cash['period_end'] not in period_end_dates:_fail('COLLECTIONS_CASHFLOW','Known period and RECOVERY/COST kind required')
                    amount=_scalar(cash['amount'],'cashflow amount');discount=df[period_end_dates.index(cash['period_end'])]
                    (recovery if cash['kind']=='RECOVERY' else cost).append(_finite(amount*discount))
                try:r=math.fsum(recovery);c=math.fsum(cost)
                except OverflowError:_fail('NUMERICAL_FAILURE','Recovery/cost sum outside finite range')
                scenarios.append(dict(branch=branch,scenario_id=path['scenario_id'],probability=mass*float(probability),
                    discounted_recovery=r,discounted_cost=c,net_value=_finite(r-c-offer_cost)))
            if abs(sum(probabilities)-1)>1e-12:_fail('COLLECTIONS_SCENARIOS','Conditional scenario masses must sum to one')
        try:mean=math.fsum(s['probability']*s['net_value'] for s in scenarios)
        except OverflowError:_fail('NUMERICAL_FAILURE','Expected recovery outside finite range')
        positive=[s['net_value'] for s in scenarios if s['probability']>0]
        output.append(dict(action_id=action['action_id'],expected_net_recovery=_finite(mean),
            expected_net_recovery_to_ead=None if exposure==0 else _finite(mean/exposure),
            scenario_min=min(positive),scenario_max=max(positive),scenarios=scenarios,
            support_status=action['support_status'],assumption_reference=action['assumption_reference'],causal_effect_identified=False))
    return _result(output,'collections.expected_collections_value',currency,context,
        limitations=('Uptake and conditional scenario masses form exclusive branches. Offer cost occurs once at time zero; scenario costs must exclude that same cost.',
                    'Scenario ranges are not confidence intervals. Values are conditional projections, not identified treatment effects; action optimization is separate.',
                    'Cashflow IDs prevent repeated records within a scenario; callers must assign one ID per economic cashflow.'))


def predict_cure_and_redefault(current_episode,model,action_conditions,horizon_grid,state_recoveries,
        *,target,currency,extrapolation='error',context=None):
    """Project native Aalen-Johansen states and ordered cure-then-redefault hits.

    Actions select declared empirical support; this method never estimates a
    counterfactual intervention effect from observational transition counts.
    """
    from .survival import predict_event_curves
    ctx=_currency_context(currency,context)
    artifact=model.model if isinstance(model,FitResult) else model
    artifact=_model_artifact(artifact);parameters=artifact['parameters']
    if artifact['method_id']!='survival.estimate_multistate_transitions' or parameters.get('kind')!='aalen_johansen':
        _fail('COLLECTIONS_MODEL','Frozen Aalen-Johansen multistate model required')
    if not isinstance(current_episode,dict) or not isinstance(action_conditions,dict):_fail('COLLECTIONS_INPUT','Episode and action conditions must be objects')
    _required(current_episode,['episode_id','state','model_time','known_at','ever_default'])
    _required(action_conditions,['action_id','assumption_reference'])
    for value in (current_episode['episode_id'],current_episode['state'],action_conditions['action_id'],action_conditions['assumption_reference']):
        if not isinstance(value,str) or not value:_fail('COLLECTIONS_INPUT','Nonempty episode/state/action/reference strings required')
    if _date(artifact['training']['known_at'],'training.known_at')>_date(current_episode['known_at'],'episode.known_at'):
        _fail('COLLECTIONS_LEAKAGE','Model unavailable at episode knowledge cutoff')
    contract=artifact.get('collections_contract')
    if not isinstance(contract,dict):_fail('COLLECTIONS_MODEL','Explicit collections applicability contract required')
    _required(contract,['cure_states','default_states','supported_actions','supported_initial_states','action_interpretation'])
    states=parameters['states']
    for field in ('cure_states','default_states','supported_initial_states'):
        x=contract[field]
        if not isinstance(x,list) or not x or any(not isinstance(s,str) for s in x) or len(set(x))!=len(x) or any(s not in states for s in x):_fail('COLLECTIONS_MODEL','Invalid collections state contract')
    actions=contract['supported_actions']
    if not isinstance(actions,list) or not actions or any(not isinstance(a,str) or not a for a in actions) or len(set(actions))!=len(actions):
        _fail('COLLECTIONS_MODEL','Explicit distinct supported action IDs required')
    if set(contract['cure_states'])&set(contract['default_states']):_fail('COLLECTIONS_MODEL','Cure and default states must differ')
    if contract['action_interpretation']!='observational_conditioning':_fail('COLLECTIONS_MODEL','Declare observational_conditioning; no causal claim supported')
    if current_episode['state'] not in contract['supported_initial_states']:_fail('COLLECTIONS_MODEL','Initial state outside model applicability')
    if type(current_episode['ever_default']) is not bool:_fail('COLLECTIONS_INPUT','Explicit historical default flag required')
    supported=action_conditions['action_id'] in contract['supported_actions']
    if not supported and extrapolation=='error':_fail('COLLECTIONS_ACTION_SUPPORT','Action outside declared empirical support')
    start=_scalar(current_episode['model_time'],'model_time')
    prediction=predict_event_curves(artifact,[],horizon_grid,feature_order=[],target=target,
        initial_state=current_episode['state'],event_states=contract['cure_states'],start_time=start,
        extrapolation=extrapolation,context=ctx)
    curve=prediction.value['curves'][0];grid=_numeric(horizon_grid,'horizon_grid',vector_only=True)
    rewards=_numeric(state_recoveries,'state_recoveries')
    if rewards.shape!=(len(grid),len(states)):_fail('COLLECTIONS_SHAPE','Recovery amounts require horizon-by-state matrix in model state order')
    cure=set(contract['cure_states']);default=set(contract['default_states']);n=len(states)
    # Three phases retain whether cure occurred, even after leaving a cure state.
    # Phase two records a first subsequent default and remains marked forever.
    initial=states.index(current_episode['state']);phase=1 if states[initial] in cure else 0
    distribution=np.zeros((3,n));distribution[phase,initial]=1.;pos=0;times=parameters['event_times'];hits=[]
    while pos<len(times) and times[pos]<=start:pos+=1
    for h in grid:
        while pos<len(times) and times[pos]<=start+h:
            transition=np.eye(n)+np.asarray(parameters['hazard_increment_matrices'][pos]);following=np.zeros_like(distribution)
            for old_phase in range(3):
                for i in range(n):
                    for j in range(n):
                        next_phase=2 if old_phase==2 or (old_phase==1 and states[j] in default) else 1 if old_phase==1 or states[j] in cure else 0
                        following[next_phase,j]+=distribution[old_phase,i]*transition[i,j]
            distribution=following;pos+=1
        hits.append(float(distribution[2].sum()))
    expected=[]
    for probabilities,amounts in zip(curve['state_occupancy'],rewards):
        try:value=math.fsum(float(p)*float(a) for p,a in zip(probabilities,amounts))
        except OverflowError:_fail('NUMERICAL_FAILURE','Expected state recovery outside finite range')
        expected.append(_finite(value))
    try:total_recovery=math.fsum(expected)
    except OverflowError:_fail('NUMERICAL_FAILURE','Expected total recovery outside finite range')
    return _result(dict(episode_id=current_episode['episode_id'],action_id=action_conditions['action_id'],
        treatment_support='SUPPORTED' if supported else 'REFERENCE_MODEL_EXTRAPOLATION',horizon_grid=grid.tolist(),
        states=states,state_occupancy=curve['state_occupancy'],sustained_cure_probability=curve['first_hit_probability'],
        cure_then_redefault_probability=hits,expected_recovery_by_period=expected,
        expected_recovery=_finite(total_recovery),ever_default=current_episode['ever_default'],causal_effect_identified=False),
        'collections.predict_cure_and_redefault',currency,ctx,
        diagnostics=dict(model_id=artifact['model_id'],model_version=artifact['model_version'],
            assumption_reference=action_conditions['assumption_reference'],survival_diagnostics=prediction.diagnostics,
            recovery_convention='conditional_period_cashflow_given_end_state; undiscounted'),
        limitations=('Cure states must already encode the specified sustained-cure probation rule in model training. Ordered re-default means a default reached after such cure, not current default occupancy.',
                    'Recovery inputs are conditional period cashflow expectations, not cumulative recoveries or state liquidation values; repeated cumulative amounts would double count.',
                    'Action support is declared training evidence. Outside-support flat projections reuse the reference model without estimating a treatment effect.'))
