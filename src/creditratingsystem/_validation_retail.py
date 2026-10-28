"""Read-only retail population, dated target and lifecycle validation."""
from collections import Counter
from copy import deepcopy
import calendar
import numpy as np
from .core import _fail,_fingerprint,_result
from .data import _date
from ._expert_common import _ex_object,_ex_ids
from ._validation_common import _vb_design,_vb_string,_vb_number


def _vr_rows(value,name):
    if hasattr(value,'value'):value=value.value
    if not isinstance(value,(list,tuple))or any(not isinstance(row,dict)for row in value):_fail('RETAIL_VALIDATION_SCHEMA',name+' must be a record array')
    return deepcopy(list(value))


def validate_retail_cohort(application_targets,account_panel,behavioral_targets,*,target_profile,split_memberships,source_lineage,design,context=None):
    """Audit existing operational tables without constructing substitute targets."""
    from .behavioral import _month
    d=_vb_design(design,context);apps=_vr_rows(application_targets,'application_targets');panel=account_panel.value if hasattr(account_panel,'value')else account_panel
    if isinstance(panel,dict):panel=panel.get('accounts')
    panel=_vr_rows(panel,'account_panel');targets=_vr_rows(behavioral_targets,'behavioral_targets');splits=_vr_rows(split_memberships,'split_memberships');sources=_vr_rows(source_lineage,'source_lineage');profile=deepcopy(_ex_object(target_profile,'target_profile'))
    for key in ('profile_id','version','reference','default_definition','default_group_rule'):_vb_string(profile.get(key),key)
    if profile.get('default_unit')not in ('facility','obligor')or type(profile.get('first_default'))is not bool or profile.get('exit_treatment')not in ('competing','censor'):_fail('RETAIL_VALIDATION_PROFILE','Explicit default unit, first-default and exit conventions required')
    events=_ex_ids([profile.get(k)for k in ('default_event','chargeoff_event','cure_event')],'distinct event concepts');rules=_ex_object(profile.get('risk_set_by_product'),'risk_set_by_product');isolation=_ex_ids(profile.get('split_isolation_keys'),'split_isolation_keys');ah=_vb_number(profile.get('application_horizon_days'),'application_horizon_days',1);bh=_vb_number(profile.get('behavioral_horizon_months'),'behavioral_horizon_months',1)
    if ah!=int(ah)or bh!=int(bh):_fail('RETAIL_VALIDATION_PROFILE','Integer target horizons required')
    for product,rule in rules.items():
        _vb_string(product,'product_id');_ex_ids(rule.get('active_states'),'active_states')
        if type(rule.get('include_zero_balance'))is not bool:_fail('RETAIL_VALIDATION_PROFILE','Explicit product-qualified zero balance risk rule required')
    profile_known=_date(profile.get('known_at'),'profile known_at');cutoff=_date(d['evaluation_cutoff'],'evaluation_cutoff');violations=[];coverage=Counter();flow=Counter();seen=set();account_keys={};prediction_dates={};units=set()
    def issue(code,unit,detail):violations.append(dict(code=code,unit_id=unit,detail=detail))
    for row in apps:
        app=_vb_string(row.get('application_id'),'application_id');customer=_vb_string(row.get('customer_id'),'customer_id');units.add(customer);key=('application',app)
        if key in seen:issue('DUPLICATE_APPLICATION',app,'Application target appears more than once')
        seen.add(key);asof=_date(row.get('information_cutoff'),'information_cutoff');prediction_dates['application:'+app]=asof
        if asof>cutoff:issue('FUTURE_APPLICATION',app,'Prediction cutoff exceeds validation cutoff')
        for flag in ('approved','accepted','booked','funded','mature','eligible','label_observed'):
            if type(row.get(flag))is not bool:_fail('RETAIL_VALIDATION_SCHEMA','Application Boolean '+flag+' required')
        flow['applications']+=1
        for flag in ('approved','accepted','booked','funded'):flow[flag]+=row[flag]
        if row['accepted']and not row['approved']or row['booked']and not row['accepted']or row['funded']and not row['booked']:issue('SELECTION_FLOW_CONFLICT',app,'Funding must follow booking, acceptance and approval')
        flow['accepted_not_funded']+=row['accepted']and not row['funded'];label=row.get('target')
        if label is not None and (type(label)is bool or label not in (0,1)):_fail('RETAIL_VALIDATION_SCHEMA','Application target must be binary or null')
        if row['label_observed']!=(label is not None):issue('LABEL_OBSERVABILITY_CONFLICT',app,'Label flag differs from actual label')
        if not row['funded']:
            if label is not None or row['eligible']:issue('UNFUNDED_OWN_CONTRACT_OUTCOME',app,'Never-funded own contract has no observed outcome')
        else:
            _vb_string(row.get('facility_id'),'facility_id');start=_date(row.get('performance_start'),'performance_start');end=_date(row.get('outcome_end'),'outcome_end')
            if start<asof or (end-start).days!=ah:issue('APPLICATION_WINDOW_CONFLICT',app,'Performance must follow prediction with declared horizon')
            if row['mature']!=(end<=cutoff):issue('MATURITY_FLAG_CONFLICT',app,'Calendar maturity differs from declared cutoff')
            coverage['application_mature_observed']+=end<=cutoff and label is not None and row['eligible']
            coverage['application_early_known_event']+=label==1 and end>cutoff
            if label==0 and (end>cutoff or row.get('observed_until')is None or _date(row['observed_until'],'observed_until')<end):issue('UNSUPPORTED_APPLICATION_NONDEFAULT',app,'Negative own-contract outcome requires mature complete follow-up')
            if label==1 and not row.get('event_id'):issue('MISSING_DEFAULT_EVENT_PROVENANCE',app,'Positive label lacks default event ID')
        if label is not None and row.get('label_origin')in ('NO_HIT','THIN_FILE','PSEUDO','HYPOTHETICAL'):issue('NON_OUTCOME_LABEL_SOURCE',app,'Missing bureau evidence or pseudolabel is not a factual outcome')
        if profile['default_unit']=='obligor'and not row.get('default_group_id'):issue('MISSING_DEFAULT_GROUP',app,'Obligor target requires an explicit default-group assignment')
    for row in panel:
        account=_vb_string(row.get('account_id'),'account_id');customer=_vb_string(row.get('customer_id'),'customer_id');units.add(customer);asof=_date(row.get('as_of'),'as_of');known=_date(row.get('known_at'),'known_at');opened=_date(row.get('opened_at'),'opened_at');unit=account+'@'+row['as_of'];key=(account,row['as_of'])
        if key in account_keys:issue('DUPLICATE_ACCOUNT_MONTH',unit,'Account-month snapshot repeated')
        account_keys[key]=row;prediction_dates['account:'+unit]=known
        if known>asof or asof>cutoff or opened>asof:issue('ACCOUNT_DATE_CONFLICT',unit,'Snapshot knowledge/opening/as-of chronology invalid')
        mob=(asof.year-opened.year)*12+asof.month-opened.month
        if row.get('months_on_book')!=mob or asof.day!=calendar.monthrange(asof.year,asof.month)[1]:issue('ACCOUNT_MOB_CONFLICT',unit,'Calendar month-end/MOB differs from origin')
        if type(row.get('at_risk'))is not bool or type(row.get('ever_default'))is not bool:_fail('RETAIL_VALIDATION_SCHEMA','Account risk/default flags required')
        rule=rules.get(row.get('product_id'));balance=row.get('features',{}).get('balance')
        if rule is None:issue('UNDECLARED_PRODUCT_RISK_SET',unit,'No product risk-set rule')
        else:
            expected=row.get('lifecycle_state')in rule['active_states']and not(profile['first_default']and row['ever_default'])and(rule['include_zero_balance']or balance is None or balance!=0)
            if row['at_risk']!=expected:issue('RISK_SET_CONFLICT',unit,'Risk membership differs from declared product/default profile')
            if balance==0 and expected:coverage['active_zero_balance_at_risk']+=1
        coverage['account_months']+=1;coverage['account_months_at_risk']+=row['at_risk'];coverage['incomplete_feature_snapshots']+=not row.get('source_complete',False)
    seen_targets=set()
    for row in targets:
        account=_vb_string(row.get('account_id'),'target account_id');asof=_date(row.get('as_of'),'target as_of');end=_date(row.get('horizon_end'),'horizon_end');key=(account,row['as_of']);unit=account+'@'+row['as_of'];origin=account_keys.get(key)
        if key in seen_targets:issue('DUPLICATE_BEHAVIORAL_TARGET',unit,'Repeated account target')
        seen_targets.add(key)
        if origin is None:issue('TARGET_WITHOUT_SNAPSHOT',unit,'Target has no matching account snapshot')
        elif row.get('customer_id')!=origin['customer_id']:issue('TARGET_CUSTOMER_CONFLICT',unit,'Target customer differs from snapshot')
        if end!=_month(asof,int(bh))or row.get('target_event')!=profile['default_event']or row.get('evaluation_cutoff')!=d['evaluation_cutoff']:issue('BEHAVIORAL_TARGET_CONFLICT',unit,'Target horizon/event/cutoff differs from profile')
        label=row.get('label');mature=end<=cutoff
        if label is not None and (type(label)is bool or label not in (0,1)):_fail('RETAIL_VALIDATION_SCHEMA','Behavioral label must be binary/null')
        if row.get('calendar_mature')!=mature or row.get('outcome_complete')!=(label is not None):issue('BEHAVIORAL_MATURITY_CONFLICT',unit,'Maturity/completeness flag conflicts with dates or label')
        if label==0:
            if row.get('status')=='COMPETING_EVENT':
                if profile['exit_treatment']!='competing' or row.get('first_event_type')not in profile.get('competing_events',[]):issue('COMPETING_TARGET_CONFLICT',unit,'Exit zero requires declared competing-risk estimand')
            elif not mature or row.get('followup_days')is None or row['followup_days']<(end-asof).days:issue('UNSUPPORTED_BEHAVIORAL_NONDEFAULT',unit,'Negative label lacks mature complete follow-up')
        if label==1 and(row.get('first_event_type')!=profile['default_event']or not row.get('first_event_id')):issue('BEHAVIORAL_EVENT_PROVENANCE',unit,'Default label lacks matching default event provenance')
        if row.get('eligible')and origin is not None and not origin['at_risk']:issue('TARGET_OUTSIDE_RISK_SET',unit,'Eligible target contradicts snapshot risk set')
        coverage['behavioral_mature_observed']+=mature and label is not None and row.get('eligible',False);coverage['behavioral_early_known_event']+=label==1 and not mature;coverage['behavioral_missing_outcomes']+=label is None
    for key,row in account_keys.items():
        if key not in seen_targets:issue('SNAPSHOT_WITHOUT_TARGET',key[0]+'@'+key[1],'Account snapshot has no target audit record')
    group_roles={};split_units=set()
    for row in splits:
        customer=_vb_string(row.get('customer_id'),'split customer_id');role=_vb_string(row.get('role'),'split role');split_units.add(customer)
        if customer not in units:issue('UNKNOWN_SPLIT_CUSTOMER',customer,'Split assignment has no application/account')
        for key in isolation:
            group=row.get(key)
            if not isinstance(group,str)or not group:issue('MISSING_SPLIT_GROUP',customer,key+' unavailable');continue
            group_roles.setdefault((key,group),set()).add(role)
    for (key,group),roles in sorted(group_roles.items()):
        if len(roles)>1:issue('SPLIT_GROUP_OVERLAP',group,key+' crosses '+','.join(sorted(roles)))
    for customer in sorted(units-split_units):issue('MISSING_SPLIT_ASSIGNMENT',customer,'No declared split-group isolation evidence')
    source_ids=[]
    for row in sources:
        source=_vb_string(row.get('source_id'),'source_id');source_ids.append(source);key=_vb_string(row.get('prediction_key'),'prediction_key');available=_date(row.get('available_at'),'available_at');observed=_date(row.get('observed_at'),'observed_at');_vb_string(row.get('source_hash'),'source_hash');kind=_vb_string(row.get('kind'),'source kind')
        if key not in prediction_dates:issue('UNKNOWN_SOURCE_PREDICTION',source,'Lineage points to unknown snapshot')
        elif available>prediction_dates[key]or observed>prediction_dates[key]:issue('FEATURE_LOOKAHEAD',source,'Feature evidence was unavailable at original prediction cutoff')
        if kind in ('TARGET','OUTCOME','POST_DECISION'):issue('OUTCOME_IN_FEATURE_PATH',source,'Future target/decision evidence cannot be a historical feature')
    for key,date in sorted(prediction_dates.items()):
        if profile_known>date:issue('PROFILE_LOOKAHEAD',key,'Target/risk profile was unavailable at original prediction cutoff')
    seen_source_keys={r['prediction_key']for r in sources}
    for key in sorted(set(prediction_dates)-seen_source_keys):issue('MISSING_FEATURE_LINEAGE',key,'No source evidence for frozen snapshot')
    value=dict(artifact_type='ValidationResult',design=d,target_profile=profile,metrics=dict(population_flow=dict(sorted(flow.items())),maturity_and_risk_coverage=dict(sorted(coverage.items()))),violations=violations,violation_counts=dict(sorted(Counter(v['code']for v in violations).items())),decision=dict(status='FAIL'if violations else'PASS',scope='DECLARED_RETAIL_COHORT_CONTRACT'),applicability='APPLICABLE'if apps or panel else'NOT_APPLICABLE',reason=None if apps or panel else'EMPTY_RETAIL_POPULATION',empirical_performance_assessed=False,source_provenance_authenticated=False,input_fingerprint=_fingerprint(dict(application_targets=apps,account_panel=panel,behavioral_targets=targets,split_memberships=splits,source_lineage=sources)))
    if not apps and not panel:value['decision']['status']='NOT_APPLICABLE'
    return _result(value,'validation.validate_retail_cohort','cohort_audit',context,limitations=('Audits supplied operational target/panel tables, not a competing target builder. Early observed defaults remain visible but do not enter mature performance coverage. Household linkage controls dependence, not default contagion. Declared source evidence is not authenticated.',))


def roll_rate_backtest(forecasts,account_panel,*,states,state_contract,design,weighting,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Align exact monthly endpoints and audit explicit cure/re-default episode IDs."""
    from ._validation_common import _vb_records
    from ._validation_migration import rating_migration_backtest
    d=_vb_design(design,context);rows=_vb_records(forecasts,d);panel=_vr_rows(account_panel,'account_panel');levels=_ex_ids(states,'states');contract=deepcopy(_ex_object(state_contract,'state_contract'))
    if d['target']['time_unit']!='months'or int(d['target']['horizon'])!=d['target']['horizon']:_fail('ROLL_RATE_HORIZON','Explicit whole-calendar-month transition horizon required')
    if weighting not in ('count','opening_balance'):_fail('ROLL_RATE_WEIGHT','Use count or opening_balance')
    defaults=set(_ex_ids(contract.get('default_states'),'default_states'));cures=set(_ex_ids(contract.get('cure_states'),'cure_states',empty=True));absorbing=set(_ex_ids(contract.get('absorbing_states'),'absorbing_states',empty=True));chargeoffs=set(_ex_ids(contract.get('chargeoff_states'),'chargeoff_states',empty=True))
    if not chargeoffs<=defaults:_fail('ROLL_RATE_STATES','Charge-off role must be a separately identified subset of default states')
    index={};seen=set()
    for row in panel:
        account=_vb_string(row.get('account_id'),'account_id');at=_date(row.get('as_of'),'as_of');known=_date(row.get('known_at'),'known_at');_vb_string(row.get('risk_episode_id'),'risk_episode_id')
        if known<at or row.get('state')not in levels:_fail('ROLL_RATE_PANEL','State knowledge cannot precede observation; state must be declared')
        if row['state']in defaults:_vb_string(row.get('default_episode_id'),'default_episode_id')
        key=(account,row['as_of'],row['known_at'])
        if key in seen:_fail('ROLL_RATE_AMBIGUITY','Duplicate equally dated state observation')
        seen.add(key);index.setdefault(account,[]).append(row)
    converted=[];matrices=[];weights=[];traces=[];episode_violations=[];origins=Counter();missing=0
    for row in rows:
        account=_vb_string(row.get('account_id'),'forecast account_id');_vb_string(row.get('risk_episode_id'),'forecast risk_episode_id');history=index.get(account,[]);available=[r for r in history if r['as_of']==row['as_of']and r['known_at']<=row['forecast_known_at']]
        if not available:_fail('ROLL_RATE_ORIGIN','Exact historically known origin state required')
        origin=max(available,key=lambda r:r['known_at'])
        if origin['state']!=row.get('origin_state')or origin['risk_episode_id']!=row['risk_episode_id']or origin.get('default_episode_id')!=row.get('default_episode_id'):_fail('ROLL_RATE_ORIGIN','Forecast origin state/episodes differ from its historical snapshot')
        origins[row['origin_state']]+=1;future=[r for r in history if row['as_of']<r['as_of']<=row['horizon_end']and r['known_at']<=d['evaluation_cutoff']];latest={}
        for r in future:
            if r['as_of']not in latest or r['known_at']>latest[r['as_of']]['known_at']:latest[r['as_of']]=r
        trajectory=[origin]+[latest[k]for k in sorted(latest)];endpoint=latest.get(row['horizon_end'])
        if endpoint is None:
            old_absorbing=[r for r in trajectory if r['state']in absorbing]
            endpoint=old_absorbing[-1]if old_absorbing else None
        converted_row=deepcopy(row);converted_row.update(scale_id=contract['scale_id'],scale_version=contract['scale_version'],observation_status='CENSORED'if endpoint is None else'OBSERVED',destination_state=None if endpoint is None else endpoint['state'],state_observed_at=None if endpoint is None else endpoint['as_of'],outcome_known_at=d['evaluation_cutoff']if endpoint is None else endpoint['known_at'])
        if endpoint is None:missing+=1
        seen_episodes=set();last_default=None;cure_count=redefault_count=0
        if origin['state']in defaults:last_default=origin['default_episode_id'];seen_episodes.add(last_default)
        local=[]
        for previous,current in zip(trajectory[:-1],trajectory[1:]):
            if previous['state']in absorbing and current['state']!=previous['state']:local.append('ABSORBING_STATE_DEPARTURE')
            if current['state']in cures and previous['state']in defaults:
                cure_count+=1
                if current['risk_episode_id']==previous['risk_episode_id']:local.append('CURE_WITHOUT_NEW_RISK_EPISODE')
            if current['state']in defaults:
                ep=current['default_episode_id']
                if previous['state']in defaults and ep!=previous['default_episode_id']:local.append('DEFAULT_EPISODE_CHANGED_WITHOUT_CURE')
                elif previous['state']not in defaults:
                    if ep in seen_episodes:local.append('REUSED_DEFAULT_EPISODE_AFTER_CURE')
                    if last_default is not None:redefault_count+=1
                seen_episodes.add(ep);last_default=ep
        if local:
            episode_violations.append(dict(prediction_id=row['prediction_id'],codes=sorted(set(local))));converted_row['observation_status']='CENSORED'
        traces.append(dict(prediction_id=row['prediction_id'],account_id=account,origin_state=row['origin_state'],horizon_end=row['horizon_end'],selected_endpoint_at=None if endpoint is None else endpoint['as_of'],selected_endpoint_known_at=None if endpoint is None else endpoint['known_at'],missing_exact_successor=row['horizon_end']not in latest,cure_transitions=cure_count,redefault_transitions=redefault_count,episode_violation_codes=sorted(set(local)),risk_episode_ids=[r['risk_episode_id']for r in trajectory],default_episode_ids=[r.get('default_episode_id')for r in trajectory]))
        matrices.append(row.get('transition_matrix'));converted.append(converted_row);weights.append(1. if weighting=='count'else _vb_number(row.get('opening_balance'),'opening_balance',0))
    if not rows:_fail('ROLL_RATE_FORECASTS','At least one frozen transition forecast required')
    checked=rating_migration_backtest(converted,matrices,states=levels,state_contract=contract,design=d,uncertainty=uncertainty,weights=weights,cluster_resamples=cluster_resamples,confidence_level=confidence_level,context=context).value
    checked['diagnostics'].update(account_panel_fingerprint=_fingerprint(panel),endpoint_traces=traces,episode_violations=episode_violations,origin_account_counts=dict(sorted(origins.items())),missing_successor_count=missing,weighting=weighting,zero_balance_origin_count=sum(r.get('opening_balance')==0 for r in rows),observed_chargeoff_destinations=sum(r.get('destination_state')in chargeoffs and r['observation_status']=='OBSERVED'for r in converted),no_missing_month_compression=True)
    return _result(checked,'validation.roll_rate_backtest','retail_transition_validation',context,limitations=('Transition scores reuse the common migration backtest. Exact horizon endpoints are required except known absorbing states; missing months never imply self-transitions. Episode identifiers are supplied provenance, not inferred hidden events. Endpoint calibration is conditional on observed valid successors.',))
