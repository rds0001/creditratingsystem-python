"""Explicit cashflow event logic and simulation accounting contracts."""
from copy import deepcopy
from collections import defaultdict
import math
from .core import _fail,_result,_numeric,_fingerprint,_context_value
from .data import _rows,_required,_date


def _number(value,name,minimum=None):
    x=_numeric(value,name)
    if x.ndim or (minimum is not None and x<minimum):_fail('SIMULATION_INPUT',name+' is outside its scalar domain')
    return float(x)


def _text(value,name):
    if not isinstance(value,str) or not value.strip():_fail('SIMULATION_INPUT',name+' must be a nonempty string')
    return value


def detect_cashflow_default(path_records,default_policy,*,currency,context=None):
    """Evaluate economic dual tests and independent Basel-reference default triggers.

    This event evaluator consumes declared material-arrears counters and UTP
    evidence. It does not turn an asset-coverage test into a regulatory veto.
    """
    rows=_rows(path_records,'path_records');policy=deepcopy(default_policy);_text(currency,'currency')
    if not isinstance(policy,dict):_fail('SIMULATION_POLICY','A versioned explicit default policy is required')
    _required(policy,['policy_id','version','basis','assessment_level','dscr_threshold','ltv_threshold','dscr_comparison','ltv_comparison','past_due_days_threshold','past_due_comparison','past_due_rule_reference','materiality_rule_reference','utp_rule_reference','dual_test_as_utp','economic_joint_timing','default_state'])
    for key in ('policy_id','version','basis','assessment_level','dscr_comparison','ltv_comparison','past_due_comparison','economic_joint_timing','default_state','past_due_rule_reference','materiality_rule_reference','utp_rule_reference'):_text(policy[key],key)
    if policy['basis']!='BASEL_CRE36_REFERENCE' or policy['assessment_level']not in ('obligor','facility') or policy['economic_joint_timing']!='SAME_DATE' or policy['default_state']!='ABSORBING_FIRST_DEFAULT':_fail('SIMULATION_POLICY','Declare Basel reference, obligor/facility scope, same-date dual test and absorbing first default')
    if policy['dscr_comparison']not in ('lt','le') or policy['ltv_comparison']not in ('gt','ge') or policy['past_due_comparison']!='gt' or not isinstance(policy['dual_test_as_utp'],bool):_fail('SIMULATION_POLICY','Declare threshold strictness, past-due gt and Boolean dual_test_as_utp')
    dscr_limit=_number(policy['dscr_threshold'],'dscr_threshold',0);ltv_limit=_number(policy['ltv_threshold'],'ltv_threshold',0);dpd_limit=_number(policy['past_due_days_threshold'],'past_due_days_threshold',0)
    if dpd_limit!=math.floor(dpd_limit)or not 90<=dpd_limit<=180:_fail('SIMULATION_POLICY','Basel-reference day threshold must be integer90..180 with a documented applicable rule')
    if dpd_limit!=90 and policy.get('supervisory_exception_eligibility')!='RETAIL_OR_PSE_APPROVED':_fail('SIMULATION_POLICY','Non90day threshold requires explicit applicable retail/PSE supervisory exception')
    if policy['assessment_level']=='facility' and policy.get('facility_scope_eligibility')!='RETAIL_APPROVED':_fail('SIMULATION_POLICY','Facility-level Basel reference requires explicit retail scope eligibility')
    ctx=_context_value(context)
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Default path currency conflicts with context')
    seen=set();row_lookup={};identity={};groups=defaultdict(list);facility_details=[]
    for row in rows:
        _required(row,['scenario_id','entity_id','facility_id','date','currency','cfads','debt_service_due','debt_outstanding','asset_value','liquidity_buffer_available','materiality_met','material_days_past_due','utp_indicators'])
        for key in ('scenario_id','entity_id','facility_id'):_text(row[key],key)
        date=_date(row['date'],'date');key=(row['scenario_id'],row['facility_id'],row['date'])
        if key in seen:_fail('SIMULATION_DUPLICATE','Each scenario/facility/date must be unique')
        seen.add(key);row_lookup[key]=row;owner_key=(row['scenario_id'],row['facility_id'])
        if owner_key in identity and identity[owner_key]!=row['entity_id']:_fail('SIMULATION_IDENTITY','Facility owner must remain fixed within this simulation')
        identity[owner_key]=row['entity_id']
        if row['currency']!=currency:_fail('TARGET_MISMATCH','Path records must use one reporting currency')
        cfads=_number(row['cfads'],'cfads');due=_number(row['debt_service_due'],'debt_service_due',0);debt=_number(row['debt_outstanding'],'debt_outstanding',0);assets=_number(row['asset_value'],'asset_value',0)
        dpd=_number(row['material_days_past_due'],'material_days_past_due',0)
        if dpd!=math.floor(dpd)or not isinstance(row['materiality_met'],bool):_fail('SIMULATION_ARREARS','Materiality flag and integer material-arrears day counter required')
        if not row['materiality_met'] and dpd!=0:_fail('SIMULATION_ARREARS','No material arrears means its continuous day counter must be zero')
        indicators=row['utp_indicators']
        if not isinstance(indicators,list)or any(not isinstance(v,str)or not v for v in indicators)or len(set(indicators))!=len(indicators):_fail('SIMULATION_UTP','UTP indicators must be distinct explicit evidence IDs')
        dscr=None if due==0 else cfads/due;dscr_breach=False if due==0 else (dscr<dscr_limit if policy['dscr_comparison']=='lt'else dscr<=dscr_limit)
        ltv=None if assets==0 else debt/assets;ltv_breach=debt>0 if assets==0 else (ltv>ltv_limit if policy['ltv_comparison']=='gt'else ltv>=ltv_limit)
        if (dscr is not None and not math.isfinite(dscr))or(ltv is not None and not math.isfinite(ltv)):_fail('NUMERICAL_FAILURE','DSCR or LTV exceeds finite representation')
        buffer=_number(row['liquidity_buffer_available'],'liquidity_buffer_available',0);shortfall=max(due-cfads-buffer,0)
        joint=dscr_breach and ltv_breach;dual_utp=policy['dual_test_as_utp']and joint and shortfall>0;utp=bool(indicators)or dual_utp;past_due=row['materiality_met']and dpd>dpd_limit
        detail=dict(scenario_id=row['scenario_id'],entity_id=row['entity_id'],facility_id=row['facility_id'],date=row['date'],dscr=dscr,dscr_status='NO_DEBT_SERVICE'if due==0 else 'DEFINED',ltv=ltv,ltv_status='NO_DEBT_OR_ASSET'if debt==0 and assets==0 else 'POSITIVE_DEBT_ZERO_ASSET'if assets==0 else 'DEFINED',dscr_breach=dscr_breach,ltv_breach=ltv_breach,economic_dual_breach=joint,utp_indicators=sorted(indicators),liquidity_buffer_available=buffer,unremedied_debt_service_shortfall=shortfall,utp_from_dual_test=dual_utp,unlikely_to_pay=utp,material_past_due_trigger=past_due,regulatory_trigger=utp or past_due)
        facility_details.append(detail);subject=row['entity_id']if policy['assessment_level']=='obligor'else row['facility_id'];groups[(row['scenario_id'],subject)].append(detail)
    assessments=[];first_events=[]
    for (scenario,subject),records in sorted(groups.items()):
        bydate=defaultdict(list)
        for row in records:bydate[row['date']].append(row)
        sets=[{r['facility_id']for r in dated}for dated in bydate.values()]
        if any(s!=sets[0]for s in sets):_fail('SIMULATION_GRID','All facilities of an assessed obligor require the same explicit date grid')
        first=None;first_reasons=[];previous={};last_nondefault=None;default_lower=None;initial_default=False
        for date,dated in sorted(bydate.items()):
            # Actual material-day counts are inputs, but cannot advance faster than calendar time.
            original={r['facility_id']:row_lookup[(scenario,r['facility_id'],date)]for r in dated}
            for fid,row in original.items():
                if fid in previous:
                    last=previous[fid];elapsed=(_date(date,'date')-_date(last['date'],'date')).days
                    if row['material_days_past_due']>last['material_days_past_due']+elapsed or (row['material_days_past_due']>elapsed and row['material_days_past_due']!=last['material_days_past_due']+elapsed):_fail('SIMULATION_ARREARS','Continuous material-arrears counter must advance by calendar time or reset within the interval')
                previous[fid]=row
            utp=any(r['unlikely_to_pay']for r in dated);past_due=any(r['material_past_due_trigger']for r in dated);trigger=utp or past_due
            reasons=(['UNLIKELY_TO_PAY']if utp else [])+(['MATERIAL_PAST_DUE']if past_due else [])
            if trigger and first is None:
                first=date;first_reasons=reasons;default_lower=last_nondefault;initial_default=last_nondefault is None
            if first is None:last_nondefault=date
            assessments.append(dict(scenario_id=scenario,subject_id=subject,date=date,assessment_level=policy['assessment_level'],unlikely_to_pay=utp,material_past_due_trigger=past_due,regulatory_trigger=trigger,in_default=first is not None,first_default_date=first,trigger_reasons=reasons,economic_dual_breach=any(r['economic_dual_breach']for r in dated)))
        first_events.append(dict(scenario_id=scenario,subject_id=subject,defaulted=first is not None,first_default_date=first,reasons=first_reasons,initial_default=initial_default,last_nondefault_date=default_lower if first is not None else last_nondefault,observed_through=max(bydate),default_time_status='NO_DEFAULT_OBSERVED'if first is None else 'LEFT_CENSORED_AT_INITIAL_STATE'if initial_default else 'DETECTED_AT_GRID_ENDPOINT'))
    value=dict(policy=policy,policy_hash=_fingerprint(policy),currency=currency,facility_tests=sorted(facility_details,key=lambda r:(r['scenario_id'],r['facility_id'],r['date'])),assessments=assessments,first_default_events=first_events)
    return _result(value,'simulation.detect_cashflow_default','default_events',ctx,diagnostics=dict(input_fingerprint=_fingerprint(rows),default_definition='UTP_OR_MATERIAL_PAST_DUE',economic_test='DSCR_BREACH_AND_LTV_BREACH_SAME_DATE'),limitations=('The DSCR/LTV intersection is an economic model indicator. It never vetoes an independently established UTP or material-arrears default. Nominal value coverage, security realisation or a guarantee do not erase borrower default.',
        'Materiality, valid continuous arrears counters, UTP evidence and applicable supervisory scope are supplied under a versioned policy. No universal national implementation or automatic regulatory approval is claimed. First default dates are detection-grid endpoints, with the previous nondefault endpoint reported. Initial defaults are left censored and must not enter an initially performing risk denominator. Default is absorbing; cure and redefault require a separate episode model.'))

from ._simulation_cashflows import build_debt_schedule, apply_cash_waterfall

from ._simulation_drivers import simulate_operating_cashflows, simulate_asset_values

from ._simulation_counterparties import simulate_contract_counterparties

from ._simulation_risk import estimate_simulation_risk

from ._simulation_risk import importance_sample_defaults

from ._simulation_recovery import simulate_recovery_waterfall

from ._simulation_transactions import simulate_transactions
