"""Frozen qualitative rubrics and dated structured evidence; no inferred facts."""
from copy import deepcopy
import math
from .core import _result,_fail,_fingerprint
from .data import _date
from ._expert_common import _ex_object,_ex_ids,_ex_number,_ex_probability,_ex_integer,_ex_target,_ex_seal,_ex_read


def _eq_records(value,name,key):
    if not isinstance(value,(list,tuple)): _fail('QUALITATIVE_CONTRACT',name+' must be a record list')
    rows=[deepcopy(_ex_object(r,name))for r in value];_ex_ids([r.get(key)for r in rows],key,True);return rows


def _eq_conditions(conditions,criteria):
    c=_ex_object(conditions,'conditions')
    if not c or set(c)-set(criteria): _fail('QUALITATIVE_CONDITION','Nonempty conditions must reference existing criteria')
    for cid,levels in c.items():
        names=_ex_ids(levels,'condition levels')
        if set(names)-{v['level_id']for v in criteria[cid]['levels']}: _fail('QUALITATIVE_CONDITION','Unknown condition level')
    return deepcopy(c)


def compile_qualitative_rubric(criteria,*,rubric_contract,score_policy,interactions=None,rating_limits=None,context=None):
    """Freeze mutually exclusive levels, evidence requirements and adverse rules."""
    rows=_eq_records(criteria,'criteria','criterion_id')
    if not rows: _fail('QUALITATIVE_CONTRACT','At least one criterion required')
    contract=deepcopy(_ex_object(rubric_contract,'rubric_contract'));_ex_target(contract.get('target'),context)
    for key in ('rubric_id','version','population_id','reference'):
        if not isinstance(contract.get(key),str) or not contract[key]: _fail('QUALITATIVE_CONTRACT',key+' required')
    _date(contract.get('known_at'),'known_at');grades=_ex_ids(contract.get('ordered_grades'),'ordered_grades',True)
    if contract.get('score_direction')!='higher_is_worse': _fail('QUALITATIVE_DIRECTION','Points must explicitly increase with risk')
    policy=deepcopy(_ex_object(score_policy,'score_policy'));floor=_ex_number(policy.get('score_floor'),'score_floor');cap=_ex_number(policy.get('score_cap'),'score_cap');coverage=_ex_probability(policy.get('minimum_evidence_coverage'),'minimum_evidence_coverage')
    if floor>=cap or policy.get('unresolved_interaction_policy')not in ('refer','apply_adverse'): _fail('QUALITATIVE_POLICY','Ordered score bounds and explicit unresolved interaction policy required')
    for criterion in rows:
        weight=_ex_number(criterion.get('weight'),'weight')
        if weight<0: _fail('QUALITATIVE_WEIGHT','Criterion weights cannot be negative')
        _ex_ids(criterion.get('required_evidence_types'),'required_evidence_types');minimum=_ex_integer(criterion.get('minimum_independent_sources'),'minimum_independent_sources',10000)
        if not minimum: _fail('QUALITATIVE_EVIDENCE','Positive number of declared independent evidence groups required')
        if criterion.get('missing_policy')not in ('refer','worst') or criterion.get('not_applicable_policy')not in ('refer','worst','fixed'): _fail('QUALITATIVE_MISSING','Explicit missing and non-applicable policies required')
        levels=_eq_records(criterion.get('levels'),'levels','level_id')
        if not levels: _fail('QUALITATIVE_LEVEL','Every criterion needs at least one level')
        points=[_ex_number(l.get('points'),'points')for l in levels];kind=criterion.get('kind')
        if kind=='categorical':
            categories=_ex_ids(criterion.get('categories'),'categories');assigned=[]
            for level in levels: assigned+=_ex_ids(level.get('values'),'level values')
            if len(set(assigned))!=len(assigned) or set(assigned)!=set(categories): _fail('QUALITATIVE_COVERAGE','Categorical levels must partition declared categories exactly')
        elif kind=='numeric':
            for level in levels:
                low=_ex_number(level.get('lower'),'lower');high=_ex_number(level.get('upper'),'upper')
                if low>=high: _fail('QUALITATIVE_LEVEL','Numeric level intervals must be nonempty')
            levels.sort(key=lambda v:v['lower'])
            if any(a['upper']!=b['lower']for a,b in zip(levels,levels[1:])): _fail('QUALITATIVE_COVERAGE','Numeric levels must be contiguous without overlap')
            direction=criterion.get('risk_order')
            if direction not in ('increasing','decreasing','none'): _fail('QUALITATIVE_DIRECTION','Declare numeric risk_order')
            changes=[b['points']-a['points']for a,b in zip(levels,levels[1:])]
            if direction=='increasing'and any(v<0 for v in changes) or direction=='decreasing'and any(v>0 for v in changes): _fail('QUALITATIVE_DIRECTION','Points violate declared numeric risk direction')
        else: _fail('QUALITATIVE_LEVEL','kind must be categorical or numeric')
        if criterion['not_applicable_policy']=='fixed':
            fixed=_ex_number(criterion.get('not_applicable_points'),'not_applicable_points')
            if not min(points)<=fixed<=max(points): _fail('QUALITATIVE_MISSING','Evidenced non-applicable fixed points must remain within declared criterion range')
        criterion['levels']=levels
    maximum=max(r['weight']for r in rows)
    if maximum<=0: _fail('QUALITATIVE_WEIGHT','Positive total criterion weight required')
    denominator=math.fsum(r['weight']/maximum for r in rows)
    for row in rows: row['normalized_weight']=(row['weight']/maximum)/denominator
    mapping={r['criterion_id']:r for r in rows};rules=_eq_records([]if interactions is None else interactions,'interactions','interaction_id');limits=_eq_records([]if rating_limits is None else rating_limits,'rating_limits','limit_id')
    for rule in rules:
        rule['conditions']=_eq_conditions(rule.get('conditions'),mapping);delta=_ex_number(rule.get('points_delta'),'points_delta')
        if delta<0: _fail('QUALITATIVE_INTERACTION','Only explicit adverse nonnegative interactions are supported')
        if not isinstance(rule.get('reference'),str) or not rule['reference']: _fail('QUALITATIVE_CONTRACT','Interaction reference required')
    for rule in limits:
        rule['conditions']=_eq_conditions(rule.get('conditions'),mapping)
        if rule.get('best_allowed_grade')not in grades or rule.get('worst_allowed_grade')not in grades or grades.index(rule['best_allowed_grade'])>grades.index(rule['worst_allowed_grade']): _fail('QUALITATIVE_LIMIT','Rating limits need a nonempty interval on declared ordered grades')
        if not isinstance(rule.get('reference'),str) or not rule['reference']: _fail('QUALITATIVE_CONTRACT','Rating-limit reference required')
    potential=math.fsum(max(l['points']for l in r['levels'])*r['normalized_weight']for r in rows)+math.fsum(r['points_delta']for r in rules)
    if not math.isfinite(potential): _fail('QUALITATIVE_SCALE','Potential score outside finite range')
    value=_ex_seal(dict(rubric_contract=contract,score_policy=policy,criteria=sorted(rows,key=lambda r:r['criterion_id']),interactions=sorted(rules,key=lambda r:r['interaction_id']),rating_limits=sorted(limits,key=lambda r:r['limit_id'])),'rubric_hash')
    return _result(value,'expert.compile_qualitative_rubric','qualitative_rubric',context,limitations=('Higher points mean higher risk. Fixed normalized weights never change with missingness. Unknown criteria refer or take worst points; evidenced non-applicability has its own declared policy. Numeric intervals are left-closed/right-open, with the final upper endpoint included.',))


def _eq_available(record,asof,knowledge,*,observation=False):
    begin=_date(record.get('valid_from'),'valid_from');end=None if record.get('valid_until')is None else _date(record['valid_until'],'valid_until');known=_date(record.get('known_at'),'known_at')
    if end is not None and end<=begin: _fail('QUALITATIVE_DATE','Empty evidence or observation validity interval')
    observed=_date(record.get('observed_at'),'observed_at')if observation else None
    if observation and known<observed: _fail('QUALITATIVE_DATE','Observation cannot be known before it occurs')
    if known>knowledge: return 'NOT_KNOWN'
    if asof<begin or end is not None and asof>=end or observation and observed>asof: return 'NOT_VALID'
    return None


def evaluate_qualitative_evidence(compiled_rubric,observations,evidence,*,as_of,known_at,conflicts=None,context=None):
    """Score only structured known/valid facts; retain evidence gaps and conflicts."""
    rubric=_ex_read(compiled_rubric,'rubric_hash');contract=rubric['rubric_contract'];_ex_target(contract['target'],context);asof=_date(as_of,'as_of');knowledge=_date(known_at,'known_at')
    if _date(contract['known_at'],'rubric known_at')>knowledge: _fail('QUALITATIVE_LOOKAHEAD','Rubric unavailable at evaluation knowledge cutoff')
    criteria={r['criterion_id']:r for r in rubric['criteria']};obs=_eq_records(observations,'observations','observation_id');ev=_eq_records(evidence,'evidence','evidence_id');cf=_eq_records([]if conflicts is None else conflicts,'conflicts','conflict_id');evidence_map={};exclusions=[]
    for row in ev:
        for key in ('evidence_type','source_id','independence_group','content_fingerprint'):
            if not isinstance(row.get(key),str) or not row[key]: _fail('QUALITATIVE_EVIDENCE',key+' required')
        if row.get('status')not in ('VERIFIED','UNVERIFIED','REJECTED'): _fail('QUALITATIVE_EVIDENCE','Evidence verification status required')
        reason=_eq_available(row,asof,knowledge)
        if reason is None and row['status']!='VERIFIED': reason=row['status']
        evidence_map[row['evidence_id']]=(row,reason)
        if reason: exclusions.append(dict(record_type='evidence',record_id=row['evidence_id'],reason=reason))
    active={cid:[]for cid in criteria};obs_map={r['observation_id']:r for r in obs}
    for row in obs:
        if row.get('criterion_id')not in criteria or not isinstance(row.get('not_applicable'),bool): _fail('QUALITATIVE_OBSERVATION','Known criterion and explicit not_applicable Boolean required')
        if 'value'not in row or row['not_applicable']and row['value']is not None: _fail('QUALITATIVE_OBSERVATION','Value required; non-applicable observations must have null value')
        eids=_ex_ids(row.get('evidence_ids'),'evidence_ids',True)
        if set(eids)-set(evidence_map): _fail('QUALITATIVE_EVIDENCE','Observation references unknown evidence')
        reason=_eq_available(row,asof,knowledge,observation=True)
        if reason: exclusions.append(dict(record_type='observation',record_id=row['observation_id'],reason=reason))
        else: active[row['criterion_id']].append(row)
    active_conflicts={cid:[]for cid in criteria}
    for row in cf:
        ids=_ex_ids(row.get('observation_ids'),'conflict observation_ids')
        if len(ids)<2 or set(ids)-set(obs_map) or len({obs_map[i]['criterion_id']for i in ids})!=1: _fail('QUALITATIVE_CONFLICT','A conflict links at least two known observations of one criterion')
        if row.get('status')not in ('OPEN','RESOLVED') or not isinstance(row.get('reference'),str) or not row['reference']: _fail('QUALITATIVE_CONFLICT','Conflict status and reference required')
        if row['status']=='RESOLVED'and row.get('resolution_observation_id')not in ids: _fail('QUALITATIVE_CONFLICT','Resolution must select one linked observation')
        reason=_eq_available(row,asof,knowledge)
        if reason: exclusions.append(dict(record_type='conflict',record_id=row['conflict_id'],reason=reason))
        else: active_conflicts[obs_map[ids[0]]['criterion_id']].append(row)
    results=[];unresolved=False;assumptions=False
    for cid,c in criteria.items():
        records=list(active[cid]);reasons=[];applicable_conflicts=active_conflicts[cid];selected={r['resolution_observation_id']for r in applicable_conflicts if r['status']=='RESOLVED'}
        if any(r['status']=='OPEN'for r in applicable_conflicts)or len(selected)>1: reasons.append('EXPLICIT_CONFLICT')
        if selected:
            chosen=next(iter(selected));removed=set(i for r in applicable_conflicts if r['status']=='RESOLVED'for i in r['observation_ids']if i!=chosen);records=[r for r in records if r['observation_id']not in removed]
            if chosen not in [r['observation_id']for r in records]: reasons.append('RESOLUTION_UNAVAILABLE')
        facts={_fingerprint(dict(value=r['value'],not_applicable=r['not_applicable']))for r in records}
        if len(facts)>1: reasons.append('CONFLICTING_OBSERVATIONS')
        eids=sorted({i for r in records for i in r['evidence_ids']});usable=[evidence_map[i][0]for i in eids if evidence_map[i][1]is None];types={r['evidence_type']for r in usable};groups={r['independence_group']for r in usable}
        covered=bool(records)and set(c['required_evidence_types'])<=types and len(groups)>=c['minimum_independent_sources']and not reasons
        points=None;level=None;status='UNRESOLVED';minimum=min(r['points']for r in c['levels']);maximum=max(r['points']for r in c['levels'])
        if reasons: status='CONFLICT';unresolved=True
        elif not covered:
            reasons.append('NO_OBSERVATION'if not records else 'EVIDENCE_GAP')
            if c['missing_policy']=='worst': points=maximum;status='ASSUMED_WORST';assumptions=True
            else: unresolved=True
        elif records[0]['not_applicable']:
            if c['not_applicable_policy']=='refer': reasons.append('NON_APPLICABLE_REFER');unresolved=True
            else: points=maximum if c['not_applicable_policy']=='worst'else c['not_applicable_points'];status='EVIDENCED_NOT_APPLICABLE'
        else:
            value=records[0]['value'];match=[]
            if c['kind']=='categorical': match=[r for r in c['levels']if isinstance(value,str)and value in r['values']]
            else:
                try: value=_ex_number(value,'observation value')
                except (ValueError,TypeError): value=None
                if value is not None: match=[r for j,r in enumerate(c['levels'])if r['lower']<=value and (value<r['upper']or j==len(c['levels'])-1 and value==r['upper'])]
            if not match: reasons.append('OUT_OF_DOMAIN');unresolved=True;covered=False
            else: level=match[0]['level_id'];points=match[0]['points'];status='OBSERVED'
        results.append(dict(criterion_id=cid,status=status,level_id=level,points=points,normalized_weight=c['normalized_weight'],weighted_contribution=None if points is None else points*c['normalized_weight'],contribution_interval=[(minimum if points is None else points)*c['normalized_weight'],(maximum if points is None else points)*c['normalized_weight']],evidence_covered=covered,observation_ids=sorted(r['observation_id']for r in records),evidence_ids=eids,eligible_evidence_ids=sorted(r['evidence_id']for r in usable),independent_groups=sorted(groups),conflict_ids=sorted(r['conflict_id']for r in applicable_conflicts),reasons=reasons))
    lookup={r['criterion_id']:r for r in results}
    def condition(conditions):
        unknown=False
        for cid,levels in conditions.items():
            level=lookup[cid]['level_id']
            if level is None: unknown=True
            elif level not in levels: return False
        return None if unknown else True
    interactions=[];low_delta=high_delta=0.;delta=0.
    for rule in rubric['interactions']:
        matched=condition(rule['conditions']);applied=matched is True or matched is None and rubric['score_policy']['unresolved_interaction_policy']=='apply_adverse'
        if matched is None:
            if not applied: unresolved=True
            else: assumptions=True
        low_delta+=rule['points_delta']if applied else 0.;high_delta+=rule['points_delta']if matched is not False else 0.;delta+=rule['points_delta']if applied else 0.
        interactions.append(dict(interaction_id=rule['interaction_id'],condition_matched=matched,applied=applied,points_delta=rule['points_delta']if applied else 0.,unresolved=matched is None))
    grades=contract['ordered_grades'];lo=0;hi=len(grades)-1;limits=[]
    for rule in rubric['rating_limits']:
        matched=condition(rule['conditions'])
        if matched is None: unresolved=True
        elif matched: lo=max(lo,grades.index(rule['best_allowed_grade']));hi=min(hi,grades.index(rule['worst_allowed_grade']))
        limits.append(dict(limit_id=rule['limit_id'],condition_matched=matched,best_allowed_grade=rule['best_allowed_grade'],worst_allowed_grade=rule['worst_allowed_grade']))
    rating_conflict=bool(grades)and lo>hi
    if rating_conflict: unresolved=True
    coverage=math.fsum(r['normalized_weight']for r in results if r['evidence_covered']);coverage_failed=coverage<rubric['score_policy']['minimum_evidence_coverage']
    if coverage_failed: unresolved=True
    clip=lambda p:max(rubric['score_policy']['score_floor'],min(rubric['score_policy']['score_cap'],p))
    bounds=[clip(math.fsum(r['contribution_interval'][0]for r in results)+low_delta),clip(math.fsum(r['contribution_interval'][1]for r in results)+high_delta)]
    score=None if any(r['points']is None for r in results)or any(r['unresolved']and not r['applied']for r in interactions)else clip(math.fsum(r['weighted_contribution']for r in results)+delta)
    return _result(dict(score=score,score_interval=bounds,status='REFER'if unresolved else 'READY_WITH_ASSUMPTIONS'if assumptions else 'READY',evidence_coverage=coverage,minimum_coverage_failed=coverage_failed,criterion_results=results,interaction_results=interactions,rating_limits=limits,rating_limit_conflict=rating_conflict,permitted_grade_interval=None if not grades or rating_conflict else [grades[lo],grades[hi]],excluded_records=sorted(exclusions,key=lambda r:(r['record_type'],r['record_id'])),observations_hash=_fingerprint(sorted(obs,key=lambda r:r['observation_id'])),evidence_hash=_fingerprint(sorted(ev,key=lambda r:r['evidence_id'])),conflicts_hash=_fingerprint(sorted(cf,key=lambda r:r['conflict_id'])),rubric_hash=rubric['rubric_hash'],as_of=as_of,known_at=known_at,target=contract['target']),
        'expert.evaluate_qualitative_evidence','qualitative_points',context,limitations=('Only supplied structured facts and verified, dated evidence are used. Evidence independence groups are caller declarations, not verified independence. Conflicting active facts always refer until a dated explicit resolution. A numeric score with REFER status is not an approval. No PD is inferred from qualitative points.',))
