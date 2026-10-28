"""Dated relationships and explicitly declared risk-adjustment mechanisms."""
from copy import deepcopy
import math
import numpy as np
from .core import CalcResult,_numeric,_probability,_fail,_result,_fingerprint,_context_value
from .data import _rows,_unique,_required,_date


def _object(x,name):
    x=x.value if isinstance(x,CalcResult) else x
    if not isinstance(x,dict):_fail('LINK_CONTRACT',name+' must be a named object')
    return x


def _ids(x,name,empty=False):
    if not isinstance(x,(list,tuple)) or (not x and not empty) or any(not isinstance(v,str) or not v for v in x) or len(set(x))!=len(x):_fail('LINK_CONTRACT',name+' requires distinct nonempty strings')
    return list(x)


def _string(x,name):
    if not isinstance(x,str) or not x:_fail('LINK_CONTRACT',name+' must be a nonempty string')
    return x


def _scalar(x,name,minimum=None):
    a=_numeric(x,name)
    if a.ndim or (minimum is not None and a<minimum):_fail('LINK_CONTRACT',name+' is outside its scalar domain')
    return float(a)


def _p(x,name):
    a=_probability(x,name=name)
    if a.ndim:_fail('LINK_CONTRACT',name+' must be scalar')
    return float(a)


def _sealed(value,field):
    value=deepcopy(value);value[field]=_fingerprint(value);return value


def _read_sealed(value,field):
    value=deepcopy(_object(value,field));expected=value.pop(field,None)
    if expected!=_fingerprint(value):_fail('LINK_INTEGRITY','Relationship artifact content or hash differs')
    value[field]=expected;return value


def _components(nodes,arcs):
    """Iterative Kosaraju strongly connected components; no recursion limit."""
    forward={n:[] for n in nodes};reverse={n:[] for n in nodes}
    for a,b in arcs:forward[a].append(b);reverse[b].append(a)
    seen=set();order=[]
    for start in nodes:
        if start in seen:continue
        stack=[(start,False)]
        while stack:
            node,finished=stack.pop()
            if finished:order.append(node);continue
            if node in seen:continue
            seen.add(node);stack.append((node,True));stack.extend((n,False)for n in reversed(sorted(forward[node]))if n not in seen)
    seen=set();components=[]
    for start in reversed(order):
        if start in seen:continue
        stack=[start];members=[];seen.add(start)
        while stack:
            node=stack.pop();members.append(node)
            for neighbor in reverse[node]:
                if neighbor not in seen:seen.add(neighbor);stack.append(neighbor)
        components.append(sorted(members))
    return sorted(components)


def build_entity_graph(entities,edges,relationship_policy,*,as_of,known_at,context=None):
    """Freeze known, valid, typed edges without turning ownership into support."""
    rows=_rows(entities,'entities');_unique(rows,'entity_id','entities');ids=sorted(r['entity_id']for r in rows)
    if not ids:_fail('LINK_ENTITIES','At least one declared entity is required')
    policy=_object(relationship_policy,'relationship_policy');_required(policy,['policy_id','version','edge_types'])
    _string(policy['policy_id'],'policy_id');_string(policy['version'],'version');types=_ids(policy['edge_types'],'edge_types')
    observation=_date(as_of,'as_of');knowledge=_date(known_at,'known_at')
    records=_rows(edges,'edges');_unique(records,'edge_id','edges');active=[];excluded=[]
    for row in records:
        _required(row,['source_id','target_id','edge_type','valid_from','known_at','evidence_reference'])
        if row['source_id'] not in ids or row['target_id'] not in ids:_fail('LINK_ENTITY','Every relationship endpoint must exist')
        if row['edge_type'] not in types:_fail('LINK_TYPE','Relationship type not declared by policy')
        _string(row['evidence_reference'],'evidence_reference');begin=_date(row['valid_from'],'valid_from');available=_date(row['known_at'],'known_at')
        end=None if row.get('valid_until') is None else _date(row['valid_until'],'valid_until')
        if end is not None and end<=begin:_fail('LINK_DATE','Relationship validity interval is empty')
        if row['edge_type']=='OWNERSHIP':
            _required(row,['ownership_fraction']);_p(row['ownership_fraction'],'ownership_fraction')
        if available>knowledge:reason='NOT_KNOWN'
        elif observation<begin or (end is not None and observation>=end):reason='NOT_VALID'
        else:reason=None
        if reason:excluded.append(dict(edge_id=row['edge_id'],reason=reason))
        else:active.append(deepcopy(row))
    active.sort(key=lambda r:r['edge_id']);duplicates=[];seen={}
    for edge in active:
        key=(edge['source_id'],edge['target_id'],edge['edge_type'])
        if key in seen:duplicates.append(dict(first_edge_id=seen[key],edge_id=edge['edge_id'],endpoints=list(key[:2]),edge_type=key[2]))
        else:seen[key]=edge['edge_id']
    cycles={}
    for kind in types:
        arcs=[(e['source_id'],e['target_id'])for e in active if e['edge_type']==kind]
        cyclic=[c for c in _components(ids,arcs)if len(c)>1 or (c[0],c[0])in arcs]
        if cyclic:cycles[kind]=cyclic
    value=_sealed(dict(as_of=as_of,known_at=known_at,entities=sorted(deepcopy(rows),key=lambda r:r['entity_id']),edges=active,policy=deepcopy(policy)),'graph_hash')
    return _result(value,'links.build_entity_graph','entity_graph',context,
        diagnostics=dict(excluded=sorted(excluded,key=lambda r:r['edge_id']),duplicate_active_relationships=duplicates,cycles=cycles),
        limitations=('Relationship types carry separate meanings. Ownership or control alone does not establish support, enforceability, shared PD or joint default.',
                    'Validity is [valid_from,valid_until); known_at may create a retrospective view and must match the historical decision cutoff for backtesting.'))


def resolve_parent_links(entity_graph,control_policy,*,context=None):
    """Resolve unique eligible parents and ultimate paths, preserving ambiguity."""
    graph=_read_sealed(entity_graph,'graph_hash');policy=_object(control_policy,'control_policy')
    _required(policy,['policy_id','version','control_edge_types','ownership_threshold','ownership_inclusive'])
    kinds=_ids(policy['control_edge_types'],'control_edge_types',True)
    if set(kinds)-set(graph['policy']['edge_types']):_fail('LINK_POLICY','Control edge types must exist in the graph policy')
    threshold=_p(policy['ownership_threshold'],'ownership_threshold')
    if not isinstance(policy['ownership_inclusive'],bool):_fail('LINK_POLICY','ownership_inclusive must be Boolean')
    ids=[r['entity_id']for r in graph['entities']];parents={n:{} for n in ids}
    for edge in graph['edges']:
        eligible=edge['edge_type'] in kinds
        if edge['edge_type']=='OWNERSHIP':eligible=edge['ownership_fraction']>=threshold if policy['ownership_inclusive'] else edge['ownership_fraction']>threshold
        if eligible:parents[edge['target_id']].setdefault(edge['source_id'],[]).append(edge['edge_id'])
    output=[]
    for entity in ids:
        candidates=sorted(parents[entity]);path=[entity];trace=[];current=entity;status='RESOLVED';ultimate=None
        while True:
            choices=sorted(parents[current])
            if not choices:ultimate=current;break
            if len(choices)>1:status='AMBIGUOUS_PARENT';break
            parent=choices[0];trace.extend(sorted(parents[current][parent]));path.append(parent)
            if parent in path[:-1]:status='CONTROL_CYCLE';break
            current=parent
        output.append(dict(entity_id=entity,direct_parent=candidates[0]if len(candidates)==1 else None,
            direct_candidates=candidates,ultimate_parent=ultimate if ultimate!=entity else None,path=path,edge_trace=trace,status=status))
    return _result(dict(graph_hash=graph['graph_hash'],control_policy_hash=_fingerprint(policy),parents=output),'links.resolve_parent_links','parent_mapping',context,
        limitations=('Explicit control edges and the supplied ownership threshold determine eligibility. Indirect share products and support capacity are not inferred. Ambiguous or cyclic chains remain unresolved.',))


def assign_default_groups(entity_graph,default_link_policy,facility_mappings,*,context=None):
    """Apply declared equivalence rules; nontransitive overlaps remain conflicts."""
    graph=_read_sealed(entity_graph,'graph_hash');policy=_object(default_link_policy,'default_link_policy')
    _required(policy,['policy_id','version','edge_types','direction_policy','transitive'])
    kinds=_ids(policy['edge_types'],'edge_types')
    if set(kinds)-set(graph['policy']['edge_types']):_fail('LINK_POLICY','Default edge types must exist in the graph policy')
    direction=policy['direction_policy']
    if direction not in ('mutual_only','undirected_explicit') or not isinstance(policy['transitive'],bool):_fail('LINK_POLICY','Declare mutual_only/undirected_explicit and Boolean transitive')
    ids=[r['entity_id']for r in graph['entities']];pairs={};trace=[]
    for edge in graph['edges']:
        if edge['edge_type'] in kinds:pairs.setdefault((edge['source_id'],edge['target_id']),[]).append(edge['edge_id'])
    neighbors={n:set()for n in ids}
    for (a,b),edge_ids in sorted(pairs.items()):
        accepted=direction=='undirected_explicit' or (b,a)in pairs
        trace.append(dict(source_id=a,target_id=b,edge_ids=sorted(edge_ids),accepted=accepted))
        if accepted:neighbors[a].add(b);neighbors[b].add(a)
    arcs=[(a,b)for a in ids for b in sorted(neighbors[a])];components=_components(ids,arcs);assignments={};groups=[];conflicts=[]
    for members in components:
        complete=all(b in neighbors[a]for a in members for b in members if a!=b)
        if not policy['transitive'] and not complete:
            conflicts.append(dict(members=members,status='NONTRANSITIVE_OVERLAP'))
            for node in members:assignments[node]=None
        else:
            group_id='default-group-'+_fingerprint(dict(members=members,policy=policy))[:20]
            groups.append(dict(group_id=group_id,members=members))
            for node in members:assignments[node]=group_id
    facilities=_rows(facility_mappings,'facility_mappings');_unique(facilities,'facility_id','facility_mappings');mapped=[]
    for row in sorted(facilities,key=lambda r:r['facility_id']):
        _required(row,['entity_id'])
        if row['entity_id'] not in assignments:_fail('LINK_ENTITY','Facility owner is absent from the graph')
        mapped.append(dict(facility_id=row['facility_id'],entity_id=row['entity_id'],group_id=assignments[row['entity_id']]))
    return _result(dict(graph_hash=graph['graph_hash'],policy_hash=_fingerprint(policy),groups=groups,
        assignments=[dict(entity_id=n,group_id=assignments[n])for n in ids],facilities=mapped,conflicts=conflicts),
        'links.assign_default_groups','default_groups',context,diagnostics=dict(edge_trace=trace),
        limitations=('Default-group equivalence is a supplied policy. A unilateral trigger is not symmetric default equivalence; mutual-only policy requires both directions.',
                    'Without transitivity, overlapping non-clique links cannot define disjoint groups and remain unresolved. No member PD is copied or averaged.'))


def _risk(risk,context):
    risk=_object(risk,'risk');_required(risk,['entity_id','pd','stage','target','applied_effects'])
    _string(risk['entity_id'],'entity_id');p=_p(risk['pd'],'pd');effects=_ids(risk['applied_effects'],'applied_effects',True)
    target=_object(risk['target'],'target');_required(target,['event','horizon','time_unit','probability_measure','conditioning'])
    if target['event']!='default' or _scalar(target['horizon'],'horizon')<=0 or target['time_unit'] not in ('days','months','years') or target['probability_measure'] not in ('PHYSICAL','RISK_NEUTRAL'):_fail('LINK_TARGET','A default target with positive horizon, time unit and measure is required')
    _string(target['conditioning'],'conditioning');ctx=_context_value(context)
    for key in ('horizon','time_unit','probability_measure','conditioning'):
        if ctx.get(key) is not None and ctx[key]!=target[key]:_fail('TARGET_MISMATCH','Risk target conflicts with context')
    return risk,p,effects,target,ctx


def apply_parent_support(standalone_risk,parent_risk,support_contract,*,joint_default_probability,context=None):
    """Integrate conditional rescue with explicit availability and allocated capacity."""
    base,p,effects,target,ctx=_risk(standalone_risk,context);parent,q,parent_effects,parent_target,_=_risk(parent_risk,context)
    if base['stage']!='STANDALONE' or parent['stage']!='STANDALONE' or base['entity_id']==parent['entity_id']:_fail('SUPPORT_STAGE','Distinct standalone child and supporter risks required; recursive support must use an explicit network model')
    if {k:v for k,v in target.items()if k not in ('entity_id','exposure_id')}!={k:v for k,v in parent_target.items()if k not in ('entity_id','exposure_id')}:_fail('TARGET_MISMATCH','Parent and child default horizon, measure and conditioning must agree')
    contract=_object(support_contract,'support_contract');_required(contract,['effect_id','assumption_reference','currency','total_capacity','reserved_elsewhere','allocated_capacity','support_required','states'])
    effect=_string(contract['effect_id'],'effect_id');_string(contract['assumption_reference'],'assumption_reference');currency=_string(contract['currency'],'currency')
    if effect in effects or effect in parent_effects:_fail('LINK_DOUBLE_COUNT','Support effect already applied to an input risk')
    if ctx.get('currency') is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Support currency conflicts with context')
    total=_scalar(contract['total_capacity'],'total_capacity',0);reserved=_scalar(contract['reserved_elsewhere'],'reserved_elsewhere',0);capacity=_scalar(contract['allocated_capacity'],'allocated_capacity',0);required=_scalar(contract['support_required'],'support_required',0)
    if required<=0 or reserved>total or capacity>total-reserved:_fail('SUPPORT_CAPACITY','Positive required amount and allocation within remaining total budget required')
    states=_object(contract['states'],'states')
    if set(states)!={'parent_alive','joint_default'}:_fail('SUPPORT_STATES','Both parent-alive and joint-default conditional states are required')
    j=_p(joint_default_probability,'joint_default_probability');lower=max(0.,math.fsum((p,q,-1.)))
    if not lower<=j<=min(p,q):_fail('INVALID_JOINT_PROBABILITY','Child/supporter intersection outside Frechet bounds')
    details=[];rescued=[]
    for state,mass in [('parent_alive',p-j),('joint_default',j)]:
        terms=_object(states[state],state);_required(terms,['available_capacity','willingness_given_state','success_given_willing_and_funded','support_permitted','evidence_reference'])
        available=_scalar(terms['available_capacity'],'available_capacity',0)
        if available>capacity or not isinstance(terms['support_permitted'],bool):_fail('SUPPORT_CAPACITY','State capacity cannot exceed earmarked capacity; permission must be Boolean')
        _string(terms['evidence_reference'],'evidence_reference');willing=_p(terms['willingness_given_state'],'willingness');success=_p(terms['success_given_willing_and_funded'],'success')
        funded=available>=required;conditional=willing*success if funded and terms['support_permitted'] else 0.;rescue=mass*conditional;rescued.append(rescue)
        details.append(dict(state=state,child_default_mass=mass,available_capacity=available,funding_sufficient=funded,support_permitted=terms['support_permitted'],rescue_probability_given_state=conditional,rescued_mass=rescue,remaining_default_mass=mass-rescue))
    supported=math.fsum((p,-rescued[0],-rescued[1]))
    if supported<0 and supported>=-8*np.finfo(float).eps:supported=0.
    if not 0<=supported<=p:_fail('NUMERICAL_FAILURE','Support event decomposition has invalid remaining mass')
    return _result(dict(entity_id=base['entity_id'],supporter_id=parent['entity_id'],stage='PRE_TRANSFER',pd=supported,standalone_pd=p,parent_pd=q,joint_default_probability=j,target=deepcopy(target),
        applied_effects=effects+[effect],support_states=details,capacity_contract=dict(currency=currency,total_capacity=total,reserved_elsewhere=reserved,allocated_capacity=capacity,support_required=required),assumption_reference=contract['assumption_reference']),
        'links.apply_parent_support','probability',ctx,limitations=('Conditional support willingness and effectiveness are declared assumptions. Allocated cash must cover the declared rescue need; no arbitrary capacity ratio is treated as a probability.',
        'Support during supporter default requires explicit permitted, funded, evidenced conditional terms. The resulting PD is before transfer adjustment; no risk contagion or rescue budget across undeclared beneficiaries is inferred.'))


def allocate_guarantee_coverage(guarantees,claims,allocation_policy,*,as_of,known_at,currency,context=None):
    """Allocate a non-reinstating shared cap once, under explicit ordered priorities."""
    gs=_rows(guarantees,'guarantees');cs=_rows(claims,'claims');_unique(gs,'guarantee_id','guarantees');_unique(cs,'claim_id','claims')
    _string(currency,'currency');date=_date(as_of,'as_of');knowledge=_date(known_at,'known_at');policy=_object(allocation_policy,'allocation_policy')
    _required(policy,['policy_id','version','guarantee_priority','claim_priority','cap_convention'])
    gp=_ids(policy['guarantee_priority'],'guarantee_priority',True);cp=_ids(policy['claim_priority'],'claim_priority',True)
    if set(gp)!={g['guarantee_id']for g in gs} or set(cp)!={c['claim_id']for c in cs} or policy['cap_convention']!='non_reinstating_total':_fail('GUARANTEE_POLICY','Complete priority orders and non_reinstating_total cap convention required')
    ctx=_context_value(context)
    if ctx.get('currency') is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Allocation currency conflicts with context')
    claim_map={};remaining={}
    for claim in cs:
        _required(claim,['entity_id','amount','currency','due_at','known_at']);_string(claim['entity_id'],'entity_id');amount=_scalar(claim['amount'],'claim amount',0);due=_date(claim['due_at'],'due_at');available=_date(claim['known_at'],'known_at')
        if claim['currency']!=currency or due<date or available>knowledge:_fail('GUARANTEE_CLAIM','Claims must be known, in reporting currency, and not past due at the allocation date')
        claim_map[claim['claim_id']]=claim;remaining[claim['claim_id']]=amount
    guarantee_map={};unused={};exclusions=[]
    for guarantee in gs:
        _required(guarantee,['guarantor_id','currency','cap','cap_used','coverage_fraction','covered_claim_ids','valid_from','valid_until','known_at','legally_effective','evidence_reference'])
        _string(guarantee['guarantor_id'],'guarantor_id');cap=_scalar(guarantee['cap'],'cap',0);used=_scalar(guarantee['cap_used'],'cap_used',0);fraction=_p(guarantee['coverage_fraction'],'coverage_fraction');covered=_ids(guarantee['covered_claim_ids'],'covered_claim_ids',True)
        if used>cap or set(covered)-set(claim_map):_fail('GUARANTEE_CAP','Consumed cap exceeds limit or references unknown claims')
        if guarantee['currency']!=currency:_fail('GUARANTEE_CURRENCY','Convert nominal caps under an explicit FX contract before allocation')
        begin=_date(guarantee['valid_from'],'valid_from');end=_date(guarantee['valid_until'],'valid_until');available=_date(guarantee['known_at'],'known_at');_string(guarantee['evidence_reference'],'evidence_reference')
        if end<=begin or not isinstance(guarantee['legally_effective'],bool):_fail('GUARANTEE_CONTRACT','Nonempty validity interval and explicit legal-effectiveness flag required')
        reason='NOT_KNOWN'if available>knowledge else 'NOT_EFFECTIVE'if not guarantee['legally_effective'] else 'EXPIRED'if date>=end else None
        guarantee_map[guarantee['guarantee_id']]=guarantee;unused[guarantee['guarantee_id']]=cap-used
        if reason:exclusions.append(dict(guarantee_id=guarantee['guarantee_id'],reason=reason))
    excluded={x['guarantee_id']for x in exclusions};allocations=[]
    for gid in gp:
        if gid in excluded:continue
        g=guarantee_map[gid]
        for cid in cp:
            if cid not in g['covered_claim_ids']:continue
            claim=claim_map[cid];due=_date(claim['due_at'],'due_at')
            if not _date(g['valid_from'],'valid_from')<=due<_date(g['valid_until'],'valid_until'):continue
            amount=min(remaining[cid],g['coverage_fraction']*claim['amount'],unused[gid])
            if amount>0:
                allocations.append(dict(guarantee_id=gid,guarantor_id=g['guarantor_id'],claim_id=cid,entity_id=claim['entity_id'],covered_amount=amount,currency=currency,due_at=claim['due_at']))
                remaining[cid]-=amount;unused[gid]-=amount
    value=_sealed(dict(as_of=as_of,known_at=known_at,currency=currency,policy=deepcopy(policy),claims=[deepcopy(claim_map[cid])for cid in sorted(claim_map)],allocations=allocations,
        uncovered_claims=[dict(claim_id=cid,amount=remaining[cid])for cid in sorted(remaining)],remaining_caps=[dict(guarantee_id=gid,amount=unused[gid])for gid in sorted(unused)],excluded=sorted(exclusions,key=lambda r:r['guarantee_id'])),'allocation_hash')
    return _result(value,'links.allocate_guarantee_coverage',currency,ctx,
        limitations=('Cap allocations are simultaneous, non-reinstating reservations over all supplied claims; priorities determine allocation and are never inferred from input row order.',
                    'A due date must fall in the guarantee validity interval. Currency conversion, legal enforceability, existing cap usage and unreported obligations are supplied assumptions.'))


def apply_guarantee_risk(borrower_risk,guarantor_risk,coverage_allocation,claim_id,loss_contract,*,joint_default_probability,context=None):
    """Apply allocated payment protection to state losses without changing borrower PD."""
    borrower,p,effects,target,ctx=_risk(borrower_risk,context);guarantor,q,ge,gt,_=_risk(guarantor_risk,context)
    semantic=lambda t:{k:v for k,v in t.items()if k not in ('entity_id','exposure_id')}
    if semantic(target)!=semantic(gt) or borrower['entity_id']==guarantor['entity_id']:_fail('TARGET_MISMATCH','Distinct borrower/guarantor with matching default event contract required')
    allocation=_read_sealed(coverage_allocation,'allocation_hash');contract=_object(loss_contract,'loss_contract')
    _required(contract,['effect_id','currency','assumption_reference','stage','states','valuation_date']);effect=_string(contract['effect_id'],'effect_id');_string(contract['assumption_reference'],'assumption_reference')
    if borrower['stage']=='GUARANTEE_ADJUSTED' or contract['stage']!='PRE_GUARANTEE' or effect in effects or effect in ge:_fail('LINK_DOUBLE_COUNT','Only unadjusted conditional losses and a new effect are accepted')
    if contract['currency']!=allocation['currency'] or ctx.get('currency',contract['currency'])!=contract['currency']:_fail('TARGET_MISMATCH','Loss and allocation currency must agree')
    _string(claim_id,'claim_id');claim=next((r for r in allocation['claims']if r['claim_id']==claim_id),None)
    if claim is None or claim['entity_id']!=borrower['entity_id']:_fail('GUARANTEE_CLAIM','Claim must belong to the borrower in the frozen allocation')
    if contract['valuation_date']!=claim['due_at']:_fail('GUARANTEE_VALUATION','Loss and allocated nominal protection must be measured at the claim due date')
    matches=[r for r in allocation['allocations']if r['claim_id']==claim_id]
    if not matches and claim_id not in [r['claim_id']for r in allocation['uncovered_claims']]:_fail('GUARANTEE_CLAIM','Claim is absent from the frozen allocation')
    if any(r['entity_id']!=borrower['entity_id'] or r['guarantor_id']!=guarantor['entity_id'] for r in matches):_fail('GUARANTEE_JOINT_STATES','All allocated protection on this claim must belong to this borrower and one guarantor; multiple guarantors require a joint event model')
    covered=math.fsum(r['covered_amount']for r in matches);j=_p(joint_default_probability,'joint_default_probability')
    if not max(0.,math.fsum((p,q,-1.)))<=j<=min(p,q):_fail('INVALID_JOINT_PROBABILITY','Borrower/guarantor intersection outside Frechet bounds')
    states=_object(contract['states'],'states')
    if set(states)!={'borrower_only','joint_default'}:_fail('GUARANTEE_STATES','Explicit conditional borrower-only and joint-default loss/payout terms required')
    rows=[]
    for name,mass in [('borrower_only',p-j),('joint_default',j)]:
        terms=_object(states[name],name);_required(terms,['loss_before_guarantee','payout_fraction','evidence_reference']);_string(terms['evidence_reference'],'evidence_reference')
        loss=_scalar(terms['loss_before_guarantee'],'loss_before_guarantee');fraction=_p(terms['payout_fraction'],'payout_fraction')
        payment=min(covered,max(loss,0.))*fraction
        rows.append(dict(state=name,probability=mass,loss_before_guarantee=loss,guarantee_payment=payment,loss_after_guarantee=loss-payment))
    before=math.fsum(r['probability']*r['loss_before_guarantee']for r in rows);after=math.fsum(r['probability']*r['loss_after_guarantee']for r in rows)
    if not math.isfinite(before) or not math.isfinite(after):_fail('NUMERICAL_FAILURE','Expected guaranteed loss outside finite range')
    return _result(dict(entity_id=borrower['entity_id'],guarantor_id=guarantor['entity_id'],claim_id=claim_id,pd=p,guarantor_pd=q,target=deepcopy(target),stage='GUARANTEE_ADJUSTED',currency=contract['currency'],covered_amount=covered,expected_loss_before=before,expected_loss_after=after,states=rows,applied_effects=effects+[effect],allocation_hash=allocation['allocation_hash']),
        'links.apply_guarantee_risk',contract['currency'],ctx,limitations=('The guarantee changes conditional monetary loss, never borrower PD. Loss and payout amounts are measured in the allocation currency at the claim due date. Discount the resulting state losses separately under an explicit convention.',
        'Only one guarantor per claim is supported by this two-event calculation. Shared nominal caps must first be allocated across all claims; sequentially applying marginal guarantees is not a joint loss model.'))


def apply_rating_constraints(base_result,rating_scale,constraints,*,representative_pd_policy,context=None):
    """Intersect explicit grade bounds; change continuous PD only under binding policy."""
    from .core import _rating_scale,_scale_index
    base=_object(base_result,'base_result');_required(base,['pd','grade']);pd=_p(base['pd'],'pd');scale=_rating_scale(rating_scale);grades=[g['grade']for g in scale['grades']]
    if base['grade']not in grades or grades[_scale_index(pd,scale)]!=base['grade']:_fail('RATING_CONSISTENCY','Base grade must map from the supplied continuous PD and scale')
    if representative_pd_policy not in ('preserve_pd','representative_on_binding'):_fail('RATING_POLICY','Declare preserve_pd or representative_on_binding')
    rows=_rows(constraints,'constraints');_unique(rows,'rule_id','constraints');lo=0;hi=len(grades)-1;trace=[];original=grades.index(base['grade'])
    for rule in rows:
        _required(rule,['kind','evidence_reference']);_string(rule['evidence_reference'],'evidence_reference');kind=rule['kind']
        if kind=='grade_bounds':
            _required(rule,['best_allowed_grade','worst_allowed_grade'])
            if rule['best_allowed_grade']not in grades or rule['worst_allowed_grade']not in grades:_fail('RATING_CONSTRAINT','Unknown bound grade')
            a=grades.index(rule['best_allowed_grade']);b=grades.index(rule['worst_allowed_grade'])
        elif kind=='notch_bounds':
            _required(rule,['reference_grade','max_improvement','max_deterioration'])
            if rule['reference_grade']not in grades:_fail('RATING_CONSTRAINT','Unknown reference grade')
            up=_scalar(rule['max_improvement'],'max_improvement',0);down=_scalar(rule['max_deterioration'],'max_deterioration',0)
            if up!=math.floor(up) or down!=math.floor(down):_fail('RATING_CONSTRAINT','Notch distances must be nonnegative integers')
            ref=grades.index(rule['reference_grade']);a=max(0,ref-int(up));b=min(len(grades)-1,ref+int(down))
        else:_fail('RATING_CONSTRAINT','Unknown constraint kind')
        if a>b:_fail('RATING_CONFLICT','A rule has empty permitted grade interval')
        lo=max(lo,a);hi=min(hi,b)
        trace.append(dict(rule_id=rule['rule_id'],best_allowed_grade=grades[a],worst_allowed_grade=grades[b],excludes_original=not a<=original<=b))
    if lo>hi:_fail('RATING_CONFLICT','Constraint intersection is empty; rule order cannot resolve a contradiction')
    final=max(lo,min(hi,original));binding=final!=original;constrained=pd if not binding or representative_pd_policy=='preserve_pd'else scale['grades'][final]['representative_pd']
    active=[r['rule_id']for r in trace if binding and ((final>original and r['best_allowed_grade']==grades[final]) or (final<original and r['worst_allowed_grade']==grades[final]))]
    return _result(dict(original=deepcopy(base),pd=constrained,grade=grades[final],stage='CONSTRAINED',binding=binding,binding_rules=active,rules=trace,representative_pd_policy=representative_pd_policy,pd_grade_consistent=grades[_scale_index(constrained,scale)]==grades[final],scale_id=scale['scale_id'],scale_version=scale['version']),
        'links.apply_rating_constraints','rating',context,limitations=('Grades are ordered from lower to higher risk. All bounds are intersected, with conflicts rejected. Nonbinding constraints preserve continuous PD exactly.',
        'preserve_pd changes only the grade; pd_grade_consistent exposes any resulting difference. representative_on_binding selects a scale representative only when the original grade is excluded.'))


def apply_override(base_result,override_request,policy,*,as_of,known_at,context=None):
    """Append a dated explicit override to a content-bound local audit chain."""
    base=_object(base_result,'base_result');request=deepcopy(_object(override_request,'override_request'));policy=_object(policy,'policy')
    _required(policy,['policy_id','version','targets','require_dual_control']);_string(policy['policy_id'],'policy_id');_string(policy['version'],'version')
    if not isinstance(policy['require_dual_control'],bool):_fail('OVERRIDE_POLICY','require_dual_control must be Boolean')
    targets=_object(policy['targets'],'targets');_required(request,['override_id','target','value','reason','author','approved_by','recorded_at','valid_from','valid_until'])
    for key in ('override_id','target','reason','author','approved_by'):_string(request[key],key)
    if policy['require_dual_control'] and request['author']==request['approved_by']:_fail('OVERRIDE_APPROVAL','Policy requires distinct author and approver')
    date=_date(as_of,'as_of');knowledge=_date(known_at,'known_at');recorded=_date(request['recorded_at'],'recorded_at');begin=_date(request['valid_from'],'valid_from');end=_date(request['valid_until'],'valid_until')
    if end<=begin:_fail('OVERRIDE_DATE','Override validity interval must be nonempty')
    if 'override_hash'in base:
        prior=_read_sealed(base,'override_hash')
        if any(s['as_of']!=as_of or s['known_at']!=known_at for s in prior['history']):_fail('OVERRIDE_REPLAY','A different evaluation cutoff requires replaying requests from the preserved original')
        original=prior['original'];current=deepcopy(prior['current']);history=deepcopy(prior['history']);previous=prior['override_hash']
    else:
        if base.get('stage')=='OVERRIDDEN':_fail('OVERRIDE_INTEGRITY','An overridden value must retain its full audit artifact')
        original=deepcopy(base);current=deepcopy(base);history=[];previous=_fingerprint(base)
    if any(s['request']['override_id']==request['override_id']for s in history):_fail('OVERRIDE_DUPLICATE','Override ID already recorded in this history')
    if request['target']not in targets:_fail('OVERRIDE_POLICY','Requested target is not allowed')
    contract=_object(targets[request['target']],'target contract');_required(contract,['path','kind']);path=_ids(contract['path'],'path')
    if path[0]in ('stage','history','original','override_hash','applied_effects','target'):_fail('OVERRIDE_POLICY','Audit, effect and event-contract fields cannot be overwritten')
    node=current
    for key in path[:-1]:
        if key not in node or not isinstance(node[key],dict):_fail('OVERRIDE_TARGET','Override path does not exist')
        node=node[key]
    if path[-1]not in node:_fail('OVERRIDE_TARGET','Override field does not exist')
    replacement=request['value'];kind=contract['kind']
    if kind=='probability':_p(replacement,'override value')
    elif kind=='numeric':
        _required(contract,['minimum','maximum']);a=_scalar(contract['minimum'],'minimum');b=_scalar(contract['maximum'],'maximum');v=_scalar(replacement,'override value')
        if not a<=v<=b:_fail('OVERRIDE_VALUE','Numeric override outside declared bounds')
    elif kind=='grade':
        allowed=_ids(contract.get('allowed_values'),'allowed_values')
        if replacement not in allowed:_fail('OVERRIDE_VALUE','Grade override outside allowed set')
    else:_fail('OVERRIDE_POLICY','Target kind must be probability, numeric or grade')
    status='NOT_KNOWN'if recorded>knowledge else 'NOT_YET_VALID'if date<begin else 'EXPIRED'if date>=end else 'APPLIED'
    before=deepcopy(node[path[-1]])
    if status=='APPLIED':node[path[-1]]=deepcopy(replacement);current['stage']='OVERRIDDEN'
    step=_sealed(dict(request=request,policy_hash=_fingerprint(policy),as_of=as_of,known_at=known_at,status=status,before=before,after=deepcopy(node[path[-1]]),previous_hash=previous),'step_hash');history.append(step)
    value=_sealed(dict(original=original,current=current,history=history,status=status),'override_hash')
    return _result(value,'links.apply_override','override_artifact',context,limitations=('This local audit artifact records declared author and approval evidence; it does not authenticate people or execute a credit action. Copying only current loses required audit history and must not create a new unadjusted baseline.',
        'Only the requested field changes. A grade override does not silently change PD or another parameter; dependent consistency checks remain explicit. Validity is [valid_from,valid_until).'))


def solve_contagion_network(liability_matrix,external_assets,external_liabilities,*,entity_ids,currency,model_assumptions,asset_scenarios=None,tolerance=1e-8,max_iterations=10000,context=None):
    """Greatest Eisenberg--Noe clearing with pari-passu outside obligations."""
    from scipy.optimize import linprog
    ids=_ids(entity_ids,'entity_ids');n=len(ids);_string(currency,'currency');ctx=_context_value(context)
    if ctx.get('currency') is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Clearing currency conflicts with context')
    required=dict(model='eisenberg_noe',seniority='single_pari_passu',bankruptcy_costs='none',outside_creditor='sink')
    if model_assumptions!=required:_fail('CLEARING_ASSUMPTIONS','Declare Eisenberg-Noe, one pari-passu debt class, no bankruptcy costs and outside creditor sink')
    matrix=_numeric(liability_matrix,'liability_matrix');outside=_numeric(external_liabilities,'external_liabilities',vector_only=True)
    if matrix.shape!=(n,n)or outside.shape!=(n,)or np.any(matrix<0)or np.any(outside<0)or np.any(np.diag(matrix)!=0):_fail('CLEARING_INPUT','Aligned nonnegative debtor-row liabilities, zero diagonal and outside obligations required')
    tol=_scalar(tolerance,'tolerance');iterations=_scalar(max_iterations,'max_iterations')
    if not 1e-12<=tol<=1e-4 or iterations!=math.floor(iterations)or not 1<=iterations<=1000000:_fail('CLEARING_CONTROL','Tolerance in [1e-12,1e-4] and integer iteration cap in [1,1000000] required')
    with np.errstate(over='ignore'):nominal=matrix.sum(axis=1)+outside
    if not np.all(np.isfinite(nominal)):_fail('NUMERICAL_FAILURE','Total nominal liabilities outside finite range')
    active=nominal>0;relative=np.divide(matrix,nominal[:,None],out=np.zeros_like(matrix),where=nominal[:,None]>0)
    def solve(assets):
        assets=_numeric(assets,'external_assets',vector_only=True)
        if assets.shape!=(n,)or np.any(assets<0):_fail('CLEARING_INPUT','One nonnegative external asset amount per entity required')
        scale=max(float(nominal.max()),float(assets.max()),1e-300)
        amounts=np.concatenate((nominal[active],assets[assets>0],matrix[matrix>0],outside[outside>0]))
        if len(amounts)and float(amounts.min())/scale<1e-10:_fail('CLEARING_SCALE','Amount dynamic range exceeds supported numerical scale; rescale or aggregate with documented economic justification')
        if not np.any(active):payment=np.zeros(n);solver='NO_LIABILITIES'
        else:
            # Positive fractional-payment objective is strictly increasing in every active payment.
            a=(np.diag(nominal)-matrix.T)/scale
            solution=linprog(-active.astype(float),A_ub=a,b_ub=assets/scale,bounds=[(0.,1. if flag else 0.)for flag in active],method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
            if not solution.success:_fail('CLEARING_SOLVER','Linear clearing optimization failed: '+solution.message)
            payment=solution.x*nominal;solver='LINEAR_PROGRAM_GREATEST'
        inflow=relative.T@payment;fixed=np.minimum(nominal,assets+inflow);denom=np.where(active,nominal,1.);residual=np.abs(payment-fixed)/denom
        if np.any(payment<0)or np.any(payment>nominal)or np.max(residual)>tol:_fail('CLEARING_RESIDUAL','Greatest solution fails node-relative fixed-point or liability bounds')
        lower=np.zeros(n);steps=0
        for step in range(int(iterations)):
            updated=np.minimum(nominal,assets+relative.T@lower);steps=step+1
            change=float(np.max(np.abs(updated-lower)/denom));lower=updated
            if change<=tol:break
        lower_residual=np.abs(lower-np.minimum(nominal,assets+relative.T@lower))/denom;gap=(payment-lower)/denom
        if np.min(gap)<-tol:_fail('CLEARING_RESIDUAL','Monotone lower iterate exceeds greatest solution')
        status='NUMERICALLY_ENCLOSED'if np.max(np.abs(gap))<=tol else 'MULTIPLE_CLEARINGS'if np.all(lower_residual==0)else 'NOT_ESTABLISHED'
        fraction=np.divide(payment,nominal,out=np.zeros(n),where=active);paid=matrix*fraction[:,None];paid_outside=outside*fraction;equity=assets+inflow-payment
        if not np.all(np.isfinite(equity)):_fail('NUMERICAL_FAILURE','Clearing equity outside finite range')
        defaults=[ids[i]for i in range(n)if active[i]and fraction[i]<1-tol]
        return dict(external_assets=assets.tolist(),nominal_payments=nominal.tolist(),clearing_payments=payment.tolist(),interbank_payments=paid.tolist(),outside_payments=paid_outside.tolist(),equity=equity.tolist(),default_set=defaults,
            diagnostics=dict(solver=solver,node_relative_residual=residual.tolist(),max_relative_residual=float(np.max(residual)),lower_iterate=lower.tolist(),lower_relative_residual=lower_residual.tolist(),greatest_lower_relative_gap=gap.tolist(),lower_iterations=steps,uniqueness=status))
    base=solve(external_assets);scenarios=[]
    if asset_scenarios is not None:
        rows=_rows(asset_scenarios,'asset_scenarios');_unique(rows,'scenario_id','asset_scenarios')
        for row in rows:
            _required(row,['external_assets']);result=solve(row['external_assets']);scenarios.append(dict(scenario_id=row['scenario_id'],result=result,payment_change=(np.array(result['clearing_payments'])-base['clearing_payments']).tolist()))
    return _result(dict(entity_ids=ids,currency=currency,model_assumptions=deepcopy(required),base=base,sensitivity=scenarios),'links.solve_contagion_network',currency,ctx,
        limitations=('Static single-currency pro-rata clearing with limited liability and absolute priority, no netting, bankruptcy costs or equity cross-holdings. Liabilities are debtor rows and creditor columns; outside creditors do not recycle payments.',
        'Greatest clearing is selected explicitly. A monotone lower iterate bounds every clearing; a small iterate change alone is not a uniqueness certificate. Defaults use the declared relative tolerance. Scenario comparisons hold liabilities fixed and do not estimate group PD.'))
