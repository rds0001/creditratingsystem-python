"""Historical decision audit and paired frozen-policy scenario validation."""
from copy import deepcopy
from collections import Counter
import math
from .core import _fail,_result,_fingerprint
from .data import _date
from .decisioning import _checked
from ._decisioning_evaluation import evaluate_decision_grid
from ._validation_common import _vb_design,_vb_records,_vb_draws,_vb_bands,_vb_number
from ._validation_system import _vs_index
from ._validation_affordability import _va_differences


def _vp_finite(value):
    if isinstance(value,dict):
        for v in value.values():_vp_finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:_vp_finite(v)
    elif isinstance(value,(int,float))and not math.isfinite(value):_fail('VALIDATION_NUMERICAL','Policy metric outside finite numeric range')

def decision_policy_backtest(evaluated_grid,grid_inputs,historical_cases,*,design,reference_candidate_id,uncertainty='none',cluster_resamples=None,confidence_level=.95,tolerance=1e-10,context=None):
    """Audit sealed policy results and actual stages without executing lending actions."""
    d=_vb_design(design,context);tol=_vb_number(tolerance,'tolerance',0);grid=_checked(evaluated_grid,'DecisionGrid');inputs=deepcopy(grid_inputs)
    if not isinstance(inputs,dict)or 'context'in inputs:_fail('VALIDATION_POLICY','Explicit original grid inputs without context required')
    if inputs.get('evaluation_cutoff')!=d['evaluation_cutoff']:_fail('VALIDATION_POLICY','Grid and design cutoffs differ')
    profile=inputs.get('evaluation_profile',{})
    version=d.get('policy_target_contract_version','1.0');bound_target=deepcopy(d['target'])
    if version not in ('1.0','1.1') or profile.get('target_contract_version','1.0')!=version:
        _fail('VALIDATION_POLICY','Policy target contract versions must explicitly match')
    if version=='1.1':
        if d.get('performance_origin')!='first_draw_at':_fail('VALIDATION_POLICY','Version1.1 requires actual first-draw performance origin')
        bound_target['performance_origin']=d['performance_origin']
    if profile.get('target')!=bound_target or profile.get('currency')!=d.get('currency')or profile.get('split_id')!=d['split_id']or profile.get('data_fingerprint')!=d['data_fingerprint']:_fail('VALIDATION_POLICY','Exact currency, target, split and data binding required')
    replay=evaluate_decision_grid(**inputs).value;violations=[]
    mismatch=_va_differences(grid,replay,tol)
    if mismatch:violations.append(dict(code='FROZEN_GRID_RECONCILIATION',case_id=None,paths=mismatch))
    rows=_vb_records(historical_cases,d);_,hm=_vs_index(rows,'case_id','historical cases');_,cm=_vs_index(inputs['cases'],'case_id','grid cases');_,pm=_vs_index(replay['candidates'],'candidate_id','grid candidates')
    if set(hm)!=set(cm)or reference_candidate_id not in pm:_fail('VALIDATION_POLICY','Exact paired historical population and existing reference required')
    if not rows:_fail('VALIDATION_POLICY','Nonempty historical cases required')
    entity_clusters={}
    for r in rows:
        if r['entity_id']in entity_clusters and entity_clusters[r['entity_id']]!=r['cluster_id']:_fail('VALIDATION_DEPENDENCE','Repeated entity cannot cross sampling clusters')
        entity_clusters[r['entity_id']]=r['cluster_id']
    draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);outcomes={r['case_id']:r for r in inputs['observed_outcomes']};ci={(r['candidate_id'],r['case_id']):r for r in inputs['candidate_inputs']};traces={(p['candidate_id'],t['case_id']):t for p in replay['candidates']for t in p['decisions']};audits=[]
    def violation(code,key,paths=None):violations.append(dict(code=code,case_id=key,paths=[]if paths is None else paths))
    for row in rows:
        key=row['case_id'];cid=row.get('historical_candidate_id');case=cm[key]
        if cid not in pm or row['as_of']!=case['decision_date']or _vb_number(row.get('weight',1),'weight',0)!=case['weight']:_fail('VALIDATION_POLICY','Historical candidate, index date and weights must bind grid case')
        t=traces[cid,key];recorded=_checked(row.get('historical_decision'),'DecisionArtifact');changed=_va_differences(recorded,t['decision'],tol)
        if changed:violation('HISTORICAL_DECISION_RECONCILIATION',key,changed)
        if 'POLICY_NOT_EFFECTIVE'in t['decision']['findings']:violation('HISTORICAL_POLICY_NOT_EFFECTIVE',key)
        for field in ('offered','taken_up'):
            if type(row.get(field))is not bool:_fail('VALIDATION_POLICY','Explicit offer and take-up stages required')
        o=outcomes.get(key)
        if o is None:_fail('VALIDATION_POLICY','Explicit actual funding or no-funding record required for every case')
        if row['taken_up']and not row['offered']or o['funded']and not row['taken_up']:_fail('VALIDATION_POLICY','Take-up requires offer; funding requires take-up')
        stage_known=_date(row.get('stages_known_at'),'stages_known_at')
        if stage_known<_date(row['as_of'],'as_of'):_fail('VALIDATION_POLICY','Actual stage knowledge cannot predate decision')
        visible=stage_known<=_date(d['evaluation_cutoff'],'cutoff')and(not o['funded']or o['funding_known_at']<=d['evaluation_cutoff'])
        terms=ci[cid,key].get('offer');limit=None;booked=None
        if row['offered']:
            if visible and t['action']!='APPROVE':violation('OFFER_WITHOUT_POLICY_APPROVAL',key)
            limit=_vb_number(row.get('offered_limit'),'offered_limit',0)
            if not isinstance(terms,dict)or not isinstance(terms.get('contract_terms'),dict):_fail('VALIDATION_POLICY','Explicit frozen offer contract terms required for offered cases')
            cap=_vb_number(terms['contract_terms'].get('limit'),'contract limit',0)
            if visible and limit>cap+tol:violation('OFFER_LIMIT_EXCEEDED',key)
        elif row.get('offered_limit')is not None:_fail('VALIDATION_POLICY','Unoffered limit must be null')
        if o['funded']:
            booked=_vb_number(row.get('booked_amount'),'booked_amount',0)
            if visible and booked>limit+tol:violation('BOOKED_LIMIT_EXCEEDED',key)
            if visible and o['booked_offer_hash']!=t['offer_hash']:violation('BOOKED_TERMS_DIFFER_FROM_HISTORICAL_OFFER',key)
        elif row.get('booked_amount')is not None:_fail('VALIDATION_POLICY','Unfunded booked amount must be null')
        cost=None
        if row.get('application_cost')is not None:
            cost=_vb_number(row['application_cost'],'application_cost',0)
            if row.get('cost_complete')is not True or _date(row.get('cost_known_at'),'cost_known_at')>_date(d['evaluation_cutoff'],'cutoff'):cost=None
        matched=next(s['status']for s in pm[cid]['observed_outcomes']['case_status']if s['case_id']==key)
        audits.append(dict(case_id=key,cluster_id=row['cluster_id'],weight=case['weight'],historical_candidate_id=cid,action=t['action'],offered=row['offered'],taken_up=row['taken_up'],funded=o['funded'],stage_evidence_visible=visible,offered_limit=limit,booked_amount=booked,known_complete_application_cost=cost,mature_matched_funded=matched=='MATURE_MATCHED_FUNDED',observed_default=int(o['defaulted'])if matched=='MATURE_MATCHED_FUNDED'else None,observed_loss=o['economic_loss']if matched=='MATURE_MATCHED_FUNDED'else None,policy_findings=t['decision']['findings']))
    def measure(mult=None):
        weights={r['case_id']:r.get('weight',1)*(1 if mult is None else mult[i])for i,r in enumerate(rows)};total=sum(weights.values());paired={};scenarios={}
        for cid,p in pm.items():
            counts={a:0. for a in ('APPROVE','REFER','DECLINE','BLOCKED')};el=ep=ead=0.;identified=True
            for key,w in weights.items():
                t=traces[cid,key];r=ci[cid,key];a=t['action'];counts[a]+=w
                if a=='APPROVE':
                    offer=r['offer']
                    if offer['support_status']!='SUPPORTED'and w>0:identified=False
                    elif offer['support_status']=='SUPPORTED':
                        el+=w*offer['takeup_probability']*offer['conditional_booked_expected_loss'];ep+=w*(offer['takeup_probability']*offer['conditional_booked_npv']-offer['offer_cost']);ead+=w*offer['takeup_probability']*offer['ead']
                else:ep-=w*r['nonapproval_costs'][a]
            scenarios[cid]={('approval'if a=='APPROVE'else a.lower())+'_rate':v/total if total else None for a,v in counts.items()}
            scenarios[cid].update(expected_loss_per_application=el/total if total and identified else None,expected_npv_per_application=ep/total if total and identified else None,expected_funded_ead_per_application=ead/total if total and identified else None)
        ref=scenarios[reference_candidate_id]
        for cid,m in scenarios.items():
            for k,v in m.items():paired[cid+'|'+k]=None if v is None or ref[k]is None else v-ref[k]
        factual={};visible=[a for a in audits if a['stage_evidence_visible']];den=sum(weights[a['case_id']]for a in visible)
        for field in ('offered','taken_up','funded'):factual[field+'_rate']=sum(weights[a['case_id']]*a[field]for a in visible)/den if den else None
        mature=[a for a in audits if a['mature_matched_funded']];mass=sum(weights[a['case_id']]for a in mature)
        factual.update(visible_stage_weight=den,mature_matched_funded_weight=mass,observed_pd=sum(weights[a['case_id']]*a['observed_default']for a in mature)/mass if mass else None,observed_mean_loss=sum(weights[a['case_id']]*a['observed_loss']for a in mature)/mass if mass else None)
        costs=[a for a in audits if a['known_complete_application_cost']is not None];cost_mass=sum(weights[a['case_id']]for a in costs)
        factual.update(known_complete_cost_weight=cost_mass,mean_application_cost=sum(weights[a['case_id']]*a['known_complete_application_cost']for a in costs)/cost_mass if cost_mass else None)
        _vp_finite([scenarios,paired,factual])
        return scenarios,paired,factual
    scenario,paired,factual=measure();bands=_vb_bands(None if draws is None else[measure(x)[1]for x in draws],confidence_level)
    for cid,p in pm.items():
        for check in p['constraints']:
            if check['status']=='VIOLATED':violation('CANDIDATE_SCENARIO_CONSTRAINT_VIOLATED',None,[cid,check['constraint_id']])
    value=dict(artifact_type='ValidationResult',design=d,decision=dict(status='BLOCKED'if violations else'NOT_ASSESSED',rule='RECONCILIATION_AND_DECLARED_SCENARIO_CONSTRAINTS'),metrics=dict(factual=factual,candidate_scenario=scenario,paired_candidate_minus_reference=paired),uncertainty=bands,p_value=None,applicability='APPLICABLE',violations=violations,violation_counts=dict(sorted(Counter(v['code']for v in violations).items())),case_audits=audits,dependence=dict(repeated_entity_rows=len(rows)-len(entity_clusters),cluster_count=len(set(entity_clusters.values())),sampling_unit='DECLARED_WHOLE_CLUSTER'),candidate_observed_scopes={cid:p['observed_outcomes']for cid,p in pm.items()},candidate_constraints={cid:p['constraints']for cid,p in pm.items()},counterfactual=dict(applicability='NOT_IDENTIFIED',full_policy_causal_value=None,reason='Frozen offer assumptions and matching factual subsets do not identify unobserved actions, changed terms, take-up or policy selection effects'),input_fingerprint=_fingerprint(dict(grid_inputs=grid_inputs,historical_cases=historical_cases)),recomputed_grid_hash=replay['sha256'],fitted_any_model=False,executed_lending_action=False)
    return _result(value,'validation.decision_policy_backtest','validation_metrics',context,limitations=('Only local frozen policy evaluation is replayed. Observed risks belong to mature funded cases at identical booked offer hashes; approved but unfunded cases have no credit labels. Price/limit changes remain assumption-conditioned scenarios, not causal policy evidence. Intervals condition on frozen policies, offer assumptions and supplied independent sampling clusters.',))
