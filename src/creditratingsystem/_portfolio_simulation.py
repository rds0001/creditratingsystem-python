"""Default-group portfolio preparation and common-factor monetary loss paths."""
from copy import deepcopy
import math
import numpy as np
from scipy.special import ndtr,ndtri
from .core import _fail,_result,_numeric,_context_value,_fingerprint,CalcResult
from .data import _rows,_required,_date
from .links import assign_default_groups,_sealed,_read_sealed,_scalar,_string
from .dependence import validate_dependence_matrix,_psd_cholesky
from .hybrid import _hy_horizon_end
from .portfolio import _ids,_weights,_sum,credit_loss_distribution


def prepare_credit_portfolio(positions,entity_graph,default_link_policy,portfolio_contract,*,guarantee_allocations=None,netting_contracts=None,context=None):
    """Bind exposure lots to legal default groups and once-only mitigation effects."""
    rows=_rows(positions,'positions');contract=deepcopy(portfolio_contract);ctx=_context_value(context)
    if not isinstance(contract,dict):_fail('PORTFOLIO_CONTRACT','Versioned portfolio contract required')
    _required(contract,['portfolio_id','version','currency','as_of','known_at','horizon','time_unit','probability_measure','conditioning','event'])
    if contract['probability_measure']!='PHYSICAL'or contract['conditioning']!='initially_performing'or contract['event']!='first_default'or contract['time_unit']not in ('days','months','years')or _scalar(contract['horizon'],'horizon',0)<=0:_fail('PORTFOLIO_CONTRACT','Physical initially-performing first-default target required')
    for k in ('portfolio_id','version','currency'):_string(contract[k],k)
    for k in ('as_of','known_at'):_date(contract[k],k)
    for k in ('currency','horizon','time_unit','probability_measure','conditioning'):
        if ctx.get(k)is not None and ctx[k]!=contract[k]:_fail('TARGET_MISMATCH','Portfolio target differs from context '+k)
    graph=_read_sealed(entity_graph,'graph_hash')
    if graph['as_of']!=contract['as_of']or graph['known_at']!=contract['known_at']:_fail('PORTFOLIO_DATE','Relationship graph must use same valuation and information cutoffs')
    netting={}
    for c in _rows([]if netting_contracts is None else netting_contracts,'netting_contracts'):
        _required(c,['netting_contract_id','facility_ids','currency','legally_effective','evidence_reference','valid_from','valid_until','known_at'])
        cid=c['netting_contract_id']
        if cid in netting or c['currency']!=contract['currency']or c['legally_effective']is not True or not _date(c['valid_from'],'valid_from')<=_date(contract['as_of'],'as_of')<_date(c['valid_until'],'valid_until')or _date(c['known_at'],'known_at')>_date(contract['known_at'],'known_at'):_fail('PORTFOLIO_NETTING','Distinct effective dated netting contracts in reporting currency required')
        _ids(c['facility_ids'],'facility_ids');_string(c['evidence_reference'],'evidence_reference');netting[cid]=c
    position_ids=set();claims={};facilities={};lots=[];used_netting=set()
    for row in sorted(rows,key=lambda r:r.get('position_id','')):
        _required(row,['position_id','claim_id','facility_id','entity_id','currency','ead','facility_ead','sector','region','exposure_basis','collateral_treatment'])
        pid=row['position_id'];fid=row['facility_id'];ead=_scalar(row['ead'],'ead',0);total=_scalar(row['facility_ead'],'facility_ead',0)
        if pid in position_ids or row['claim_id']in claims:_fail('PORTFOLIO_DUPLICATE','Position and legal coverage claim IDs must each be unique')
        position_ids.add(pid);claims[row['claim_id']]=row
        for field in ('sector','region'):_string(row[field],field)
        if row['currency']!=contract['currency']or row['collateral_treatment']not in ('NONE','ALREADY_IN_CONDITIONAL_LGD'):_fail('PORTFOLIO_EFFECT','Explicit reporting currency and nonduplicated collateral treatment required')
        if row['exposure_basis']=='NETTED':
            cid=row.get('netting_contract_id')
            if cid not in netting or fid not in netting[cid]['facility_ids']:_fail('PORTFOLIO_NETTING','Netted exposure requires an effective contract covering the facility')
            used_netting.add(cid)
        elif row['exposure_basis']!='GROSS'or row.get('netting_contract_id')is not None:_fail('PORTFOLIO_NETTING','Declare gross or explicitly contracted already-netted EAD; no automatic offset')
        if fid in facilities:
            f=facilities[fid]
            if f['entity_id']!=row['entity_id']or f['total']!=total or f['basis']!=row['exposure_basis']or f['netting_id']!=row.get('netting_contract_id'):_fail('PORTFOLIO_FACILITY','Facility owner, total and exposure basis must agree across lots')
            f['amounts'].append(ead)
        else:facilities[fid]=dict(entity_id=row['entity_id'],total=total,basis=row['exposure_basis'],netting_id=row.get('netting_contract_id'),amounts=[ead])
        lots.append(deepcopy(row))
    if not lots:_fail('PORTFOLIO_EMPTY','Nonempty exposure portfolio required')
    for f in facilities.values():
        if not math.isclose(_sum(f['amounts']),f['total'],rel_tol=1e-12,abs_tol=0):_fail('PORTFOLIO_ALLOCATION','Lots must conserve the declared facility exposure')
    if used_netting!=set(netting):_fail('PORTFOLIO_NETTING','Unused netting contracts rejected')
    assignment=assign_default_groups(entity_graph,default_link_policy,[dict(facility_id=fid,entity_id=f['entity_id'])for fid,f in sorted(facilities.items())],context=ctx).value
    mapping={r['entity_id']:r['group_id']for r in assignment['assignments']}
    if any(mapping.get(row['entity_id'])is None for row in lots):_fail('PORTFOLIO_DEFAULT_GROUP','Unresolved default equivalence cannot be used for simulation')
    for row in lots:row['default_group_id']=mapping[row['entity_id']]
    protection=[];allocation_hashes=[];used_guarantees=set();reserved={pid:0. for pid in position_ids};group_ids={row['default_group_id']for row in lots}
    allocations=[]if guarantee_allocations is None else guarantee_allocations
    if not isinstance(allocations,list):_fail('PORTFOLIO_GUARANTEE','List of sealed simultaneous coverage allocations required')
    if allocations and contract.get('guarantee_loss_rule')not in ('FIRST_LOSS_CAP_IF_GUARANTOR_SURVIVES_HORIZON','PRO_RATA_EAD_CAPPED_IF_GUARANTOR_SURVIVES_HORIZON'):_fail('PORTFOLIO_GUARANTEE','Explicit terminal-horizon guarantee payoff rule required')
    terminal_date=_hy_horizon_end(contract['as_of'],contract).isoformat()
    for raw in allocations:
        allocation=_read_sealed(raw,'allocation_hash')
        if allocation['currency']!=contract['currency']or allocation['as_of']!=contract['as_of']or allocation['known_at']!=contract['known_at']:_fail('PORTFOLIO_GUARANTEE','Guarantee allocations must match portfolio currency and cutoffs')
        gids={r['guarantee_id']for r in allocation['allocations']}
        if gids&used_guarantees:_fail('PORTFOLIO_GUARANTEE','Guarantee capacity cannot be reused through multiple allocation artifacts')
        used_guarantees|=gids;allocation_hashes.append(allocation['allocation_hash'])
        for coverage in allocation['allocations']:
            cid=coverage['claim_id']
            if coverage['due_at']!=terminal_date:_fail('PORTFOLIO_GUARANTEE','Reserved guarantee must cover the declared terminal-horizon claim date')
            if cid not in claims:_fail('PORTFOLIO_GUARANTEE','Every reserved coverage claim must map to one portfolio lot')
            row=claims[cid];guarantor_group=mapping.get(coverage['guarantor_id'])
            if guarantor_group is None or coverage['entity_id']!=row['entity_id']:_fail('PORTFOLIO_GUARANTEE','Known guarantor group and matching claim owner required')
            reserved[row['position_id']]+=coverage['covered_amount'];group_ids.add(guarantor_group)
            protection.append(dict(position_id=row['position_id'],guarantee_id=coverage['guarantee_id'],guarantor_group_id=guarantor_group,covered_amount=coverage['covered_amount']))
    if any(reserved[row['position_id']]>row['ead']+1e-12*max(1,row['ead'])for row in lots):_fail('PORTFOLIO_GUARANTEE','Reserved guarantee coverage exceeds current lot exposure')
    value=_sealed(dict(contract=contract,positions=lots,default_group_ids=sorted(group_ids),default_group_assignment=assignment,guarantees=protection,guarantee_allocation_hashes=allocation_hashes,netting_contracts=[netting[k]for k in sorted(netting)],total_ead=_sum(r['ead']for r in lots)),'portfolio_hash')
    return _result(value,'portfolio.prepare_credit_portfolio','credit_portfolio',ctx,diagnostics=dict(position_count=len(lots),facility_count=len(facilities),default_group_count=len(group_ids)),limitations=('Default equivalence is the declared graph policy, not PD averaging. Already-netted EAD requires legal evidence and is never netted again. Collateral belongs to conditional LGD, not an additional recovery here. Guarantee allocations are fixed non-reinstating reservations; each artifact capacity and claim are consumed once. Legal evidence and complete facility totals are supplied, not authenticated.',))


def _portfolio_defaults(group_ids,pds,dependence_contract,shared_innovations,n):
    _required(dependence_contract,['kind','group_ids'])
    if dependence_contract['group_ids']!=group_ids:_fail('PORTFOLIO_ORDER','Dependence group order must match frozen portfolio')
    if not isinstance(pds,dict)or set(pds)!=set(group_ids):_fail('PORTFOLIO_PD','Exactly one PD per shared default group required')
    pd=np.array([_scalar(pds[g],'pd',0)for g in group_ids])
    if np.any(pd>1):_fail('PORTFOLIO_PD','PD must not exceed one')
    if dependence_contract['kind']=='SUPPLIED_DEPENDENT_UNIFORMS':
        u=_numeric(shared_innovations.get('group_uniforms'),'group_uniforms')
        if u.shape!=(n,len(group_ids))or np.any((u<0)|(u>1)):_fail('PORTFOLIO_INNOVATIONS','Aligned group-uniform paths in [0,1] required')
        defaults=(u<pd)|(pd==1);latent=None;systematic_variance=None
    elif dependence_contract['kind']=='GAUSSIAN_FACTORS':
        _required(dependence_contract,['factor_ids','factor_correlation','loadings']);factors=_ids(dependence_contract['factor_ids'],'factor_ids');corr=np.asarray(validate_dependence_matrix(dependence_contract['factor_correlation'],factors,matrix_kind='correlation',repair_policy='none').value['matrix']);loadings=_numeric(dependence_contract['loadings'],'loadings')
        z=_numeric(shared_innovations.get('factor_normals'),'factor_normals');e=_numeric(shared_innovations.get('group_normals'),'group_normals')
        if loadings.shape!=(len(group_ids),len(factors))or z.shape!=(n,len(factors))or e.shape!=(n,len(group_ids)):_fail('PORTFOLIO_INNOVATIONS','Aligned factor loadings and common/residual normal matrices required')
        variance=np.einsum('ij,jk,ik->i',loadings,corr,loadings)
        if np.any(variance< -1e-12)or np.any(variance>1+1e-12):_fail('PORTFOLIO_VARIANCE','Systematic latent variance must be in [0,1]')
        variance=np.minimum(1,np.maximum(variance,0));systematic=z@_psd_cholesky(corr).T;latent=systematic@loadings.T+e*np.sqrt(1-variance)
        if np.any(~np.isfinite(latent)):_fail('NUMERICAL_FAILURE','Latent factor paths exceed finite range')
        defaults=latent<ndtri(pd);u=ndtr(latent);systematic_variance=variance.tolist()
    else:_fail('PORTFOLIO_DEPENDENCE','Unsupported explicit dependence contract')
    return defaults,u,latent,systematic_variance


def simulate_credit_portfolio(prepared_portfolio,group_pds,dependence_contract,shared_innovations,conditional_loss_paths,*,levels=(.95,.99),context=None):
    """Generate shared-group defaults and jointly conditioned facility loss paths."""
    portfolio=_read_sealed(prepared_portfolio,'portfolio_hash');contract=portfolio['contract'];ctx=_context_value(context)
    for k in ('currency','horizon','time_unit','probability_measure','conditioning'):
        if ctx.get(k)is not None and ctx[k]!=contract[k]:_fail('TARGET_MISMATCH','Portfolio target differs from context')
    if not isinstance(shared_innovations,dict)or not isinstance(conditional_loss_paths,dict):_fail('PORTFOLIO_PATHS','Explicit shared innovations and conditional loss paths required')
    _required(shared_innovations,['scenario_ids','weights','group_ids','sampling_design']);ids=_ids(shared_innovations['scenario_ids'],'scenario_ids');groups=portfolio['default_group_ids'];position_ids=[p['position_id']for p in portfolio['positions']]
    if shared_innovations['group_ids']!=groups:_fail('PORTFOLIO_ORDER','Innovation default-group order must match portfolio')
    if shared_innovations['sampling_design']not in ('IID','FIXED_WEIGHTED_SCENARIOS','DEPENDENT_OR_QMC','EXACT_STATES'):_fail('PORTFOLIO_SAMPLING','Explicit supported sampling design required')
    w=_weights(shared_innovations['weights'],len(ids));defaults,u,latent,variance=_portfolio_defaults(groups,group_pds,dependence_contract,shared_innovations,len(ids))
    driver_hash=_fingerprint(dict(dependence_contract=dependence_contract,shared_innovations=shared_innovations))
    _required(conditional_loss_paths,['scenario_ids','position_ids','driver_hash','ead','lgd','guarantee_treatment'])
    if conditional_loss_paths['scenario_ids']!=ids or conditional_loss_paths['position_ids']!=position_ids or conditional_loss_paths['driver_hash']!=driver_hash or conditional_loss_paths['guarantee_treatment']!='NOT_YET_APPLIED':_fail('PORTFOLIO_PATHS','Loss paths must bind the same ordered scenarios/positions/drivers before guarantee treatment')
    ead=_numeric(conditional_loss_paths['ead'],'ead');lgd=_numeric(conditional_loss_paths['lgd'],'lgd')
    if ead.shape!=(len(ids),len(position_ids))or lgd.shape!=ead.shape or np.any(ead<0):_fail('PORTFOLIO_PATHS','Nonnegative EAD and raw LGD require aligned scenario-by-position arrays')
    group_columns=[groups.index(p['default_group_id'])for p in portfolio['positions']];position_defaults=defaults[:,group_columns]
    with np.errstate(over='ignore',invalid='ignore'):gross=np.where(position_defaults,ead*lgd,0.)
    if np.any(~np.isfinite(gross))or np.any(position_defaults&(ead!=0)&(lgd!=0)&(gross==0)):_fail('NUMERICAL_FAILURE','Conditional default losses exceed finite range')
    net=gross.copy();payments=[]
    for coverage in portfolio['guarantees']:
        j=position_ids.index(coverage['position_id']);g=groups.index(coverage['guarantor_group_id']);cap=coverage['covered_amount']
        entitlement=np.full(len(ids),cap)
        if contract['guarantee_loss_rule']=='PRO_RATA_EAD_CAPPED_IF_GUARANTOR_SURVIVES_HORIZON':
            fraction=np.divide(cap,ead[:,j],out=np.zeros(len(ids)),where=ead[:,j]>0);entitlement=np.minimum(np.maximum(gross[:,j],0)*np.minimum(fraction,1),cap)
        amount=np.minimum(np.maximum(net[:,j],0),entitlement)*position_defaults[:,j]*(~defaults[:,g]);net[:,j]-=amount
        payments.append(dict(position_id=coverage['position_id'],guarantee_id=coverage['guarantee_id'],payments=amount.tolist()))
    total=np.array([_sum(row)for row in net]);distribution=credit_loss_distribution(total,w,scenario_ids=ids,levels=levels,currency=contract['currency'],context=ctx).value
    value=dict(portfolio_hash=portfolio['portfolio_hash'],target=contract,scenario_ids=ids,position_ids=position_ids,default_group_ids=groups,group_defaults=defaults.tolist(),position_defaults=position_defaults.tolist(),group_uniforms=u.tolist(),group_latent_values=None if latent is None else latent.tolist(),systematic_asset_variance=variance,ead_paths=ead.tolist(),lgd_paths=lgd.tolist(),gross_position_losses=gross.tolist(),net_position_losses=net.tolist(),guarantee_payments=payments,portfolio_loss_paths=total.tolist(),distribution=distribution,sampling_design=shared_innovations['sampling_design'],driver_hash=driver_hash)
    return _result(value,'portfolio.simulate_credit_portfolio','portfolio_loss_paths',ctx,diagnostics=dict(shared_innovations_hash=_fingerprint(shared_innovations),loss_paths_hash=_fingerprint(conditional_loss_paths)),limitations=('Gaussian inputs require independent standard-normal common/residual draws under the declared factor law; covariance describes latent assets, not Bernoulli events. Supplied dependent uniforms may come from granular copula routines. Conditional EAD/LGD driver identity is checked, not their empirical realism. All facilities in a group use one default event. Guarantees pay only positive remaining loss when borrower defaults and guarantor survives, within non-reinstating reserved caps. No hidden default-PD uplift, collateral recovery or extra netting is added.',))


def stress_portfolio_factor_structure(prepared_portfolio,group_pds,dependence_contract,shared_innovations,conditional_loss_paths,stress_cases,*,levels=(.95,.99),context=None):
    """Replay explicit dependence/recovery stresses on identical random coordinates."""
    from .portfolio import incremental_credit_risk
    base=simulate_credit_portfolio(prepared_portfolio,group_pds,dependence_contract,shared_innovations,conditional_loss_paths,levels=levels,context=context).value;cases=_rows(stress_cases,'stress_cases');seen=set();out=[]
    for case in sorted(cases,key=lambda c:c.get('case_id','')):
        _required(case,['case_id','dependence_contract','group_pds','conditional_loss_paths','recovery_response'])
        if case['case_id']in seen:_fail('PORTFOLIO_STRESS','Distinct stress case IDs required')
        seen.add(case['case_id'])
        if case['recovery_response']not in ('FIXED_PATHS','REPLACED_CONDITIONAL_PATHS'):_fail('PORTFOLIO_STRESS','Declare whether recovery/exposure paths respond to stress')
        if case['recovery_response']=='FIXED_PATHS':
            for k in ('scenario_ids','position_ids','ead','lgd','guarantee_treatment'):
                if case['conditional_loss_paths'].get(k)!=conditional_loss_paths[k]:_fail('PORTFOLIO_STRESS','Fixed recovery declaration requires unchanged EAD/LGD paths and identities')
        stressed=simulate_credit_portfolio(prepared_portfolio,case['group_pds'],case['dependence_contract'],shared_innovations,case['conditional_loss_paths'],levels=levels,context=context).value
        changes=np.asarray(stressed['portfolio_loss_paths'])-np.asarray(base['portfolio_loss_paths'])
        comparison=incremental_credit_risk(base['portfolio_loss_paths'],changes,shared_innovations['weights'],scenario_ids=base['scenario_ids'],levels=levels,currency=base['target']['currency'],change_id=case['case_id'],context=context).value
        out.append(dict(case_id=case['case_id'],recovery_response=case['recovery_response'],stressed=stressed,paired_comparison=comparison))
    return _result(dict(base=base,stress_cases=out,shared_innovations_hash=_fingerprint(shared_innovations)),'portfolio.stress_portfolio_factor_structure','portfolio_stress',context,diagnostics=dict(stress_contracts_hash=_fingerprint(cases)),limitations=('Every case reuses the same supplied systematic/residual random coordinates or group uniforms. Covariance/loadings are validated without substantive hidden repair. Fixed versus replaced conditional EAD/LGD is explicit. Comparisons are paired scenario differences; finite-sample marginal default frequencies may change even if contractual marginal PDs are unchanged.',))


def simulate_portfolio_migrations(initial_grades,transition_matrices,shared_uniforms,migration_contract,*,valuation_contract=None,context=None):
    """Physical group Markov migration paths and optional explicit marked values."""
    contract=deepcopy(migration_contract)
    if not isinstance(contract,dict)or not isinstance(initial_grades,dict)or not isinstance(shared_uniforms,dict):_fail('PORTFOLIO_MIGRATION','Explicit migration, grade and innovation contracts required')
    _required(contract,['contract_id','version','group_ids','states','default_state','allow_cure','probability_measure','time_grid','time_unit','matrix_mode','innovation_law'])
    groups=_ids(contract['group_ids'],'group_ids');states=_ids(contract['states'],'states');times=_numeric(contract['time_grid'],'time_grid',vector_only=True);g=len(groups);k=len(states);h=len(times)
    if contract['probability_measure']!='PHYSICAL'or contract['default_state']not in states or not isinstance(contract['allow_cure'],bool)or contract['time_unit']not in ('days','months','years')or np.any(times<=0)or np.any(np.diff(times)<=0)or contract['innovation_law']!='CONDITIONALLY_UNIFORM_GIVEN_PRIOR_STATES':_fail('PORTFOLIO_MIGRATION','Physical positive time grid, explicit default/cure and sequential conditional-uniform law required')
    if set(initial_grades)!=set(groups)or any(initial_grades[x]not in states for x in groups):_fail('PORTFOLIO_MIGRATION','Known initial grade for each group required')
    a=_numeric(transition_matrices,'transition_matrices')
    if contract['matrix_mode']=='HOMOGENEOUS'and a.shape==(k,k):a=np.repeat(a[None,:,:],h,axis=0)
    elif contract['matrix_mode']!='TIME_VARYING'or a.shape!=(h,k,k):_fail('PORTFOLIO_MIGRATION','Explicit homogeneous square or time-varying transition tensor required')
    if np.any(a<0)or np.any(a>1)or np.any(np.abs(a.sum(axis=2)-1)>1e-12):_fail('PORTFOLIO_MIGRATION','Transition rows must be probabilities summing to one')
    a=a/a.sum(axis=2,keepdims=True);d=states.index(contract['default_state'])
    if not contract['allow_cure']and (np.any(a[:,d,d]!=1)or np.any(a[:,d,np.arange(k)!=d]!=0)):_fail('PORTFOLIO_MIGRATION','Default must be absorbing unless explicit cure is allowed')
    _required(shared_uniforms,['scenario_ids','group_ids','values','weights']);ids=_ids(shared_uniforms['scenario_ids'],'scenario_ids');u=_numeric(shared_uniforms['values'],'values');w=_weights(shared_uniforms['weights'],len(ids))
    if shared_uniforms['group_ids']!=groups or u.shape!=(len(ids),h,g)or np.any((u<0)|(u>=1)):_fail('PORTFOLIO_MIGRATION','Aligned scenario/time/group uniforms in [0,1) required')
    ctx=_context_value(context)
    if ctx.get('probability_measure')not in (None,'PHYSICAL')or ctx.get('time_unit')not in (None,contract['time_unit'])or ctx.get('horizon')not in (None,float(times[-1])):_fail('TARGET_MISMATCH','Migration target conflicts with context')
    current=np.tile([states.index(initial_grades[x])for x in groups],(len(ids),1));history=np.empty((len(ids),h+1,g),dtype=int);history[:,0,:]=current;first=np.full((len(ids),g),np.nan);first[current==d]=0.
    for t in range(h):
        for s in range(len(ids)):
            for j in range(g):
                cdf=np.cumsum(a[t,current[s,j],:]);cdf[-1]=1;current[s,j]=int(np.searchsorted(cdf,u[s,t,j],side='right'))
                if current[s,j]==d and np.isnan(first[s,j]):first[s,j]=times[t]
        history[:,t+1,:]=current
    terminal=[];theoretical=[]
    product=np.eye(k)
    for matrix in a:product=product@matrix
    for j,group in enumerate(groups):
        terminal.append(dict(group_id=group,probabilities={state:_sum(w[current[:,j]==i])for i,state in enumerate(states)}))
        theoretical.append(dict(group_id=group,probabilities={state:float(product[states.index(initial_grades[group]),i])for i,state in enumerate(states)}))
    marked=None
    if valuation_contract is not None:
        v=deepcopy(valuation_contract)
        if not isinstance(v,dict):_fail('PORTFOLIO_VALUATION','Explicit fixed value-by-grade contract required')
        _required(v,['valuation_id','version','currency','valuation_basis','group_ids','states','initial_values','terminal_values_by_grade','cash_distributions'])
        if v['group_ids']!=groups or v['states']!=states or v['valuation_basis']!='SUPPLIED_FROZEN_MARKED_VALUES':_fail('PORTFOLIO_VALUATION','Aligned fixed marked-value table required; physical transitions are not a pricing measure')
        _string(v['currency'],'currency')
        if ctx.get('currency')not in (None,v['currency']):_fail('TARGET_MISMATCH','Valuation currency conflicts with context')
        initial=_numeric(v['initial_values'],'initial_values',vector_only=True);terminal_values=_numeric(v['terminal_values_by_grade'],'terminal_values_by_grade');cash=_numeric(v['cash_distributions'],'cash_distributions')
        if initial.shape!=(g,)or terminal_values.shape!=(g,k)or cash.shape!=(len(ids),g):_fail('PORTFOLIO_VALUATION','Group initial values, group/state terminal table and scenario/group cash distributions required')
        marks=np.array([[terminal_values[j,current[s,j]]for j in range(g)]for s in range(len(ids))]);loss=initial-marks-cash
        if np.any(~np.isfinite(loss)):_fail('NUMERICAL_FAILURE','Marked value changes exceed finite range')
        marked=dict(currency=v['currency'],terminal_values=marks.tolist(),cash_distributions=cash.tolist(),marked_loss_paths=loss.tolist(),portfolio_marked_loss_paths=[_sum(row)for row in loss],valuation_hash=_fingerprint(v))
    value=dict(contract=contract,scenario_ids=ids,group_ids=groups,states=states,time_grid=[0.]+times.tolist(),grade_paths=[[[states[i]for i in row]for row in path]for path in history],first_default_times=[[None if np.isnan(v)else float(v)for v in row]for row in first],terminal_empirical=terminal,terminal_markov_probabilities=theoretical,marked_values=marked)
    return _result(value,'portfolio.simulate_portfolio_migrations','rating_migration_paths',ctx,diagnostics=dict(transition_hash=_fingerprint(a.tolist()),innovations_hash=_fingerprint(shared_uniforms)),limitations=('Physical conditional transition probabilities require sequential uniforms conditional on past states; correlated simultaneous group uniforms preserve cross-group dependence. Reusing temporally dependent uniforms without that property does not establish the stated Markov law. Default is absorbing unless explicitly allowed to cure; first default time remains absorbing even after cure. Optional terminal marked values and cash distributions are supplied valuations, not prices inferred from physical probabilities.',))
