"""Shared-default, dated collateral realization and priority allocation."""
import math
from collections import defaultdict
from .core import _fail, _result, _context_value, _fingerprint
from .data import _rows, _required, _date
from .simulation import _number, _text


def _sum_finite(values):
    try:value=math.fsum(values)
    except OverflowError:_fail("NUMERICAL_FAILURE","Recovery totals exceed finite representation")
    if not math.isfinite(value):_fail("NUMERICAL_FAILURE","Recovery totals exceed finite representation")
    return value

def simulate_recovery_waterfall(default_events, claims, realization_paths, recovery_policy, *, context=None):
    """Allocate complete supplied collateral realizations against legal claims."""
    events=_rows(default_events,'default_events');claims=_rows(claims,'claims');paths=_rows(realization_paths,'realization_paths');policy=recovery_policy
    if not isinstance(policy,dict):_fail('RECOVERY_POLICY','Explicit recovery policy required')
    _required(policy,['policy_id','version','currency','allocation','cost_rule','completion'])
    for key in ('policy_id','version','currency','allocation','cost_rule','completion'):_text(policy[key],key)
    if policy['allocation']not in ('LEGAL_CAP','UNCAPPED_PRO_RATA')or policy['cost_rule']!='PROCEEDS_FIRST_EXPLICIT_DEFICIT_SHARES'or policy['completion']!='ULTIMATE_COMPLETE':_fail('RECOVERY_POLICY','Supported allocation, explicit cost treatment and complete ultimate realization required')
    ctx=_context_value(context)
    if ctx.get('currency')is not None and ctx['currency']!=policy['currency']:_fail('TARGET_MISMATCH','Recovery currency conflicts with context')
    emap={};groups=defaultdict(list);pmap=defaultdict(list);seen=set();realizations=set();collateral=set();subjects=set()
    for e in events:
        _required(e,['scenario_id','default_event_id','subject_id','default_date'])
        for key in ('scenario_id','default_event_id','subject_id'):_text(e[key],key)
        _date(e['default_date'],'default_date');key=(e['scenario_id'],e['default_event_id'])
        if key in emap or (e['scenario_id'],e['subject_id'])in subjects:_fail('RECOVERY_DUPLICATE','Unique first-default event per subject and scenario required')
        subjects.add((e['scenario_id'],e['subject_id']))
        emap[key]=e
    for c in claims:
        _required(c,['scenario_id','default_event_id','subject_id','claim_id','pool_id','currency','ead_at_default','legal_balance','priority','deficit_cost_share'])
        for k in ('scenario_id','default_event_id','subject_id','claim_id','pool_id','currency'):_text(c[k],k)
        key=(c['scenario_id'],c['default_event_id']);ck=(c['scenario_id'],c['claim_id'])
        if key not in emap or c['subject_id']!=emap[key]['subject_id']or c['currency']!=policy['currency']:_fail('RECOVERY_EVENT','Claim must match shared event, subject and currency')
        if ck in seen:_fail('RECOVERY_DUPLICATE','Claim may occur in only one collateral pool per scenario')
        seen.add(ck)
        for k in ('ead_at_default','legal_balance','deficit_cost_share'):_number(c[k],k,0)
        _number(c['priority'],'priority');groups[(*key,c['pool_id'])].append(c)
    for p in paths:
        _required(p,['scenario_id','default_event_id','pool_id','currency','realization_id','collateral_id','date','gross_value','realization_fraction','cost','discount_factor','discount_anchor_date'])
        for k in ('scenario_id','default_event_id','pool_id','currency','realization_id','collateral_id'):_text(p[k],k)
        key=(p['scenario_id'],p['default_event_id'],p['pool_id']);event=emap.get(key[:2])
        if key not in groups or event is None or p['currency']!=policy['currency']or p['discount_anchor_date']!=event['default_date']:_fail('RECOVERY_EVENT','Realization must match event/pool/currency and default-date discount anchor')
        date=_date(p['date'],'date')
        if date<_date(event['default_date'],'default_date'):_fail('RECOVERY_DATE','Recovery cannot precede shared default')
        rid=(p['scenario_id'],p['realization_id']);cid=(p['scenario_id'],p['collateral_id'])
        if rid in realizations or cid in collateral:_fail('RECOVERY_DUPLICATE','Each realization and collateral asset may be consumed once per scenario')
        realizations.add(rid);collateral.add(cid)
        for k in ('gross_value','realization_fraction','cost','discount_factor'):_number(p[k],k,0)
        if p['realization_fraction']>1 or p['discount_factor']<=0:_fail('RECOVERY_VALUE','Realization fraction in [0,1] and positive discount factor required')
        pmap[key].append(p)
    if set(groups)!=set(pmap):_fail('RECOVERY_COMPLETENESS','Every claim pool needs explicit complete realization evidence, including zero recovery')
    if {k[:2]for k in groups}!=set(emap):_fail('RECOVERY_COMPLETENESS','Every supplied default event must have claims and a complete recovery path')
    out=[];ledger=[];pools=[]
    for key,cs in sorted(groups.items()):
        share=_sum_finite(c['deficit_cost_share']for c in cs)
        if not math.isclose(share,1,rel_tol=0,abs_tol=1e-12):_fail('RECOVERY_COST','Explicit deficit cost shares must sum to one in each pool')
        if policy['allocation']=='UNCAPPED_PRO_RATA'and(len({c['priority']for c in cs})!=1 or _sum_finite(c['legal_balance']for c in cs)<=0):_fail('RECOVERY_POLICY','Uncapped participation requires one rank and positive legal allocation weights')
        balance={c['claim_id']:float(c['legal_balance'])for c in cs};pv={c['claim_id']:0. for c in cs};received={c['claim_id']:0. for c in cs};funded={c['claim_id']:0. for c in cs};surplus=0.;gross_sum=cost_sum=0.
        for p in sorted(pmap[key],key=lambda a:(a['date'],a['realization_id'])):
            gross=p['gross_value']*p['realization_fraction'];cost=p['cost'];available=max(gross-cost,0);deficit=max(cost-gross,0);allocated={c['claim_id']:0. for c in cs}
            if policy['allocation']=='UNCAPPED_PRO_RATA':
                total=_sum_finite(c['legal_balance']for c in cs)
                for c in cs:allocated[c['claim_id']]=available*(c['legal_balance']/total)
                available=0.
            else:
                for rank in sorted({c['priority']for c in cs}):
                    members=[c for c in cs if c['priority']==rank];total=_sum_finite(balance[c['claim_id']]for c in members)
                    amount=min(available,total)
                    if total>0:
                        for c in members:
                            cid=c['claim_id'];allocation=amount*(balance[cid]/total);allocated[cid]=allocation;balance[cid]=max(balance[cid]-allocation,0)
                    available=max(available-amount,0)
            for c in cs:
                cid=c['claim_id'];contribution=deficit*(c['deficit_cost_share']/share);net=allocated[cid]-contribution;present=net*p['discount_factor']
                received[cid]+=allocated[cid];funded[cid]+=contribution;pv[cid]+=present
                ledger.append(dict(scenario_id=key[0],default_event_id=key[1],pool_id=key[2],claim_id=cid,realization_id=p['realization_id'],date=p['date'],gross_allocation=allocated[cid],deficit_cost_contribution=contribution,net_recovery=net,discount_factor=p['discount_factor'],discounted_net_recovery=present))
            surplus+=available;gross_sum+=gross;cost_sum+=cost
        for c in cs:
            cid=c['claim_id'];ead=c['ead_at_default'];loss=ead-pv[cid]
            if not all(math.isfinite(v)for v in (pv[cid],loss,received[cid],funded[cid])):_fail('NUMERICAL_FAILURE','Recovery exceeds finite representation')
            lgd=None if ead==0 else loss/ead
            if lgd is not None and not math.isfinite(lgd):_fail('NUMERICAL_FAILURE','LGD exceeds finite representation')
            out.append(dict(scenario_id=key[0],default_event_id=key[1],subject_id=c['subject_id'],claim_id=cid,pool_id=key[2],currency=policy['currency'],default_date=emap[key[:2]]['default_date'],ead_at_default=ead,gross_recovery=received[cid],deficit_cost_contribution=funded[cid],discounted_net_recovery=pv[cid],economic_loss=loss,lgd=lgd,ultimate_complete=True))
        if not all(math.isfinite(v)for v in (gross_sum,cost_sum,surplus)):_fail('NUMERICAL_FAILURE','Pool totals exceed finite representation')
        pools.append(dict(scenario_id=key[0],default_event_id=key[1],pool_id=key[2],realized_proceeds=gross_sum,costs=cost_sum,creditor_gross_recovery=_sum_finite(received.values()),creditor_deficit_funding=_sum_finite(funded.values()),unallocated_surplus=surplus))
    return _result(dict(claim_results=out,allocation_ledger=ledger,pool_results=pools,policy=policy),'simulation.simulate_recovery_waterfall','recovery_waterfall',ctx,diagnostics=dict(input_fingerprint=_fingerprint(dict(default_events=events,claims=claims,realization_paths=paths))),limitations=('Ultimate completeness and legal priority/entitlement are supplied evidence. Each collateral identity is realized once per scenario; split asset sales require distinct nonoverlapping asset identities. Legal caps apply to undiscounted claims; EAD is a separate loss denominator. Costs are paid once from proceeds with only deficits charged to explicit creditor shares. Raw LGD is uncapped and undefined at zero EAD. No independent recovery or default resampling is performed.',))
