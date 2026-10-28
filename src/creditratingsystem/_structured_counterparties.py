"""Counterparty monetary claims and signed replacement cash without double use."""
from copy import deepcopy
from datetime import timedelta
from .core import _fail,_result,_fingerprint
from .data import _rows,_required,_date
from ._structured_pool import _number,_text,_sum,_bound,_nullable
from ._structured_trade import _trade_scenarios,_party_events


def stress_pool_counterparties(baseline_cashflows,exposure_contracts,netting_sets,exposure_paths,collateral_paths,settlement_terms,scenario_contract,party_mapping,party_default_paths,shared_innovations,*,as_of,context=None):
    """Apply shared counterparty default to distinct existing cash/replaceable claims."""
    _date(as_of,'as_of');contract,ctx=_trade_scenarios(scenario_contract,as_of,context);sids=contract['scenario_ids'];dates=contract['dates'];horizon=dates[-1]
    baseline={};ids=set()
    for row in _rows(baseline_cashflows,'baseline_cashflows'):
        _required(row,['scenario_id','cashflow_id','date','amount']);key=(row['scenario_id'],_text(row['cashflow_id'],'cashflow_id'))
        if key in baseline or row['scenario_id']not in sids or not _date(as_of,'as_of')<=_date(row['date'],'date')<=_date(horizon,'horizon'):_fail('STRUCTURED_CASHFLOW','Unique aligned baseline cashflows within horizon required')
        _number(row['amount'],'amount',0);baseline[key]=row;ids.add(row['cashflow_id'])
    if set(baseline)!={(s,i)for s in sids for i in ids}:_fail('STRUCTURED_CASHFLOW','Each baseline monetary ID needs every scenario')
    exposures={};linked=set();parties=set();sets={}
    for row in _rows(exposure_contracts,'exposure_contracts'):
        _required(row,['exposure_id','counterparty_id','netting_set_id','role','basis','cashflow_ids']);eid=_text(row['exposure_id'],'exposure_id')
        for k in ('counterparty_id','netting_set_id'):_text(row[k],k)
        supported={'SERVICER':'CASH_HELD','ACCOUNT_BANK':'CASH_HELD','GUARANTOR':'GUARANTEE_RECEIVABLE','SWAP':'REPLACEMENT'}
        if eid in exposures or row['role']not in supported or row['basis']!=supported[row['role']]:_fail('STRUCTURED_COUNTERPARTY','Unique exposures and explicit supported role/basis required')
        cashids=row['cashflow_ids']
        if not isinstance(cashids,list)or len(set(cashids))!=len(cashids)or any(x not in ids or x in linked for x in cashids)or (row['basis']=='REPLACEMENT'and cashids)or(row['basis']!='REPLACEMENT'and not cashids):_fail('STRUCTURED_DOUBLE_COUNT','Distinct existing monetary IDs required; replacement marks cannot duplicate baseline cash')
        linked.update(cashids);exposures[eid]=row;parties.add(row['counterparty_id'])
    if not exposures:_fail('STRUCTURED_COUNTERPARTY','Nonempty exposure contracts required')
    collateral_ids=set()
    for row in _rows(netting_sets,'netting_sets'):
        _required(row,['netting_set_id','counterparty_id','enforceable','collateral_contract_id','collateral_convention','default_timing']);nid=_text(row['netting_set_id'],'netting_set_id');cid=_text(row['collateral_contract_id'],'collateral_contract_id')
        if nid in sets or cid in collateral_ids or row['counterparty_id']not in parties or not isinstance(row['enforceable'],bool)or row['collateral_convention']!='RECEIVED_TRANSFERABLE' or row['default_timing']!='AFTER_COLLATERAL_SNAPSHOT_BEFORE_CASHFLOWS':_fail('STRUCTURED_NETTING','Unique set/collateral IDs and explicit legal/default timing conventions required')
        members=[e for e in exposures.values()if e['netting_set_id']==nid]
        if not members or any(e['counterparty_id']!=row['counterparty_id']for e in members)or(not row['enforceable']and len(members)!=1):_fail('STRUCTURED_NETTING','Unenforceable netting requires singleton sets; legal counterparty must agree')
        sets[nid]=row;collateral_ids.add(cid)
    if {e['netting_set_id']for e in exposures.values()}!=set(sets):_fail('STRUCTURED_NETTING','Every exposure needs exactly one defined netting set')
    pmap,defaults,events=_party_events(contract,party_mapping,parties,party_default_paths,shared_innovations)
    marks={};collateral={};terms={}
    for row in _rows(exposure_paths,'exposure_paths'):
        _required(row,['scenario_id','exposure_id','date','signed_replacement_mark']);key=(row['scenario_id'],row['exposure_id'],row['date'])
        if key in marks or row['scenario_id']not in sids or row['exposure_id']not in exposures or row['date']not in dates:_fail('STRUCTURED_EXPOSURE','Complete unique aligned signed replacement snapshots required')
        mark=_number(row['signed_replacement_mark'],'signed_replacement_mark')
        if exposures[row['exposure_id']]['basis']!='REPLACEMENT'and mark!=0:_fail('STRUCTURED_DOUBLE_COUNT','Cash-linked exposure derives its claim from baseline; no added replacement mark')
        marks[key]=mark
    if set(marks)!={(s,e,d)for s in sids for e in exposures for d in dates}:_fail('STRUCTURED_EXPOSURE','Full exposure/scenario/risk-date grid required')
    for row in _rows(collateral_paths,'collateral_paths'):
        _required(row,['scenario_id','netting_set_id','date','known_at','collateral_value','haircut']);key=(row['scenario_id'],row['netting_set_id'],row['date'])
        if key in collateral or row['scenario_id']not in sids or row['netting_set_id']not in sets or row['date']not in dates or _date(row['known_at'],'known_at')>_date(row['date'],'date'):_fail('STRUCTURED_COLLATERAL','Known-at or before default snapshot and unique aligned collateral grid required')
        _number(row['collateral_value'],'collateral_value',0);_number(row['haircut'],'haircut',0,1);collateral[key]=row
    if set(collateral)!={(s,n,d)for s in sids for n in sets for d in dates}:_fail('STRUCTURED_COLLATERAL','Full set/scenario/risk-date collateral grid required')
    for row in _rows(settlement_terms,'settlement_terms'):
        _required(row,['scenario_id','counterparty_id','estate_recovery_fraction','estate_capacity','estate_delay_days','collateral_delay_days','replacement_cost','replacement_delay_days']);key=(row['scenario_id'],row['counterparty_id'])
        if key in terms or row['scenario_id']not in sids or row['counterparty_id']not in parties:_fail('STRUCTURED_COUNTERPARTY','Unique aligned counterparty settlement terms required')
        _number(row['estate_recovery_fraction'],'estate_recovery_fraction',0,1)
        for k in ('estate_capacity','replacement_cost'):_number(row[k],k,0)
        for k in ('estate_delay_days','collateral_delay_days','replacement_delay_days'):
            n=_number(row[k],k,0)
            if n!=int(n):_fail('STRUCTURED_CALENDAR','Integer nonnegative settlement delays required')
        terms[key]=row
    if set(terms)!={(s,p)for s in sids for p in parties}:_fail('STRUCTURED_COUNTERPARTY','Full counterparty/scenario settlement terms required')
    adjusted=[];removed=[];netting=[];counterparty=[];ending=[]
    def flow(sid,date,source,party,nid,amount,cashflow_id=None):
        if amount:adjusted.append(dict(scenario_id=sid,date=date,source=source,counterparty_id=party,netting_set_id=nid,cashflow_id=cashflow_id,amount=_sum([amount])))
    for sid in sids:
        removed_ids=set();total_loss=total_cost=0.
        for party in sorted(parties):
            default=defaults[sid,pmap[party]]
            if default is None:
                counterparty.append(dict(scenario_id=sid,counterparty_id=party,default_date=None,credit_loss=0.,replacement_cost=0.,estate_recovery=0.,collateral_recovery=0.));continue
            term=terms[sid,party];settlement_dates={}
            for key in ('estate_delay_days','collateral_delay_days','replacement_delay_days'):
                try:date=(_date(default,'default_date')+timedelta(days=int(term[key]))).isoformat()
                except (OverflowError,ValueError):_fail('STRUCTURED_CALENDAR','Settlement date outside supported calendar')
                if date>horizon:_fail('STRUCTURED_UNRESOLVED','Settlement tail outside explicit horizon; extend scenarios rather than drop recovery')
                settlement_dates[key]=date
            parts=[]
            for nid in sorted(sets):
                if sets[nid]['counterparty_id']!=party:continue
                members=[e for e in exposures.values()if e['netting_set_id']==nid];cash_claims=[];swap_marks=[];positive_by_role={role:[] for role in ('SERVICER','ACCOUNT_BANK','GUARANTOR','SWAP')}
                for e in members:
                    mark=marks[sid,e['exposure_id'],default];swap_marks.append(mark);contributions=[mark]
                    for cashid in e['cashflow_ids']:
                        row=baseline[sid,cashid]
                        if row['date']>=default:
                            cash_claims.append(row['amount']);contributions.append(row['amount']);removed_ids.add(cashid);removed.append(dict(scenario_id=sid,cashflow_id=cashid,exposure_id=e['exposure_id'],counterparty_id=party,netting_set_id=nid,default_date=default,original_date=row['date'],amount=row['amount']))
                    positive_by_role[e['role']].append(max(0.,_sum(contributions)))
                held=_sum(cash_claims);replacement=_sum(swap_marks);net=_sum([held,replacement]);positive=max(0.,net);snap=collateral[sid,nid,default];coll=min(positive,snap['collateral_value']*(1-snap['haircut']));unsecured=max(0.,positive-coll)
                parts.append(dict(positive_claims_by_role={role:_sum(values)for role,values in positive_by_role.items()},netting_set_id=nid,linked_cash_claim=held,signed_replacement_mark=replacement,net_closeout=net,positive_exposure=positive,collateral_recovery=coll,unsecured_claim=unsecured))
            unsecured_total=_sum(p['unsecured_claim']for p in parts);estate_total=min(unsecured_total*term['estate_recovery_fraction'],term['estate_capacity']);credit=[];colls=[]
            for p in parts:
                nid=p['netting_set_id'];estate=estate_total*(p['unsecured_claim']/unsecured_total)if unsecured_total else 0.;loss=max(0.,p['unsecured_claim']-estate);credit.append(loss);colls.append(p['collateral_recovery'])
                flow(sid,default,'SIGNED_REPLACEMENT_OUTLAY',party,nid,-p['signed_replacement_mark'])
                flow(sid,default,'NEGATIVE_NET_CLOSEOUT',party,nid,min(0.,p['net_closeout']))
                flow(sid,settlement_dates['collateral_delay_days'],'COLLATERAL_REALIZATION',party,nid,p['collateral_recovery'])
                flow(sid,settlement_dates['estate_delay_days'],'ESTATE_RECOVERY',party,nid,estate)
                gross_positive=_sum(p['positive_claims_by_role'].values())
                attribution={role:loss*(amount/gross_positive)if gross_positive else 0. for role,amount in p['positive_claims_by_role'].items()}
                netting.append(dict(scenario_id=sid,counterparty_id=party,default_date=default,**p,estate_recovery=estate,credit_loss=loss,proportional_loss_by_role=attribution))
            flow(sid,settlement_dates['replacement_delay_days'],'ADDITIONAL_REPLACEMENT_COST',party,None,-term['replacement_cost']);loss=_sum(credit);total_loss=_sum([total_loss,loss]);total_cost=_sum([total_cost,term['replacement_cost']]);counterparty.append(dict(scenario_id=sid,counterparty_id=party,default_date=default,credit_loss=loss,replacement_cost=term['replacement_cost'],estate_recovery=estate_total,collateral_recovery=_sum(colls)))
        for (s,cashid),row in baseline.items():
            if s==sid and cashid not in removed_ids:flow(sid,row['date'],'BASELINE_RETAINED',None,None,row['amount'],cashid)
        original=_sum(r['amount']for(s,_),r in baseline.items()if s==sid);actual=_sum(r['amount']for r in adjusted if r['scenario_id']==sid);residual=_sum([original,-actual,-total_loss,-total_cost])
        if abs(residual)>1e-10*max(1.,original,total_loss,total_cost):_fail('STRUCTURED_CONSERVATION','Baseline/adjusted/credit/replacement cash reconciliation failed')
        ending.append(dict(scenario_id=sid,baseline_cash=original,adjusted_net_cash=actual,credit_loss=total_loss,additional_replacement_cost=total_cost,conservation_residual=residual))
    adjusted.sort(key=lambda r:(r['scenario_id'],r['date'],r['source'],r['counterparty_id']or'',r['netting_set_id']or'',r['cashflow_id']or''))
    value=_bound(dict(schema_version='1.0',kind='PoolCounterpartyStress',origin=as_of,exposure_contracts=deepcopy(exposure_contracts),netting_sets=deepcopy(netting_sets),scenario_contract=contract,innovation_hash=_fingerprint(shared_innovations),input_hash=_fingerprint(dict(baseline_cashflows=baseline_cashflows,exposure_contracts=exposure_contracts,netting_sets=netting_sets,exposure_paths=exposure_paths,collateral_paths=collateral_paths,settlement_terms=settlement_terms,party_mapping=party_mapping,party_default_paths=party_default_paths)),group_events=events,removed_baseline_cashflows=removed,adjusted_cashflows=adjusted,netting_results=netting,counterparty_results=counterparty,ending_states=ending))
    return _result(value,'structured_credit.stress_pool_counterparties','counterparty_cash_impacts',ctx,limitations=('Linked baseline cash denotes existing incoming monetary claims held from origin until remittance; unearned future income must not be labeled held cash. Shared default occurs after the declared known collateral snapshot and before same-day cash. Enforceable netting is caller-declared; otherwise each set is a singleton. Received transferable collateral is allocated uniquely; posted collateral claims require the separate counterparty exposure engine. Supplied signed replacement marks create one replacement outlay or inflow and one net closeout claim/payment; derivative prices are not estimated here. Estate capacity is capped once per counterparty across sets and allocated pro rata; additional replacement cost is charged once per counterparty. All recovery dates must fit the horizon. Netting-set loss attribution to roles is proportional to positive gross claims after netting/collateral; it is an allocation convention, not a separate loss calculation. Baseline cash removed and closeout/replacement cash reconcile exactly to credit loss plus additional costs, with no repeated monetary IDs.',))
