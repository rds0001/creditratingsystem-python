"""Invoice credit, seller recourse and segregated operational-risk accounting."""
from copy import deepcopy
from .core import _fail,_result,_fingerprint,_context_value
from .data import _required,_rows,_date
from ._structured_pool import _number,_text,_sum,_bound,_checked


def _trade_scenarios(contract,origin,context):
    c=deepcopy(contract);_required(c,['contract_id','version','scenario_ids','weights','dates','currency','probability_measure','sampling_design','conditioning'])
    for key in ('contract_id','version','currency','conditioning'):_text(c[key],key)
    for key in ('scenario_ids','dates'):
        if not isinstance(c[key],list)or not c[key]or len(set(c[key]))!=len(c[key]):_fail('STRUCTURED_SCENARIOS','Unique nonempty scenario/date axes required')
    for sid in c['scenario_ids']:_text(sid,'scenario_id')
    previous=origin
    for date in c['dates']:
        if _date(date,'date')<=_date(previous,'previous'):_fail('STRUCTURED_CALENDAR','Risk grid must increase after origin')
        previous=date
    if not isinstance(c['weights'],list)or len(c['weights'])!=len(c['scenario_ids']):_fail('STRUCTURED_SCENARIOS','Aligned weights required')
    weights=[_number(w,'weight',0)for w in c['weights']]
    if abs(_sum(weights)-1)>1e-12 or c['probability_measure']!='PHYSICAL' or c['sampling_design']not in ('IID_UNWEIGHTED','FIXED_WEIGHTED_SCENARIOS','DEPENDENT_OR_QMC'):_fail('STRUCTURED_SCENARIOS','Normalized physical scenario contract required')
    if c['sampling_design']=='IID_UNWEIGHTED'and any(abs(w-1/len(weights))>1e-14 for w in weights):_fail('STRUCTURED_SCENARIOS','IID requires equal weights')
    ctx=_context_value(context);expected=dict(currency=c['currency'],probability_measure='PHYSICAL',conditioning=c['conditioning'],time_unit='days',horizon=(_date(c['dates'][-1],'date')-_date(origin,'origin')).days)
    if any(ctx.get(k)not in(None,v)for k,v in expected.items()):_fail('TARGET_MISMATCH','Physical scenario and actual day horizon must agree with context')
    return c,ctx


def _party_events(contract,mapping,required_parties,group_paths,innovations):
    party_map={}
    for row in _rows(mapping,'party_mapping'):
        _required(row,['party_id','default_group_id']);party=_text(row['party_id'],'party_id');group=_text(row['default_group_id'],'default_group_id')
        if party in party_map:_fail('STRUCTURED_DEFAULT_GROUP','A legal party must have one shared default group across roles')
        party_map[party]=group
    if set(party_map)!=set(required_parties):_fail('STRUCTURED_DEFAULT_GROUP','Exact participating-party mapping required')
    groups=set(party_map.values());sids=contract['scenario_ids'];dates=contract['dates'];imap={};used=set();gp={}
    for row in _rows(innovations,'shared_innovations'):
        _required(row,['scenario_id','date','channel_id','uniform']);key=(row['scenario_id'],row['date'],_text(row['channel_id'],'channel_id'));u=_number(row['uniform'],'uniform',0,1)
        if key in imap or row['scenario_id']not in sids or row['date']not in dates or u==1:_fail('STRUCTURED_INNOVATION','Unique aligned uniforms in [0,1) required')
        imap[key]=u
    for row in _rows(group_paths,'party_default_paths'):
        _required(row,['scenario_id','default_group_id','date','conditional_pd','default_channel']);key=(row['scenario_id'],row['default_group_id'],row['date']);ik=(row['scenario_id'],row['date'],row['default_channel'])
        if key in gp or row['scenario_id']not in sids or row['default_group_id']not in groups or row['date']not in dates:_fail('STRUCTURED_DEFAULT_GROUP','Aligned unique group hazard grid required')
        _number(row['conditional_pd'],'conditional_pd',0,1)
        if ik not in imap:_fail('STRUCTURED_INNOVATION','Missing group innovation')
        gp[key]=row;used.add(ik)
    if set(gp)!={(s,g,d)for s in sids for g in groups for d in dates}or used!=set(imap):_fail('STRUCTURED_DEFAULT_GROUP','Complete group grid and exactly used innovations required')
    defaults={};events=[]
    for sid in sids:
        for group in sorted(groups):
            first=None
            for date in dates:
                row=gp[sid,group,date];new=first is None and imap[sid,date,row['default_channel']]<row['conditional_pd']
                if new:first=date
                events.append(dict(scenario_id=sid,default_group_id=group,date=date,new_default=new,in_default=first is not None))
            defaults[sid,group]=first
    return party_map,defaults,events


def simulate_trade_receivables(invoice_pool,scenario_contract,party_mapping,party_default_paths,invoice_scenarios,seller_terms,servicer_terms,shared_innovations,*,context=None):
    """Evaluate actual invoice payment/default and capped shared seller/servicer risk."""
    pool=_checked(invoice_pool,'CreditAssetPool');contract,ctx=_trade_scenarios(scenario_contract,pool['as_of'],context)
    if pool['profile']['product_kind']!='TRADE_RECEIVABLES'or pool['profile']['pool_mode']!='STATIC'or contract['currency']!=pool['profile']['currency']:_fail('STRUCTURED_ROUTE','Static observed invoice pool and matching currency required')
    origin=pool['as_of'];horizon=contract['dates'][-1];shares={};tape={}
    for rec in pool['active_assignments']:
        a=rec['assignment'];aid=a['asset_id']
        if a['effective_until']is not None and a['effective_until']<=horizon:_fail('STRUCTURED_TRANSFER','Invoice assignment must remain effective through horizon')
        shares[aid]=_sum([shares.get(aid,0.),a['share']]);tape[aid]=pool['asset_contracts'][aid]
    if not tape:_fail('STRUCTURED_TRADE','Nonempty owned invoice pool required')
    for a in tape.values():
        if a['asset_class']!='INVOICE'or len(a['schedule'])!=1 or a['schedule'][0]['interest_due']!=0:_fail('STRUCTURED_TRADE','One zero-interest bullet per invoice required')
    parties={a[k]for a in tape.values()for k in ('obligor_id','seller_id','servicer_id')};party_map,defaults,events=_party_events(contract,party_mapping,parties,party_default_paths,shared_innovations)
    if any(party_map[a['obligor_id']]!=a['default_group_id']for a in tape.values()):_fail('STRUCTURED_DEFAULT_GROUP','Debtor mapping must match frozen pool group')
    def date_in(value,name):
        if not _date(origin,'origin')<=_date(value,name)<=_date(horizon,'horizon'):_fail('STRUCTURED_CALENDAR','All modeled cash/event dates must fit explicit horizon')
        return value
    scenarios={}
    for row in _rows(invoice_scenarios,'invoice_scenarios'):
        _required(row,['scenario_id','asset_id','payment_date','dilution_date','dilution_fraction','debtor_recovery_fraction','debtor_recovery_date','debtor_recovery_cost']);key=(row['scenario_id'],row['asset_id'])
        if key in scenarios or row['scenario_id']not in contract['scenario_ids']or row['asset_id']not in tape:_fail('STRUCTURED_TRADE','Unique aligned invoice scenarios required')
        for k in ('payment_date','dilution_date','debtor_recovery_date'):date_in(row[k],k)
        for k in ('dilution_fraction','debtor_recovery_fraction'):_number(row[k],k,0,1)
        _number(row['debtor_recovery_cost'],'debtor_recovery_cost',0);a=tape[row['asset_id']];default=defaults[row['scenario_id'],a['default_group_id']]
        if row['payment_date']<a['schedule'][0]['date']:_fail('STRUCTURED_CALENDAR','Invoice early settlement needs explicit discount terms; payment cannot precede due date')
        event_date=min(row['payment_date'],default)if default else row['payment_date']
        if row['dilution_date']>event_date or row['debtor_recovery_date']<event_date:_fail('STRUCTURED_CALENDAR','Dilution must precede payment/default; recovery cannot precede event')
        scenarios[key]=row
    if set(scenarios)!={(s,a)for s in contract['scenario_ids']for a in tape}:_fail('STRUCTURED_TRADE','Full invoice/scenario grid required')
    sellers={a['seller_id']for a in tape.values()};servicers={a['servicer_id']for a in tape.values()};seller_map={};servicer_map={}
    for rows,kind,ids,dest in ((seller_terms,'seller',sellers,seller_map),(servicer_terms,'servicer',servicers,servicer_map)):
        for row in _rows(rows,kind+'_terms'):
            idkey=kind+'_id';_required(row,['scenario_id',idkey]);key=(row['scenario_id'],row[idkey])
            if key in dest or row['scenario_id']not in contract['scenario_ids']or row[idkey]not in ids:_fail('STRUCTURED_TRADE','Unique aligned seller/servicer terms required')
            if kind=='seller':
                _required(row,['settlement_date','performing_capacity','default_estate_capacity','default_recovery_fraction','recourse_scope']);date_in(row['settlement_date'],'settlement_date')
                for k in ('performing_capacity','default_estate_capacity'):_number(row[k],k,0)
                _number(row['default_recovery_fraction'],'default_recovery_fraction',0,1)
                if row['recourse_scope']not in ('CREDIT_ONLY','CREDIT_AND_DILUTION'):_fail('STRUCTURED_TRADE','Explicit seller recourse scope required')
            else:
                _required(row,['remittance_date','default_commingling_recovery_fraction','default_commingling_capacity','replacement_cost']);date_in(row['remittance_date'],'remittance_date');_number(row['default_commingling_recovery_fraction'],'default_commingling_recovery_fraction',0,1)
                for k in ('default_commingling_capacity','replacement_cost'):_number(row[k],k,0)
            dest[key]=row
        if set(dest)!={(s,i)for s in contract['scenario_ids']for i in ids}:_fail('STRUCTURED_TRADE','Full seller/servicer scenario terms required')
    invoice_results=[];seller_results=[];servicer_results=[];cashflows=[];ending=[]
    def cash(sid,date,source,party,amount,asset_id=None):
        cashflows.append(dict(scenario_id=sid,date=date,source=source,party_id=party,asset_id=asset_id,amount=_sum([amount])))
    for sid in contract['scenario_ids']:
        claims={};collections={v:[]for v in servicers};costs=[]
        for aid in sorted(tape):
            a=tape[aid];row=scenarios[sid,aid];amount=a['principal_balance']*shares[aid];dilution=amount*row['dilution_fraction'];net=amount-dilution;default=defaults[sid,a['default_group_id']];credit_event=net>0 and default is not None and default<=row['payment_date'];received=net*row['debtor_recovery_fraction']if credit_event else net;collection_date=row['debtor_recovery_date']if credit_event else row['payment_date'];cost=row['debtor_recovery_cost']*shares[aid]if credit_event else 0.
            if seller_map[sid,a['seller_id']]['settlement_date']<collection_date or servicer_map[sid,a['servicer_id']]['remittance_date']<collection_date:_fail('STRUCTURED_CALENDAR','Seller residual settlement and servicer remittance must follow debtor cash')
            claims[aid]=dict(scenario_id=sid,asset_id=aid,invoice_id=a['invoice_id'],owned_principal=amount,dilution_loss=dilution,debtor_credit_event=credit_event,debtor_default_date=default,actual_collection_date=collection_date,debtor_collection=received,debtor_credit_loss=net-received,seller_credit_recourse=0.,seller_dilution_recourse=0.,debtor_recovery_cost=cost)
            collections[a['servicer_id']].append((aid,collection_date,received));costs.append(cost)
            if cost:cash(sid,collection_date,'DEBTOR_WORKOUT_COST',a['obligor_id'],-cost,aid)
        for seller in sorted(sellers):
            term=seller_map[sid,seller];default=defaults[sid,party_map[seller]];failed=default is not None and default<=term['settlement_date'];eligible={}
            for aid,a in tape.items():
                if a['seller_id']==seller and a['recourse']=='SELLER':
                    q=claims[aid];eligible[aid]=_sum([q['debtor_credit_loss'],q['dilution_loss']if term['recourse_scope']=='CREDIT_AND_DILUTION'else 0.])
            total=_sum(eligible.values());payment=min(total*(term['default_recovery_fraction']if failed else 1.),term['default_estate_capacity']if failed else term['performing_capacity']);allocated=[]
            for aid,claim in sorted(eligible.items()):
                take=payment*(claim/total)if total else 0.;q=claims[aid];credit=take*(q['debtor_credit_loss']/claim)if claim else 0.;q['seller_credit_recourse']=credit;q['seller_dilution_recourse']=take-credit;allocated.append(take)
                if take:cash(sid,term['settlement_date'],'SELLER_RECOURSE',seller,take,aid)
            seller_results.append(dict(scenario_id=sid,seller_id=seller,in_default=failed,eligible_claim=total,aggregate_payment=_sum(allocated),capacity_used=_sum(allocated),settlement_date=term['settlement_date']))
        commingling_total=0.;replacement_total=0.
        for servicer in sorted(servicers):
            term=servicer_map[sid,servicer];default=defaults[sid,party_map[servicer]];failed=default is not None and default<=term['remittance_date'];held=[];redirected=0.
            for aid,date,amount in collections[servicer]:
                if failed and date>=default:
                    if amount:cash(sid,date,'REDIRECTED_DEBTOR_COLLECTION',servicer,amount,aid)
                    redirected=_sum([redirected,amount])
                else:held.append((aid,amount))
            held_total=_sum(x[1]for x in held);remitted=min(held_total*term['default_commingling_recovery_fraction'],term['default_commingling_capacity'])if failed else held_total
            if remitted:cash(sid,term['remittance_date'],'SERVICER_REMITTANCE',servicer,remitted)
            replacement=term['replacement_cost']if failed else 0.
            if replacement:cash(sid,default,'SERVICER_REPLACEMENT_COST',servicer,-replacement)
            lost=held_total-remitted;commingling_total=_sum([commingling_total,lost]);replacement_total=_sum([replacement_total,replacement]);servicer_results.append(dict(scenario_id=sid,servicer_id=servicer,in_default_before_remittance=failed,held_cash=held_total,redirected_cash=redirected,remittance=remitted,commingling_loss=lost,replacement_cost=replacement))
        invoice_results.extend(claims.values());net_credit=_sum(q['debtor_credit_loss']-q['seller_credit_recourse']for q in claims.values());net_dilution=_sum(q['dilution_loss']-q['seller_dilution_recourse']for q in claims.values());net_cash=_sum(r['amount']for r in cashflows if r['scenario_id']==sid);total_principal=_sum(q['owned_principal']for q in claims.values());total_cost=_sum([*costs,replacement_total]);residual=_sum([total_principal,-net_credit,-net_dilution,-commingling_total,-total_cost,-net_cash])
        if abs(residual)>1e-10*max(1.,total_principal):_fail('STRUCTURED_CONSERVATION','Invoice principal/recourse/operational cash reconciliation failed')
        ending.append(dict(scenario_id=sid,owned_principal=total_principal,net_credit_loss=net_credit,net_dilution_loss=net_dilution,commingling_loss=commingling_total,workout_and_replacement_cost=total_cost,net_investor_cash=net_cash,conservation_residual=residual))
    cashflows.sort(key=lambda r:(r['scenario_id'],r['date'],r['source'],r['party_id'],r['asset_id']or''))
    value=_bound(dict(schema_version='1.0',kind='TradeReceivablePaths',pool_hash=pool['sha256'],scenario_contract=contract,innovation_hash=_fingerprint(shared_innovations),party_mapping=deepcopy(party_mapping),input_hash=_fingerprint(dict(party_default_paths=party_default_paths,invoice_scenarios=invoice_scenarios,seller_terms=seller_terms,servicer_terms=servicer_terms)),group_events=events,invoice_results=invoice_results,seller_results=seller_results,servicer_results=servicer_results,cashflows=cashflows,ending_states=ending))
    return _result(value,'structured_credit.simulate_trade_receivables','invoice_cashflows',ctx,limitations=('Observed static bullet invoices only; actual delayed payment remains exposed to shared debtor default until payment, with default before same-day payment. A legal party has one shared default group across debtor, seller and servicer roles. Commercial dilution precedes credit/payment and remains separate; full dilution creates no debtor workout cost. Residual seller recourse follows debtor recoveries, is capped once across invoices and allocated pro rata; it is paid directly to investor outside servicer accounts. Before-failure collections held by the servicer share one capped commingling claim; on/after-failure collections redirect directly, and replacement cost is charged once. All event and cash dates must fit the explicit horizon; unresolved or future-originated claims are not silently completed.',))
