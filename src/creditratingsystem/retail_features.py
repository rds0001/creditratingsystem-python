"""Point-in-time retail evidence, economic relationships and granular features."""
from copy import deepcopy
import math
from .core import CalcResult,CreditRiskError,_result,_fail,_numeric,_context_value,_fingerprint
from .data import _rows,_required,_unique,_date,_days
from ._loss_paths import _sum,_multiply


def _text(value,name):
    if not isinstance(value,str) or not value.strip():_fail('RETAIL_CONTRACT',name+' must be a nonempty string')
    return value


def _strings(value,name,empty=True):
    if not isinstance(value,(list,tuple)) or any(not isinstance(v,str)or not v for v in value) or len(set(value))!=len(value) or (not empty and not value):_fail('RETAIL_CONTRACT',name+' requires distinct nonempty strings')
    return list(value)


def _number(value,name,minimum=None):
    x=_numeric(value,name)
    if x.ndim or (minimum is not None and x<minimum):_fail('RETAIL_INPUT',name+' requires a finite scalar in its domain')
    return float(x)


def _cutoffs(as_of,information_cutoff,context=None):
    anchor=_date(as_of,'as_of');cutoff=_date(information_cutoff,'information_cutoff')
    if cutoff>anchor:_fail('RETAIL_DATE','Feature information cutoff cannot follow its economic evaluation date')
    ctx=_context_value(context)
    for k,v in [('as_of',as_of),('information_cutoff',information_cutoff),('known_at',information_cutoff)]:
        if ctx.get(k) is not None and ctx[k]!=v:_fail('TARGET_MISMATCH','Context '+k+' differs from feature cutoff')
    hashed='context_hash' in ctx;ctx.pop('context_hash',None);ctx.update(mode='DATED',as_of=as_of,known_at=information_cutoff)
    if hashed:ctx['context_hash']=_fingerprint(ctx)
    return anchor,cutoff,ctx


def _releases(records,key,cutoff,name):
    """Choose the latest known integer release before economic-date filtering."""
    rows=_rows(records,name);groups={};excluded=[];seen=set()
    for row in rows:
        _required(row,[key,'release_seq','known_at']);_text(row[key],key);_days(row['release_seq'],'release_seq',1)
        identity=(row[key],row['release_seq'])
        if identity in seen:_fail('DATA_DUPLICATE','Duplicate logical record and release sequence')
        seen.add(identity);known=_date(row['known_at'],'known_at')
        if known>cutoff:excluded.append(dict(record_id=row[key],release_seq=row['release_seq'],reason='NOT_YET_KNOWN'));continue
        previous=groups.get(row[key])
        if previous is None or (row['known_at'],row['release_seq'])>(previous['known_at'],previous['release_seq']):groups[row[key]]=row
    for row in rows:
        if _date(row['known_at'],'known_at')<=cutoff and row is not groups.get(row[key]):excluded.append(dict(record_id=row[key],release_seq=row['release_seq'],reason='SUPERSEDED_RELEASE'))
    return [groups[k]for k in sorted(groups)],sorted(excluded,key=lambda r:(r['record_id'],r['release_seq']))


def _coverage(coverage,cutoff):
    if not isinstance(coverage,dict):_fail('RETAIL_COVERAGE','Explicit source coverage object required')
    _required(coverage,['status','sources','known_at'])
    if coverage['status']not in ('COMPLETE','PARTIAL','UNKNOWN') or _date(coverage['known_at'],'coverage.known_at')>cutoff:_fail('RETAIL_COVERAGE','Coverage must be known at cutoff and COMPLETE/PARTIAL/UNKNOWN')
    _strings(coverage['sources'],'coverage.sources')
    if coverage['status']=='COMPLETE' and not coverage['sources']:_fail('RETAIL_COVERAGE','Complete coverage needs at least one named source')
    return deepcopy(coverage)


def _bound(value):
    value=deepcopy(value);value['artifact_hash']=_fingerprint(value);return value


def _artifact(value,kind):
    if isinstance(value,CalcResult):value=value.value
    if not isinstance(value,dict) or value.get('artifact_type')!=kind or value.get('schema_version')!='1.0':_fail('RETAIL_ARTIFACT','Expected '+kind+' artifact')
    raw=deepcopy(value);digest=raw.pop('artifact_hash',None)
    if digest!=_fingerprint(raw):_fail('INTEGRITY_ERROR','Retail artifact differs from its frozen content')
    return value


def validate_retail_dataset(tables,data_contract,*,as_of,information_cutoff,context=None):
    """Validate explicit relational schemas and feature eligibility without creating labels."""
    anchor,cutoff,ctx=_cutoffs(as_of,information_cutoff,context)
    if not isinstance(tables,dict) or not isinstance(data_contract,dict) or data_contract.get('schema_version')!='1.0':_fail('RETAIL_CONTRACT','Named local tables and version1.0 RetailDataContract required')
    _required(data_contract,['contract_id','version','table_specs','prohibited_feature_fields','source_coverage']);_text(data_contract['version'],'version');specs=_rows(data_contract['table_specs'],'table_specs');_unique(specs,'table','table_specs')
    forbidden=set(_strings(data_contract['prohibited_feature_fields'],'prohibited_feature_fields'))|{'target','outcome','label','default_in_horizon','future_default','lgd_target','recovery_target'}
    known_specs={r['table']:r for r in specs};issues=[];data={};eligible={};coverage=[]
    def issue(table,index,code,field):issues.append(dict(table=table,row_index=index,code=code,field=field))
    for name in sorted(set(tables)-set(known_specs)):issue(name,None,'UNDECLARED_TABLE',None)
    for spec in specs:
        _required(spec,['role','primary_key','fields','foreign_keys','date_order','event_field','known_field','feature_fields','allow_extra_fields'])
        if spec['role']not in ('IDENTITY','FEATURE','OUTCOME','DECISION') or type(spec['allow_extra_fields'])is not bool:_fail('RETAIL_CONTRACT','Declare table role and boolean extra-field policy')
        keys=_strings(spec['primary_key'],'primary_key',False);features=_strings(spec['feature_fields'],'feature_fields');fields=_rows(spec['fields'],'fields');_unique(fields,'name','fields');fieldmap={f['name']:f for f in fields}
        if any(k not in fieldmap for k in keys+[spec['event_field'],spec['known_field']]+features):_fail('RETAIL_CONTRACT','Keys, times and feature names must have declared field types')
        if any(fieldmap[k]['type']!='date'for k in (spec['event_field'],spec['known_field'])):_fail('RETAIL_CONTRACT','Economic and knowledge fields must have date types')
        for field in fields:
            if field.get('type')not in ('string','date','number','integer','boolean') or type(field.get('nullable'))is not bool:_fail('RETAIL_CONTRACT','Field type and nullable boolean required')
        name=spec['table'];data[name]=_rows(tables.get(name,[]),name);eligible[name]=[]
        if name not in tables:issue(name,None,'MISSING_TABLE',None)
        if spec['role']!='FEATURE' and features:_fail('RETAIL_CONTRACT','Only FEATURE-role tables can designate model features')
        if any(f in forbidden for f in features):issue(name,None,'OUTCOME_AS_FEATURE',','.join(sorted(set(features)&forbidden)))
        seen=set()
        for i,row in enumerate(data[name]):
            valid_fields=set()
            for field in fields:
                key=field['name'];value=row.get(key);kind=field['type']
                if key not in row or value is None:
                    if not field['nullable'] or key in keys:issue(name,i,'MISSING_REQUIRED',key)
                    continue
                valid=False
                if kind=='string':valid=isinstance(value,str)and bool(value)
                elif kind=='boolean':valid=type(value)is bool
                elif kind in ('number','integer'):valid=type(value)in (int,float)and math.isfinite(value)and (kind!='integer'or value==math.floor(value))
                elif kind=='date':
                    try:_date(value,key);valid=True
                    except CreditRiskError:pass
                if not valid:issue(name,i,'INVALID_TYPE',key)
                else:valid_fields.add(key)
            if not spec['allow_extra_fields']:
                for key in sorted(set(row)-set(fieldmap)):issue(name,i,'UNDECLARED_FIELD',key)
            if all(k in valid_fields for k in keys):
                key=tuple(row[k]for k in keys)
                if key in seen:issue(name,i,'DUPLICATE_KEY',','.join(keys))
                seen.add(key)
            for pair in spec['date_order']:
                if not isinstance(pair,list) or len(pair)!=2 or any(k not in fieldmap or fieldmap[k]['type']!='date'for k in pair):_fail('RETAIL_CONTRACT','date_order requires declared date-field pairs')
                if all(k in valid_fields for k in pair)and row[pair[0]]>row[pair[1]]:issue(name,i,'DATE_ORDER',','.join(pair))
            ef,kf=spec['event_field'],spec['known_field']
            if ef in valid_fields and kf in valid_fields and _date(row[ef],ef)<=anchor and _date(row[kf],kf)<=cutoff:eligible[name].append(i)
    for spec in specs:
        name=spec['table']
        for fk in spec['foreign_keys']:
            _required(fk,['field','table','target_field'])
            if fk['table']not in known_specs or fk['field']not in {f['name']for f in spec['fields']} or fk['target_field']not in {f['name']for f in known_specs[fk['table']]['fields']}:_fail('RETAIL_CONTRACT','Foreign key references an undeclared table or field')
            parent=data[fk['table']];ids={r.get(fk['target_field'])for r in parent if isinstance(r.get(fk['target_field']),(str,int,float))};visible_ids={parent[j].get(fk['target_field'])for j in eligible[fk['table']]if isinstance(parent[j].get(fk['target_field']),(str,int,float))}
            for i,row in enumerate(data[name]):
                value=row.get(fk['field'])
                if value is None:continue
                if not isinstance(value,(str,int,float))or value not in ids:issue(name,i,'FOREIGN_KEY',fk['field'])
                elif i in eligible[name]and value not in visible_ids:issue(name,i,'PARENT_NOT_KNOWN_AT_CUTOFF',fk['field'])
    source_rows=_rows(data_contract['source_coverage'],'source_coverage');_unique(source_rows,'source_id','source_coverage')
    for source in source_rows:
        _required(source,['source_id','status','known_at'])
        if source['status']not in ('COMPLETE','PARTIAL','UNKNOWN'):_fail('RETAIL_COVERAGE','Source status must be COMPLETE/PARTIAL/UNKNOWN')
        coverage.append(dict(source_id=source['source_id'],status=source['status']if _date(source['known_at'],'known_at')<=cutoff else 'UNKNOWN'))
    return _result(dict(valid=not issues,applicability='APPLICABLE',violations=issues,row_counts={k:len(data[k])for k in sorted(data)},eligible_row_indices={k:eligible[k]for k in sorted(eligible)},source_coverage=sorted(coverage,key=lambda x:x['source_id'])),'retail_features.validate_retail_dataset','data_validation',ctx,diagnostics=dict(contract_fingerprint=_fingerprint(data_contract),as_of=as_of,information_cutoff=information_cutoff),limitations=('Schema and point-in-time relational checks do not validate predictive usefulness, lawful feature use or completeness of unreported sources. Unknown coverage is not evidence of zero obligations.',))


def build_customer_household_graph(entities,relationships,*,as_of,information_cutoff,context=None):
    """Freeze explicitly supplied typed relationships and separate liability from budgeting."""
    anchor,cutoff,ctx=_cutoffs(as_of,information_cutoff,context);nodes=_rows(entities,'entities');_unique(nodes,'entity_id','entities');lookup={};visible=[]
    for row in nodes:
        _required(row,['entity_type','known_at'])
        if row['entity_type']not in ('CUSTOMER','HOUSEHOLD','APPLICATION','FACILITY','BUSINESS'):_fail('RETAIL_ENTITY','Unknown entity type')
        lookup[row['entity_id']]=row
        if _date(row['known_at'],'entity.known_at')<=cutoff:visible.append(row)
    selected,excluded=_releases(relationships,'relationship_id',cutoff,'relationships');edges=[];members={};seen=set();budget={};types={'MEMBER_OF':'HOUSEHOLD','COAPPLICANT_ON':'APPLICATION','BORROWER_ON':'FACILITY','GUARANTOR_ON':'FACILITY','OWNER_OF':'BUSINESS'}
    for row in selected:
        _required(row,['source_id','target_id','role','effective_from','evidence_reference']);_text(row['evidence_reference'],'evidence_reference')
        if any(k not in row for k in ('effective_until','liability_fraction','budget_share')):_fail('RETAIL_CONTRACT','Explicit nullable end date, liability fraction and budget share required')
        start=_date(row['effective_from'],'effective_from');end=None if row['effective_until']is None else _date(row['effective_until'],'effective_until')
        if end is not None and end<=start:_fail('RETAIL_DATE','Relationship uses nonempty [effective_from,effective_until) interval')
        if start>anchor or (end is not None and anchor>=end):excluded.append(dict(record_id=row['relationship_id'],release_seq=row['release_seq'],reason='OUTSIDE_EFFECTIVE_INTERVAL'));continue
        source=lookup.get(row['source_id']);target=lookup.get(row['target_id']);role=row['role']
        if role not in types or source is None or target is None or source['entity_type']!='CUSTOMER' or target['entity_type']!=types[role]:_fail('RETAIL_ENTITY','Relationship role endpoints have incompatible entity types')
        if _date(source['known_at'],'source.known_at')>cutoff or _date(target['known_at'],'target.known_at')>cutoff:_fail('RETAIL_DATE','Visible relationship references an entity not known at cutoff')
        key=(row['source_id'],row['target_id'],role)
        if key in seen:_fail('DOUBLE_COUNTING','Duplicate active person/target/role relationship')
        seen.add(key)
        for field in ('liability_fraction','budget_share'):
            if row[field]is not None and not 0<=_number(row[field],field)<=1:_fail('RETAIL_LIABILITY','Fractions must be in [0,1]')
        if role not in ('BORROWER_ON','GUARANTOR_ON')and (row['liability_fraction']is not None or row['budget_share']is not None):_fail('RETAIL_LIABILITY','Debt fractions belong to borrower or guarantor roles only')
        if role=='MEMBER_OF':
            if row['source_id']in members:_fail('RETAIL_MEMBERSHIP','A customer needs a unique active economic household at this snapshot')
            members[row['source_id']]=row['target_id']
        if role=='BORROWER_ON'and row['budget_share']is not None:budget.setdefault(row['target_id'],[]).append(row['budget_share'])
        edges.append(row)
    if any(_sum(v)>1 for v in budget.values()):_fail('RETAIL_LIABILITY','Internal borrower budget shares may not sum above one; legal joint liability is separate')
    paths=[dict(customer_id=r['source_id'],household_id=members.get(r['source_id']),facility_id=r['target_id'],role=r['role'],liability_fraction=r['liability_fraction'],budget_share=r['budget_share'],relationship_id=r['relationship_id'])for r in edges if r['role']in ('BORROWER_ON','GUARANTOR_ON')]
    value=_bound(dict(schema_version='1.0',artifact_type='RETAIL_RELATIONSHIP_GRAPH',as_of=as_of,information_cutoff=information_cutoff,entities=sorted(visible,key=lambda r:r['entity_id']),relationships=sorted(edges,key=lambda r:r['relationship_id']),memberships=[dict(customer_id=k,household_id=members[k])for k in sorted(members)],liability_paths=sorted(paths,key=lambda r:(r['customer_id'],r['facility_id'],r['role']))))
    return _result(value,'retail_features.build_customer_household_graph','typed_graph',ctx,diagnostics=dict(excluded_releases=sorted(excluded,key=lambda r:(r['record_id'],r['release_seq']))),limitations=('Identity is supplied through typed keys, never inferred from address or name. A household is economic context, not an automatic legal default unit. Legal liability fractions can each equal one for joint borrowers; internal budget shares cannot be added as legal liability.',))


def aggregate_household_obligations(obligations,household_graph,coverage,*,currency,period,guarantee_budget_policy,context=None):
    """Aggregate each facility once per household with separate legal and budget views."""
    graph=_artifact(household_graph,'RETAIL_RELATIONSHIP_GRAPH');anchor,cutoff,ctx=_cutoffs(graph['as_of'],graph['information_cutoff'],context);coverage=_coverage(coverage,cutoff);_text(currency,'currency');_text(period,'period')
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Context currency differs from obligations')
    if guarantee_budget_policy not in ('exclude','include_declared_share'):_fail('RETAIL_LIABILITY','Explicit guarantee budget inclusion policy required')
    rows,excluded=_releases(obligations,'facility_id',cutoff,'obligations');facilities={r['entity_id']for r in graph['entities']if r['entity_type']=='FACILITY'};values={};lineage=[]
    for row in rows:
        _required(row,['source_id','observed_at','currency','period','status'])
        if any(k not in row for k in ('debt_stock','scheduled_debt_service')):_fail('RETAIL_CONTRACT','Debt and service require explicit value or null')
        observed=_date(row['observed_at'],'observed_at')
        if observed>_date(row['known_at'],'known_at'):_fail('RETAIL_DATE','Historical obligation snapshot cannot be known before observation')
        if observed>anchor:continue
        if row['facility_id']not in facilities:_fail('RETAIL_ENTITY','Obligation has no visible facility in supplied graph')
        if row['currency']!=currency or row['period']!=period:_fail('TARGET_MISMATCH','Obligation currency and debt-service period must match')
        if row['source_id']not in coverage['sources']:_fail('RETAIL_COVERAGE','Obligation source not declared in coverage')
        if row['status']not in ('ACTIVE','CLOSED','UNKNOWN'):_fail('RETAIL_CONTRACT','Declare obligation ACTIVE/CLOSED/UNKNOWN status')
        for key in ('debt_stock','scheduled_debt_service'):
            if row[key]is not None:_number(row[key],key,0)
            if row['status']=='CLOSED'and row[key]!=0:_fail('RETAIL_CONTRACT','Closed obligations require explicit zero residual amounts')
        values[row['facility_id']]=row;lineage.append(dict(facility_id=row['facility_id'],release_seq=row['release_seq'],source_id=row['source_id'],age_days=(anchor-observed).days))
    outputs=[];person_rows=[]
    for node in sorted(graph['entities'],key=lambda r:r['entity_id']):
        if node['entity_type']!='HOUSEHOLD':continue
        hid=node['entity_id'];members=[r['customer_id']for r in graph['memberships']if r['household_id']==hid];paths=[r for r in graph['liability_paths']if r['household_id']==hid];ids=sorted({r['facility_id']for r in paths if r['role']=='BORROWER_ON' or guarantee_budget_policy=='include_declared_share'});known_stock=[];known_service=[];budget_stock=[];budget_service=[];uncertain=[];uncertain_stock=[];uncertain_service=[];unknown_budget=[]
        for fid in ids:
            row=values.get(fid);selected=[r for r in paths if r['facility_id']==fid and (r['role']=='BORROWER_ON' or guarantee_budget_policy=='include_declared_share')]
            if row is None or row['status']=='UNKNOWN' or row['debt_stock']is None:uncertain_stock.append(fid)
            if row is None or row['status']=='UNKNOWN' or row['scheduled_debt_service']is None:uncertain_service.append(fid)
            if fid in uncertain_stock or fid in uncertain_service:uncertain.append(fid)
            if row is not None:
                if row['debt_stock']is not None:known_stock.append(row['debt_stock'])
                if row['scheduled_debt_service']is not None:known_service.append(row['scheduled_debt_service'])
            if row is None or any(r['budget_share']is None for r in selected):unknown_budget.append(fid);continue
            share=_sum(r['budget_share']for r in selected)
            if share>1:_fail('DOUBLE_COUNTING','Selected borrower/guarantee budget shares exceed one for household facility')
            if row['debt_stock']is not None:budget_stock.append(float(_multiply(share,row['debt_stock'])))
            if row['scheduled_debt_service']is not None:budget_service.append(float(_multiply(share,row['scheduled_debt_service'])))
        stock_complete=coverage['status']=='COMPLETE'and not uncertain_stock;service_complete=coverage['status']=='COMPLETE'and not uncertain_service
        outputs.append(dict(household_id=hid,customer_ids=sorted(members),facility_ids=ids,known_linked_debt_stock=_sum(known_stock),known_linked_debt_service=_sum(known_service),total_linked_debt_stock=_sum(known_stock)if stock_complete else None,total_linked_debt_service=_sum(known_service)if service_complete else None,known_budget_debt_stock=_sum(budget_stock),known_budget_debt_service=_sum(budget_service),total_budget_debt_stock=_sum(budget_stock)if stock_complete and not unknown_budget else None,total_budget_debt_service=_sum(budget_service)if service_complete and not unknown_budget else None,uncertain_facility_ids=uncertain,uncertain_stock_facility_ids=uncertain_stock,uncertain_service_facility_ids=uncertain_service,unknown_budget_share_facility_ids=unknown_budget,coverage_status=coverage['status']))
    for path in graph['liability_paths']:
        row=values.get(path['facility_id']);fraction=path['liability_fraction'];person_rows.append(dict(customer_id=path['customer_id'],facility_id=path['facility_id'],role=path['role'],legal_debt_stock=None if row is None or row['debt_stock']is None or fraction is None else float(_multiply(row['debt_stock'],fraction)),legal_debt_service=None if row is None or row['scheduled_debt_service']is None or fraction is None else float(_multiply(row['scheduled_debt_service'],fraction))))
    return _result(dict(households=outputs,person_legal_liabilities=person_rows,currency=currency,period=period,coverage=coverage,guarantee_budget_policy=guarantee_budget_policy),'retail_features.aggregate_household_obligations',currency,ctx,diagnostics=dict(graph_hash=graph['artifact_hash'],obligation_lineage=lineage,excluded_releases=excluded),limitations=('Each linked facility is counted once per household, irrespective of the number of borrowers. Household linked debt, internal budget allocation and person-level legal liability are different views and must not be summed together. Unknown or partial source coverage leaves total debt unknown even when the known subtotal is zero.',))


def _mapped_rows(records,mapping,fields,name):
    if not isinstance(mapping,dict) or any(k not in fields or not isinstance(v,str)or not v for k,v in mapping.items()):_fail('RETAIL_MAPPING','Mapping keys must be supported canonical fields and values source field names')
    rows=_rows(records,name);out=[]
    for row in rows:
        if any(mapping.get(k,k)not in row for k in fields):_fail('RETAIL_MAPPING','Missing mapped '+name+' field; nullable values must be explicit')
        out.append({k:row[mapping.get(k,k)]for k in fields})
    return out


def build_bureau_features(snapshots,tradelines,inquiries,provider_mapping,*,customer_id,provider_id,as_of,information_cutoff,lookback_days,max_age_days,currency,context=None):
    """Describe one locally supplied provider file; no-hit and missing evidence differ."""
    anchor,cutoff,ctx=_cutoffs(as_of,information_cutoff,context);_text(customer_id,'customer_id');_text(provider_id,'provider_id');_text(currency,'currency');days=_days(lookback_days,'lookback_days',1);max_age=_days(max_age_days,'max_age_days')
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Context currency differs from bureau reporting currency')
    if not isinstance(provider_mapping,dict):_fail('RETAIL_MAPPING','Dated local provider mapping required')
    _required(provider_mapping,['mapping_id','version','known_at','snapshots','tradelines','inquiries'])
    if _date(provider_mapping['known_at'],'mapping.known_at')>cutoff:_fail('RETAIL_DATE','Provider mapping was unavailable at feature cutoff')
    fields=dict(snapshots=['snapshot_id','release_seq','customer_id','provider_id','observed_at','reported_at','known_at','coverage_status','currency'],tradelines=['snapshot_id','snapshot_release_seq','tradeline_id','opened_at','closed_at','status','balance','limit','max_dpd'],inquiries=['inquiry_id','release_seq','customer_id','provider_id','occurred_at','known_at','kind'])
    snaps=_mapped_rows(snapshots,provider_mapping['snapshots'],fields['snapshots'],'snapshots');lines=_mapped_rows(tradelines,provider_mapping['tradelines'],fields['tradelines'],'tradelines');queries=_mapped_rows(inquiries,provider_mapping['inquiries'],fields['inquiries'],'inquiries')
    selected,excluded=_releases([r for r in snaps if r['customer_id']==customer_id and r['provider_id']==provider_id],'snapshot_id',cutoff,'bureau snapshots');visible=[]
    for row in selected:
        observed,reported,known=(_date(row[k],k)for k in ('observed_at','reported_at','known_at'))
        if not observed<=reported<=known:_fail('RETAIL_DATE','Bureau observation must precede reporting and local receipt')
        if row['coverage_status']not in ('COMPLETE','PARTIAL','NO_HIT') or row['currency']!=currency:_fail('RETAIL_COVERAGE','Bureau coverage or reporting currency invalid')
        if observed<=anchor:visible.append(row)
    snapshot=max(visible,key=lambda r:(r['observed_at'],r['known_at'],r['release_seq'],r['snapshot_id']))if visible else None
    features=dict(reported_tradeline_count=None,open_tradeline_count=None,closed_tradeline_count=None,unknown_status_count=None,history_days=None,known_open_balance=None,total_open_balance=None,known_closed_balance=None,known_open_limit=None,total_open_limit=None,aggregate_open_utilization=None,reported_max_dpd=None,hard_inquiries=None,soft_inquiries=None,unknown_inquiries=None)
    queries,query_excluded=_releases([r for r in queries if r['customer_id']==customer_id and r['provider_id']==provider_id],'inquiry_id',cutoff,'bureau inquiries')
    selected_lines=[];query_ids=[];stale=None;age=None;state='ABSENT';coverage='UNKNOWN'
    if snapshot is not None:
        observed=_date(snapshot['observed_at'],'observed_at');age=(anchor-observed).days;stale=age>max_age;coverage=snapshot['coverage_status'];state='NO_HIT'if coverage=='NO_HIT'else 'STALE'if stale else 'CURRENT'
        selected_lines=[r for r in lines if r['snapshot_id']==snapshot['snapshot_id']and r['snapshot_release_seq']==snapshot['release_seq']];_unique(selected_lines,'tradeline_id','snapshot tradelines')
        if coverage=='NO_HIT'and selected_lines:_fail('RETAIL_COVERAGE','No-hit snapshot cannot contain reported tradelines')
        opened=[];opens=[];closed=[];unknown=[];dpd=[]
        for line in selected_lines:
            start=_date(line['opened_at'],'opened_at');end=None if line['closed_at']is None else _date(line['closed_at'],'closed_at')
            if start>observed or (end is not None and not start<=end<=observed):_fail('RETAIL_DATE','Tradeline events cannot follow their historical snapshot')
            if line['status']not in ('OPEN','CLOSED','UNKNOWN') or (line['status']=='CLOSED')!=(end is not None):_fail('RETAIL_COVERAGE','Actual closure date and reported tradeline state must agree')
            for key in ('balance','limit','max_dpd'):
                if line[key]is not None:_number(line[key],key,0)
            if line['max_dpd']is not None:_days(line['max_dpd'],'max_dpd');dpd.append(line['max_dpd'])
            opened.append(start);(opens if line['status']=='OPEN'else closed if line['status']=='CLOSED'else unknown).append(line)
        _unique(queries,'inquiry_id','inquiries');counts={'HARD':0,'SOFT':0,'UNKNOWN':0}
        for row in queries:
            if row['customer_id']!=customer_id or row['provider_id']!=provider_id:continue
            occurred,known=(_date(row[k],k)for k in ('occurred_at','known_at'))
            if known<occurred or row['kind']not in counts:_fail('RETAIL_DATE','Inquiry type or receipt chronology invalid')
            if known<=cutoff and 0<=(anchor-occurred).days<days:counts[row['kind']]+=1;query_ids.append(row['inquiry_id'])
        balance=_sum(r['balance']for r in opens if r['balance']is not None);limit=_sum(r['limit']for r in opens if r['limit']is not None);complete_balance=coverage=='COMPLETE'and not unknown and all(r['balance']is not None for r in opens);complete_limit=coverage=='COMPLETE'and not unknown and all(r['limit']is not None for r in opens)
        features=dict(reported_tradeline_count=len(selected_lines),open_tradeline_count=len(opens),closed_tradeline_count=len(closed),unknown_status_count=len(unknown),history_days=None if not opened else (observed-min(opened)).days,known_open_balance=balance,total_open_balance=balance if complete_balance else None,known_closed_balance=_sum(r['balance']for r in closed if r['balance']is not None),known_open_limit=limit,total_open_limit=limit if complete_limit else None,aggregate_open_utilization=balance/limit if complete_balance and complete_limit and limit>0 else None,reported_max_dpd=max(dpd)if dpd else None,hard_inquiries=counts['HARD'],soft_inquiries=counts['SOFT'],unknown_inquiries=counts['UNKNOWN'])
        if features['aggregate_open_utilization']is not None and not math.isfinite(features['aggregate_open_utilization']):_fail('NUMERICAL_FAILURE','Bureau utilization outside finite range')
    return _result(dict(customer_id=customer_id,provider_id=provider_id,as_of=as_of,information_cutoff=information_cutoff,currency=currency,file_state=state,coverage_status=coverage,age_days=age,is_stale=stale,features=features),'retail_features.build_bureau_features','retail_features',ctx,diagnostics=dict(mapping_id=provider_mapping['mapping_id'],mapping_version=provider_mapping['version'],mapping_fingerprint=_fingerprint(provider_mapping),snapshot_id=None if snapshot is None else snapshot['snapshot_id'],snapshot_release_seq=None if snapshot is None else snapshot['release_seq'],tradeline_ids=sorted(r['tradeline_id']for r in selected_lines),inquiry_ids=sorted(query_ids),excluded_snapshot_releases=excluded,excluded_inquiry_releases=query_excluded),limitations=('One provider file is described without reproducing a proprietary score. Tradeline fields inherit the selected snapshot receipt time. No-hit, absent and stale files are evidence states, not good/bad credit labels. Reported closed balances remain visible and are not assumed forgiven; open-file totals do not establish all household debt. Inquiries use the supplied local history and the interval (as_of-lookback_days,as_of].',))


def build_thin_file_features(bureau_features,evidence,eligibility_profile,*,customer_id,as_of,information_cutoff,context=None):
    """Classify availability and supported feature sets without assigning credit quality."""
    anchor,cutoff,ctx=_cutoffs(as_of,information_cutoff,context);_text(customer_id,'customer_id')
    if isinstance(bureau_features,CalcResult):
        if bureau_features.method_id!='retail_features.build_bureau_features':_fail('RETAIL_EVIDENCE','Expected bureau feature result')
        bureau_features=bureau_features.value
    if not isinstance(bureau_features,dict) or any(bureau_features.get(k)!=v for k,v in [('customer_id',customer_id),('as_of',as_of),('information_cutoff',information_cutoff)]):_fail('RETAIL_EVIDENCE','Bureau customer and information snapshot must match')
    profile=eligibility_profile
    if not isinstance(profile,dict):_fail('RETAIL_CONTRACT','Explicit dated feature eligibility profile required')
    _required(profile,['profile_id','version','known_at','minimum_tradelines','minimum_history_days','maximum_evidence_age_days','allow_partial_evidence','feature_requirements'])
    if _date(profile['known_at'],'profile.known_at')>cutoff or type(profile['allow_partial_evidence'])is not bool:_fail('RETAIL_DATE','Eligibility profile unavailable or partial-evidence policy invalid')
    minimum=_days(profile['minimum_tradelines'],'minimum_tradelines');history=_days(profile['minimum_history_days'],'minimum_history_days');max_age=_days(profile['maximum_evidence_age_days'],'maximum_evidence_age_days')
    state=bureau_features.get('file_state');features=bureau_features.get('features',{})
    if not isinstance(features,dict) or bureau_features.get('coverage_status')not in ('UNKNOWN','COMPLETE','PARTIAL','NO_HIT'):_fail('RETAIL_EVIDENCE','Bureau feature dictionary and coverage required')
    for field in ('reported_tradeline_count','history_days'):
        if features.get(field)is not None:_days(features[field],field)
    if state not in ('ABSENT','NO_HIT','STALE','CURRENT'):_fail('RETAIL_EVIDENCE','Unsupported bureau evidence state')
    stratum='BUREAU_ABSENT'if state=='ABSENT'else 'BUREAU_NO_HIT'if state=='NO_HIT'else 'BUREAU_STALE'if state=='STALE'else 'THIN'if features.get('reported_tradeline_count')is None or features['reported_tradeline_count']<minimum or features.get('history_days')is None or features['history_days']<history else 'ESTABLISHED'
    available={'BUREAU':state=='CURRENT'and (bureau_features['coverage_status']=='COMPLETE'or (profile['allow_partial_evidence']and bureau_features['coverage_status']=='PARTIAL'))};details=[]
    rows,excluded=_releases(evidence,'evidence_id',cutoff,'financial evidence');bytype={}
    for row in rows:
        _required(row,['customer_id','evidence_type','observed_from','observed_until','coverage_status']);start=_date(row['observed_from'],'observed_from');end=_date(row['observed_until'],'observed_until')
        if not start<=end<=_date(row['known_at'],'known_at') or end>anchor:_fail('RETAIL_DATE','Evidence observation interval or availability invalid')
        if row['customer_id']!=customer_id:continue
        kind=_text(row['evidence_type'],'evidence_type')
        if kind=='BUREAU' or row['coverage_status']not in ('COMPLETE','PARTIAL','UNKNOWN'):_fail('RETAIL_EVIDENCE','Other evidence types must be distinct from bureau and carry coverage')
        bytype.setdefault(kind,[]).append(row)
    for kind in sorted(bytype):
        row=max(bytype[kind],key=lambda r:(r['observed_until'],r['known_at'],r['release_seq'],r['evidence_id']));age=(anchor-_date(row['observed_until'],'observed_until')).days;usable=age<=max_age and (row['coverage_status']=='COMPLETE'or(profile['allow_partial_evidence']and row['coverage_status']=='PARTIAL'));available[kind]=usable;details.append(dict(evidence_type=kind,evidence_id=row['evidence_id'],available=usable,coverage_status=row['coverage_status'],age_days=age,history_days=(_date(row['observed_until'],'observed_until')-_date(row['observed_from'],'observed_from')).days))
    requirements=profile['feature_requirements']
    if not isinstance(requirements,dict):_fail('RETAIL_CONTRACT','feature_requirements maps feature names to required evidence types')
    supported=[];unsupported=[]
    for feature in sorted(requirements):
        _text(feature,'feature')
        if feature.lower()in ('target','outcome','label','default_in_horizon','future_default','lgd_target','recovery_target'):_fail('RETAIL_LEAKAGE','Outcome labels cannot be eligible input features')
        needs=_strings(requirements[feature],'required evidence types',False);missing=[k for k in needs if not available.get(k,False)]
        if missing:unsupported.append(dict(feature=feature,unavailable_evidence=missing))
        else:supported.append(feature)
    return _result(dict(customer_id=customer_id,as_of=as_of,information_cutoff=information_cutoff,thin_file_stratum=stratum,evidence_availability={k:available[k]for k in sorted(available)},supported_features=supported,unsupported_features=unsupported,evidence_details=details),'retail_features.build_thin_file_features','evidence_status',ctx,diagnostics=dict(profile_id=profile['profile_id'],profile_version=profile['version'],profile_fingerprint=_fingerprint(profile),excluded_releases=excluded),limitations=('Thin-file strata describe evidence, not default classes. Eligibility thresholds and required sources are supplied policy; availability alone does not establish predictive value. Current partial bureau evidence can support features only when the explicit profile permits it.',))


def reconcile_income_evidence(declarations,documentary_evidence,transaction_evidence,inclusion_policy,*,customer_id,as_of,information_cutoff,currency,period,basis,context=None):
    """Reconcile the same income component across sources without summing duplicates."""
    anchor,cutoff,ctx=_cutoffs(as_of,information_cutoff,context);_text(customer_id,'customer_id');_text(currency,'currency');_text(period,'period')
    if ctx.get('currency')is not None and ctx['currency']!=currency:_fail('TARGET_MISMATCH','Context currency differs from income reporting currency')
    if basis not in ('NET','GROSS'):_fail('RETAIL_INCOME','Explicit NET or GROSS target basis required')
    policy=inclusion_policy
    if not isinstance(policy,dict):_fail('RETAIL_CONTRACT','Explicit dated income inclusion policy required')
    _required(policy,['policy_id','version','known_at','source_priority','included_component_types','expected_component_ids','coverage_status','maximum_age_days','allow_stale','relative_discrepancy_threshold'])
    if _date(policy['known_at'],'policy.known_at')>cutoff or type(policy['allow_stale'])is not bool:_fail('RETAIL_DATE','Income policy unavailable or stale-evidence policy invalid')
    priority=_strings(policy['source_priority'],'source_priority',False)
    if set(priority)!={'DECLARED','DOCUMENT','TRANSACTION'}:_fail('RETAIL_INCOME','Prioritize each declared/document/transaction source once')
    included=_strings(policy['included_component_types'],'included_component_types');allowed={'SALARY','BONUS','PENSION','RENT_NET','NET_BUSINESS_INCOME','OTHER_NET_INCOME'}
    if not set(included)<=allowed or policy['coverage_status']not in ('COMPLETE','PARTIAL','UNKNOWN'):_fail('RETAIL_INCOME','Explicit income component types and source coverage required; revenue is not income')
    expected=_strings(policy['expected_component_ids'],'expected_component_ids');max_age=_days(policy['maximum_age_days'],'maximum_age_days');threshold=_number(policy['relative_discrepancy_threshold'],'relative_discrepancy_threshold',0);combined=[]
    for source,records in [('DECLARED',declarations),('DOCUMENT',documentary_evidence),('TRANSACTION',transaction_evidence)]:
        for row in _rows(records,source):combined.append(dict(row,source_type=source))
    rows,excluded=_releases(combined,'evidence_id',cutoff,'income evidence');components={};ineligible=[]
    for row in rows:
        _required(row,['customer_id','component_id','component_type','observed_from','observed_until','amount','basis','period','currency','evidence_reference']);_text(row['evidence_reference'],'evidence_reference');start=_date(row['observed_from'],'observed_from');end=_date(row['observed_until'],'observed_until')
        if not start<=end<=_date(row['known_at'],'known_at') or end>anchor:_fail('RETAIL_DATE','Income observation interval or receipt chronology invalid')
        if row['customer_id']!=customer_id:continue
        _number(row['amount'],'income amount',0)
        if row['basis']not in ('NET','GROSS'):_fail('RETAIL_INCOME','Every evidence amount requires NET/GROSS basis')
        component=_text(row['component_id'],'component_id');_text(row['component_type'],'component_type');components.setdefault(component,[]).append(row)
        reasons=[]
        if row['component_type']not in included:reasons.append('COMPONENT_TYPE_EXCLUDED')
        if row['basis']!=basis:reasons.append('BASIS_MISMATCH')
        if row['period']!=period:reasons.append('PERIOD_MISMATCH')
        if row['currency']!=currency:reasons.append('CURRENCY_MISMATCH')
        if (anchor-end).days>max_age and not policy['allow_stale']:reasons.append('STALE')
        if reasons:ineligible.append(dict(evidence_id=row['evidence_id'],reasons=reasons))
    banned={r['evidence_id']for r in ineligible};outputs=[];chosen=[];verified=[];uncertain=[];missing=[]
    # Excluded component types remain visible but are not implicitly expected income.
    ids=sorted(set(expected)|{k for k,v in components.items()if any(r['component_type']in included for r in v)})
    for cid in ids:
        records=components.get(cid,[]);types={r['component_type']for r in records}
        if len(types)>1:_fail('RETAIL_INCOME','One component identity cannot mix salary, bonus or other income kinds')
        eligible=[r for r in records if r['evidence_id']not in banned];sources={}
        for source in priority:
            candidates=[r for r in eligible if r['source_type']==source]
            if candidates:sources[source]=max(candidates,key=lambda r:(r['observed_until'],r['known_at'],r['release_seq'],r['evidence_id']))
        selected=next((sources[k]for k in priority if k in sources),None);amount=None if selected is None else selected['amount'];stale=None if selected is None else (anchor-_date(selected['observed_until'],'observed_until')).days>max_age
        if selected is None:missing.append(cid)
        else:
            chosen.append(amount);(verified if selected['source_type']=='DOCUMENT'and not stale else uncertain).append(amount)
        declared=sources.get('DECLARED');reference=None if declared is None else declared['amount'];difference=None if amount is None or reference is None else _sum((amount,-reference));relative=None if difference is None or reference==0 else difference/reference
        if relative is not None and not math.isfinite(relative):_fail('NUMERICAL_FAILURE','Income discrepancy ratio outside finite range')
        outputs.append(dict(component_id=cid,component_type=next(iter(types))if types else None,selected_amount=amount,selected_source=None if selected is None else selected['source_type'],selected_evidence_id=None if selected is None else selected['evidence_id'],is_stale=stale,source_amounts={k:sources[k]['amount']if k in sources else None for k in priority},declared_difference=difference,declared_relative_difference=relative,discrepancy_status='MISSING_REFERENCE'if reference is None else 'ZERO_REFERENCE'if reference==0 else 'AVAILABLE',discrepancy_flag=None if relative is None else abs(relative)>threshold))
    subtotal=_sum(chosen);complete=policy['coverage_status']=='COMPLETE'and not missing
    return _result(dict(customer_id=customer_id,as_of=as_of,information_cutoff=information_cutoff,currency=currency,period=period,basis=basis,components=outputs,known_selected_income=subtotal,total_selected_income=subtotal if complete else None,document_supported_current_income=_sum(verified),uncertain_or_non_document_income=_sum(uncertain),missing_component_ids=missing,coverage_status=policy['coverage_status']),'retail_features.reconcile_income_evidence',currency,ctx,diagnostics=dict(policy_id=policy['policy_id'],policy_version=policy['version'],policy_fingerprint=_fingerprint(policy),ineligible_evidence=sorted(ineligible,key=lambda r:r['evidence_id']),excluded_releases=excluded),limitations=('Each stable income component is selected once according to supplied source priority; salary and bonus need distinct component identities. Document-supported means the user supplied documentary evidence, not independent authenticity verification. Transaction evidence must already represent the declared income basis; borrowing, owner transfers, refunds and business revenue are not inferred to be disposable income. Basis, period and currency mismatches are not silently converted.',))


def build_payment_history_features(obligations,payments,coverage,*,account_id,as_of,information_cutoff,lookback_days,allocation_policy,currency,grace_days,context=None):
    """Replay historical contractual payments with contemporaneous knowledge."""
    from ._retail_behavior import build_payment_history_features as calculate
    return calculate(obligations,payments,coverage,account_id=account_id,as_of=as_of,information_cutoff=information_cutoff,lookback_days=lookback_days,allocation_policy=allocation_policy,currency=currency,grace_days=grace_days,context=context)


def build_revolving_features(balance_history,statements,coverage,*,account_id,as_of,information_cutoff,lookback_days,currency,context=None):
    """Separate observed utilization, balance/limit changes and repayment states."""
    from ._retail_behavior import build_revolving_features as calculate
    return calculate(balance_history,statements,coverage,account_id=account_id,as_of=as_of,information_cutoff=information_cutoff,lookback_days=lookback_days,currency=currency,context=context)


def build_transaction_features(transactions,daily_balances,category_mapping,coverage,*,customer_id,own_account_ids,as_of,information_cutoff,lookback_days,currency,context=None):
    """Describe dated booked cashflows, transfers, category uncertainty and buffers."""
    from ._retail_cashflows import build_transaction_features as calculate
    return calculate(transactions,daily_balances,category_mapping,coverage,customer_id=customer_id,own_account_ids=own_account_ids,as_of=as_of,information_cutoff=information_cutoff,lookback_days=lookback_days,currency=currency,context=context)


def build_bnpl_obligation_features(plans,installments,events,provider_coverage,*,customer_id,as_of,information_cutoff,lookback_days,near_term_days,currency,context=None):
    """Aggregate explicit BNPL installments, refunds, disputes and concurrent plans."""
    from ._retail_obligations import build_bnpl_obligation_features as calculate
    return calculate(plans,installments,events,provider_coverage,customer_id=customer_id,as_of=as_of,information_cutoff=information_cutoff,lookback_days=lookback_days,near_term_days=near_term_days,currency=currency,context=context)


def build_retail_sme_features(business_cashflows,owner_drawings,household_obligations,liability_links,coverage,*,business_id,owner_customer_ids,business_debt_facility_ids,household_debt_service_window,as_of,information_cutoff,lookback_days,currency,context=None):
    """Reconcile business cash, owner transfers and explicitly linked household debt."""
    from ._retail_sme import build_retail_sme_features as calculate
    return calculate(business_cashflows,owner_drawings,household_obligations,liability_links,coverage,business_id=business_id,owner_customer_ids=owner_customer_ids,business_debt_facility_ids=business_debt_facility_ids,household_debt_service_window=household_debt_service_window,as_of=as_of,information_cutoff=information_cutoff,lookback_days=lookback_days,currency=currency,context=context)


def fit_retail_risk_pools(observations,pooling_spec,*,target,training,model_id,model_version,context=None):
    """Fit frozen segment-specific score pools with adjacent merge diagnostics."""
    from ._retail_pools import fit_retail_risk_pools as fit
    return fit(observations,pooling_spec,target=target,training=training,model_id=model_id,model_version=model_version,context=context)


def assign_retail_risk_pool(observations,fitted_model,*,as_of,information_cutoff,expected_target,uncovered_policy,context=None):
    """Replay frozen retail pool rules with explicit uncovered-profile handling."""
    from ._retail_pools import assign_retail_risk_pool as assign
    return assign(observations,fitted_model,as_of=as_of,information_cutoff=information_cutoff,expected_target=expected_target,uncovered_policy=uncovered_policy,context=context)


def estimate_pool_risk_parameters(pool_assignments,cohort_outcomes,workouts,estimation_profile,*,evaluation_cutoff,calibration_id,calibration_version,context=None):
    """Estimate mature-cohort and default-conditioned local pool risk parameters."""
    from ._retail_pools import estimate_pool_risk_parameters as estimate
    return estimate(pool_assignments,cohort_outcomes,workouts,estimation_profile,evaluation_cutoff=evaluation_cutoff,calibration_id=calibration_id,calibration_version=calibration_version,context=context)
