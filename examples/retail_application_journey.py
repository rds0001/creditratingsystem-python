"""Dated retail applications become their own funded accounts and observed outcomes."""
from copy import deepcopy
from datetime import date,timedelta
from importlib.resources import files
import json
import math
import calendar
import creditratingsystem as cr


def fingerprint(value):return cr.content_fingerprint(value).value


def load_sources(data,applications_path=None,releases_path=None):
    resource=files('creditratingsystem.datasets')
    a=cr.read_local_dataset(applications_path or resource.joinpath('journey_applications.csv'),data['left_schema'])
    r=cr.read_local_dataset(releases_path or resource.joinpath('journey_releases.csv'),data['right_schema'])
    control=data['source_controls']
    if sorted(x['record_id']for x in a.value)!=sorted(control['record_ids'])or len(a.value)!=control['total_count']or sum(x['balance']for x in a.value)!=control['total_requested_amount']:raise ValueError('Application archive does not reconcile to supplied independent control')
    joined=cr.join_as_of(a.value,r.value).value
    if any(x['release_id']is None for x in joined):raise ValueError('No source evidence available at application cutoff')
    audits=[]
    for cutoff in sorted({x['as_of']for x in joined}):
        left=[x for x in a.value if x['as_of']==cutoff];out=[x for x in joined if x['as_of']==cutoff];ids={x['release_id']for x in out};right=[x for x in r.value if x['release_id']in ids]
        def node(identity,kind,rows,source):
            records=[dict(record_id=x['release_id']if source=='release'else x['record_id'],join_key=x['entity_id'],weight=1,amounts=dict(amount=0 if source=='release'else x['balance']))for x in rows]
            observed=max(x['observed_at']if source=='release'else x['as_of']for x in rows);known=max(x['available_at']if source=='release'else x['known_at']for x in rows)
            n=dict(node_id=identity,kind=kind,schema_version='1',vintage=cutoff,revision='original',transformation_version='as-of-1',content_hash=fingerprint(rows),observed_at=observed,available_at=known,records=records)
            if kind=='SOURCE':n.update(source_id=identity,semantic_role='FEATURE')
            return n
        nodes=[node('applications','SOURCE',left,'app'),node('releases','SOURCE',right,'release'),node('joined','FEATURE',out,'app')]
        refs=[dict(reconciliation_id='control-'+n['node_id'],node_id=n['node_id'],reference='supplied-through-door-control',expected_record_ids=[x['record_id']for x in n['records']],expected_total_weight=len(left),expected_amounts=dict(amount=0 if n['node_id']=='releases'else sum(x['balance']for x in left)))for n in nodes]
        join=dict(join_id='asof',left_node_id='applications',right_node_id='releases',output_node_id='joined',cardinality='one_to_one',join_type='left',weight_contract='LEFT_WEIGHT_ALLOCATION',amount_origin='LEFT',mapping=[dict(output_id=x['record_id'],left_id=x['record_id'],right_id=x['release_id'],allocation_weight=1)for x in out],excluded_left_ids=[],unmatched_right_ids=[])
        contract=dict(contract_id='application-source-control',version='1',reference='declared-date-specific-control',amount_fields=dict(amount='EUR'),absolute_tolerance=0,relative_tolerance=0,required_node_ids=['applications','releases','joined'],required_use_ids=['application'],required_join_ids=['asof'])
        arguments=dict(nodes=nodes,edges=[dict(parent_id=n,child_id='joined')for n in ['applications','releases']],joins=[join],reconciliations=refs,uses=[dict(use_id='application',node_id='joined',purpose='FEATURE',as_of=cutoff,known_at=cutoff,source_snapshot_hashes={n['node_id']:n['content_hash']for n in nodes[:2]})],contract=contract)
        audits.append(dict(cutoff=cutoff,inputs=arguments,result=cr.validate_source_lineage(**arguments).value))
    return dict(rows=joined,audits=audits,archive_hashes=dict(applications=a.provenance['source_sha256'],releases=r.provenance['source_sha256']))


def fit_product_models(product,c,target):
    """Fit actual prior cells; references do not substitute for the estimators."""
    def checked(role):
        rows=c[role]
        for r in rows:
            if not r['feature_observed_at']<=r['feature_known_at']<r['application_at']<=r['first_draw_at']<r['outcome_end']<=r['outcome_known_at']:raise ValueError('Prior cohort chronology is invalid')
            event=r['default_at'];observed=int(event is not None and r['first_draw_at']<event<=r['outcome_end'])
            if observed!=r['outcome']or(date.fromisoformat(r['outcome_end'])-date.fromisoformat(r['first_draw_at'])).days!=target['horizon']:raise ValueError('Prior outcome does not reconcile to first-default window')
        return rows
    train=checked('training');cal=checked('calibration')
    if {r['customer_id']for r in train}&{r['customer_id']for r in cal}:raise ValueError('Training and calibration customer overlap')
    known=max(r['outcome_known_at']for r in train)
    if known>=min(r['application_at']for r in cal):raise ValueError('Training not available before calibration')
    model=cr.fit_binary_glm([[r['driver']]for r in train],[r['outcome']for r in train],feature_order=['driver'],link='logit',fit_intercept=True,target=target,training=dict(known_at=known,data_fingerprint=fingerprint(train),split_id=product+'-prior-training',data_origin='SYNTHETIC'),model_id=product+'-application',model_version='1').model
    model['application_contract']=dict(product_ids=[product],population_ids=[product+'-through-door'],performance_origin='first_draw_at',term_stage='requested',term_features={})
    raw=list(cr.predict_pd(model,[[r['driver']]for r in cal],feature_order=['driver'],expected_target=target).value)
    metadata=dict(target=target,calibration=dict(start=min(r['application_at']for r in cal),end=max(r['outcome_end']for r in cal),known_at=max(r['outcome_known_at']for r in cal),split_id=product+'-prior-calibration',prediction_origin='HOLDOUT'),base_model_version='1',population_definition=product+' funded prior applicants',outcome_policy='First observed default in30days from first draw',weight_type='frequency',data_origin='SYNTHETIC')
    calibrator=cr.calibrate_logistic(raw,[r['outcome']for r in cal],metadata=metadata).value
    oracle=[]
    for x in [-1,1]:
        cell=[r for r in train if r['driver']==x];oracle.append(sum(r['outcome']for r in cell)/len(cell))
    estimates=list(cr.predict_pd(model,[[-1],[1]],feature_order=['driver'],expected_target=target).value)
    if any(abs(a-b)>1e-8 for a,b in zip(estimates,oracle)):raise ValueError('Fitted cell probabilities differ from independent binomial MLE')
    return dict(model=model,calibrator=calibrator,calibration_inputs=dict(values=raw,outcomes=[r['outcome']for r in cal]),training_rows=train,calibration_rows=cal,independent_cell_mle=oracle,fit_predictions=estimates)


def payment_contract(terms):
    kind=terms['kind']
    if kind=='LEVEL_PRINCIPAL':return [terms['principal']/terms['periods']]*terms['periods']
    if kind=='BALLOON':return [(terms['principal']-terms['balloon'])/terms['periods']]*(terms['periods']-1)+[(terms['principal']-terms['balloon'])/terms['periods']+terms['balloon']]
    if kind=='OPERATING_RENT':
        if terms['customer_residual_liability']!=0:raise ValueError('Operating rent contract cannot silently include a residual purchase obligation')
        return [terms['rent']]*12
    if kind=='GRACE':return [terms['grace_payment']]*2+[terms['post_grace_payment']]*8
    if kind=='OVERLAPPING_PLANS':return [sum(terms['plan_payments'])]*terms['periods']
    if kind=='WEEKLY':return terms['weekly_dues']
    raise ValueError('Unsupported payment contract')


def capacity(app,household,terms,source):
    key=app['application_id'];D=app['application_at'];prior=household['source_date'];income=source['income']
    if income!=household['monthly_income']:raise ValueError('Budget must use the historically selected income evidence')
    component_type=household.get('income_component_type','SALARY')
    evidence=dict(release_seq=1,evidence_id='payroll-'+key,customer_id=app['customer_id'],component_id='salary-'+key,component_type=component_type,observed_from=prior,observed_until=prior,known_at=prior,amount=income,basis='NET',period='month',currency='EUR',evidence_reference=source['release_id'])
    policy=dict(policy_id='verified-income',version='1',known_at='2024-01-01',source_priority=['DOCUMENT','TRANSACTION','DECLARED'],included_component_types=[component_type],expected_component_ids=[evidence['component_id']],coverage_status='COMPLETE',maximum_age_days=31,allow_stale=False,relative_discrepancy_threshold=.1)
    reconciled=cr.reconcile_income_evidence([], [evidence], [],policy,customer_id=app['customer_id'],as_of=D,information_cutoff=D,currency='EUR',period='month',basis='NET').value
    norm=cr.normalize_income_period([dict(economic_id=evidence['component_id'],amount=income,frequency='month',basis='net',currency='EUR',kind='recurring')],target_period='month',target_basis='net',currency='EUR',payments_per_year=dict(month=12,year=1,week=52),variable_income_factor=1).value
    expense=cr.estimate_recurring_expenses([dict(economic_id='living-'+key,category='living',kind='essential',amount=household['monthly_expense'],frequency='month',currency='EUR')],target_period='month',currency='EUR',payments_per_year=dict(month=12,year=1),household_profile=dict(profile_id='living-cost',required_categories=['living'])).value
    schedule=payment_contract(terms)
    if terms['kind']=='WEEKLY':
        residual=cr.residual_income(terms['weekly_income']if income>100 else [income/4]*4,terms['weekly_expenses'],[0]*4,schedule,period='weekly',currency='EUR').value
    else:residual=cr.residual_income([income]*len(schedule),[household['monthly_expense']]*len(schedule),[0]*len(schedule),schedule,period='monthly',currency='EUR').value
    dsti=cr.debt_service_to_income(schedule,(terms['weekly_income']if income>100 else [income/4]*4)if terms['kind']=='WEEKLY'else [income]*len(schedule),period='weekly'if terms['kind']=='WEEKLY'else'monthly',currency='EUR').value
    # A monthly projection of one explicitly supplied scheduled payment; weekly
    # and grace/balloon profiles above remain separate full contractual checks.
    payment=sum(schedule[:4])if terms['kind']=='WEEKLY'else schedule[0]
    baseline=dict(currency='EUR',as_of=D,liquid_buffer=0,cltv_scope='not_applicable',incomes=[dict(economic_id='income-'+key,currency='EUR',monthly_amount=income,basis='net')],expenses=[dict(economic_id='living-'+key,currency='EUR',monthly_amount=household['monthly_expense']),dict(economic_id='payment-'+key,currency='EUR',monthly_amount=payment)],contracts=[],collateral=[],thresholds=dict(minimum_residual=0,maximum_dsti=1,maximum_cltv=1))
    dt=date.fromisoformat(D);year=dt.year+(dt.month==12);month=dt.month%12+1;nextdate=date(year,month,min(dt.day,calendar.monthrange(year,month)[1])).isoformat()
    scenario=dict(schema_version='1.0',scenario_id='verified-budget-income-shock',periods=[dict(date=nextdate,fx_to_reporting={'EUR':1},income_factors={'income-'+key:.9},expense_factors={'living-'+key:1,'payment-'+key:1},collateral_factors={},rate_shifts={})])
    stress=cr.stress_affordability(baseline,scenario).value
    return dict(dsti=dsti,income_evidence=reconciled,normalized_income=norm,expenses=expense,schedule=schedule,residual=residual,baseline=baseline,scenario=scenario,stress=stress,status='AFFORDABLE'if residual['minimum']>=0 else'NOT_AFFORDABLE',projection_scope='One supplied monthly contractual cash payment is an expense; full weekly/grace/balloon schedule is checked separately. No refinancing or rate reset is inferred.')


def policy_spec(product,target):
    target={k:v for k,v in target.items()if k!='performance_origin'}
    return dict(schema_version='1.1',policy_id=product+'-credit-policy',version='1',approval_reference='independently-prespecified-synthetic-capacity-and-risk-policy',known_at='2024-01-01',effective_from='2024-01-01',effective_until='2026-01-01',target=target,default_action='REFER',missing_action='BLOCKED',fields={'risk.pd':dict(type='number'),'affordability.status':dict(type='category',values=['AFFORDABLE','NOT_AFFORDABLE','INSUFFICIENT_EVIDENCE'])},rules=[dict(rule_id='capacity',priority=0,action='DECLINE',reason_code='CAPACITY',when=[dict(field='affordability.status',op='eq',value='NOT_AFFORDABLE')]),dict(rule_id='eligible-risk',priority=1,action='APPROVE',reason_code='DECLARED_PD',when=[dict(field='risk.pd',op='le',value=.8)])],approval_contract=dict(domain='RETAIL_CREDIT',affordability=dict(mode='REQUIRED'),required_gates=[]))


def origination(product,c,target,sources):
    live={a['customer_id']for a in c['applications']}
    if live&{r['customer_id']for role in ['training','calibration']for r in c[role]}:raise ValueError('Journey customers overlap fitted training or calibration')
    fitted=fit_product_models(product,c,target);model=fitted['model'];calibrator=fitted['calibrator'];policy=cr.compile_credit_decision_policy(policy_spec(product,target)).value
    source_by_id={x['record_id']:x for x in sources['rows']};features=[]
    for app in c['applications']:
        source=source_by_id[app['application_id']]
        features.append(dict(application_id=app['application_id'],feature='driver',value=source['driver'],observed_at=source['observed_at'],known_at=source['available_at'],source_id=source['release_id']))
    snapshots=cr.build_application_snapshot(c['applications'],features,required_features=['driver']).value
    labels=cr.build_application_targets(c['applications'],c['decisions'],c['bookings'],c['defaults'],c['exits'],target['horizon'],'2025-02-10').value
    registry=cr.register_rating_model_version(None,dict(model=model,dependencies=dict(calibrator=calibrator),configuration=dict(product_id=product)),dict(model_id=model['model_id'],model_version='1',registered_at='2021-01-01')).value
    mh=registry['entries'][0]['model_hash'];rows=[];requests=[]
    for app,snapshot,historical,h in zip(c['applications'],snapshots,c['decisions'],c['households']):
        D=app['application_at'];K=historical['decided_at'];key=app['application_id'];subject=dict(subject_id=app['customer_id'],subject_type='customer')
        subjects=cr.build_rating_subjects(dict(customer=[dict(subject_id=subject['subject_id'],version='1',effective_from=D,effective_to=None,known_at=D,attributes=dict(product=product))]),[],D,D).value
        rs=dict(subject,as_of=D,known_at=D,attributes=dict(product=dict(value=product,observed_at=D,known_at=D),driver=dict(value=snapshot['features']['driver'],observed_at=source_by_id[key]['observed_at'],known_at=source_by_id[key]['available_at'])))
        rules=[dict(rule_id=product,profile_id=product+'-application',profile_version='1',priority=1,effective_from='2024-01-01',effective_to=None,known_at='2024-01-01',conditions=[dict(field='product',operator='eq',value=product)])]
        route=cr.route_rating_profile(rs,rules,D,D).value
        app_contract=dict(contract_id=product+'-support',version='1',required_features=['product','driver'],allowed_values=dict(product=[product]),support_bounds=dict(driver=[-1,1]),maximum_age_days=dict(driver=31))
        applicability=cr.assess_rating_applicability(rs,app_contract,D,D).value
        selected=cr.select_effective_model(registry,[dict(assignment_id=product+'-champion',profile_id=route['profile_id'],purpose='application',model_id=model['model_id'],model_version='1',model_hash=mh,effective_from='2021-01-01',effective_to='2026-01-01',known_at='2021-01-01')],route['profile_id'],'application',D,D).value
        risk=cr.predict_application_risk(snapshot,model,target,calibrator=calibrator).value
        bureau=cr.build_bureau_features(**next(b for b in c['bureau_inputs']if b['customer_id']==app['customer_id'])).value
        budget=capacity(app,h,c['terms'],source_by_id[key]);dates=dict(as_of=K,known_at=K)
        policy_risk=dict(risk);policy_risk['target']={k:v for k,v in target.items()if k!='performance_origin'}
        inputs=dict(risk=dict(policy_risk,**dates,freshness='CURRENT'),affordability=dict(status=budget['status'],**dates),eligibility=dict(status='ELIGIBLE',**dates),operations=dict(status='CLEAR',**dates))
        decision=cr.apply_credit_decision_policy(policy,**inputs,**dates).value
        if(decision['action']=='APPROVE')!=(historical['status']=='approved'):raise ValueError('Frozen policy does not reconcile to recorded origination decision')
        request=dict(row_id=key,known_at=D,model_id=model['model_id'],model_version='1',model_hash=mh,recipe=dict(method='application_risk',version='1.0',parameters=dict(target=target)),input=dict(snapshot=snapshot))
        # Batch recipe is uncalibrated; calibrated decision is separately sealed.
        requests.append(request)
        binding=dict(profile_id=route['profile_id'],profile_version='1',model_id=model['model_id'],model_version='1',model_hash=mh,input_hash=fingerprint(snapshot))
        assessment=cr.compose_risk_assessment(key+'-assessment',subject,[dict(component_id=key+'-pd',quantity='PD',value=risk['pd'],**subject,variant='application_given_funding',definition=target)],binding,D,D,D).value
        rows.append(dict(application=app,snapshot=snapshot,label=next(r for r in labels if r['application_id']==key),subjects=subjects,route=route,applicability=applicability,selected_model=selected,risk=risk,capacity=budget,bureau=bureau,assessment=assessment,decision=decision,decision_inputs=inputs,policy_explanation=cr.decompose_policy_decision(decision,dict(CAPACITY='Negative full-schedule verified income residual',DECLARED_PD='Within predeclared risk threshold')).value))
    batch=cr.rate_subject_batch(requests,registry,product+'-batch','2025-02-10').value
    replay=cr.replay_assessment_history(requests,registry,batch['rows'],'2025-02-10').value
    return dict(joint_application=cr.build_customer_household_graph(**c['joint_application']).value,fitted=fitted,policy=policy,registry=registry,applications=rows,targets=labels,batch=batch,replay=replay,requests=requests)


def run(data=None,verify_reference=True,product_ids=None,applications_path=None,releases_path=None,return_artifacts=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('retail_application_journey.json').read_text())
    data=deepcopy(data);sources=load_sources(data,applications_path,releases_path);out={}
    for product,c in data['products'].items():
        if product_ids is not None and product not in product_ids:continue
        out[product]=origination(product,c,data['target'],sources)
        out[product]['performing']=performing(product,c,out[product],data['evaluation_cutoff'])
        out[product]['policy_validation']=policy_audit(product,c,out[product],data['target'],data['evaluation_cutoff'])
        out[product]['affordability_validation']=affordability_audit(product,c,out[product],data['evaluation_cutoff'])
        out[product]['monitoring']=monitor_and_replay(product,c,out[product],data['target'],data['evaluation_cutoff'])
        out[product]['pools']=application_pools(product,c,out[product],data['target'])
    result=dict(data_origin=data['data_origin'],journey_id=data['journey_id'],sources=sources,products=out)
    if verify_reference: verify_journey(result)
    return result if return_artifacts else presentation(result)





def performing(product,c,origin,cutoff):
    prior=c['behavioral_training'];trpanel=cr.build_account_month_panel(prior['accounts'],prior['statements'],prior['events'],observation_dates=['2017-12-31']).value['accounts']
    trlabels=cr.build_behavioral_targets(trpanel,prior['events'],horizon_months=1,target_event='DEFAULT',competing_events=['CLOSED'],exit_treatment='competing',evaluation_cutoff='2018-02-01',coverage=prior['coverage']).value
    target=dict(event='default',horizon=1,time_unit='months',probability_measure='PHYSICAL',conditioning='performing')
    import numpy as np
    model=cr.fit_binary_glm(np.empty((len(trlabels),0)),[x['label']for x in trlabels],feature_order=[],link='logit',fit_intercept=True,target=target,training=dict(known_at='2018-02-01',data_fingerprint=fingerprint(prior),split_id=product+'-behavioral-prior',data_origin='SYNTHETIC'),model_id=product+'-behavioral',model_version='1').model
    model['behavioral_contract']=dict(risk_unit='account',event_policy='first_default',product_ids=[product],allowed_states=['ACTIVE'],maximum_source_age_days=0)
    apps={a['application_id']:a for a in c['applications']};accounts=[];statements=[];events=[];coverage=[];obligations=[];payments=[]
    for b in c['bookings']:
        app=apps[b['application_id']];aid=b['facility_id'];D=b['first_draw_at'];accounts.append(dict(original_balance=1000,account_id=aid,customer_id=app['customer_id'],product_id=product,opened_at=D,known_at=D,currency='EUR'))
        statements.append(dict(statement_id='statement-'+aid,account_id=aid,observed_at=D,known_at=D,balance=1000,credit_limit=1000,dpd=0,past_due_amount=0,currency='EUR'))
        coverage.append(dict(account_id=aid,observed_through='2025-01-31'if D=='2024-12-31'else cutoff,known_at=cutoff))
        due=(date.fromisoformat(D)+timedelta(days=10)).isoformat();amount=payment_contract(c['terms'])[0]
        obligations.append(dict(obligation_id='due-'+aid,account_id=aid,due_at=due,known_at=D,amount=amount,currency='EUR'))
        if aid.endswith('SURVIVOR')and amount>0:payments.append(dict(kind='payment',payment_id='paid-'+aid,account_id=aid,paid_at=due,known_at=due,amount=amount,currency='EUR'))
    for e in c['defaults']:events.append(dict(event_id=e['event_id'],account_id=e['facility_id'],event_type='DEFAULT',occurred_at=e['default_at'],known_at=e['known_at']))
    for e in c['exits']:events.append(dict(event_id=e['event_id'],account_id=e['facility_id'],event_type='CLOSED',occurred_at=e['exit_at'],known_at=e['known_at']))
    # At each facility's first actual funded month-end; no synthetic pre-funding account.
    panel=[]
    for D in ['2024-12-31','2025-01-31']:
        aa=[a for a in accounts if a['opened_at']==D];ids={a['account_id']for a in aa}
        panel.extend(cr.build_account_month_panel(aa,[s for s in statements if s['account_id']in ids],[e for e in events if e['account_id']in ids],observation_dates=[D]).value['accounts'])
    labels=cr.build_behavioral_targets(panel,events,horizon_months=1,target_event='DEFAULT',competing_events=['CLOSED'],exit_treatment='competing',evaluation_cutoff=cutoff,coverage=coverage).value
    risk=[cr.predict_behavioral_risk(s,model,target).value for s in panel]
    memberships=[dict(customer_id=a['customer_id'],household_id='household-'+a['application_id'],role='test')for a in c['applications']]
    source_records=[dict(source_id='feature-'+a['application_id'],prediction_key='application:'+a['application_id'],source_hash=fingerprint(next(x['snapshot']for x in origin['applications']if x['application']['application_id']==a['application_id'])),kind='DOCUMENT',observed_at=h['source_date'],available_at=h['source_date'])for a,h in zip(c['applications'],c['households'])]+[dict(source_id=s['statement_id'],prediction_key='account:'+s['account_id']+'@'+s['observed_at'],source_hash=fingerprint(s),kind='STATEMENT',observed_at=s['observed_at'],available_at=s['known_at'])for s in statements]
    profile=dict(profile_id=product+'-retail-target',version='1',reference='Observed first default before closure; no household contagion',known_at='2024-01-01',default_definition='DECLARED_FIRST_DEFAULT',default_group_rule='facility-only-no-household-contagion',default_unit='facility',first_default=True,exit_treatment='competing',default_event='DEFAULT',chargeoff_event='CHARGEOFF',cure_event='CURED',competing_events=['CLOSED'],risk_set_by_product={product:dict(active_states=['ACTIVE','CURED'],include_zero_balance=True)},split_isolation_keys=['customer_id','household_id'],application_horizon_days=30,behavioral_horizon_months=1)
    design=dict(model_id=model['model_id'],model_version='1',parameter_version='1',data_fingerprint=fingerprint(c),population_id=product+'-through-door',split_id='journey-independent-holdout',prediction_origin='HOLDOUT',evaluation_cutoff=cutoff,target=target)
    validation_inputs=dict(application_targets=origin['targets'],account_panel=panel,behavioral_targets=labels,target_profile=profile,split_memberships=memberships,source_lineage=source_records,design=design)
    validation=cr.validate_retail_cohort(**validation_inputs).value
    payment_history=[]
    state_profile=dict(schema_version='1.0',profile_id='declared-retail-delinquency',version='1',materiality=dict(absolute=1,relative=0,logic='and'),immaterial_state='CURRENT',buckets=[dict(state='CURRENT',lower=0,upper=1),dict(state='ARREARS',lower=1,upper=None)],event_priority=[dict(event='utp',state='DEFAULT',at_risk=False,absorbing=True)])
    for account in accounts:
        if account['opened_at']!='2024-12-31':continue
        aid=account['account_id'];obs=[o for o in obligations if o['account_id']==aid];pays=[p for p in payments if p['account_id']==aid]
        due=cr.compute_days_past_due(obs,pays,as_of='2025-01-15',known_at='2025-01-15',currency='EUR').value
        state=cr.assign_delinquency_state(due['dpd'],due['past_due_amount'],1000,{'utp':False},state_profile).value
        history=cr.build_payment_history_features(obs,pays,dict(status='COMPLETE',known_at='2025-01-15',observed_from='2024-12-31',complete_through='2025-01-15'),account_id=aid,as_of='2025-01-15',information_cutoff='2025-01-15',lookback_days=15,allocation_policy='fifo',currency='EUR',grace_days=0).value
        payment_history.append(dict(account_id=aid,dpd=due,state=state,features=history))
    vintage=cr.build_vintage_performance(accounts,panel,events,months_on_book=[0,1],evaluation_cutoff=cutoff,coverage=coverage,default_event='DEFAULT',delinquency_states=[],exit_events=['CLOSED']).value
    return dict(payment_history=payment_history,vintage=vintage,training_panel=trpanel,training_targets=trlabels,model=model,accounts=accounts,statements=statements,events=events,coverage=coverage,obligations=obligations,payments=payments,panel=panel,targets=labels,risk=risk,validation_inputs=validation_inputs,validation=validation)





def policy_audit(product,c,origin,target,cutoff):
    target={k:v for k,v in target.items()if k!='performance_origin'}
    candidate=dict(candidate_id='historical',policy=origin['policy'],offer_rule_reference='frozen-product-terms-v1');cases=[];inputs=[];outcomes=[];historical=[];bookings={b['application_id']:b for b in c['bookings']}
    for row,d in zip(origin['applications'],c['decisions']):
        key=row['application']['application_id'];D=d['decided_at'];label=row['label'];funded=label['funded'];b=bookings.get(key);observed=funded and label['label_observed'] and label['mature'];pd=row['risk']['pd']
        offer=dict(offer_id=key,version='1',known_at=D,currency='EUR',target=target,risk_conditioning='CONDITIONAL_ON_THIS_ACCEPTED_OFFER',assumption_reference='Explicit illustrative uptake0.5 and LGD0.4; not estimated causal effects of price or limit',support_status='SUPPORTED',takeup_probability=.5,conditional_booked_expected_loss=pd*400,conditional_booked_npv=20-pd*400,ead=1000,offer_cost=1,contract_terms=dict(limit=1000,annual_rate=0,tenor_days=row['application']['requested_terms']['term_months']*30,currency='EUR',product_terms=c['terms']))
        cases.append(dict(case_id=key,decision_date=D,weight=1,split_id='journey-independent-holdout',prediction_reference='product-specific-prior-model-and-calibration',excluded_from_model_training=True,excluded_from_calibration=True))
        inputs.append(dict(candidate_id='historical',case_id=key,**row['decision_inputs'],offer=offer,nonapproval_costs=dict(REFER=1,DECLINE=0,BLOCKED=0)))
        end=(date.fromisoformat(b['first_draw_at'])+timedelta(days=30)).isoformat()if funded else None
        economics=next((e for e in c['realized_economics']if e['application_id']==key),None)
        if observed and(economics is None or not economics['complete']or economics['known_at']>cutoff):raise ValueError('Mature economic scoring requires complete actual cash ledger')
        outcomes.append(dict(case_id=key,funded=funded,complete=bool(observed),booked_offer_hash=fingerprint(offer)if funded else None,funded_at=b['first_draw_at']if funded else None,funding_known_at=b['first_draw_known_at']if funded else None,outcome_end=end,known_at='2025-01-31'if observed else None,defaulted=bool(label['target'])if observed else None,economic_loss=economics['principal_funded']-economics['principal_cash']+economics['collection_cost']if observed else None,net_profit=economics['principal_cash']+economics['interest_cash']-economics['principal_funded']-economics['collection_cost']if observed else None))
        approved=d['status']=='approved';taken='accepted_at'in d
        historical.append(dict(prediction_id=key,entity_id=row['application']['customer_id'],cluster_id=row['application']['customer_id'],segment=product,case_id=key,weight=1,as_of=D,forecast_known_at=D,horizon_end=(date.fromisoformat(D)+timedelta(days=30)).isoformat(),model_id=product+'-policy-audit',model_version='1',parameter_version='1',historical_candidate_id='historical',historical_decision=row['decision'],offered=approved,taken_up=taken,offered_limit=1000 if approved else None,booked_amount=1000 if funded else None,stages_known_at=b['first_draw_known_at']if funded else D,application_cost=1,cost_known_at=D,cost_complete=True))
    profile=dict(profile_id=product+'-policy-evaluation',version='1',known_at='2024-01-01',currency='EUR',registered_at='2024-01-01',outcome_access_date=cutoff,evaluation_role='TEST',prediction_design='HELD_OUT',split_id='journey-independent-holdout',data_fingerprint=fingerprint(c),target=target,candidate_hashes={'historical':fingerprint(candidate)},objective='expected_npv_per_application',constraints=[],tie_break='candidate_id',tolerance=1e-12,loss_bounds=[0,1000],profit_bounds=[-1000,100])
    grid_inputs=dict(policy_candidates=[candidate],cases=cases,candidate_inputs=inputs,observed_outcomes=outcomes,evaluation_profile=profile,evaluation_cutoff=cutoff)
    grid=cr.evaluate_decision_grid(**grid_inputs).value
    design=dict(model_id=product+'-policy-audit',model_version='1',parameter_version='1',data_fingerprint=profile['data_fingerprint'],population_id=product+'-through-door',split_id=profile['split_id'],prediction_origin='HOLDOUT',evaluation_cutoff=cutoff,currency='EUR',target=target)
    args=dict(evaluated_grid=grid,grid_inputs=grid_inputs,historical_cases=historical,design=design,reference_candidate_id='historical')
    result=cr.decision_policy_backtest(**args).value
    return dict(inputs=args,result=result)





def affordability_audit(product,c,origin,cutoff):
    cases=[];evidence=[];allocations=[];payments=[]
    for row,h in zip(origin['applications'],c['households']):
        app=row['application'];key=app['application_id'];b=row['capacity']['baseline'];scenario=row['capacity']['scenario'];D=app['application_at'];hh=h['household_id'];bindings={}
        for kind in ['incomes','expenses','contracts','collateral']:
            for item in b[kind]:
                eid=item['economic_id'];proof=dict(evidence_id='proof-'+eid,economic_id=eid,kind=kind,reference='verified-source-'+key,known_at=h['source_date'],observed_at=h['source_date'],verified=True,terms={k:v for k,v in item.items()if k!='economic_id'})
                if kind=='incomes':proof['source_basis']='net'
                evidence.append(proof);bindings[eid]=proof['evidence_id'];allocations.append(dict(allocation_id='allocation-'+eid,economic_id=eid,household_id=hh,as_of=D,known_at=h['source_date'],share=1,mode='FULL_RECOURSE'if kind=='contracts'else'PROPORTIONAL'))
        end=scenario['periods'][0]['date'];cases.append(dict(prediction_id=key,entity_id=app['customer_id'],cluster_id=hh,segment=product,as_of=D,forecast_known_at=D,horizon_end=end,model_id=product+'-budget',model_version='1',parameter_version='1',household_id=hh,baseline=b,scenario=scenario,assessment=row['capacity']['stress'],evidence_bindings=bindings,expected_economic_ids={kind:[v['economic_id']for v in b[kind]]for kind in ['incomes','expenses','contracts','collateral']},scenario_known_at=D,profile_known_at='2024-01-01',profile_version='1'))
        aid=row['label'].get('facility_id')
        if aid is not None and end<=cutoff:
            ledger=origin['performing'];dues=sum(v['amount']for v in ledger['obligations']if v['account_id']==aid and v['due_at']<=end);paid=sum(v['amount']for v in ledger['payments']if v['account_id']==aid and v['paid_at']<=end)
            payments.append(dict(prediction_id=key,period_date=end,known_at=cutoff,observed_through=end,complete=True,currency='EUR',scheduled_payment=dues,paid_amount=paid))
    design=dict(model_id=product+'-budget',model_version='1',parameter_version='1',data_fingerprint=fingerprint(c),population_id=product+'-through-door',split_id='journey-independent-holdout',prediction_origin='HOLDOUT',evaluation_cutoff=cutoff,currency='EUR',target=dict(event='default',horizon=1,time_unit='months',probability_measure='PHYSICAL',conditioning='household_budget'))
    args=dict(cases=cases,economic_evidence=evidence,allocations=allocations,design=design,payment_outcomes=payments)
    return dict(inputs=args,result=cr.affordability_stress_backtest(**args).value)





def monitor_and_replay(product,c,origin,target,cutoff):
    mt=dict(event='default',probability_measure='PHYSICAL',default_definition='DECLARED_FIRST_DEFAULT',unit='facility',horizon_days=30,default_grade='D');forecasts=[];events=[];coverage=[];apps={r['application']['application_id']:r for r in origin['applications']}
    for b in c['bookings']:
        r=apps[b['application_id']];forecasts.append(dict(forecast_id='forecast-'+b['facility_id'],case_id=b['application_id'],entity_id=b['facility_id'],cluster_id=r['application']['customer_id'],cohort=b['first_draw_at'][:7],segment=product,model_id=origin['fitted']['model']['model_id'],model_version='1',forecast_at=b['first_draw_at'],known_at=b['first_draw_at'],original_prediction_at=r['application']['application_at'],pd=r['risk']['pd'],target=mt,prediction_origin='HOLDOUT'))
        coverage.append(dict(entity_id=b['facility_id'],complete_through=b['followup_end'],known_at=b['outcome_available_at']))
    for e in c['defaults']:events.append(dict(event_id=e['event_id'],entity_id=e['facility_id'],event='default',event_at=e['default_at'],known_at=e['known_at'],default_definition='DECLARED_FIRST_DEFAULT'))
    for e in c['exits']:events.append(dict(event_id=e['event_id'],entity_id=e['facility_id'],event='exit',event_at=e['exit_at'],known_at=e['known_at'],default_definition='DECLARED_FIRST_DEFAULT'))
    cohort=cr.build_monitoring_cohorts(forecasts,events,coverage,target_contract=mt,evaluation_cutoff=cutoff).value
    monitor=cr.monitor_rating_cohort(cohort,group_by=['segment'],reference_window=dict(start='2024-12-31',end='2024-12-31'),minimum_count=30,uncertainty='none').value
    log=[];dependencies=[]
    for r,d in zip(origin['applications'],c['decisions']):
        key=r['application']['application_id'];assessment=r['assessment']['assessment_id'];dependencies.append(dict(assessment_id=assessment,dependency_type='application',dependency_id=key))
        phases=[('APPLICATION',r['application']['application_at'],dict(snapshot=r['snapshot']['snapshot_hash'])),('DECISION',d['decided_at'],dict(decision_hash=r['decision']['sha256']))]
        book=next((b for b in c['bookings']if b['application_id']==key),None)
        if book:phases.extend([('FUNDING',book['first_draw_at'],dict(facility_id=book['facility_id'])),('MONITORING',cutoff,dict(facility_id=book['facility_id'],cohort_hash=cohort['cohort_hash']))])
        for phase,D,payload in phases:
            event=dict(event_id=key+'-'+phase,idempotency_key=key+'-'+phase,occurred_at=D,known_at=D,sequence=len(log),payload=payload,revision_of=None);event['payload_hash']=fingerprint(payload);event['application_id']=key;log.append(event)
    log.append(deepcopy(log[0]))
    def execute():
        state=None;seen=set();applied=[]
        for event in sorted(log,key=lambda x:(x['known_at'],x['sequence'],x['event_id'])):
            if event['idempotency_key']in seen:continue
            seen.add(event['idempotency_key']);applied.append(event['event_id'])
            change=dict(event_id=event['event_id'],revision='1',occurred_at=event['occurred_at'],known_at=event['known_at'],dependencies=[dict(type='application',id=event['application_id'])],payload=event['payload'])
            state=cr.apply_assessment_event(state,change,dependencies,cutoff).value['state']
        rerated=cr.rate_subject_batch(origin['requests'],origin['registry'],product+'-batch',cutoff).value
        return applied,dict(dependency_state=state,ratings=rerated['rows'])
    applied,state=execute();again,state_replayed=execute()
    models=origin['fitted'];snapshots=[dict(snapshot_id=kind,kind=kind,content_hash=fingerprint(value),known_at=known,effective_from=known,effective_until=None)for kind,value,known in [('MODEL',models['model'],'2018-02-03'),('CALIBRATION',models['calibrator'],'2020-02-03'),('POLICY',origin['policy'],'2024-01-01'),('SOURCE',c['features'],'2025-01-28')]]
    hashes={s['snapshot_id']:s['content_hash']for s in snapshots};base=dict(checkpoint_id='final',as_of=cutoff,known_at=cutoff,event_log_hash=fingerprint(log),snapshot_hashes=hashes)
    original=dict(base,applied_event_ids=applied,state=state);replayed=dict(base,applied_event_ids=again,state=state_replayed,mode='REPLAY')
    contract=dict(contract_id=product+'-actual-execution-replay',version='1',reference='Independent rerating and public dependency-event execution on original frozen inputs',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=['final'],absolute_tolerance=1e-12,relative_tolerance=0)
    args=dict(event_log=log,original_checkpoints=[original],replayed_checkpoints=[replayed],snapshot_manifest=snapshots,replay_contract=contract)
    return dict(cohort=cohort,monitor=monitor,replay_inputs=args,replay_validation=cr.replay_consistency(**args).value)





def verify_journey(result):
    for p in result['products'].values():
        assert [r['target']for r in p['targets']]==[0,1,None,None,None]
        assert [r['decision']['action']for r in p['applications']]==['APPROVE','APPROVE','DECLINE','APPROVE','APPROVE']
        assert p['performing']['validation']['decision']['status']=='PASS'
        assert not p['policy_validation']['result']['violation_counts'] and not p['affordability_validation']['result']['violation_counts']
        assert p['monitoring']['replay_validation']['decision']['status']=='PASS'
        metrics=p['monitoring']['monitor'][0]['metrics'];lo,hi=p['fitted']['independent_cell_mle']
        assert metrics['n']==2 and math.isclose(metrics['brier'],(lo**2+(1-hi)**2)/2,abs_tol=1e-8)
        assert p['policy_validation']['result']['metrics']['factual']['observed_mean_loss']==200


def presentation(result):
    """Compact reported values only; raw fitted artifacts are never rounded."""
    products={}
    for name,p in result['products'].items():
        records=[]
        for row in p['applications']:
            a=row['application'];l=row['label']
            records.append(dict(application_id=a['application_id'],customer_id=a['customer_id'],facility_id=l['facility_id'],application_at=a['application_at'],model_id=row['risk']['model_id'],model_version=row['risk']['model_version'],policy_id=p['policy']['specification']['policy_id'],policy_version='1',pd=round(row['risk']['pd'],8),action=row['decision']['action'],budget_minimum=round(row['capacity']['residual']['minimum'],2),bureau_state=row['bureau']['file_state'],funded=l['funded'],mature=l.get('mature'),observed=l.get('label_observed'),target=l.get('target'),route_profile=row['route']['profile_id']))
        m=p['monitoring']['monitor'][0];f=p['policy_validation']['result']['metrics']['factual']
        products[name]=dict(pool_assignments=[dict(observation_id=r['observation_id'],pool_id=r['pool_id'])for a in p['pools']['assignments']for r in a['assignments']],target=p['fitted']['model']['target'],application_population=p['fitted']['model']['application_contract']['population_ids'],calibration_parameter_version=p['fitted']['calibrator']['parameter_version'],applications=records,prior_model=dict(training_known_at=p['fitted']['model']['training']['known_at'],calibration_known_at=p['fitted']['calibrator']['metadata']['calibration']['known_at'],cell_mle=p['fitted']['independent_cell_mle'],fitted_cell_pd=[round(x,8)for x in p['fitted']['fit_predictions']]),account_targets=p['performing']['targets'],payment_history=[dict(account_id=r['account_id'],dpd=r['dpd']['dpd'],past_due_amount=r['dpd']['past_due_amount'],state=r['state']['state'])for r in p['performing']['payment_history']],retail_cohort_status=p['performing']['validation']['decision']['status'],policy_factual=f,policy_violations=p['policy_validation']['result']['violation_counts'],affordability_status=p['affordability_validation']['result']['decision']['status'],payment_bias=p['affordability_validation']['result']['metrics']['factual_scheduled_minus_paid']['bias'],monitoring=dict(status=m['status'],n=m['metrics']['n'],observed_rate=m['metrics']['observed_rate'],mean_pd=round(m['metrics']['mean_pd'],8),brier=round(m['metrics']['brier'],8),excluded_count=m['excluded_count']),replay_status=p['monitoring']['replay_validation']['decision']['status'],duplicate_deliveries=p['monitoring']['replay_validation']['metrics']['duplicate_identical_deliveries'],joint_application=p['joint_application'])
    return dict(data_origin=result['data_origin'],journey_id=result['journey_id'],reporting=dict(probability_decimal_places=8,currency_decimal_places=2,raw_artifacts_available=True,raw_artifact_argument='return_artifacts=True',model_parameters_rounded=False),source_archives=result['sources']['archive_hashes'],source_validation=[dict(cutoff=r['cutoff'],status=r['result']['decision']['status'],node_totals=r['result']['node_totals'])for r in result['sources']['audits']],products=products)





def application_pools(product,c,origin,target):
    model=origin['fitted']['model'];strata=dict(product=product,borrower_type='person',facility_type=c['terms']['kind'],delinquency_state='not_defaulted');T=dict(target,conditioning='not_defaulted_at_index',default_unit='facility',default_definition='DECLARED_FIRST_DEFAULT',currency='EUR');T.pop('performance_origin')
    rows=[]
    for r in c['training']:
        score=float(cr.predict_pd(model,[[r['driver']]],feature_order=['driver'],expected_target=target).value[0])
        rows.append(dict(observation_id=r['application_id'],index_date=r['first_draw_at'],feature_observed_at=r['feature_observed_at'],feature_known_at=r['feature_known_at'],score=score,**strata,default_unit_id=r['facility_id'],role='TRAIN',label_window_end=r['outcome_end'],label_known_at=r['outcome_known_at'],defaulted=bool(r['outcome']),default_date=r['default_at'],lgd_status='OPEN'if r['outcome']else'NOT_DEFAULTED',lgd=None,ead=None,lgd_known_at=None,ead_known_at=None))
    cut=sum(origin['fitted']['independent_cell_mle'])/2
    spec=dict(spec_id=product+'-application-pools',version='1',known_at='2017-01-01',stratum_fields=list(strata),score_field='score',segments=[dict(segment_id=product,values=strata,cutpoints=[cut],score_min=0,score_max=1,monotonicity=dict(pd='increasing',lgd='none',ead='none'))],minimum_cases=2,minimum_defaults=0,minimum_lgd_count=0,minimum_ead_count=0,maximum_bin_mean_spread=dict(pd=None,lgd=None,ead=None))
    fit=cr.fit_retail_risk_pools(rows,spec,target=T,training=dict(model['training'],role='TRAIN'),model_id=product+'-pools',model_version='1')
    assignments=[]
    for r in origin['applications']:
        a=r['application'];D=a['application_at'];obs=dict(observation_id=a['application_id'],index_date=D,feature_observed_at=D,feature_known_at=D,score=r['risk']['pd'],**strata)
        assignments.append(cr.assign_retail_risk_pool([obs],fit,as_of=D,information_cutoff=D,expected_target=T,uncovered_policy='error').value)
    return dict(training_inputs=rows,model=fit.model,assignments=assignments)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
