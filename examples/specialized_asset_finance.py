"""Explicit physical cash-pool contracts for sixteen specialized-finance variants."""
from importlib.resources import files
from copy import deepcopy
from datetime import date
import json
import math
import creditratingsystem as cr


def _close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)


def _ledger_oracle(case,scenario,price,transaction):
    """Independent two-date cash accounting, event and discounted recovery algebra."""
    principal=case['principal'];interest=0.;reserve=case['reserve'];arrears=0.;oparrears=0.;previous_material=False;clock=0
    expected=[];default_date=None;loss=0.;ead_default=0.
    origin=date(2023,1,1);previous=origin
    for i,when in enumerate([date(2024,1,1),date(2025,1,1)]):
        multiplier=round(1-case['floating_share'][i]+case['floating_share'][i]*price[i],12)
        cash=case['revenue'][i]*multiplier-case['fixed_cost'][i]-case['capex'][i]
        for j,amounts in enumerate(case['contract_amounts']):
            if scenario['counterparty_stress'] and j not in case.get('safe_counterparty_indices',[]):cash-=amounts[i]
        coupon=case['principal']*.1*(when-previous).days/365
        due=coupon+(case['principal']if i==1 else 0)+arrears
        opening_reserve=reserve;opening_oparrears=oparrears;available=max(cash,0)+reserve
        opdue=max(-cash,0)+oparrears;paid_op=min(available,opdue);available-=paid_op;oparrears=opdue-paid_op
        payment=min(available,due);available-=payment;arrears=due-payment
        # No reserve target and retain-excess contract: excess cash is distributed,
        # only unused opening reserve remains available to the next period.
        cash_consumed=max(paid_op+payment-max(cash,0),0);reserve=max(opening_reserve-cash_consumed,0)
        interest+=coupon;paid_interest=min(interest,payment);interest-=paid_interest;paid_principal=payment-paid_interest;principal-=paid_principal
        ead=principal+interest;material=arrears>1 and (arrears/ead if ead else 0)>.01
        clock=clock+(when-previous).days if material and previous_material else 0
        market=round(price[i]*case.get('completion_multiplier',[1,1])[i],12)
        elapsed=(when-origin).days/365;assets=[]
        for value,salvage in zip(case['asset_values'],case['salvage']):
            ageing=math.exp(-.35*elapsed)if case.get('obsolescence')else max(case['life']-elapsed,0)/case['life']
            assets.append((salvage+(value-salvage)*ageing)*market)
        buffer=max(opening_reserve-opening_oparrears,0)
        # These contracts never carry an unpaid operating deficit alongside a
        # positive opening reserve; opening buffer is therefore unambiguous.
        utp=(cash+buffer<due) and ead>sum(assets)
        default=clock>90 or utp
        expected.append(dict(date=when.isoformat(),cfads=cash,principal_paid=paid_principal,interest_paid=paid_interest,ead=ead,reserve=reserve,arrears=arrears,clock=clock,assets=assets))
        if default:
            default_date=when.isoformat();ead_default=ead
            proceeds=sum(assets)+reserve;cost=sum(case['recovery_cost'])
            df=1.05**(-case['delay']/365)
            # Collateral and costs settle together after the declared delay;
            # remaining reserve is realized at default without discount.
            net=proceeds-cost
            paid=min(ead,max(net,0))
            # In these fixtures reserve is exhausted before any default.
            assert reserve==0
            loss=ead-paid*df+max(-net,0)*df
            break
        previous=when;previous_material=material
    path=next(p for p in transaction['paths']if p['scenario_id']==scenario['scenario_id'])
    assert path['default_event']['first_default_date']==default_date
    _close(path['ead_at_default'],ead_default)
    assert len(path['going_concern_ledger'])==len(expected)
    for actual,want in zip(path['going_concern_ledger'],expected):
        _close(actual['cfads'],want['cfads']);_close(actual['liquidity']['closing_reserve'],want['reserve'])
        for key in ['principal_paid','interest_paid','ead']:_close(actual['facilities'][0][key],want[key])
        _close(actual['material_arrears'],want['arrears']);assert actual['material_days_past_due']==want['clock']
        for a,b in zip(actual['asset_values'].values(),want['assets']):_close(a,b)
    actual_loss=sum(x['economic_loss']for x in transaction['recovery']['claim_results']if x['scenario_id']==scenario['scenario_id'])
    _close(actual_loss,loss)
    return dict(defaulted=default_date is not None,loss=loss)


def _guarantee(case,transaction,cube,convention,verify):
    if 'guarantee_cap' not in case:return None
    key=case['case_id'];guarantor=key+'-guarantor';origin=cube['origin'];claim_id=key+'-claim';rows=[]
    for sid,weight in zip(cube['scenario_ids'],cube['weights']):
        path=next(p for p in transaction['paths']if p['scenario_id']==sid)
        if not path['default_event']['defaulted']:continue
        due=path['default_event']['first_default_date'];q=convention['guarantor_cumulative_pd'][due]
        loss=sum(x['economic_loss']for x in transaction['recovery']['claim_results']if x['scenario_id']==sid)
        claim=dict(claim_id=claim_id,entity_id=key,amount=path['ead_at_default'],currency='EUR',due_at=due,known_at=origin)
        expiry=convention['vendor_valid_until']if case['variant']=='vendor_supported'else convention['other_valid_until']
        guarantee=dict(guarantee_id=key+'-guarantee',guarantor_id=guarantor,currency='EUR',cap=case['guarantee_cap'],cap_used=0,coverage_fraction=1,covered_claim_ids=[claim_id],valid_from=origin,valid_until=expiry,known_at=origin,legally_effective=True,evidence_reference='Synthetic dated capped contract')
        policy=dict(policy_id=key+'-allocation',version='1',guarantee_priority=[guarantee['guarantee_id']],claim_priority=[claim_id],cap_convention='non_reinstating_total')
        allocation=cr.allocate_guarantee_coverage([guarantee],[claim],policy,as_of=origin,known_at=origin,currency='EUR')
        target=dict(event='default',horizon=1 if due=='2024-01-01'else 2,time_unit='years',probability_measure='PHYSICAL',conditioning='given_macro_lessee_path')
        borrower=dict(entity_id=key,pd=1,stage='PRE_TRANSFER',target=dict(target,exposure_id=claim_id),applied_effects=[])
        supporter=dict(entity_id=guarantor,pd=q,stage='STANDALONE',target=target,applied_effects=[])
        joint=cr.joint_default_probability(1,q,event_order=[key,guarantor],joint_probability=q).value
        contract=dict(effect_id=key+'-protection',currency='EUR',assumption_reference=convention['dependence'],stage='PRE_GUARANTEE',valuation_date=due,states=dict(borrower_only=dict(loss_before_guarantee=loss,payout_fraction=1,evidence_reference='Capped claim paid if guarantor survives'),joint_default=dict(loss_before_guarantee=loss,payout_fraction=0,evidence_reference='No payment on guarantor default')))
        adjusted=cr.apply_guarantee_risk(borrower,supporter,allocation,claim_id,contract,joint_default_probability=joint).value
        paid=min(case['guarantee_cap'],max(loss,0))if due<expiry else 0
        states=[dict(state_id='survives',events=dict(borrower_default=True,guarantor_default=False),probability=1-q,conditional_loss=loss-paid),dict(state_id='fails',events=dict(borrower_default=True,guarantor_default=True),probability=q,conditional_loss=loss)]
        states += [dict(state_id='no_borrower_'+str(g),events=dict(borrower_default=False,guarantor_default=g),probability=0,conditional_loss=0)for g in [False,True]]
        exclusive=cr.event_conditioned_loss(states,event_names=['borrower_default','guarantor_default'],currency='EUR',probability_measure='PHYSICAL').value
        if verify:
            _close(adjusted['expected_loss_after'],loss-(1-q)*paid);_close(exclusive['expected_loss'],adjusted['expected_loss_after']);assert adjusted['pd']==1
            if case['variant']=='vendor_supported':assert adjusted['covered_amount']==0
        # Report the loss algebra, keeping the complete sealed allocation local;
        # no independently fitted artifacts are substituted during replay.
        rows.append(dict(scenario_id=sid,weight=weight,claim_due=due,guarantee_expiry=expiry,covered_amount=adjusted['covered_amount'],guarantor_pd=q,joint_default_probability=joint,loss_before=loss,conditional_expected_loss_after=adjusted['expected_loss_after'],conditional_borrower_pd=adjusted['pd']))
    after=sum(x['weight']*x['conditional_expected_loss_after']for x in rows)
    if verify:assert after<=transaction['risk']['expected_loss']+1e-10
    return dict(contract=convention,paths=rows,borrower_pd=transaction['risk']['pd'],expected_loss_before=transaction['risk']['expected_loss'],expected_loss_after=after)


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_asset_finance.json').read_text())
    lt=data['lessee_training'];tr=lt['score_training'];ca=lt['calibration']
    lessee_model=cr.fit_scorecard(tr['X'],tr['y'],feature_order=tr['feature_order'],fit_spec=tr['fit_spec'],weights=tr['weights'],target=lt['target'],training=tr['training'],model_id='equipment-lessee',model_version='1',tolerance=1e-13)
    lessee_cal=cr.calibrate_logistic(ca['values'],ca['outcomes'],ca['weights'],metadata=ca['metadata'])
    lessee_raw=cr.score_scorecard(lessee_model,[[lt['application_leverage']]],feature_order=tr['feature_order'],expected_target=lt['target']).value['pd']
    lessee_pd=cr.apply_calibrator(lessee_cal,lessee_raw,target=lt['target'],base_model_version='1').value[0]
    if verify_reference:assert math.isclose(lessee_pd,.1,abs_tol=1e-8)
    lessee_pd=round(lessee_pd,8)
    model=cr.fit_var(data['macro_history'],lags=1,training=data['macro_training'],target=data['macro_target'],model_id='specialized-macro',model_version='1',covariance_estimator='mle')
    macro=cr.simulate_macro_scenarios(model,data['innovations'])
    satellite=cr.fit_satellite_model(data['satellite_records'],data['macro_history'],**data['satellite_arguments'])
    state=dict(as_of=macro.value['origin'],known_at=macro.value['information_cutoff'],mode='absolute',macro_history=[[0]],residual_state=0)
    noise=dict(kind='STANDARD_NORMAL',scenario_ids=macro.value['scenario_ids'],time_grid=macro.value['time_grid'],values=[[0,0]for _ in macro.value['scenario_ids']])
    price=cr.apply_satellite_model(satellite,macro,noise,state).value
    ttc=cr.simulate_macro_scenarios(model,data['ttc_innovations'],sampling_contract=data['ttc_sampling_contract']).value
    cube=dict(schema_version='1.0',kind='ScenarioCube',axes=['scenario','time','factor'],scenario_ids=macro.value['scenario_ids'],time_grid=macro.value['time_grid'],weights=macro.value['weights'],origin=macro.value['origin'],frequency='annual',probability_measure='PHYSICAL',sampling_contract=dict(law='FORECAST_CONDITIONAL',conditioning_scope='PIT',quadrature='Three-point Gaussian innovation tensor crossed with declared common counterparty Bernoulli state; fixed weights, no IID confidence interval.'),factor_ids=[],units=[],values=[[[]for _ in range(2)]for _ in macro.value['scenario_ids']])
    for case in data['cases']:
        key=case['case_id'];cube['factor_ids'] += [key+'-revenue',key+'-asset'];cube['units']+=['relative_index']*2
        for s in range(len(cube['scenario_ids'])):
            for t in range(2):
                p=price['values'][s][t][0]
                cube['values'][s][t]+=[round(1-case['floating_share'][t]+case['floating_share'][t]*p,12),round(p*case.get('completion_multiplier',[1,1])[t],12)]
    parameter_snapshot=deepcopy(data['parameter_snapshot'])
    for case in data['cases']:
        if case['product']=='EQUIPMENT_LEASING':
            for row in parameter_snapshot['parameters'][case['case_id']]['counterparties']['paths']:
                if row['period_end']=='2024-01-01':row['conditional_pd']=lessee_pd
    recipes=[dict(transaction_id=c['case_id'],version='1',subject_id=c['case_id'],parameter_id=c['case_id'],calendar=data['calendar'],parameter_hash=cr.content_fingerprint(parameter_snapshot['parameters'][c['case_id']]).value,cube_hash=cr.content_fingerprint(cube).value)for c in data['cases']]
    execution=cr.simulate_transactions(recipes,cube,data['entity_innovations'],parameter_snapshot).value
    variants={}
    for case in data['cases']:
        key=case['case_id'];tx=next(t for t in execution['transactions']if t['transaction_id']==key)
        facts=dict(product=case['product'],variant=case['variant'],phase=case['phase'],contracts_verified=True)
        snapshot=dict(subject_id=key,subject_type='facility',as_of=cube['origin'],known_at=cube['origin'],attributes={k:dict(value=v,observed_at=cube['origin'],known_at=cube['origin'])for k,v in facts.items()})
        rule=dict(rule_id=key,profile_id=key,profile_version='1',priority=1,effective_from=cube['origin'],effective_to=None,known_at=cube['origin'],conditions=[dict(field=k,operator='eq',value=v)for k,v in facts.items()])
        route=cr.route_rating_profile(snapshot,[rule],cube['origin'],cube['origin']).value
        snapshot['attributes']['contracts_verified']['value']=False
        negative=cr.route_rating_profile(snapshot,[rule],cube['origin'],cube['origin']).value
        default=[];lgd=[];ead=[]
        for sid in cube['scenario_ids']:
            path=next(p for p in tx['paths']if p['scenario_id']==sid);di=[0,0];li=[0.,0.];ei=[0.,0.]
            if path['default_event']['defaulted']:
                t=[p['end_date']for p in data['calendar']].index(path['default_event']['first_default_date']);di[t]=1;ei[t]=path['ead_at_default'];loss=sum(r['economic_loss']for r in tx['recovery']['claim_results']if r['scenario_id']==sid);li[t]=loss/ei[t]
            default.append(di);lgd.append(li);ead.append(ei)
        loss=cr.expected_loss_from_paths(default,lgd,ead,cube['weights'],[1,1],scenario_ids=cube['scenario_ids'],time_grid=cube['time_grid'],currency='EUR',probability_measure='PHYSICAL').value
        if verify_reference:
            assert route['status']=='ROUTED'and negative['status']=='NO_ROUTE'
            checks=[_ledger_oracle(case,sc,[price['values'][s][t][0]for t in range(2)],tx)for s,sc in enumerate(data['scenario_contract'])]
            _close(tx['risk']['pd'],sum(sc['weight']*v['defaulted']for sc,v in zip(data['scenario_contract'],checks)))
            _close(tx['risk']['expected_loss'],sum(sc['weight']*v['loss']for sc,v in zip(data['scenario_contract'],checks)))
            _close(loss['expected_loss'],tx['risk']['expected_loss'])
        protection=_guarantee(case,tx,cube,data['guarantee_convention'],verify_reference)
        variants[key]=dict(guarantee=protection,lessee_pd=lessee_pd if case['product']=='EQUIPMENT_LEASING'else None,product=case['product'],variant=case['variant'],profile_id=route['profile_id'],missing_contract_status=negative['status'],transaction=tx,joint_path_loss=loss)
    moments=[]
    for t in range(2):
        mean=sum(w*v[t][0]for w,v in zip(ttc['weights'],ttc['values']));variance=sum(w*(v[t][0]-mean)**2 for w,v in zip(ttc['weights'],ttc['values']))
        moments.append(dict(mean=mean,variance=variance))
        if verify_reference:_close(mean,-3/35);_close(variance,data['hidden_truth']['stationary_variance'])
    p=model.model['parameters']
    if verify_reference:
        _close(p['intercept'][0],-1/7);_close(p['lag_coefficients'][0][0][0],-2/3);_close(p['covariance'][0][0],data['hidden_truth']['innovation_variance'])
        for s,sc in enumerate(data['scenario_contract']):
            first=-1/7+math.sqrt(data['hidden_truth']['innovation_variance'])*sc['innovations'][0][0]
            _close(price['values'][s][0][0],1);_close(price['values'][s][1][0],math.exp(.2*first))
    return dict(data_origin='SYNTHETIC',scenario_cube_hash=execution['scenario_cube_hash'],parameter_snapshot_hash=execution['parameter_snapshot_hash'],innovation_hash=execution['innovation_hash'],scenario_precision_contract=data['scenario_precision_contract'],pit_sampling_contract=cube['sampling_contract'],ttc_sampling_contract=ttc['sampling_contract'],stationary_moments=moments,macro_parameters=p,satellite_parameters=satellite.model['parameters'],variants=variants)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
