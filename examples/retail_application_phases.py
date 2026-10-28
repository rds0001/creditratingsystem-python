"""Same funded accounts: grace/rent/principal claims and separate legal settlements."""
from copy import deepcopy
from datetime import date,timedelta
from importlib.resources import files
from pathlib import Path
import importlib.util,json
import creditratingsystem as cr

def helper():
    s=importlib.util.spec_from_file_location('phase_application',Path(__file__).with_name('retail_application_journey.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def assess(case,ledger,cutoff,bindings):
    x=ledger;aid=x['account_id'];key=x['application_id'];booking=next(b for b in case['bookings']if b['application_id']==key);default=next(d for d in case['defaults']if d['facility_id']==aid);exit=next(e for e in case['exits']if e['facility_id']==aid);econ=next(e for e in case['realized_economics']if e['application_id']==key);lease=case['terms']['kind']=='OPERATING_RENT'
    if x['settlement_known_at']>cutoff or x['settlement_date']<x['claim_cash_date']or x['settlement_known_at']<x['settlement_date']or x['account_exit_date']!=exit['exit_at']:raise ValueError('Separate dated account exit and legal settlement required')
    if x['claim_cash']+x['claim_waiver']!=x['claim_amount']or x['investment_recovery']!=econ['principal_cash']or x['investment_cost']!=econ['principal_funded']:raise ValueError('Claim settlement and original investment cash must reconcile separately')
    if lease and (x['scope']!='UNPAID_RENT_CLAIM'or x['claim_amount']!=case['terms']['rent']or x['claim_cash']!=0 or case['terms']['customer_residual_liability']!=0):raise ValueError('Asset disposal is not tenant rent or residual liability')
    if not lease and (x['scope']!='PRINCIPAL_CLAIM'or x['claim_amount']!=x['investment_cost']or x['claim_cash']!=x['investment_recovery']):raise ValueError('Principal claim must match original funded principal')
    if x['proposal_status']!='SCENARIO_NOT_ACCEPTED'or not x['proposal_date']<=x['proposal_known_at']<default['default_at']:raise ValueError('Only a dated predefault unaccepted proposal is supplied')
    due=(date.fromisoformat(booking['first_draw_at'])+timedelta(days=10)).isoformat();amount=helper().payment_contract(case['terms'])[0]
    obligations=[dict(obligation_id=aid+'-due',account_id=aid,due_at=due,known_at=booking['first_draw_known_at'],amount=amount,currency='EUR')]
    dpd=cr.compute_days_past_due(obligations,[],as_of=x['proposal_date'],known_at=x['proposal_known_at'],currency='EUR').value
    baseline=dict(plan_id='claim-no-new-agreement',payments=[0,0,0],period_interest_rates=[0,0,0],new_fees=[0,0,0],servicing_costs=[0,0,0],receipt_fractions=[1,1,1],capitalize_unpaid_interest=False);candidate=dict(baseline,plan_id='proposed-only',payments=x['proposal_payments'])
    proposal=cr.simulate_repayment_arrangement(dict(currency='EUR',principal=x['claim_amount'],principal_arrears=min(dpd['past_due_amount'],x['claim_amount']),accrued_interest=0,fees=0,as_of=x['proposal_date']),baseline,candidate,affordability_paths=[dict(path_id='declared-capacity',currency='EUR',available_cash=x['proposal_available_cash'])],discount_factors=[1,1,1],period_end_dates=x['proposal_dates'],currency='EUR').value
    ep=aid+'-legal-workout';cf=[dict(cashflow_id=ep+'-cash',episode_id=ep,cashflow_at=x['claim_cash_date'],known_at=x['settlement_known_at'],amount=x['claim_cash'],kind='recovery',currency='EUR')]
    defaults=[dict(episode_id=ep,entity_id=aid,default_at=default['default_at'],known_at=default['known_at'],ead=x['claim_amount'],currency='EUR')];closures=[dict(closure_id=ep+'-terminal',episode_id=ep,closed_at=x['settlement_date'],known_at=x['settlement_known_at'],closure_reason='SETTLEMENT')]
    linked=cr.build_recovery_episodes(defaults,cf,cures=[],closures=closures,evaluation_cutoff=cutoff).value
    early=cr.build_recovery_episodes(defaults,cf,cures=[],closures=closures,evaluation_cutoff=exit['known_at']).value
    cash=[dict(cashflow_id=cf[0]['cashflow_id'],cashflow_date=x['claim_cash_date'],known_at=x['settlement_known_at'],amount=x['claim_cash'],kind='RECOVERY',currency='EUR')]
    discounted=cr.discount_recoveries(cash,dict(anchor_date=default['default_at'],known_at=default['known_at'],discount_basis='DECLARED_ZERO_RATE_CLAIM',points=[dict(cashflow_date=x['claim_cash_date'],discount_factor=1)]),dict(reporting_currency='EUR',known_at=default['known_at'],conversion_basis='cashflow_date',rates=[dict(currency='EUR',date=x['claim_cash_date'],rate=1)]),default_date=default['default_at'],evaluation_cutoff=cutoff,reporting_currency='EUR')
    loss=cr.workout_lgd(x['claim_amount'],discounted,episode_status='CLOSED',default_date=default['default_at'],evaluation_cutoff=cutoff,currency='EUR').value
    # Realized investment loss distribution is descriptive, never a future LGD forecast.
    realized=[e['principal_funded']-e['principal_cash']for e in case['realized_economics']];distribution=cr.credit_loss_distribution(realized,[1]*len(realized),scenario_ids=[e['facility_id']for e in case['realized_economics']],levels=[.95],currency='EUR').value
    payloads=[('ORIGINATION',booking['first_draw_at'],booking['first_draw_known_at'],dict(funded=True)),('PERFORMING',booking['first_draw_at'],booking['first_draw_known_at'],dict(account_active=True)),('REVIEW_EARLY_WARNING',x['proposal_date'],x['proposal_known_at'],dict(dpd=dpd['dpd'])),('DELINQUENCY',x['proposal_date'],x['proposal_known_at'],dict(past_due=dpd['past_due_amount'],in_arrears=dpd['dpd']>0)),('RESTRUCTURING',x['proposal_date'],x['proposal_known_at'],dict(proposal_only=True,contract_changed=False)),('DEFAULT_WORKOUT',default['default_at'],default['known_at'],dict(ever_default=True,legal_claim=x['claim_amount'])),('EXIT',exit['exit_at'],exit['known_at'],dict(account_closed=True,legal_claim_remaining=x['claim_amount'])),('LEGAL_SETTLEMENT',x['settlement_date'],x['settlement_known_at'],dict(legal_claim_remaining=0,waived=x['claim_waiver'],ever_default=True)),('PORTFOLIO_MONITORING',x['settlement_date'],x['settlement_known_at'],dict(realized_investment_loss=x['investment_cost']-x['investment_recovery']))]
    events=[dict(event_id=aid+'-'+kind,idempotency_key=aid+'-'+kind,occurred_at=D,known_at=K,sequence=i,payload=p,revision_of=None,payload_hash=cr.content_fingerprint(p).value)for i,(kind,D,K,p)in enumerate(payloads)];events.append(deepcopy(events[4]));deps=[dict(assessment_id=aid+'-assessment',dependency_type='account',dependency_id=aid)]
    def execute():
        state=None;seen=set();ids=[];account_state={}
        for e in sorted(events,key=lambda r:(r['known_at'],r['sequence'],r['event_id'])):
            if e['idempotency_key']in seen:continue
            seen.add(e['idempotency_key']);ids.append(e['event_id']);account_state.update(deepcopy(e['payload']));state=cr.apply_assessment_event(state,dict(event_id=e['event_id'],revision='1',occurred_at=e['occurred_at'],known_at=e['known_at'],dependencies=[dict(type='account',id=aid)],payload=e['payload']),deps,cutoff).value['state']
        return ids,dict(dependencies=state,account=account_state)
    ids,state=execute();ids2,state2=execute();snaps=[dict(snapshot_id=k,kind=k,content_hash=bindings[field],known_at='2024-01-01',effective_from='2024-01-01',effective_until=None)for k,field in [('MODEL','model'),('CALIBRATION','calibrator'),('POLICY','policy')]]+[dict(snapshot_id='SOURCE',kind='SOURCE',content_hash=cr.content_fingerprint(dict(case=case,ledger=x,bindings=bindings or {})).value,known_at=cutoff,effective_from=cutoff,effective_until=None)];base=dict(checkpoint_id=aid+'-final',as_of=cutoff,known_at=cutoff,event_log_hash=cr.content_fingerprint(events).value,snapshot_hashes={r['snapshot_id']:r['content_hash']for r in snaps})
    replay_inputs=dict(event_log=events,original_checkpoints=[dict(base,applied_event_ids=ids,state=state)],replayed_checkpoints=[dict(base,applied_event_ids=ids2,state=state2,mode='REPLAY')],snapshot_manifest=snaps,replay_contract=dict(contract_id=aid+'-allphase',version='1',reference='Two actual public reducer executions; account exit and legal settlement separate',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=[base['checkpoint_id']],absolute_tolerance=1e-12,relative_tolerance=0))
    replay=cr.replay_consistency(**replay_inputs).value
    assert replay['decision']['status']=='PASS' and not proposal['contract_changed'] and abs(loss['economic_loss']-x['claim_waiver'])<1e-10, (replay,loss)
    summary=dict(account_id=aid,application_id=key,product_contract=case['terms'],claim_scope=x['scope'],dpd=dpd['dpd'],past_due=dpd['past_due_amount'],proposal_only=True,baseline_scope='NO_NEW_RECEIPT_SCENARIO_NOT_REPLACEMENT_CONTRACT_SCHEDULE',account_exit=exit['exit_at'],legal_settlement=x['settlement_date'],legal_claim_at_account_exit=x['claim_amount'],claim_cash=x['claim_cash'],claim_waiver=x['claim_waiver'],remaining_claim=0,ever_default=True,investment_loss=x['investment_cost']-x['investment_recovery'],loss_scopes_not_added=lease,realized_book_mean_loss=distribution['expected_loss'],replay_status=replay['decision']['status'],phases=[p[0]for p in payloads if p[0]!='LEGAL_SETTLEMENT'])
    return dict(summary=summary,proposal=proposal,workout=loss,recovery_episodes=linked,episodes_at_account_exit=early,replay_inputs=replay_inputs,replay=replay,distribution=distribution)

def run(data=None,application_data=None,application_chain=None,product_ids=None,return_artifacts=False):
    resource=files('creditratingsystem.datasets');data=json.loads(resource.joinpath('retail_application_phases.json').read_text())if data is None else data;source=json.loads(resource.joinpath('retail_application_journey.json').read_text())if application_data is None else application_data;chosen=list(data['products'])if product_ids is None else product_ids
    chain=helper().run(source,product_ids=chosen,return_artifacts=True)if application_chain is None else application_chain;out={}
    for p in chosen:
        origin=chain['products'][p];binding={k:cr.content_fingerprint(v).value for k,v in dict(model=origin['fitted']['model'],calibrator=origin['fitted']['calibrator'],policy=origin['policy']).items()};out[p]=assess(source['products'][p],data['products'][p],data['evaluation_cutoff'],binding)
    result=dict(journey_id='J_SYSTEM_EVENT_REPLAY',data_origin='SYNTHETIC',products={p:x['summary']for p,x in out.items()},scope='Same original funded accounts, product-specific claims; no new fitted model and no causal proposal benefit or portfolio forecast')
    if return_artifacts:result['raw_artifacts']=out
    return result
if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
