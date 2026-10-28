"""Sector cash rights, liquidity stress, member calls and legal-priority losses."""
from importlib.resources import files
from copy import deepcopy
import json
import math
import creditratingsystem as cr


def _receivable_paths(case):
    variant=case['variant'];capital=variant in ['multilateral_lender','supranational_body']
    if not capital and variant!='reinsurer':return None
    if capital:
        promises=[dict(promise_id='member-A',amount=40 if variant=='multilateral_lender'else 20,currency='EUR',fx=1,due='2024-06-01'),dict(promise_id='member-B',amount=30 if variant=='multilateral_lender'else 15,currency='USD'if variant=='multilateral_lender'else'EUR',fx=.9 if variant=='multilateral_lender'else 1,due='2024-07-01')]
    else:promises=[dict(promise_id='retrocession-A',amount=50,currency='EUR',fx=1,due='2024-05-01')]
    dates=['2024-05-01','2024-06-01','2024-07-01'];paths=[];draws=[];contracts=[]
    for p in promises:
        contracts.append(dict(contract_id=p['promise_id'],counterparty_id=p['promise_id'],currency='EUR',default_payment_timing='BEFORE_DUE',cure_probability=0,cure_delay_days=0,replacement_enabled=False,replacement_delay_days=0,replacement_payment_fraction=0,cashflows=[dict(date=d,amount=p['amount']*p['fx']if d==p['due']else 0)for d in dates]))
    for scenario in ['normal','first_member_default','joint_member_default']:
        for i,p in enumerate(promises):
            start='2024-04-01'
            for t,end in enumerate(dates):
                default=t==0 and (scenario=='joint_member_default'or scenario=='first_member_default'and i==0)
                paths.append(dict(scenario_id=scenario,counterparty_id=p['promise_id'],period_start=start,period_end=end,conditional_pd=1 if default else 0,uniform=.5,probability_measure='PHYSICAL'));start=end
            draws.append(dict(scenario_id=scenario,contract_id=p['promise_id'],uniform=.5))
    result=cr.simulate_contract_counterparties(paths,contracts,draws,currency='EUR').value
    uncalled=None
    if capital:
        inactive=deepcopy(contracts)
        for contract in inactive:
            for flow in contract['cashflows']:flow['amount']=0
        uncalled=cr.simulate_contract_counterparties(paths,inactive,draws,currency='EUR').value
    return dict(promises=promises,contract=result,untriggered_contract=uncalled,call_terms=dict(contract_known_at='2024-04-01',scheduled_notice_at='2024-05-01',minimum_notice_days=31,trigger='Explicit synthetic call-authorized stress state; untriggered state schedules no payment',currency_conversion='Declared fixed contractual EUR conversion, not market FX',no_double_credit='Each promise scheduled once; no parallel PD support benefit applied')if capital else None,interpretation='Explicit fixed stress states, no probability-weighted loss estimate; capital is financing, not operating revenue'if capital else 'Retrocession receipts depend on the same named counterparty event, not insurer default')


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_sector_cashflows.json').read_text())
    out=[]
    for case in data['cases']:
        t=case['terms'];sid=case['case_id'];receivables=_receivable_paths(case);capital=case['variant']in ['multilateral_lender','supranational_body']
        ops=[];water=[];expected=[]
        for scenario in ['base','stress']:
            transfers=t['transfer_receipts']*(.5 if scenario=='stress'else 1)
            cp=0
            if case['variant']=='reinsurer':cp=sum(x['received_cash']for x in receivables['contract']['cashflows']if x['scenario_id']==('normal'if scenario=='base'else'first_member_default')and x['date']=='2024-05-01')
            row=dict(scenario_id=scenario,period_start='2024-04-01',period_end='2024-05-01',currency='EUR',base_quantity=1,base_unit_price=t['own_receipts'],base_unit_variable_cost=0,base_fixed_cost=t['operating_due'],volume_multiplier=1,price_multiplier=1,variable_cost_multiplier=1,fixed_cost_multiplier=1,working_capital_revenue_ratio=0,cash_taxes=0,capex=0,asset_disposals=0,other_operating_cash=transfers+cp)
            ops.append(row);cash=t['initial_cash']+t['own_receipts']+transfers+cp
            water.append(dict(scenario_id=scenario,date='2024-05-01',currency='EUR',cash_available=cash,reserve_target=0,amounts_due=dict(operating=t['operating_due'],financial_credit=t['financial_due'])))
            op_paid=min(cash,t['operating_due']);credit_paid=min(cash-op_paid,t['financial_due'])
            expected.append(dict(scenario_id=scenario,own_receipts=t['own_receipts'],transfers=transfers,counterparty_receipts=cp,planned_cfads=t['own_receipts']+transfers+cp-t['operating_due'],operating_unpaid=t['operating_due']-op_paid,financial_credit_unpaid=t['financial_due']-credit_paid))
        operating=cr.simulate_operating_cashflows(ops,dict(contract_id=sid+'-operating',version='1',currency='EUR',period_unit='month',quantity_unit='explicit_period_cash_receipt',initial_working_capital=0,include_asset_disposals_in_cfads=False)).value
        liquidity=cr.apply_cash_waterfall(water,dict(policy_id=sid+'-liquidity',version='1',initial_reserve=0,reserve_priority=3,excess_reserve_policy='retain',arrears_policy='carry_without_interest',claims=[dict(claim_id='operating',priority=1,reserve_eligible=False),dict(claim_id='financial_credit',priority=2,reserve_eligible=False)]),currency='EUR').value
        balance=None
        if 'asset_book'in t:
            loss=t['asset_book']*t['asset_haircut'];balance=dict(initial_capital=t['asset_book']-t['liabilities'],market_value_loss=loss,noncash_reserve_strengthening=t['reserve_strengthening'],stressed_capital=t['asset_book']-loss-t['liabilities']-t['reserve_strengthening'],restricted_capital_not_cash=t['restricted_capital'])
        recovery=cr.simulate_recovery_waterfall(**case['recovery']).value
        expected_recovery=min(max(t['recovery_gross']-t['prior_claim'],0),t['credit_claim']);expected_lgd=1-expected_recovery/t['credit_claim']
        facts=dict(product=case['product'],variant=case['variant'],borrower_class=t['borrower_class'],claim_type=t['claim_type'],cashflow_type=t['cashflow_type'])
        snapshot=dict(subject_id=sid,subject_type='obligor',as_of='2024-04-01',known_at='2024-04-01',attributes={k:dict(value=v,observed_at='2024-04-01',known_at='2024-04-01')for k,v in facts.items()})
        rule=dict(rule_id=sid,profile_id=sid,profile_version='1',priority=1,effective_from='2023-01-01',effective_to=None,known_at='2023-01-01',conditions=[dict(field=k,operator='eq',value=v)for k,v in facts.items()])
        route=cr.route_rating_profile(snapshot,[rule],'2024-04-01','2024-04-01').value;snapshot['attributes']['claim_type']['value']=None
        negative=cr.route_rating_profile(snapshot,[rule],'2024-04-01','2024-04-01').value
        if verify_reference:
            for e,op,w in zip(expected,operating['periods'],liquidity['periods']):
                assert math.isclose(op['cfads'],e['planned_cfads'],abs_tol=1e-12)
                payments={p['claim_id']:p for p in w['payments']}
                assert payments['operating']['closing_arrears']==e['operating_unpaid']and payments['financial_credit']['closing_arrears']==e['financial_credit_unpaid']
            claim=next(x for x in recovery['claim_results']if x['claim_id']=='financial_credit');assert math.isclose(claim['lgd'],expected_lgd,abs_tol=1e-12)
            assert negative['status']=='NO_ROUTE'
            if capital:
                assert sum(x['received_cash']for x in receivables['untriggered_contract']['cashflows'])==0
                flows=receivables['contract']['cashflows'];assert sum(x['received_cash']for x in flows if x['date']=='2024-05-01')==0
                assert sum(x['received_cash']for x in flows if x['scenario_id']=='joint_member_default')==0
                assert sum(x['received_cash']for x in flows if x['scenario_id']=='normal')==sum(p['amount']*p['fx']for p in receivables['promises'])
        out.append(dict(product=case['product'],variant=case['variant'],case_id=sid,terms=t,operating=operating,liquidity=liquidity,balance_stress=balance,receivables=receivables,recovery=recovery,independent_cash_oracle=expected,independent_credit_lgd=expected_lgd,profile=route['profile_id'],missing_claim_status=negative['status']))
    return dict(data_origin='SYNTHETIC',legal_convention=data['legal_convention'],cases=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
