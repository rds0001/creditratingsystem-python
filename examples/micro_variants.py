"""Weekly microcredit, seasonal cash deficits and a shared group guarantee."""
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('micro_variants.json').read_text())
    debt=cr.build_debt_schedule(**data['debt']).value
    obligations=[dict(obligation_id='week'+str(i+1),account_id='weekly',due_at=row['payment_date'],known_at=data['debt']['calendar'][0]['start_date'],amount=row['net_payment_due'],currency='EUR')for i,row in enumerate(debt['schedule'])]
    due=cr.compute_days_past_due(obligations,data['payments'],as_of=data['observed_at'],known_at=data['observed_at'],currency='EUR').value
    residual=cr.residual_income(**data['residual']).value
    graph=cr.build_entity_graph(**data['graph'])
    groups=cr.assign_default_groups(graph,data['default_policy'],data['facility_mappings']).value
    assignments={x['entity_id']:x['group_id']for x in groups['assignments']}
    event_groups={assignments[x]for x in data['observed_default_entities']}
    membership_default={name:group in event_groups for name,group in assignments.items()}
    allocation=cr.allocate_guarantee_coverage(**data['allocation'])
    protected={name:cr.apply_guarantee_risk(**args,coverage_allocation=allocation).value for name,args in data['guarantee_risks'].items()}
    if verify_reference:
        assert [x['net_payment_due']for x in debt['schedule']]==[100]*4
        assert due['dpd']==5 and due['unpaid_due_amount']==100
        assert residual['residual']==[-200,-200,700]*4 and sum(residual['residual'])/12==100
        assert membership_default['memberA'] and not membership_default['memberB'] and not membership_default['group']
        assert [x['covered_amount']for x in allocation.value['allocations']]==[80,20]
        assert math.isclose(protected['memberA']['expected_loss_after'],1.6,abs_tol=1e-12)
        assert math.isclose(protected['memberB']['expected_loss_after'],6.4,abs_tol=1e-12)
        assert all(x['pd']==.1 for x in protected.values())
    return dict(data_origin=data['data_origin'],product=data['product'],weekly_schedule=debt,weekly_delinquency=due,seasonal_residual=residual,default_groups=groups,membership_default=membership_default,guarantee_allocation=allocation.value,protected_claims=protected)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
