"""Mixed-case IDs, input permutation and sealed graph/portfolio parity."""
import creditratingsystem as cr


def canonical_id_fixture(reverse=False):
    ids=['a','B','A','b','é'];entities=[dict(entity_id=x)for x in ids]
    edges=[dict(edge_id=i,source_id=a,target_id=b,edge_type='CROSS_DEFAULT',valid_from='2024-01-01',valid_until=None,known_at='2024-01-01',evidence_reference='explicit mutual cross-default')for i,a,b in [('z','A','a'),('Z','a','A')]]
    rows=[dict(position_id='p'+x,claim_id='c'+x,facility_id='f'+x,entity_id=x,currency='EUR',ead=100,facility_ead=100,sector=x,region='r',exposure_basis='GROSS',collateral_treatment='NONE')for x in ids]
    if reverse:entities.reverse();edges.reverse();rows.reverse()
    graph=cr.build_entity_graph(entities,edges,dict(policy_id='graph',version='1',edge_types=['CROSS_DEFAULT']),as_of='2024-01-01',known_at='2024-01-01').value
    policy=dict(policy_id='default',version='1',edge_types=['CROSS_DEFAULT'],direction_policy='mutual_only',transitive=True)
    contract=dict(portfolio_id='mixed-case',version='1',currency='EUR',as_of='2024-01-01',known_at='2024-01-01',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='initially_performing',event='first_default')
    portfolio=cr.prepare_credit_portfolio(rows,graph,policy,contract).value
    concentration=cr.credit_concentration_measures(rows,weight_field='ead',group_fields=['sector'],weight_unit='EUR').value
    return dict(graph=graph,portfolio=portfolio,concentration=concentration)


def test_mixed_case_identifiers_and_permutation_preserve_sealed_content():
    value=canonical_id_fixture();assert value==canonical_id_fixture(reverse=True)
    assert [x['entity_id']for x in value['graph']['entities']]==['A','B','a','b','é']
    assert [x['edge_id']for x in value['graph']['edges']]==['Z','z']
    assert [x['position_id']for x in value['portfolio']['positions']]==['pA','pB','pa','pb','pé']
    assert [x['group_id']for x in value['concentration']['dimensions'][0]['groups']]==['pA','pB','pa','pb','pé']
    for name,field in [('graph','graph_hash'),('portfolio','portfolio_hash')]:
        sealed=value[name];assert cr.content_fingerprint({k:v for k,v in sealed.items()if k!=field}).value==sealed[field]
    group=next(x for x in value['portfolio']['default_group_assignment']['groups']if x['members']==['A','a'])
    assert group['group_id'].startswith('default-group-')
