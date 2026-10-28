from copy import deepcopy
from itertools import product
import math
import pytest
from creditratingsystem.expert import compile_discrete_bayesian_network,infer_discrete_credit_network
T=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
C=dict(network_id='synthetic-network',version='1',population_id='synthetic',default_node='C',default_state='bad',known_at='2024-01-01',as_of='2024-01-01',target=T)
def node(name,parents,cpt,states=None):return dict(node_id=name,states=['good','bad']if states is None else states,parents=parents,cpt=cpt,parameter_origin='FIXED',reference='analytic oracle',known_at='2024-01-01')
def root(name,p,states=None):return node(name,[],[dict(given={},probabilities=p)],states)
def network():
 a=root('A',[.2,.3,.5],['low','medium','high']);b=root('B',[.7,.3]);c=node('C',['B','A'],[dict(given=dict(B=bv,A=av),probabilities=[1-p,p])for bv,av,p in [('good','low',.01),('good','medium',.1),('good','high',.3),('bad','low',.2),('bad','medium',.5),('bad','high',.9)]])
 return [c,b,a]

def test_network_exact_variable_elimination_against_full_enumeration():
 nodes=network();fitted=compile_discrete_bayesian_network(nodes,network_contract=C,elimination_order=['A','B','C'])
 mapping={n['node_id']:n for n in nodes};total=bad=0
 for a,b,c in product(mapping['A']['states'],mapping['B']['states'],mapping['C']['states']):
  pa=mapping['A']['cpt'][0]['probabilities'][mapping['A']['states'].index(a)];pb=mapping['B']['cpt'][0]['probabilities'][mapping['B']['states'].index(b)];row=next(r for r in mapping['C']['cpt']if r['given']==dict(A=a,B=b));pc=row['probabilities'][mapping['C']['states'].index(c)]
  if b=='bad':total+=pa*pb*pc;bad+=pa*pb*pc*(c=='bad')
 result=infer_discrete_credit_network(fitted,{'B':'bad'}).value
 assert result['evidence_probability']==pytest.approx(total)
 assert result['default_probability']==pytest.approx(bad/total)
 observed=infer_discrete_credit_network(fitted,{'B':'bad','C':'bad'}).value
 assert observed['default_probability']==1 and observed['evidence_probability']==pytest.approx(bad)
 marginal=infer_discrete_credit_network(fitted,{}).value
 assert marginal['default_probability']==pytest.approx(.7*(.2*.01+.3*.1+.5*.3)+.3*(.2*.2+.3*.5+.5*.9))
 assert nodes==network()

def test_network_cycles_incomplete_cpt_impossible_and_underflow_evidence():
 nodes=network();nodes[2]['parents']=['C']
 with pytest.raises(ValueError,match='CYCLE'):compile_discrete_bayesian_network(nodes,network_contract=C)
 nodes=network();nodes[0]['cpt']=nodes[0]['cpt'][:-1]
 with pytest.raises(ValueError,match='CPT'):compile_discrete_bayesian_network(nodes,network_contract=C)
 fixed=compile_discrete_bayesian_network([root('C',[1,0])],network_contract=C)
 with pytest.raises(ValueError,match='IMPOSSIBLE'):infer_discrete_credit_network(fixed,{'C':'bad'})
 nodes=[root('C',[.9,.1])]+[root('E'+str(i),[1.,1e-30])for i in range(30)]
 fitted=compile_discrete_bayesian_network(nodes,network_contract=C,max_factor_cells=2)
 result=infer_discrete_credit_network(fitted,{'E'+str(i):'bad'for i in range(30)}).value
 assert result['evidence_probability_underflow'] and result['log_evidence_probability']==pytest.approx(30*math.log(1e-30))
 assert result['default_probability']==pytest.approx(.1)

def test_network_intermediate_factor_limit_is_enforced():
 parents=['A','B'];cpt=[dict(given=dict(A=a,B=b),probabilities=[.5,.5])for a,b in product(['good','bad'],repeat=2)]
 nodes=[root('A',[.5,.5]),root('B',[.5,.5]),node('C',parents,cpt),node('D',parents,cpt)]
 fitted=compile_discrete_bayesian_network(nodes,network_contract=dict(C,default_node='A'),elimination_order=['B','C','D','A'],max_factor_cells=8)
 with pytest.raises(ValueError,match='FACTOR_LIMIT'):infer_discrete_credit_network(fitted,{})
