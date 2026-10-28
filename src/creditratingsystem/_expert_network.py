"""Finite exact Bayesian networks with frozen CPTs and bounded variable elimination."""
from copy import deepcopy
from itertools import product
import math
import numpy as np
from scipy.special import logsumexp
from .core import _result,_fail,_numeric
from .data import _date
from ._expert_common import _ex_object,_ex_ids,_ex_integer,_ex_target,_ex_seal,_ex_read


def compile_discrete_bayesian_network(nodes,*,network_contract,elimination_order=None,max_factor_cells=1000000,context=None):
    """Validate a DAG and complete CPTs; freeze explicit finite probability factors."""
    if not isinstance(nodes,(list,tuple)) or not nodes: _fail('EXPERT_NETWORK','Nonempty node list required')
    ids=_ex_ids([n.get('node_id')for n in nodes],'node_ids');contract=deepcopy(_ex_object(network_contract,'network_contract'));target=_ex_target(contract.get('target'),context);limit=_ex_integer(max_factor_cells,'max_factor_cells',10000000)
    if not limit: _fail('EXPERT_NETWORK','Positive factor-size limit required')
    for key in ('network_id','version','population_id','default_node','default_state'):
        if not isinstance(contract.get(key),str) or not contract[key]: _fail('EXPERT_NETWORK',key+' required')
    cutoff=_date(contract.get('known_at'),'known_at');asof=_date(contract.get('as_of'),'as_of');mapping={n['node_id']:deepcopy(n)for n in nodes}
    for name,node in mapping.items():
        states=_ex_ids(node.get('states'),'states');parents=_ex_ids(node.get('parents'),'parents',True)
        if name in parents or set(parents)-set(ids): _fail('EXPERT_NETWORK','Unknown or self parent')
        if node.get('parameter_origin') not in ('ELICITED','ESTIMATED','FIXED') or not isinstance(node.get('reference'),str) or not node['reference']: _fail('EXPERT_NETWORK','Parameter origin and reference required')
        if _date(node.get('known_at'),'node known_at')>cutoff: _fail('EXPERT_LOOKAHEAD','CPT not available at network knowledge cutoff')
    topological=[];remaining=set(ids)
    while remaining:
        eligible=sorted(i for i in remaining if set(mapping[i]['parents'])<=set(topological))
        if not eligible: _fail('EXPERT_NETWORK_CYCLE','Directed network contains a cycle')
        topological.extend(eligible);remaining-=set(eligible)
    for name in topological:
        node=mapping[name];parents=node['parents'];dimensions=[len(mapping[p]['states'])for p in parents];size=math.prod(dimensions)*len(node['states'])
        if size>limit: _fail('EXPERT_FACTOR_LIMIT','A CPT exceeds declared factor-size limit')
        cpt=node.get('cpt')
        if not isinstance(cpt,(list,tuple)) or len(cpt)!=math.prod(dimensions): _fail('EXPERT_CPT','Exactly one CPT row per parent configuration required')
        indexed={}
        for row in cpt:
            given=_ex_object(row.get('given'),'given')
            if set(given)!=set(parents) or any(given[p]not in mapping[p]['states']for p in parents): _fail('EXPERT_CPT','CPT parent states do not match declared parents')
            key=tuple(mapping[p]['states'].index(given[p])for p in parents);probs=_numeric(row.get('probabilities'),'probabilities',vector_only=True)
            if key in indexed or probs.shape!=(len(node['states']),) or np.any(probs<0) or abs(float(probs.sum())-1)>1e-12: _fail('EXPERT_CPT','CPT rows must be unique, nonnegative and sum to one')
            indexed[key]=dict(given=deepcopy(given),probabilities=probs.tolist())
        node['cpt']=[indexed[key]for key in product(*(range(d)for d in dimensions))]
    if contract['default_node'] not in mapping or contract['default_state']not in mapping[contract['default_node']]['states']: _fail('EXPERT_NETWORK','Declared default node/state absent')
    order=list(reversed(topological))if elimination_order is None else _ex_ids(elimination_order,'elimination_order')
    if set(order)!=set(ids): _fail('EXPERT_NETWORK','Elimination order must contain every node once')
    value=_ex_seal(dict(network_contract=contract,nodes=[mapping[n]for n in topological],topological_order=topological,elimination_order=order,max_factor_cells=limit),'network_hash')
    return _result(value,'expert.compile_discrete_bayesian_network','discrete_bayesian_network',context,limitations=('CPTs are supplied, not estimated here. Parameter origins and knowledge dates remain explicit. Conditional-independence factorization is assumed; a directed graph alone establishes no causal interpretation.',))


def infer_discrete_credit_network(compiled_network,evidence,*,query_node=None,context=None):
    """Log-domain exact variable elimination; reject impossible evidence explicitly."""
    net=_ex_read(compiled_network,'network_hash');contract=net['network_contract'];_ex_target(contract['target'],context);nodes={n['node_id']:n for n in net['nodes']};evidence=_ex_object(evidence,'evidence');query=contract['default_node']if query_node is None else query_node;limit=net['max_factor_cells']
    if query not in nodes or set(evidence)-set(nodes) or any(v not in nodes[n]['states']for n,v in evidence.items()): _fail('EXPERT_NETWORK_EVIDENCE','Unknown query, evidence node or state')
    sizes={n:len(v['states'])for n,v in nodes.items()};factors=[];peak=0
    for name,node in nodes.items():
        scope=node['parents']+[name];dimensions=[sizes[n]for n in scope];values=np.array([p for row in node['cpt']for p in row['probabilities']]).reshape(dimensions)
        with np.errstate(divide='ignore'): values=np.log(values)
        index=tuple(nodes[n]['states'].index(evidence[n])if n in evidence else slice(None)for n in scope);values=values[index];scope=[n for n in scope if n not in evidence];factors.append((scope,values));peak=max(peak,int(np.size(values)))
    def join(group):
        nonlocal peak
        scope=[n for n in net['topological_order']if any(n in f[0]for f in group)];dimensions=[sizes[n]for n in scope];cells=math.prod(dimensions)
        if cells>limit: _fail('EXPERT_FACTOR_LIMIT','Variable elimination intermediate exceeds frozen factor-size limit; supply a better elimination order or explicit larger limit at compile time')
        peak=max(peak,cells);out=np.zeros(dimensions)
        for sub,val in group:
            order=sorted(range(len(sub)),key=lambda i:scope.index(sub[i]));aligned=np.transpose(val,order)if sub else val;shape=[sizes[n]if n in sub else 1 for n in scope];out=out+np.reshape(aligned,shape)
        return scope,out
    eliminated=[]
    for name in net['elimination_order']:
        if name==query or name in evidence: continue
        involved=[f for f in factors if name in f[0]]
        if not involved: continue
        factors=[f for f in factors if name not in f[0]];scope,values=join(involved);axis=scope.index(name);values=logsumexp(values,axis=axis);scope.remove(name);factors.append((scope,values));eliminated.append(name)
    scope,values=join(factors);log_evidence=float(logsumexp(values))
    if not math.isfinite(log_evidence): _fail('EXPERT_IMPOSSIBLE_EVIDENCE','Evidence has probability zero under the frozen network')
    if query in evidence: posterior=np.array([float(state==evidence[query])for state in nodes[query]['states']])
    else:
        if scope!=[query]: _fail('NUMERICAL_FAILURE','Elimination left unexpected variables')
        posterior=np.exp(values-log_evidence)
    probability=float(np.exp(log_evidence));pd=float(posterior[nodes[query]['states'].index(contract['default_state'])])if query==contract['default_node']else None
    return _result(dict(query_node=query,states=nodes[query]['states'],posterior_probabilities=posterior.tolist(),default_probability=pd,evidence=deepcopy(evidence),evidence_probability=probability,log_evidence_probability=log_evidence,evidence_probability_underflow=probability==0,eliminated_variables=eliminated,peak_factor_cells=peak,network_hash=net['network_hash'],target=contract['target']),
        'expert.infer_discrete_credit_network','probability_distribution',context,limitations=('Exact finite factor elimination conditions on supplied evidence and marginalizes unobserved states. Floating-point underflow in the evidence probability is separately flagged while its finite log probability remains available.',))
