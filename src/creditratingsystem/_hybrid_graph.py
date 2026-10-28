"""Pure, typed frozen-model graph execution and explicitly nonadditive attribution."""
from copy import deepcopy
import numpy as np
from scipy.special import expit,logit
from .core import _result,_numeric,_fail,_fingerprint,_context_value,_json_data,_model_artifact
from .data import _date
from ._expert_common import _ex_object,_ex_ids,_ex_target,_ex_seal,_ex_read


def _hg_contract(raw,context):
    c=deepcopy(_ex_object(raw,'node contract'))
    if set(c)-{'unit','target','row_ids','as_of','known_at','currency','feature_order'}:_fail('GRAPH_CONTRACT','Unknown contract field')
    if c.get('unit')not in ('design','probability','loss_fraction','money'):_fail('GRAPH_UNIT','Unknown node unit')
    _ex_target(c.get('target'));_ex_ids(c.get('row_ids'),'row_ids')
    for key in ('as_of','known_at'):
        _date(c.get(key),key)
        if c[key]!=context.get(key):_fail('GRAPH_CONTEXT','All nodes must share immutable information dates')
    for key in ('event','horizon','time_unit','probability_measure'):
        if context.get(key)is not None and context[key]!=c['target'][key]:_fail('GRAPH_TARGET','Node target differs from graph context')
    if c['unit']=='money':
        if not isinstance(c.get('currency'),str)or not c['currency']:_fail('GRAPH_CURRENCY','Money requires currency')
        if context.get('currency')is not None and context['currency']!=c['currency']:_fail('GRAPH_CURRENCY','Currency differs from graph context')
    elif 'currency'in c:_fail('GRAPH_CURRENCY','Currency only applies to money')
    if c['unit']=='design':_ex_ids(c.get('feature_order'),'feature_order',empty=True)
    elif 'feature_order'in c:_fail('GRAPH_CONTRACT','Feature order only applies to designs')
    if c['unit']in ('design','probability')and context.get('conditioning')is not None and context['conditioning']!=c['target']['conditioning']:_fail('GRAPH_TARGET','PD conditioning differs from graph context')
    return c


def _hg_value(value,contract):
    x=_numeric(value,'node value',allow_empty=contract['unit']=='design')
    shape=(len(contract['row_ids']),len(contract['feature_order']))if contract['unit']=='design'else(len(contract['row_ids']),)
    if x.shape!=shape:_fail('GRAPH_SHAPE','Node values must match ordered rows and declared features without recycling')
    if contract['unit']=='probability'and np.any((x<0)|(x>1)):_fail('GRAPH_PROBABILITY','Probability outside [0,1]')
    return x.tolist()


def _hg_ports(inputs,expected):
    if set(inputs)!=set(expected):_fail('GRAPH_PORT','Operation requires exact declared input ports')


def _hg_same(a,b,conditioning=True):
    a=deepcopy(a);b=deepcopy(b)
    for x in (a,b):
        x.pop('unit',None);x.pop('feature_order',None);x.pop('currency',None)
        if not conditioning:x['target'].pop('conditioning',None)
    if a!=b:_fail('GRAPH_ALIGNMENT','Node rows, information dates, target or conditioning differ')


def evaluate_model_graph(model_graph,input_data,parameter_snapshots,*,context):
    """Execute only declared frozen operations; retain immutable replayable snapshots."""
    graph=_json_data(deepcopy(_ex_object(model_graph,'model_graph')));data=_json_data(deepcopy(_ex_object(input_data,'input_data')));snapshots=_json_data(deepcopy(_ex_object(parameter_snapshots,'parameter_snapshots')));ctx=_context_value(context)
    if ctx.get('mode')!='DATED' or _date(ctx.get('known_at'),'known_at')>_date(ctx.get('as_of'),'as_of'):_fail('GRAPH_CONTEXT','A dated non-lookahead calculation context is required')
    for key in ('graph_id','graph_version'):
        if not isinstance(graph.get(key),str)or not graph[key]:_fail('GRAPH_CONTRACT',key+' required')
    nodes=graph.get('nodes')
    if not isinstance(nodes,list)or not nodes or any(not isinstance(n,dict)for n in nodes):_fail('GRAPH_CONTRACT','Nonempty graph nodes required')
    ids=_ex_ids([n.get('node_id')for n in nodes],'node_ids');by_id=dict(zip(ids,nodes));output=graph.get('output_node')
    if output not in by_id:_fail('GRAPH_CONTRACT','Unknown output node')
    for n in nodes:
        inputs=_ex_object(n.get('inputs'),'node inputs');_ex_object(n.get('parameters'),'parameters');_hg_contract(n.get('output_contract'),ctx)
        if any(v not in by_id for v in inputs.values()):_fail('GRAPH_EDGE','Unknown parent node')
    order=[];remaining=set(ids)
    while remaining:
        ready=sorted(i for i in remaining if all(p in order for p in by_id[i]['inputs'].values()))
        if not ready:_fail('GRAPH_CYCLE','Acyclic execution graph required; cyclic domain models must be solved before insertion')
        order.extend(ready);remaining.difference_update(ready)
    results={};trace=[];cache={}
    for node_id in order:
        node=by_id[node_id];op=node.get('operation');params=node['parameters'];parents={p:results[i]for p,i in node['inputs'].items()};declared=_hg_contract(node['output_contract'],ctx);effects=sorted(set(e for parent in parents.values()for e in parent['effects']));new_effect=None
        allowed={'input':{'input_id'},'predict_pd':{'snapshot_id'},'apply_calibrator':{'snapshot_id','base_model_version','groups'},'combine_pd_signals':{'signal_order','weights','combination_scale','combination_contract'},'log_odds_shift':{'shift','effect_id','reference'},'expected_loss':{'loss_assumption','reference'}}
        if op not in allowed or set(params)-allowed[op]:_fail('GRAPH_OPERATION','Unsupported operation or parameter')
        snapshot=None
        if op in ('predict_pd','apply_calibrator'):
            snapshot=snapshots.get(params.get('snapshot_id'))
            if snapshot is None:_fail('GRAPH_SNAPSHOT','Missing frozen parameter snapshot')
        key=_fingerprint(dict(operation=op,parameters=params,parents=parents,declared_contract=declared,snapshot=snapshot,context=ctx,input_snapshot=data.get(params.get('input_id'))if op=='input'else None))
        cached=key in cache
        if cached:result=deepcopy(cache[key])
        else:
            if op=='input':
                _hg_ports(parents,[]);record=_ex_object(data.get(params.get('input_id')),'input snapshot');contract=_hg_contract(record.get('contract'),ctx);value=_hg_value(record.get('value'),contract)
                if record.get('effects')is not None:effects=_ex_ids(record['effects'],'input effects',empty=True)
            elif op=='predict_pd':
                from .classification import predict_pd
                _hg_ports(parents,['design']);parent=parents['design'];contract=deepcopy(parent['contract'])
                if contract['unit']!='design':_fail('GRAPH_UNIT','predict_pd requires a design input')
                model=_model_artifact(snapshot)
                if _date(model['training']['known_at'],'model known_at')>_date(ctx['known_at'],'known_at'):_fail('GRAPH_LOOKAHEAD','Frozen model was unavailable at graph cutoff')
                value=_json_data(predict_pd(model,parent['value'],feature_order=contract.pop('feature_order'),expected_target=contract['target'],context=ctx).value);contract['unit']='probability'
            elif op=='apply_calibrator':
                from .calibration import apply_calibrator
                _hg_ports(parents,['pd']);parent=parents['pd'];contract=deepcopy(parent['contract'])
                if contract['unit']!='probability':_fail('GRAPH_UNIT','Calibrator requires a probability input')
                if snapshot.get('calibration_type')not in ('intercept','logistic','isotonic','beta','hierarchical'):_fail('GRAPH_UNIT','Only PD calibrators are allowed on probability edges')
                known=snapshot.get('metadata',{}).get('calibration',{}).get('known_at')
                if _date(known,'calibration known_at')>_date(ctx['known_at'],'known_at'):_fail('GRAPH_LOOKAHEAD','Calibrator unavailable at cutoff')
                new_effect='calibration:'+str(snapshot.get('artifact_hash'))
                value=_json_data(apply_calibrator(snapshot,parent['value'],target=contract['target'],base_model_version=params.get('base_model_version'),groups=params.get('groups'),context=ctx).value)
            elif op=='combine_pd_signals':
                from .hybrid import combine_pd_signals
                ports=_ex_ids(params.get('signal_order'),'signal_order');_hg_ports(parents,ports);first=parents[ports[0]];contract=deepcopy(first['contract']);signals=[]
                for port in ports:
                    parent=parents[port];_hg_same(contract,parent['contract'])
                    if parent['contract']['unit']!='probability':_fail('GRAPH_UNIT','Only PD signals may be pooled')
                    signals.append(dict(signal_id=port,model_id=port,model_version=_fingerprint(parent),row_ids=contract['row_ids'],target=contract['target'],as_of=contract['as_of'],known_at=contract['known_at'],pd=parent['value']))
                value=combine_pd_signals(signals,params.get('weights'),combination_scale=params.get('combination_scale'),combination_contract=params.get('combination_contract'),context=ctx).value['pd']
            elif op=='log_odds_shift':
                _hg_ports(parents,['pd']);parent=parents['pd'];contract=deepcopy(parent['contract'])
                if contract['unit']!='probability':_fail('GRAPH_UNIT','Log-odds adjustment requires probabilities')
                shift=_numeric(params.get('shift'),'shift',vector_only=True);p=np.asarray(parent['value'])
                if shift.shape!=p.shape:_fail('GRAPH_SHAPE','One log-odds shift per row is required')
                for name in ('effect_id','reference'):
                    if not isinstance(params.get(name),str)or not params[name]:_fail('GRAPH_ADJUSTMENT','Declared effect ID and reference required')
                new_effect=params['effect_id'];value=p.copy();inside=(p>0)&(p<1);value[inside]=expit(logit(p[inside])+shift[inside]);value=value.tolist()
            elif op=='expected_loss':
                from .losses import expected_loss
                _hg_ports(parents,['pd','lgd','ead']);pd,lgd,ead=[parents[p]for p in ('pd','lgd','ead')];contract=deepcopy(pd['contract'])
                if [r['contract']['unit']for r in (pd,lgd,ead)]!=['probability','loss_fraction','money']:_fail('GRAPH_UNIT','Expected loss requires PD, LGD fraction and money EAD')
                _hg_same(contract,lgd['contract'],False);_hg_same(contract,ead['contract'],False)
                if contract['target']['conditioning']!='performing'or any(r['contract']['target']['conditioning']!='default'for r in (lgd,ead)):_fail('GRAPH_CONDITIONING','Performing PD and default-conditional severity/exposure required')
                if params.get('loss_assumption')not in ('deterministic_conditional_amounts','conditional_independence_EAD_LGD')or not isinstance(params.get('reference'),str)or not params['reference']:_fail('GRAPH_LOSS_ASSUMPTION','Product-of-means loss requires an explicit justified conditional assumption')
                seen=set()
                for parent in (pd,lgd,ead):
                    if seen.intersection(parent['effects']):_fail('GRAPH_DOUBLE_COUNT','Same risk effect occurs on several loss legs')
                    seen.update(parent['effects'])
                if np.any(np.asarray(ead['value'])<0):_fail('GRAPH_EAD','EAD cannot be negative')
                value=_json_data(expected_loss(pd['value'],lgd['value'],ead['value'],context=ctx).value);contract['unit']='money';contract['currency']=ead['contract']['currency']
            else:_fail('GRAPH_OPERATION','Unsupported graph operation')
            if new_effect is not None:
                if new_effect in effects:_fail('GRAPH_DOUBLE_COUNT','Effect already applied in the node ancestry')
                effects=sorted(effects+[new_effect])
            if contract!=declared:_fail('GRAPH_CONTRACT','Computed output contract differs from declared contract')
            result=dict(value=_hg_value(value,contract),contract=contract,effects=effects);cache[key]=deepcopy(result)
        results[node_id]=result;trace.append(dict(node_id=node_id,operation=op,inputs=deepcopy(node['inputs']),cache_key=key,cache_hit=cached,result=deepcopy(result)))
    value=_ex_seal(dict(schema_version='1.0',graph=graph,input_snapshots=data,parameter_snapshots=snapshots,context=ctx,node_results=results,trace=trace,output_node=output,final=deepcopy(results[output])),'result_hash')
    return _result(value,'hybrid.evaluate_model_graph',results[output]['contract']['unit'],ctx,diagnostics=dict(node_count=len(nodes),unique_computations=len(cache),cache_hits=len(nodes)-len(cache)),limitations=('Frozen operations only; provenance declarations and content hashes do not authenticate caller evidence. Cyclic financial networks must use an explicit domain solver outside this execution graph.',))


def decompose_hybrid_prediction(hybrid_result,baseline_inputs,*,decomposition_scale='native',context=None):
    """Baseline plus one-input changes plus an explicit interaction remainder."""
    result=_ex_read(hybrid_result,'result_hash');ctx=result['context']
    if context is not None and _context_value(context)!=ctx:_fail('GRAPH_CONTEXT','Decomposition must preserve original immutable context')
    baseline=deepcopy(_ex_object(baseline_inputs,'baseline_inputs'));actual=result['input_snapshots']
    if set(baseline)!=set(actual):_fail('GRAPH_BASELINE','Baseline must specify exactly the frozen input snapshots')
    for name in actual:
        if baseline[name].get('contract')!=actual[name].get('contract')or baseline[name].get('effects')!=actual[name].get('effects'):_fail('GRAPH_BASELINE','Baseline may change values only, preserving typed contracts and effect lineage')
    if result['final']['contract']['unit']=='design':_fail('GRAPH_DECOMPOSITION','Final output must be PD or money, not a design')
    if decomposition_scale not in ('native','log_odds'):_fail('GRAPH_DECOMPOSITION','Use native or log_odds scale')
    if decomposition_scale=='log_odds'and result['final']['contract']['unit']!='probability':_fail('GRAPH_UNIT','Log-odds decomposition requires a PD output')
    def replay(data):return evaluate_model_graph(result['graph'],data,result['parameter_snapshots'],context=ctx).value
    def transform(value):
        x=np.asarray(value,dtype=float)
        if decomposition_scale=='log_odds':
            if np.any((x<=0)|(x>=1)):_fail('GRAPH_BOUNDARY','Finite log-odds decomposition requires interior outputs for all replay scenarios')
            x=logit(x)
        return x
    # Verify the sealed graph actually reproduces its purported trace; a hash alone is not proof.
    verified=replay(actual)
    for name,node in result['node_results'].items():
        check=verified['node_results'].get(name)
        if check is None or check['contract']!=node['contract']or check['effects']!=node['effects']or not np.allclose(check['value'],node['value'],rtol=1e-12,atol=1e-14):_fail('GRAPH_REPLAY','Frozen graph does not reproduce supplied node values and contracts')
    if result['final']!=result['node_results'][result['output_node']]:_fail('GRAPH_REPLAY','Final result differs from its output node')
    if len(result['trace'])!=len(verified['trace']):_fail('GRAPH_REPLAY','Trace length differs from graph')
    for step,check in zip(result['trace'],verified['trace']):
        if any(step.get(k)!=check[k]for k in ('node_id','operation','inputs'))or step.get('result')!=result['node_results'][check['node_id']]:_fail('GRAPH_REPLAY','Trace differs from graph node results')
    base=replay(baseline);a=transform(result['final']['value']);b=transform(base['final']['value']);contributions=[];total=np.zeros_like(a)
    for name in sorted(actual):
        one=deepcopy(baseline);one[name]=deepcopy(actual[name]);change=transform(replay(one)['final']['value'])-b;total+=change;contributions.append(dict(input_id=name,contribution=change.tolist()))
    remainder=a-b-total;steps=[]
    for step in result['trace']:
        parents=[result['node_results'][p]for p in step['inputs'].values()];out=step['result'];delta=None
        if len(parents)==1 and parents[0]['contract']==out['contract']:
            delta=(np.asarray(out['value'])-np.asarray(parents[0]['value'])).tolist()
        steps.append(dict(node_id=step['node_id'],operation=step['operation'],unit=out['contract']['unit'],output=deepcopy(out['value']),local_native_delta=delta,additive_global_contribution=False))
    return _result(dict(baseline=b.tolist(),actual=a.tolist(),input_contributions=contributions,interaction_remainder=remainder.tolist(),reconstructed=(b+total+remainder).tolist(),reconciliation_error=(a-(b+total+remainder)).tolist(),decomposition_scale=decomposition_scale,steps=steps,graph_result_hash=result['result_hash'],baseline_result_hash=base['result_hash'],attribution='ONE_INPUT_AT_A_TIME_FROM_COMMON_BASELINE_WITH_EXPLICIT_INTERACTION_REMAINDER'),
        'hybrid.decompose_hybrid_prediction','log_odds'if decomposition_scale=='log_odds'else result['final']['contract']['unit'],ctx,limitations=('Contributions are changes from a caller reference, not causal effects or Shapley values. Nonlinear and multi-input interactions are retained as a remainder; local step deltas must not be summed across branches.',))
