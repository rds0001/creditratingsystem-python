from copy import deepcopy
import math
from .core import CalcResult, _numeric, _fail, _fingerprint, _context_value
from .data import _date

def _ex_object(x, name):
    x = x.value if isinstance(x, CalcResult) else x
    if not isinstance(x, dict): _fail('EXPERT_CONTRACT', name+' must be an object')
    return x


def _ex_ids(x, name, empty=False):
    if not isinstance(x,(list,tuple)) or (not x and not empty) or any(not isinstance(v,str) or not v for v in x) or len(set(x))!=len(x):
        _fail('EXPERT_CONTRACT',name+' must contain distinct nonempty strings')
    return list(x)


def _ex_number(x,name,positive=False):
    a=_numeric(x,name)
    if a.ndim or (positive and a<=0): _fail('EXPERT_CONTRACT',name+' requires a finite '+('positive 'if positive else '')+'scalar')
    return float(a)


def _ex_probability(x,name,interior=False):
    v=_ex_number(x,name)
    if not 0<=v<=1 or (interior and v in (0,1)): _fail('EXPERT_CONTRACT',name+' outside probability domain')
    return v


def _ex_integer(x,name,maximum=1000000):
    v=_ex_number(x,name)
    if not 0<=v<=maximum or v!=math.floor(v): _fail('EXPERT_CONTRACT',name+' outside integer domain')
    return int(v)


def _ex_seal(value,field):
    out=deepcopy(value);out[field]=_fingerprint(out);return out


def _ex_read(value,field):
    out=deepcopy(_ex_object(value,field));seal=out.pop(field,None)
    if seal!=_fingerprint(out): _fail('EXPERT_INTEGRITY','Artifact content or hash changed')
    out[field]=seal;return out


def _ex_target(target,context=None,allowed_events=('default',)):
    t=_ex_object(target,'target')
    if set(t)!={'event','horizon','time_unit','probability_measure','conditioning'} or t.get('event') not in allowed_events or t.get('time_unit') not in ('days','months','years') or t.get('probability_measure') not in ('PHYSICAL','RISK_NEUTRAL') or not isinstance(t.get('conditioning'),str) or not t['conditioning']:
        _fail('EXPERT_TARGET','Explicit default horizon, measure and conditioning required')
    _ex_number(t['horizon'],'horizon',True);ctx=_context_value(context)
    for key in ('horizon','time_unit','probability_measure','conditioning'):
        if ctx.get(key) is not None and ctx[key]!=t[key]: _fail('TARGET_MISMATCH','Context differs from expert target')
    return deepcopy(t)


def _ex_elicitation(contract,context=None):
    c=deepcopy(_ex_object(contract,'elicitation_contract'))
    for key in ('population_id','reference','known_at'):
        if not isinstance(c.get(key),str) or not c[key]: _fail('EXPERT_CONTRACT',key+' required')
    _date(c['known_at'],'known_at');_ex_ids(c.get('expert_ids'),'expert_ids');_ex_ids(c.get('evidence_ids'),'evidence_ids');c['target']=_ex_target(c.get('target'),context)
    return c


