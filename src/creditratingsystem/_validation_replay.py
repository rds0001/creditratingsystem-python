"""Historical replay comparison with frozen snapshots and append-only revisions."""
from copy import deepcopy
import math
from .core import _fail, _fingerprint
from .data import _date
from ._expert_common import _ex_object, _ex_ids
from ._validation_common import _vb_string, _vb_number
from ._validation_retail import _vr_rows
from ._validation_system import _vs_index, _vs_finish
from ._validation_release import _vs_hashes


def replay_consistency(event_log, original_checkpoints, replayed_checkpoints, *, snapshot_manifest, replay_contract, context=None):
    """Compare supplied real replay checkpoints; do not execute a surrogate engine."""
    events = _vr_rows(event_log, 'event_log')
    originals, oi = _vs_index(original_checkpoints, 'checkpoint_id', 'original_checkpoints')
    replays, ri = _vs_index(replayed_checkpoints, 'checkpoint_id', 'replayed_checkpoints')
    snapshots, si = _vs_index(snapshot_manifest, 'snapshot_id', 'snapshot_manifest')
    cfg = deepcopy(_ex_object(replay_contract, 'replay_contract'))
    for k in ('contract_id', 'version', 'reference'):
        _vb_string(cfg.get(k), k)
    expected_contract = dict(ordering='KNOWN_AT_SEQUENCE_EVENT_ID', idempotency='FIRST_IDENTICAL_DELIVERY', corrections='APPEND_REVISION')
    if any(cfg.get(k) != v for k,v in expected_contract.items()):
        _fail('REPLAY_CONTRACT', 'Explicit stable knowledge/sequence ordering, identical idempotency and append-revision semantics required')
    required = _ex_ids(cfg.get('required_checkpoint_ids'), 'required_checkpoint_ids')
    atol = _vb_number(cfg.get('absolute_tolerance'), 'absolute_tolerance', 0)
    rtol = _vb_number(cfg.get('relative_tolerance'), 'relative_tolerance', 0)
    violations, findings = [], []
    def issue(code, unit, detail):
        violations.append(dict(code=code, unit_id=unit, detail=detail))
    event_index = {}; idempotency = {}; duplicates = 0
    for e in events:
        eid = _vb_string(e.get('event_id'), 'event_id'); key = _vb_string(e.get('idempotency_key'), 'idempotency_key')
        occurred = _date(e.get('occurred_at'), 'event occurred_at'); known = _date(e.get('known_at'), 'event known_at')
        sequence = _vb_number(e.get('sequence'), 'event sequence', 0)
        if sequence != int(sequence):
            _fail('REPLAY_SEQUENCE', 'Sequence must be a nonnegative integer')
        if known < occurred:
            _fail('REPLAY_DATE', 'Historical occurrence must precede event knowledge')
        _ex_object(e.get('payload'), 'event payload')
        if e.get('payload_hash') != _fingerprint(e['payload']):
            issue('EVENT_PAYLOAD_HASH', eid, 'Payload hash differs from actual immutable payload')
        digest = _fingerprint(e)
        if eid in event_index and digest != _fingerprint(event_index[eid]):
            _fail('REPLAY_EVENT_CONFLICT', 'Repeated event ID contains different immutable content')
        if key in idempotency:
            prior = idempotency[key]
            if digest != _fingerprint(prior):
                _fail('REPLAY_IDEMPOTENCY_CONFLICT', 'An idempotency key can only repeat the identical event')
            duplicates += 1
        else:
            idempotency[key] = e
        event_index[eid] = e
    ordered = sorted(idempotency.values(), key=lambda e: (e['known_at'],e['sequence'],e['event_id']))
    position = {e['event_id']: i for i,e in enumerate(ordered)}
    for e in ordered:
        revision = e.get('revision_of')
        if revision is not None:
            if revision not in event_index or position[revision] >= position[e['event_id']]:
                _fail('REPLAY_REVISION', 'Revision must append after an existing earlier-known event')
    for s in snapshots:
        if s.get('kind') not in ('MODEL','CALIBRATION','POLICY','SOURCE'):
            _fail('REPLAY_SNAPSHOT', 'Snapshot kind must be MODEL, CALIBRATION, POLICY or SOURCE')
        _vs_hashes({'snapshot': s.get('content_hash')}, 'snapshot hash')
        _date(s.get('known_at'), 'snapshot known_at'); start = _date(s.get('effective_from'), 'snapshot effective_from')
        if s.get('effective_until') is not None and _date(s['effective_until'],'snapshot effective_until') <= start:
            _fail('REPLAY_SNAPSHOT', 'Snapshot effective interval must be nonempty')
    log_hash = _fingerprint(events)
    def state_difference(a,b,path=''):
        paths, maximum = [], 0.
        if type(a) in (int,float) and type(b) in (int,float):
            maximum = abs(a-b)
            if not math.isfinite(maximum): _fail('REPLAY_NUMERICAL', 'State difference exceeds finite range')
            if maximum > atol + rtol*abs(a): paths.append(path)
        elif isinstance(a,dict) and isinstance(b,dict):
            if set(a) != set(b): paths.append(path + '/<keys>')
            for k in sorted(set(a)&set(b)):
                p,m = state_difference(a[k],b[k],path+'/'+k); paths.extend(p); maximum=max(maximum,m)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a) != len(b): paths.append(path + '/<length>')
            for i,(x,y) in enumerate(zip(a,b)):
                p,m=state_difference(x,y,path+'/'+str(i)); paths.extend(p); maximum=max(maximum,m)
        elif type(a) is not type(b) or a != b:
            paths.append(path)
        return paths,maximum
    for cid in required:
        if cid not in oi: issue('MISSING_ORIGINAL_CHECKPOINT',cid,'Required frozen original absent')
        if cid not in ri: issue('MISSING_REPLAY_CHECKPOINT',cid,'Required actual replay output absent')
    for cid in sorted(set(oi)|set(ri)):
        if cid not in oi or cid not in ri:
            if cid not in required: issue('UNPAIRED_CHECKPOINT',cid,'Original and replay checkpoint sets differ')
            continue
        original = oi[cid]; replay = ri[cid]; before = len(violations)
        asof = _date(original.get('as_of'),'checkpoint as_of'); known = _date(original.get('known_at'),'checkpoint known_at')
        if known < asof: _fail('REPLAY_CHECKPOINT_DATE','Knowledge cutoff precedes state as-of')
        eligible = [e for e in ordered if _date(e['known_at'],'event known')<=known and _date(e['occurred_at'],'event occurred')<=asof]
        expected_ids = [e['event_id']for e in eligible]
        hashes = _vs_hashes(original.get('snapshot_hashes'),'checkpoint snapshot_hashes')
        kinds = set()
        for sid,digest in hashes.items():
            s=si.get(sid)
            if s is None or s['content_hash']!=digest:
                issue('SNAPSHOT_BINDING_MISMATCH',cid,sid+' missing or different content hash'); continue
            kinds.add(s['kind'])
            if _date(s['known_at'],'snapshot known')>known or _date(s['effective_from'],'snapshot effective')>asof or s.get('effective_until')is not None and asof>=_date(s['effective_until'],'snapshot until'):
                issue('SNAPSHOT_NOT_HISTORICALLY_APPLICABLE',cid,sid+' unavailable or outside its effective interval')
        if kinds != {'MODEL','CALIBRATION','POLICY','SOURCE'}:
            issue('MISSING_SNAPSHOT_KIND',cid,'Original must bind model, calibration, policy and source snapshots')
        for role,cp in [('original',original),('replay',replay)]:
            _ex_object(cp.get('state'),'checkpoint state'); _fingerprint(cp['state'])
            applied = cp.get('applied_event_ids')
            if not isinstance(applied,(list,tuple)) or any(not isinstance(x,str)for x in applied):
                _fail('REPLAY_APPLIED_EVENTS','Applied events must be an ordered ID array')
            if list(applied)!=expected_ids:
                issue('EVENT_ORDER_OR_IDEMPOTENCY_MISMATCH',cid,role+' applied events differ from the eligible unique ordered log')
            if cp.get('event_log_hash')!=log_hash:
                issue('EVENT_LOG_BINDING_MISMATCH',cid,role+' does not bind the exact immutable delivery log')
        if replay.get('mode')!='REPLAY':
            issue('COUNTERFACTUAL_NOT_REPLAY',cid,'Recomputation under changed methods is not historical replay')
        if any(replay.get(k)!=original.get(k)for k in ('as_of','known_at')):
            issue('CHECKPOINT_CUTOFF_MISMATCH',cid,'Replay uses different historical observation/knowledge cutoffs')
        replay_hashes = _vs_hashes(replay.get('snapshot_hashes'),'replay snapshot_hashes')
        if replay_hashes!=hashes:
            issue('REPLAY_SNAPSHOT_CHANGED',cid,'Replay changed model, calibration, policy or source snapshots')
        paths,maximum=state_difference(original['state'],replay['state'])
        if paths:issue('CHECKPOINT_STATE_MISMATCH',cid,'State differs at '+','.join(paths))
        findings.append(dict(checkpoint_id=cid,status='PASS'if len(violations)==before else'BLOCKED',
            expected_applied_event_ids=expected_ids,known_revision_count=sum(e.get('revision_of')is not None for e in eligible),
            differing_state_paths=paths,maximum_absolute_numeric_difference=maximum,
            original_state_hash=_fingerprint(original['state']),replay_state_hash=_fingerprint(replay['state'])))
    payload=dict(contract=cfg,applicability='APPLICABLE',checkpoint_findings=findings,event_log_hash=log_hash,
        snapshot_manifest_hash=_fingerprint(snapshots),metrics=dict(deliveries=len(events),unique_events=len(ordered),
        duplicate_identical_deliveries=duplicates,required_checkpoints=len(required),compared_checkpoints=len(findings)),
        event_execution_performed=False,original_decisions_modified=False,comparison_scope='SUPPLIED_ACTUAL_REPLAY_VERSUS_FROZEN_ORIGINAL')
    return _vs_finish('replay_consistency',payload,violations,dict(event_log=events,original_checkpoints=originals,
        replayed_checkpoints=replays,snapshot_manifest=snapshots,replay_contract=cfg),context)
