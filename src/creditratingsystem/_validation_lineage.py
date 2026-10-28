"""Dated source DAG and actual join-ledger reconciliation audits."""
from collections import Counter
from copy import deepcopy
import math
from .core import _fail, _fingerprint
from .data import _date
from ._expert_common import _ex_object, _ex_ids
from ._validation_common import _vb_string, _vb_number
from ._validation_retail import _vr_rows
from ._validation_system import _vs_index, _vs_finish
from ._validation_release import _vs_hashes


def validate_source_lineage(nodes, edges, joins, reconciliations, uses, *, contract, context=None):
    """Audit supplied row-level lineage, multiplicity, dated use and totals."""
    rows, ni = _vs_index(nodes, 'node_id', 'nodes')
    links = _vr_rows(edges, 'edges')
    jr, _ = _vs_index(joins, 'join_id', 'joins')
    recs, _ = _vs_index(reconciliations, 'reconciliation_id', 'reconciliations')
    ur, _ = _vs_index(uses, 'use_id', 'uses')
    cfg = deepcopy(_ex_object(contract, 'contract'))
    for key in ('contract_id', 'version', 'reference'):
        _vb_string(cfg.get(key), key)
    fields = _ex_object(cfg.get('amount_fields'), 'amount_fields')
    for field, currency in fields.items():
        _vb_string(field, 'amount field'); _vb_string(currency, 'amount currency')
    atol = _vb_number(cfg.get('absolute_tolerance'), 'absolute_tolerance', 0)
    rtol = _vb_number(cfg.get('relative_tolerance'), 'relative_tolerance', 0)
    required = _ex_ids(cfg.get('required_node_ids'), 'required_node_ids')
    required_uses = _ex_ids(cfg.get('required_use_ids'), 'required_use_ids')
    required_joins = _ex_ids(cfg.get('required_join_ids'), 'required_join_ids', empty=True)
    violations, totals, join_findings, use_findings = [], {}, [], []
    def issue(code, unit, detail):
        violations.append(dict(code=code, unit_id=unit, detail=detail))
    def close(a, b):
        return abs(a-b) <= atol + rtol*abs(b)
    records = {}
    for row in rows:
        nid = row['node_id']
        if row.get('kind') not in ('SOURCE', 'TRANSFORM', 'FEATURE', 'TARGET', 'PREDICTION', 'DECISION'):
            _fail('LINEAGE_KIND', 'Explicit source/transform/feature/target/prediction/decision node required')
        for key in ('schema_version', 'vintage', 'revision', 'transformation_version'):
            _vb_string(row.get(key), key)
        _vs_hashes({'node': row.get('content_hash')}, 'node content_hash')
        observed = _date(row.get('observed_at'), 'node observed_at'); available = _date(row.get('available_at'), 'node available_at')
        if available < observed:
            issue('AVAILABILITY_PRECEDES_OBSERVATION', nid, 'Node available date precedes its observation')
        if row['kind'] == 'SOURCE':
            _vb_string(row.get('source_id'), 'source_id')
            if row.get('semantic_role') not in ('FEATURE', 'OUTCOME', 'POST_DECISION'):
                _fail('LINEAGE_SOURCE_ROLE', 'Source requires explicit semantic role')
        data, ri = _vs_index(row.get('records'), 'record_id', 'node records')
        records[nid] = ri
        weights, amounts = [], {f: [] for f in fields}
        for record in data:
            weights.append(_vb_number(record.get('weight'), 'record weight', 0))
            values = _ex_object(record.get('amounts'), 'record amounts')
            if set(values) != set(fields):
                _fail('LINEAGE_AMOUNTS', 'Every node record must declare exactly the contracted amount fields')
            for f in fields:
                amounts[f].append(_vb_number(values[f], 'record amount'))
        try:
            sums = {f: math.fsum(v) for f, v in amounts.items()}; weight = math.fsum(weights)
        except OverflowError:
            _fail('LINEAGE_NUMERICAL', 'Reconciled total exceeds finite range')
        totals[nid] = dict(row_count=len(data), record_ids=sorted(ri), total_weight=weight, amounts=sums)
    parents = {nid: [] for nid in ni}; children = {nid: [] for nid in ni}; seen = set()
    for e in links:
        a = e.get('parent_id'); b = e.get('child_id')
        if a not in ni or b not in ni:
            _fail('LINEAGE_EDGE', 'Every edge endpoint must be a supplied node')
        if (a,b) in seen:
            _fail('LINEAGE_EDGE', 'Duplicate DAG edge')
        seen.add((a,b)); parents[b].append(a); children[a].append(b)
        if ni[a]['available_at'] > ni[b]['available_at']:
            issue('TRANSFORMATION_LOOKAHEAD', b, a + ' was unavailable when child was produced')
    indegree = {n: len(parents[n]) for n in ni}; queue = sorted(n for n in ni if indegree[n] == 0); order = []
    while queue:
        node = queue.pop(0); order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child); queue.sort()
    if len(order) != len(ni):
        _fail('LINEAGE_CYCLE', 'Lineage must be an acyclic dependency graph')
    ancestors = {}
    for nid in order:
        ancestors[nid] = set(parents[nid])
        for p in parents[nid]:
            ancestors[nid].update(ancestors[p])
        if ni[nid]['kind'] == 'SOURCE' and parents[nid]:
            issue('SOURCE_HAS_TRANSFORM_PARENTS', nid, 'A source snapshot must be a lineage root')
        if ni[nid]['kind'] != 'SOURCE' and not parents[nid]:
            issue('UNTRACED_TRANSFORMATION', nid, 'Derived node lacks a source/transform parent')
    for rec in recs:
        nid = rec.get('node_id')
        if nid not in ni:
            _fail('LINEAGE_RECONCILIATION', 'Reconciliation references unknown node')
        _vb_string(rec.get('reference'), 'reconciliation reference')
        ids = _ex_ids(rec.get('expected_record_ids'), 'expected_record_ids', empty=True)
        expected_weight = _vb_number(rec.get('expected_total_weight'), 'expected total weight', 0)
        expected_amounts = _ex_object(rec.get('expected_amounts'), 'expected_amounts')
        if set(expected_amounts) != set(fields):
            _fail('LINEAGE_RECONCILIATION', 'Expected amounts must cover contracted fields')
        if sorted(ids) != totals[nid]['record_ids']:
            issue('RECONCILIATION_IDS', rec['reconciliation_id'], 'Expected and actual record identities differ')
        if not close(totals[nid]['total_weight'], expected_weight):
            issue('RECONCILIATION_WEIGHT', rec['reconciliation_id'], 'Expected and actual total weight differ')
        for f in fields:
            expected = _vb_number(expected_amounts[f], 'expected amount')
            if not close(totals[nid]['amounts'][f], expected):
                issue('RECONCILIATION_AMOUNT', rec['reconciliation_id'], f + ' does not reconcile')
    for nid in required:
        if nid not in ni:
            issue('MISSING_REQUIRED_LINEAGE_NODE', nid, 'Required source/output node absent')
        elif not any(r['node_id'] == nid for r in recs):
            issue('MISSING_NODE_RECONCILIATION', nid, 'Required node has no independent row/weight/amount reconciliation')
    for join in jr:
        jid = join['join_id']; left = join.get('left_node_id'); right = join.get('right_node_id'); out = join.get('output_node_id')
        if any(n not in ni for n in (left,right,out)) or left == right:
            _fail('LINEAGE_JOIN', 'Distinct supplied left/right nodes and an output node required')
        if (left,out) not in seen or (right,out) not in seen:
            issue('JOIN_DAG_CONFLICT', jid, 'Both join inputs must directly parent its output')
        cardinality = join.get('cardinality')
        if cardinality not in ('one_to_one','one_to_many','many_to_one','many_to_many') or join.get('join_type') not in ('inner','left'):
            _fail('LINEAGE_JOIN', 'Declare join cardinality and inner/left semantics')
        if join.get('weight_contract') != 'LEFT_WEIGHT_ALLOCATION' or join.get('amount_origin') != 'LEFT':
            _fail('LINEAGE_JOIN', 'Explicit left-weight and left-amount allocation contract required')
        ledger, oi = _vs_index(join.get('mapping'), 'output_id', 'join mapping')
        lcounts = Counter(_vb_string(r.get('join_key'), 'left join key') for r in records[left].values())
        rcounts = Counter(_vb_string(r.get('join_key'), 'right join key') for r in records[right].values())
        if cardinality in ('one_to_one','one_to_many') and any(v > 1 for v in lcounts.values()) or cardinality in ('one_to_one','many_to_one') and any(v > 1 for v in rcounts.values()):
            issue('JOIN_CARDINALITY', jid, 'Input business-key multiplicity exceeds declared cardinality')
        if set(oi) != set(records[out]):
            issue('JOIN_OUTPUT_IDENTITIES', jid, 'Mapping does not cover exactly the output rows')
        allocated = {}; used_right = set(); pairs = set()
        for m in ledger:
            lid = m.get('left_id'); rid = m.get('right_id'); output = m['output_id']; allocation = _vb_number(m.get('allocation_weight'), 'allocation_weight', 0)
            if lid not in records[left] or rid is not None and rid not in records[right] or output not in records[out]:
                _fail('LINEAGE_JOIN_MAPPING', 'Mapping must reference supplied row identities')
            if (lid,rid) in pairs:
                issue('DUPLICATE_JOIN_PAIR', jid, 'Same input pair produces multiple rows')
            pairs.add((lid,rid)); l = records[left][lid]; o = records[out][output]
            if rid is None:
                if join['join_type'] != 'left' or l['join_key'] in rcounts:
                    issue('INVALID_UNMATCHED_JOIN_ROW', jid, 'Unmatched output conflicts with join keys/type')
            else:
                used_right.add(rid)
                if l['join_key'] != records[right][rid]['join_key']:
                    issue('JOIN_KEY_MISMATCH', jid, 'Joined rows have different business keys')
            allocated[lid] = allocated.get(lid,0) + allocation
            if not close(o['weight'], l['weight']*allocation):
                issue('JOIN_WEIGHT_MULTIPLICATION', jid, output + ' weight differs from explicit left allocation')
            for f in fields:
                if not close(o['amounts'][f], l['amounts'][f]*allocation):
                    issue('JOIN_AMOUNT_MULTIPLICATION', jid, output + '/' + f + ' differs from left allocation')
        for lid, fraction in allocated.items():
            if not close(fraction, 1):
                issue('JOIN_ALLOCATION_NOT_CONSERVING', jid, lid + ' allocations do not sum to one')
        excluded = sorted(set(records[left]) - set(allocated)); unmatched = sorted(set(records[right]) - used_right)
        supplied_excluded = sorted(_ex_ids(join.get('excluded_left_ids'), 'excluded_left_ids', empty=True))
        supplied_unmatched = sorted(_ex_ids(join.get('unmatched_right_ids'), 'unmatched_right_ids', empty=True))
        if excluded != supplied_excluded or unmatched != supplied_unmatched:
            issue('JOIN_EXCLUSION_RECONCILIATION', jid, 'Declared excluded/unmatched rows do not match actual mapping')
        if join['join_type'] == 'left' and excluded:
            issue('LEFT_JOIN_DROPPED_ROWS', jid, 'Left join must retain each left row')
        expected_pairs = {(lid,rid) for lid,l in records[left].items() for rid,r in records[right].items() if l['join_key'] == r['join_key']}
        if not expected_pairs <= pairs:
            issue('INCOMPLETE_JOIN_MATCHES', jid, 'Declared equality join omitted matching input pairs')
        join_findings.append(dict(join_id=jid, left_rows=len(records[left]), right_rows=len(records[right]), output_rows=len(records[out]),
            excluded_left_ids=excluded, unmatched_right_ids=unmatched, allocated_left_weights=dict(sorted(allocated.items()))))
    for use in ur:
        uid = use['use_id']; nid = use.get('node_id')
        if nid not in ni:
            _fail('LINEAGE_USE', 'Use references an unknown node')
        purpose = use.get('purpose')
        if purpose not in ('FEATURE','TARGET','PREDICTION','DECISION'):
            _fail('LINEAGE_USE', 'Explicit feature, target, prediction or decision use required')
        asof = _date(use.get('as_of'), 'use as_of'); known = _date(use.get('known_at'), 'use known_at')
        if known < asof:
            issue('USE_KNOWLEDGE_BEFORE_ASOF', uid, 'Audit knowledge cutoff precedes observation cutoff')
        lineage = ancestors[nid] | {nid}; sources = sorted(n for n in lineage if ni[n]['kind'] == 'SOURCE')
        expected = {n: ni[n]['content_hash'] for n in sources}
        supplied = _vs_hashes(use.get('source_snapshot_hashes'), 'use source_snapshot_hashes', empty=True)
        if not sources or supplied != expected:
            issue('SOURCE_SNAPSHOT_BINDING', uid, 'Use does not bind exactly to its historical source snapshots')
        for n in sorted(lineage):
            if _date(ni[n]['available_at'], 'node available') > known:
                issue('SOURCE_USE_LOOKAHEAD', uid, n + ' was unavailable at use knowledge cutoff')
            if purpose != 'TARGET' and _date(ni[n]['observed_at'], 'node observed') > asof:
                issue('FUTURE_OBSERVATION_IN_FEATURE_PATH', uid, n + ' observation follows original as-of')
            if purpose != 'TARGET' and (ni[n]['kind'] == 'TARGET' or ni[n].get('semantic_role') in ('OUTCOME','POST_DECISION')):
                issue('OUTCOME_IN_PREDICTIVE_PATH', uid, n + ' carries target/post-decision evidence')
        use_findings.append(dict(use_id=uid, node_id=nid, source_node_ids=sources, ancestor_count=len(lineage)-1))
    for kind, required_ids, actual_ids in [('USE', required_uses, {u['use_id'] for u in ur}), ('JOIN', required_joins, {j['join_id'] for j in jr})]:
        for identity in required_ids:
            if identity not in actual_ids:
                issue('MISSING_REQUIRED_' + kind, identity, 'Required dated use/join audit absent')
    payload = dict(contract=cfg, applicability='APPLICABLE', node_totals=totals, topological_order=order,
        join_findings=join_findings, use_findings=use_findings,
        metrics=dict(nodes=len(rows), edges=len(links), joins=len(jr), reconciliations=len(recs), dated_uses=len(ur)),
        source_content_authenticated=False)
    return _vs_finish('validate_source_lineage', payload, violations,
        dict(nodes=rows, edges=links, joins=jr, reconciliations=recs, uses=ur, contract=cfg), context)
