"""Local typed data contracts and explicitly dated credit-event preparation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path

from .core import _fail, _result


def _rows(table, name):
    if hasattr(table, 'to_dict'):
        table = table.to_dict(orient='records')
    if not isinstance(table, (list, tuple)) or any(not isinstance(r, dict) for r in table):
        _fail('DATA_SCHEMA', f'{name} must be a list of row mappings')
    return [dict(r) for r in table]


def _date(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        _fail('DATA_DATE', f'{name} requires an ISO YYYY-MM-DD date')
    try:
        return date.fromisoformat(value)
    except ValueError:
        _fail('DATA_DATE', f'Invalid date in {name}')


def _required(row, fields):
    for f in fields:
        if f not in row or row[f] is None:
            _fail('DATA_SCHEMA', f'Missing required field {f}')
        if f.endswith('_id') and (not isinstance(row[f], str) or not row[f]):
            _fail('DATA_SCHEMA', f'{f} must be a nonempty string')


def _days(value, name, minimum=0):
    # JSON has numeric values, not a portable int-versus-double type distinction.
    # Accept integral binary64 values from native R while rejecting coercion of
    # booleans, strings, fractions and nonfinite values.
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or isinstance(value, float) and (not math.isfinite(value) or not value.is_integer())
            or value < minimum):
        _fail('DATA_SCHEMA', f'{name} must be an integer >= {minimum}')
    return int(value)


def _unique(rows, keys, name):
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, (list, tuple)) or not keys or any(not isinstance(k, str) or not k for k in keys):
        _fail('DATA_SCHEMA', 'Keys must be nonempty column names')
    seen = set()
    for row in rows:
        _required(row, keys)
        if any(not isinstance(row[f], str) or not row[f] for f in keys):
            _fail('DATA_SCHEMA', f'Key fields in {name} must be nonempty strings')
        k = tuple(row[f] for f in keys)
        if k in seen:
            _fail('DATA_DUPLICATE', f'Duplicate key in {name}: {k}')
        seen.add(k)


def _json_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            _fail('DATA_DUPLICATE', f'Duplicate JSON key {key}')
        out[key] = value
    return out


def read_local_dataset(path, schema, encoding='utf-8', delimiter=',', decimal_mark='.'):
    """Read local CSV or JSON rows using a versioned, explicit column contract."""
    path_string = str(path)
    if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*://', path_string) or path_string.startswith(('//', '\\\\')):
        _fail('DATA_REMOTE_PATH', 'Only local filesystem paths are supported')
    if not isinstance(schema, dict) or schema.get('version') != '1.0' or not isinstance(schema.get('columns'), dict) or not schema['columns']:
        _fail('DATA_SCHEMA', 'schema requires version 1.0 and a nonempty columns mapping')
    if decimal_mark not in ('.', ',') or not isinstance(delimiter, str) or len(delimiter) != 1:
        _fail('DATA_SCHEMA', 'Invalid decimal mark or delimiter')
    columns = schema['columns']
    types = {'string', 'integer', 'number', 'boolean', 'date'}
    for field, spec in columns.items():
        if not isinstance(spec, dict) or spec.get('type') not in types:
            _fail('DATA_SCHEMA', f'Unsupported type for {field}')
        if spec['type'] in ('integer', 'number') and not isinstance(spec.get('unit'), str):
            _fail('DATA_SCHEMA', f'Numeric field {field} requires an explicit unit')
    try:
        raw = Path(path).read_bytes()
        content = raw.decode(encoding)
        if Path(path).suffix.lower() == '.csv':
            import io
            reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                _fail('DATA_SCHEMA', 'Missing or duplicate CSV column names')
            if set(reader.fieldnames) != set(columns):
                _fail('DATA_SCHEMA', 'File columns must equal schema columns')
            rows = list(reader)
        elif Path(path).suffix.lower() == '.json':
            rows = json.loads(content, object_pairs_hook=_json_object, parse_constant=lambda x: _fail('DATA_TYPE', f'Nonfinite JSON token {x}'))
        else:
            _fail('DATA_FORMAT', 'Supported formats are .csv and .json')
    except (OSError, UnicodeError, LookupError, csv.Error, json.JSONDecodeError) as exc:
        _fail('DATA_READ', str(exc))
    rows = _rows(rows, 'dataset')
    if not rows and not schema.get('allow_empty', False):
        _fail('DATA_EMPTY', 'Empty table requires allow_empty=true')
    output = []
    for i, row in enumerate(rows):
        if set(row) != set(columns):
            _fail('DATA_SCHEMA', f'Unexpected or missing columns at row {i + 1}')
        out = {}
        for field, spec in columns.items():
            value = row[field]
            if value is None or value == '':
                if not spec.get('nullable', False):
                    _fail('DATA_NULL', f'{field} is not nullable at row {i + 1}')
                out[field] = None
                continue
            typ = spec['type']
            try:
                if typ == 'string':
                    if not isinstance(value, str):
                        raise ValueError('IDs and strings must already be strings')
                    parsed = value
                elif typ == 'date':
                    parsed = _date(value, field).isoformat()
                elif typ == 'boolean':
                    if value is True or value == 'true':
                        parsed = True
                    elif value is False or value == 'false':
                        parsed = False
                    else:
                        raise ValueError('Use true or false')
                else:
                    if isinstance(value, bool):
                        raise ValueError('Boolean is not numeric')
                    text = str(value)
                    if isinstance(value, str) and decimal_mark == ',':
                        if '.' in text:
                            raise ValueError('Unexpected decimal point')
                        text = text.replace(',', '.')
                    if isinstance(value, str) and not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', text.strip()):
                        raise ValueError('Invalid decimal numeric syntax')
                    if typ == 'integer' and isinstance(value, str) and not re.fullmatch(r'[+-]?[0-9]+', text.strip()):
                        raise ValueError('Integer strings must contain only signed digits')
                    parsed = float(text)
                    if not math.isfinite(parsed):
                        raise ValueError('Nonfinite number')
                    if typ == 'integer':
                        if not parsed.is_integer() or abs(parsed) > 9007199254740991:
                            raise ValueError('Integer must be exact within JSON safe range')
                        parsed = int(parsed)
                out[field] = parsed
            except (ValueError, TypeError, OverflowError) as exc:
                _fail('DATA_TYPE', f'Row {i + 1}, {field}: {exc}')
        output.append(out)
    if schema.get('primary_key'):
        keys = [schema['primary_key']] if isinstance(schema['primary_key'], str) else schema['primary_key']
        if not isinstance(keys, (list, tuple)) or any(k not in columns for k in keys):
            _fail('DATA_SCHEMA', 'Primary key columns must exist in schema')
        _unique(output, schema['primary_key'], 'dataset')
    return _result(output, 'data.read_local_dataset', diagnostics={'row_count': len(output), 'schema_version': '1.0', 'units': {k: v.get('unit') for k, v in columns.items()}}, provenance={'source_sha256': hashlib.sha256(raw).hexdigest(), 'data_origin': 'USER_SUPPLIED'})


def validate_credit_dataset(tables, contract_version='1.0'):
    """Report all relational violations in supported canonical credit tables."""
    if contract_version != '1.0' or not isinstance(tables, dict):
        _fail('DATA_SCHEMA', 'Expected table mapping and contract version 1.0')
    definitions = {'obligors': ('entity_id',), 'observations': ('entity_id', 'observed_at'), 'facilities': ('facility_id',), 'defaults': ('episode_id',), 'recoveries': ('cashflow_id',)}
    if set(tables) - set(definitions):
        _fail('DATA_SCHEMA', 'Unknown canonical table')
    required = {'obligors': ['entity_id'], 'observations': ['entity_id', 'observed_at', 'available_at'], 'facilities': ['facility_id', 'entity_id', 'ead'], 'defaults': ['episode_id', 'entity_id', 'default_at'], 'recoveries': ['cashflow_id', 'episode_id', 'cashflow_at', 'amount', 'kind']}
    data = {k: _rows(v, k) for k, v in tables.items()}
    issues = []
    def issue(t, i, code, field):
        issues.append({'table': t, 'row': i + 1, 'code': code, 'field': field})
    for table, rows in data.items():
        seen = set()
        for i, row in enumerate(rows):
            for f in required[table]:
                if row.get(f) is None:
                    issue(table, i, 'MISSING_FIELD', f)
            key = tuple(row.get(f) for f in definitions[table])
            if any(v is not None and (not isinstance(v, str) or not v) for v in key):
                issue(table, i, 'INVALID_KEY', ','.join(definitions[table]))
            if all(isinstance(v, str) and v for v in key):
                if key in seen:
                    issue(table, i, 'DUPLICATE_KEY', ','.join(definitions[table]))
                seen.add(key)
            for f in ('ead', 'amount'):
                if f in row and (isinstance(row[f], bool) or not isinstance(row[f], (int, float)) or not math.isfinite(row[f]) or row[f] < 0):
                    issue(table, i, 'INVALID_AMOUNT', f)
            for f in ('observed_at', 'available_at', 'default_at', 'cashflow_at', 'cure_at', 'closed_at'):
                if row.get(f) is not None:
                    try:
                        _date(row[f], f)
                    except Exception:
                        issue(table, i, 'INVALID_DATE', f)
            for early, late in [('observed_at', 'available_at'), ('default_at', 'cure_at'), ('default_at', 'closed_at')]:
                if row.get(early) is not None and row.get(late) is not None and str(row[early]) > str(row[late]):
                    issue(table, i, 'DATE_ORDER', late)
            if table == 'recoveries' and row.get('kind') not in ('recovery', 'cost'):
                issue(table, i, 'INVALID_KIND', 'kind')
    for table, field, parent, parent_key in [('observations', 'entity_id', 'obligors', 'entity_id'), ('facilities', 'entity_id', 'obligors', 'entity_id'), ('defaults', 'entity_id', 'obligors', 'entity_id'), ('recoveries', 'episode_id', 'defaults', 'episode_id')]:
        parent_ids = {r.get(parent_key) for r in data.get(parent, []) if isinstance(r.get(parent_key), str)}
        for i, row in enumerate(data.get(table, [])):
            if not isinstance(row.get(field), str) or row.get(field) not in parent_ids:
                issue(table, i, 'FOREIGN_KEY', field)
    defaults = {r.get('episode_id'): r for r in data.get('defaults', []) if isinstance(r.get('episode_id'), str)}
    for i, row in enumerate(data.get('recoveries', [])):
        d = defaults.get(row.get('episode_id')) if isinstance(row.get('episode_id'), str) else None
        if d and row.get('cashflow_at') and d.get('default_at') and str(row['cashflow_at']) < str(d['default_at']):
            issue('recoveries', i, 'BEFORE_DEFAULT', 'cashflow_at')
    return _result({'valid': not issues, 'violations': issues, 'row_counts': {k: len(v) for k, v in data.items()}}, 'data.validate_credit_dataset')


def join_as_of(left, right, entity_keys=('entity_id',), as_of='as_of', known_at='known_at', observed_at='observed_at', available_at='available_at', release_id='release_id', suffix='_right'):
    """Select the latest observable fact and then its latest known revision."""
    left, right = _rows(left, 'left'), _rows(right, 'right')
    if not entity_keys or not suffix:
        _fail('DATA_SCHEMA', 'entity_keys and suffix must be nonempty')
    _unique(right, [release_id], 'right releases')
    for row in right:
        _required(row, list(entity_keys) + [observed_at, available_at])
        _date(row[observed_at], observed_at)
        _date(row[available_at], available_at)
    cols = sorted(set().union(*(r.keys() for r in right)) - set(entity_keys)) if right else []
    out = []
    for row in left:
        _required(row, list(entity_keys) + [as_of, known_at])
        a, k = _date(row[as_of], as_of), _date(row[known_at], known_at)
        candidates = [r for r in right if all(r[f] == row[f] for f in entity_keys) and _date(r[observed_at], observed_at) <= a and _date(r[available_at], available_at) <= k]
        selected = None
        if candidates:
            order = max((r[observed_at], r[available_at]) for r in candidates)
            best = [r for r in candidates if (r[observed_at], r[available_at]) == order]
            if len(best) != 1:
                _fail('DATA_AMBIGUOUS_JOIN', 'Multiple releases tie on observation and availability dates')
            selected = best[0]
        result = dict(row)
        for col in cols:
            target = col + suffix if col in row else col
            if target in result:
                _fail('DATA_COLUMN_COLLISION', f'Join output collision: {target}')
            result[target] = selected[col] if selected and col in selected else None
        out.append(result)
    return _result(out, 'data.join_as_of', diagnostics={'ordering': [observed_at, available_at], 'right_rows': len(right)})


def build_observation_windows(ratings, default_events, exits, horizon_days, evaluation_cutoff, entity_key='entity_id'):
    """Create first-default targets on (origin, origin+horizon], preserving censoring."""
    horizon_days = _days(horizon_days, 'horizon_days', 1)
    cutoff = _date(evaluation_cutoff, 'evaluation_cutoff')
    ratings, events, exits = _rows(ratings, 'ratings'), _rows(default_events, 'default_events'), _rows(exits, 'exits')
    _unique(ratings, ['observation_id'], 'ratings')
    _unique(events, ['event_id'], 'default_events')
    for collection, time in [(events, 'default_at'), (exits, 'exit_at')]:
        for row in collection:
            _required(row, [entity_key, time, 'known_at'])
            if _date(row['known_at'], 'known_at') < _date(row[time], time):
                _fail('DATA_DATE_ORDER', 'Event cannot be known before occurrence')
    out = []
    for r in ratings:
        _required(r, [entity_key, 'as_of'])
        start = _date(r.get('performance_start', r['as_of']), 'performance_start')
        if start < _date(r['as_of'], 'as_of'):
            _fail('DATA_DATE_ORDER', 'Performance origin precedes prediction')
        end = start + timedelta(days=horizon_days)
        followup = _date(r.get('followup_end', evaluation_cutoff), 'followup_end')
        stop = min(end, cutoff, followup)
        known_events = [e for e in events if e[entity_key] == r[entity_key] and _date(e['known_at'], 'known_at') <= cutoff]
        prior = any(_date(e['default_at'], 'default_at') <= start for e in known_events)
        relevant_exits = [_date(e['exit_at'], 'exit_at') for e in exits if e[entity_key] == r[entity_key] and _date(e['known_at'], 'known_at') <= cutoff and _date(e['exit_at'], 'exit_at') > start]
        exit_date = min(relevant_exits) if relevant_exits else None
        if exit_date:
            stop = min(stop, exit_date)
        matches = sorted([e for e in known_events if start < _date(e['default_at'], 'default_at') <= stop], key=lambda e: (e['default_at'], e['event_id']))
        funded = r.get('funded', True)
        if not isinstance(funded, bool):
            _fail('DATA_SCHEMA', 'funded must be boolean')
        event = matches[0] if matches else None
        pending = r.get('outcome_available_at') is not None and _date(r['outcome_available_at'], 'outcome_available_at') > cutoff
        label, reason = (None, 'not_funded') if not funded else (None, 'prior_default') if prior else (None, 'not_started') if stop < start else (1, 'default') if event else (None, 'exit') if exit_date and exit_date <= min(end, cutoff, followup) else (None, 'outcome_pending') if pending else (0, 'mature') if stop == end else (None, 'right_censored')
        observed_until = _date(event['default_at'], 'default_at') if label == 1 else max(start, stop)
        out.append({**r, 'performance_start': start.isoformat(), 'outcome_end': end.isoformat(), 'observed_until': observed_until.isoformat(), 'target': label, 'censored': reason in ('exit', 'right_censored', 'not_started', 'outcome_pending'), 'eligible': funded and not prior and stop >= start, 'label_status': reason, 'event_id': event['event_id'] if label == 1 else None})
    return _result(out, 'data.build_observation_windows', unit='binary', diagnostics={'boundary': '(start,end]', 'target': 'first_default', 'exit_policy': 'censor; default wins same-day ties'})


def build_recovery_episodes(default_events, recovery_cashflows, cures=None, reopening_days=90, evaluation_cutoff=None, *, closures=None):
    """Link repeated defaults to workouts using explicit cures and dated cashflows."""
    reopening_days = _days(reopening_days, 'reopening_days')
    defaults = _rows(default_events, 'default_events')
    cashflows = _rows(recovery_cashflows, 'recovery_cashflows')
    cures = _rows(cures or [], 'cures')
    cutoff = _date(evaluation_cutoff, 'evaluation_cutoff') if evaluation_cutoff else date.max
    _unique(defaults, ['episode_id'], 'default_events')
    _unique(cashflows, ['cashflow_id'], 'recovery_cashflows')
    _unique(cures, ['episode_id'], 'cures')
    for d in defaults:
        _required(d, ['episode_id', 'entity_id', 'default_at', 'known_at', 'ead', 'currency'])
        if not isinstance(d['ead'], (int, float)) or isinstance(d['ead'], bool) or not math.isfinite(d['ead']) or d['ead'] < 0:
            _fail('DATA_AMOUNT', 'EAD must be finite and nonnegative')
        if _date(d['known_at'], 'known_at') < _date(d['default_at'], 'default_at'):
            _fail('DATA_DATE_ORDER', 'Default known before occurrence')
    default_map = {d['episode_id']: d for d in defaults}
    cure_map = {}
    for c in cures:
        _required(c, ['episode_id', 'cure_at', 'known_at'])
        if c['episode_id'] not in default_map:
            _fail('DATA_FOREIGN_KEY', 'Cure references unknown default')
        if _date(c['cure_at'], 'cure_at') < _date(default_map[c['episode_id']]['default_at'], 'default_at') or _date(c['known_at'], 'known_at') < _date(c['cure_at'], 'cure_at'):
            _fail('DATA_DATE_ORDER', 'Invalid cure dates')
        if _date(c['known_at'], 'known_at') <= cutoff:
            cure_map[c['episode_id']] = c
    visible = sorted([d for d in defaults if _date(d['known_at'], 'known_at') <= cutoff], key=lambda d: (d['entity_id'], d['default_at'], d['episode_id']))
    _unique(visible, ['entity_id', 'default_at'], 'entity defaults')
    episodes, links, last = [], {}, {}
    for d in visible:
        previous = last.get(d['entity_id'])
        reopen = False
        if previous:
            c = cure_map.get(previous['episode_id'])
            reopen = c is None or _date(d['default_at'], 'default_at') <= _date(c['cure_at'], 'cure_at') + timedelta(days=reopening_days)
        if reopen:
            episode = links[previous['episode_id']]
            if episode['currency'] != d['currency']:
                _fail('DATA_CURRENCY', 'Reopened workout currency differs')
            episode['default_ids'].append(d['episode_id'])
        else:
            episode = {'workout_id': d['episode_id'], 'entity_id': d['entity_id'], 'default_at': d['default_at'], 'ead_at_first_default': d['ead'], 'currency': d['currency'], 'default_ids': [d['episode_id']], 'cashflow_ids': [], 'recoveries': 0.0, 'costs': 0.0, 'closed_at': None, 'status': 'open'}
            episodes.append(episode)
        links[d['episode_id']] = episode
        last[d['entity_id']] = d
        episode['closed_at'] = None
        episode['status'] = 'open'
        cure = cure_map.get(d['episode_id'])
        if cure and cure.get('closed_at') is not None:
            closed = _date(cure['closed_at'], 'closed_at')
            if closed < _date(cure['cure_at'], 'cure_at'):
                _fail('DATA_DATE_ORDER', 'Closure precedes cure')
            _required(cure, ['closed_known_at'])
            closed_known = _date(cure['closed_known_at'], 'closed_known_at')
            if closed_known < closed:
                _fail('DATA_DATE_ORDER', 'Closure cannot be known before closure date')
            if closed_known <= cutoff:
                episode['closed_at'], episode['status'] = closed.isoformat(), 'closed'
    # Legal termination does not imply cure or reset the borrower default state.
    terminals = _rows([] if closures is None else closures, 'closures')
    _unique(terminals, ['closure_id'], 'closures')
    _unique(terminals, ['episode_id'], 'closures')
    closed_workouts = set()
    for terminal in terminals:
        _required(terminal, ['closure_id', 'episode_id', 'closed_at', 'known_at', 'closure_reason'])
        key = terminal['episode_id']
        if key not in default_map:
            _fail('DATA_FOREIGN_KEY', 'Closure references unknown default')
        if terminal['closure_reason'] not in ('SETTLEMENT', 'WRITE_OFF', 'LEGAL_TERMINATION'):
            _fail('DATA_SCHEMA', 'Unknown legal closure reason')
        closed, known = _date(terminal['closed_at'], 'closed_at'), _date(terminal['known_at'], 'known_at')
        if closed < _date(default_map[key]['default_at'], 'default_at') or known < closed:
            _fail('DATA_DATE_ORDER', 'Invalid legal closure dates')
        if known > cutoff:
            continue
        if key not in links:
            _fail('DATA_FOREIGN_KEY', 'Closure visible before its default')
        ep = links[key]
        if ep['workout_id'] in closed_workouts or any(cure_map.get(i, {}).get('closed_at') is not None and _date(cure_map[i]['closed_known_at'], 'closed_known_at') <= cutoff for i in ep['default_ids']):
            _fail('DATA_TERMINAL_CONFLICT', 'Multiple terminal records for one workout')
        if any(_date(default_map[i]['default_at'], 'default_at') > closed for i in ep['default_ids']):
            _fail('DATA_DATE_ORDER', 'Legal closure precedes a default in its continuous workout')
        closed_workouts.add(ep['workout_id'])
        ep.update(closed_at=closed.isoformat(), status='closed', closure_id=terminal['closure_id'], closure_reason=terminal['closure_reason'])
    allocated = []
    for cf in cashflows:
        _required(cf, ['cashflow_id', 'episode_id', 'cashflow_at', 'known_at', 'amount', 'kind', 'currency'])
        if cf['episode_id'] not in default_map:
            _fail('DATA_FOREIGN_KEY', 'Cashflow references unknown default')
        when, known = _date(cf['cashflow_at'], 'cashflow_at'), _date(cf['known_at'], 'known_at')
        if when < _date(default_map[cf['episode_id']]['default_at'], 'default_at') or known < when:
            _fail('DATA_DATE_ORDER', 'Invalid recovery dates')
        if cf['kind'] not in ('recovery', 'cost') or isinstance(cf['amount'], bool) or not isinstance(cf['amount'], (float, int)) or not math.isfinite(cf['amount']) or cf['amount'] < 0:
            _fail('DATA_AMOUNT', 'Cashflow needs nonnegative amount and recovery/cost kind')
        if known > cutoff:
            continue
        if cf['episode_id'] not in links:
            _fail('DATA_FOREIGN_KEY', 'Cashflow is visible before its default')
        ep = links[cf['episode_id']]
        if cf['currency'] != ep['currency']:
            _fail('DATA_CURRENCY', 'Cashflow currency differs from workout')
        if ep['closed_at'] and when > _date(ep['closed_at'], 'closed_at'):
            _fail('DATA_CLOSED_EPISODE', 'Cashflow after explicit closure requires revised closure or reopening')
        ep['cashflow_ids'].append(cf['cashflow_id'])
        ep['recoveries' if cf['kind'] == 'recovery' else 'costs'] += cf['amount']
        allocated.append({**cf, 'workout_id': ep['workout_id'], 'days_since_first_default': (when - _date(ep['default_at'], 'default_at')).days})
    for ep in episodes:
        ep['net_recovery'] = ep['recoveries'] - ep['costs']
    return _result({'episodes': episodes, 'cashflows': allocated}, 'data.build_recovery_episodes', limitations=('Undiscounted currency amounts; original cashflow dates retained for discounting.', 'Cure alone does not establish workout closure; only explicit closed_at does.'))


def make_grouped_temporal_splits(panel, train_end, calibration_end, test_end, embargo_days=0, entity_key='entity_id', group_key='group_id', time_key='observed_at', outcome_end_key='outcome_end'):
    """Assign connected entities/groups wholly to their latest temporal partition."""
    panel = _rows(panel, 'panel')
    embargo = timedelta(days=_days(embargo_days, 'embargo_days'))
    t1, t2, t3 = (_date(v, n) for v, n in [(train_end, 'train_end'), (calibration_end, 'calibration_end'), (test_end, 'test_end')])
    if not t1 < t2 < t3:
        _fail('DATA_DATE_ORDER', 'Split boundaries must strictly increase')
    _unique(panel, ['observation_id'], 'panel')
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in panel:
        _required(row, [entity_key, time_key, outcome_end_key])
        if not isinstance(row[entity_key], str) or (row.get(group_key) is not None and not isinstance(row[group_key], str)):
            _fail('DATA_SCHEMA', 'Entity and group identifiers must be strings')
        a = root(('e', row[entity_key]))
        if row.get(group_key) is not None:
            b = root(('g', row[group_key]))
            parent[b] = a
    candidates, latest = [], {}
    for row in panel:
        time, end = _date(row[time_key], time_key), _date(row[outcome_end_key], outcome_end_key)
        if end < time:
            _fail('DATA_DATE_ORDER', 'Outcome end precedes observation')
        part = 0 if time <= t1 else 1 if time <= t2 else 2 if time <= t3 else 3
        component = root(('e', row[entity_key]))
        latest[component] = max(latest.get(component, -1), part)
        candidates.append((row, time, end, part, component))
    names = ['train', 'calibration', 'test']
    assignments = []
    for row, time, end, part, component in candidates:
        reason = None
        if part == 3:
            reason = 'outside_test_window'
        elif part != latest[component]:
            reason = 'group_isolation'
        elif (part == 0 and end + embargo > t1) or (part == 1 and end + embargo > t2) or (part == 2 and end > t3):
            reason = 'outcome_overlap'
        elif (part == 1 and time <= t1 + embargo) or (part == 2 and time <= t2 + embargo):
            reason = 'embargo'
        assignments.append({'observation_id': row['observation_id'], 'split': 'excluded' if reason else names[part], 'reason': reason})
    return _result({'assignments': assignments, 'counts': {name: sum(a['split'] == name for a in assignments) for name in names + ['excluded']}, 'group_overlap': False}, 'data.make_grouped_temporal_splits', diagnostics={'group_rule': 'connected entity/group components reserved to latest partition, including excluded future rows', 'embargo_days': embargo_days})
