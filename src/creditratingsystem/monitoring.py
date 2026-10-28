"""Local dated model-role histories and explicit monitoring escalation policies."""
from copy import deepcopy

from .core import CalcResult, _fail, _fingerprint, _result
from .data import _date, _rows, _required, _unique
from .rating_system import _registry, _object, _string, _seal, _sealed


def _history(history):
    if history is None:
        return _seal(dict(schema_version="1.0", kind="ModelAssignmentHistory", events=[]), "history_hash")
    h = _sealed(history, "history_hash", "assignment history")
    if h.get("kind") != "ModelAssignmentHistory":
        _fail("INVALID_ASSIGNMENT_HISTORY", "Expected a model-assignment history")
    rows = _rows(h.get("events"), "assignment events")
    _unique(rows, ["event_id"], "assignment events")
    seen = set()
    for row in rows:
        _assignment_fields(row)
        key = tuple(row[k] for k in ("profile_id", "purpose", "role", "slot_id", "effective_from"))
        if key in seen:
            _fail("ASSIGNMENT_CONFLICT", "Multiple events at the same role/slot effective boundary")
        seen.add(key)
    return h


def _assignment_fields(event):
    _required(event, ["event_id", "profile_id", "purpose", "role", "slot_id", "model_id",
                      "model_version", "model_hash", "effective_from", "known_at",
                      "evaluation_reference", "reason", "required_feature_order"])
    for key in ("event_id", "profile_id", "purpose", "slot_id", "model_id", "model_version",
                "model_hash", "evaluation_reference", "reason"):
        _string(event[key], key)
    if event["role"] not in ("champion", "challenger"):
        _fail("INVALID_MODEL_ROLE", "Declare champion or challenger")
    if event["role"] == "champion" and event["slot_id"] != "champion":
        _fail("INVALID_MODEL_ROLE", "A scope has only one champion slot")
    _date(event["effective_from"], "effective_from")
    _date(event["known_at"], "known_at")
    names = event["required_feature_order"]
    if not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        _fail("FEATURE_MISMATCH", "Declare a unique ordered feature schema")


def _assignment_projection(history, known_at, role="champion"):
    h = _history(history)
    cutoff = _date(known_at, "known_at")
    visible = [e for e in h["events"] if _date(e["known_at"], "event.known_at") <= cutoff and e["role"] == role]
    groups = {}
    for e in visible:
        groups.setdefault(tuple(e[k] for k in ("profile_id", "purpose", "role", "slot_id")), []).append(e)
    assignments = []
    for key in sorted(groups):
        events = sorted(groups[key], key=lambda e: e["effective_from"])
        for i, event in enumerate(events):
            assignments.append(dict(assignment_id=event["event_id"], profile_id=event["profile_id"],
                purpose=event["purpose"], role=event["role"], slot_id=event["slot_id"],
                model_id=event["model_id"], model_version=event["model_version"], model_hash=event["model_hash"],
                effective_from=event["effective_from"],
                effective_to=events[i+1]["effective_from"] if i+1<len(events) else None,
                known_at=event["known_at"]))
    return assignments


def assign_model_role(history, registry, assignment, *, context=None):
    """Append a content-bound model-role event; historical states stay reconstructible.

    Role changes implicitly end the preceding assignment in that role slot at
    the new effective date. End dates are derived at the requested knowledge
    cutoff; the original events are never rewritten. Scheduled later changes
    remain active even if an earlier rollback is subsequently added.
    """
    h = _history(history)
    registry = _registry(registry)
    event = _object(assignment, "assignment")
    _assignment_fields(event)
    existing = [e for e in h["events"] if e["event_id"] == event["event_id"]]
    if existing:
        if existing[0] != event:
            _fail("ASSIGNMENT_CONFLICT", "Event id already binds different content")
        return _result(h, "monitoring.assign_model_role", "assignment_history", context,
                       diagnostics={"idempotent": True})
    if h["events"] and event["known_at"] < max(e["known_at"] for e in h["events"]):
        _fail("ASSIGNMENT_DATE_ORDER", "Append events in nondecreasing knowledge order")
    model = [e for e in registry["entries"] if e["model_id"] == event["model_id"] and e["model_version"] == event["model_version"]]
    if not model or model[0]["registered_at"] > event["known_at"]:
        _fail("MISSING_MODEL", "Model must be registered and known when assigned")
    if model[0]["model_hash"] != event["model_hash"]:
        _fail("RATING_INTEGRITY", "Assignment must pin the registered content hash")
    if model[0]["bundle"]["model"]["feature_order"] != event["required_feature_order"]:
        _fail("FEATURE_MISMATCH", "Model feature order differs from declared deployment schema")
    h["events"].append(event)
    h = _seal(h, "history_hash")
    _history(h)
    return _result(h, "monitoring.assign_model_role", "assignment_history", context,
        diagnostics={"idempotent": False},
        limitations=("Role and evaluation references are local user assertions; no model deployment or institutional approval occurs.",))


def rollback_model_assignment(history, registry, target_event_id, rollback_event, *, context=None):
    """Append a new dated assignment to a pinned previously assigned model."""
    h = _history(history)
    _string(target_event_id, "target_event_id")
    target = [e for e in h["events"] if e["event_id"] == target_event_id]
    if not target:
        _fail("MISSING_ASSIGNMENT", "Rollback requires an explicit historical event")
    event = _object(rollback_event, "rollback_event")
    _required(event, ["event_id", "effective_from", "known_at", "reason", "evaluation_reference", "required_feature_order"])
    prior = target[0]
    if _date(prior["known_at"], "target.known_at") > _date(event["known_at"], "known_at"):
        _fail("ASSIGNMENT_DATE_ORDER", "Rollback cannot use an unknown target event")
    if _date(prior["effective_from"], "target.effective_from") >= _date(event["effective_from"], "effective_from"):
        _fail("ASSIGNMENT_DATE_ORDER", "Rollback must occur after the target assignment began")
    for key in ("profile_id", "purpose", "role", "slot_id", "model_id", "model_version", "model_hash"):
        if key in event and event[key] != prior[key]:
            _fail("ASSIGNMENT_CONFLICT", "Rollback cannot change the pinned target model or scope")
        event[key] = prior[key]
    event["rollback_of"] = target_event_id
    result = assign_model_role(h, registry, event, context=context)
    return _result(result.value, "monitoring.rollback_model_assignment", "assignment_history", context,
                   diagnostics=result.diagnostics, limitations=result.limitations)


def classify_monitoring_findings(metric_panel, policy, *, context=None):
    """Apply predeclared thresholds and consecutive-period requirements locally.

    Each series/metric needs a record at every declared period. Missing and
    non-applicable values break persistence and remain visible as insufficient
    information. Thresholds and directions are supplied, not regulatory defaults.
    """
    p = _object(policy, "monitoring policy")
    _required(p, ["schema_version", "policy_id", "version", "periods", "series_ids", "rules"])
    if p["schema_version"] != "1.0":
        _fail("UNSUPPORTED_SCHEMA_VERSION", "Unsupported monitoring policy")
    _string(p["policy_id"], "policy_id"); _string(p["version"], "version")
    periods = p["periods"]
    if not isinstance(periods, list) or not periods or any(not isinstance(x,str) or not x for x in periods) or len(set(periods)) != len(periods):
        _fail("INVALID_POLICY", "Declare a distinct ordered period grid")
    rules = _rows(p["rules"], "rules"); _unique(rules, ["rule_id"], "rules")
    if not rules:
        _fail("INVALID_POLICY", "At least one monitoring rule required")
    panel = _rows(metric_panel, "metric_panel")
    _unique(panel, ["series_id", "metric", "period"], "metric_panel")
    for row in panel:
        _required(row, ["series_id", "metric", "period", "status"])
        if "value" not in row:
            _fail("INVALID_INPUT", "Metric value must be present, possibly null")
        if row["period"] not in periods or row["status"] not in ("OK", "MISSING_OUTCOME", "INSUFFICIENT_INFORMATION", "TECHNICAL_ERROR", "NOT_APPLICABLE"):
            _fail("INVALID_INPUT", "Unknown monitoring period or metric status")
    series = p["series_ids"]
    if not isinstance(series, list) or not series or any(not isinstance(x,str) or not x for x in series) or len(set(series))!=len(series):
        _fail("INVALID_POLICY", "Declare all expected series identifiers")
    if any(r["series_id"] not in series for r in panel):
        _fail("INVALID_INPUT", "Metric panel contains an undeclared series")
    series = sorted(series)
    traces = []
    compare = {"gt": lambda a,b:a>b, "ge": lambda a,b:a>=b, "lt": lambda a,b:a<b, "le":lambda a,b:a<=b}
    for rule in rules:
        _required(rule, ["rule_id", "metric", "operator", "threshold", "consecutive_periods", "severity", "action_code"])
        if rule["operator"] not in compare or type(rule["consecutive_periods"]) is not int or rule["consecutive_periods"]<1 or rule["severity"] not in ("WARNING", "CRITICAL"):
            _fail("INVALID_POLICY", "Invalid threshold direction, persistence or severity")
        from .core import _numeric
        threshold = _numeric(rule["threshold"], "threshold")
        if threshold.ndim:
            _fail("INVALID_POLICY", "Scalar threshold required")
        _string(rule["action_code"], "action_code")
        for sid in series:
            streak = 0
            rows = {r["period"]:r for r in panel if r["series_id"]==sid and r["metric"]==rule["metric"]}
            for period in periods:
                row = rows.get(period)
                status = "INSUFFICIENT_INFORMATION" if row is None else row["status"]
                if status=="OK" and row["value"] is not None:
                    value = _numeric(row["value"], "metric value")
                    if value.ndim:
                        _fail("INVALID_INPUT", "Metric value must be scalar")
                    breached = bool(compare[rule["operator"]](float(value), float(threshold)))
                    streak = streak+1 if breached else 0
                    assessment = rule["severity"] if streak>=rule["consecutive_periods"] else "WATCH" if breached else "WITHIN_POLICY"
                else:
                    if status=="OK": status="INSUFFICIENT_INFORMATION"
                    breached = None; streak = 0; assessment = status
                traces.append(dict(series_id=sid,metric=rule["metric"],period=period,rule_id=rule["rule_id"],
                    breached=breached,streak=streak,assessment=assessment,
                    action_code=rule["action_code"] if assessment in ("WARNING","CRITICAL") else None))
    if not traces:
        status="INSUFFICIENT_INFORMATION"
    else:
        states={r["assessment"] for r in traces if r["period"]==periods[-1]}
        order=("TECHNICAL_ERROR","CRITICAL","WARNING","MISSING_OUTCOME","INSUFFICIENT_INFORMATION","NOT_APPLICABLE","WATCH","WITHIN_POLICY")
        status=next(s for s in order if s in states)
    return _result(dict(status=status,policy_id=p["policy_id"],policy_version=p["version"],
                        policy_hash=_fingerprint(p),trace=traces),"monitoring.classify_monitoring_findings","monitoring_assessment",context,
        limitations=("Thresholds and persistence are supplied policy, not universal significance levels. No automatic recalibration, model promotion or external message occurs.",))

from ._monitoring_metrics import (
    build_monitoring_cohorts, monitor_rating_cohort, detect_credit_feature_drift, detect_pd_calibration_drift, compare_challenger_cohorts, diagnose_fairness_slices
)
