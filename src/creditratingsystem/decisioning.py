"""Deterministic local credit policies, separate from risk estimation."""
from copy import deepcopy
from datetime import date
import math

from .core import CalcResult, _fail, _fingerprint, _result

_ACTIONS = ("APPROVE", "REFER", "DECLINE", "BLOCKED")
_SOURCES = ("risk", "affordability", "eligibility", "operations")


def _text(x, name):
    if not isinstance(x, str) or not x.strip():
        _fail("INVALID_POLICY", f"{name} must be nonempty text")


def _date(x):
    try:
        if not isinstance(x, str) or date.fromisoformat(x).isoformat() != x:
            raise ValueError()
    except (ValueError, TypeError):
        _fail("INVALID_DATE", "Dates must be ISO YYYY-MM-DD")
    return x


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _compare(x, op, y):
    return {"eq": lambda: type(x) is type(y) and x == y or _number(x) and _number(y) and x == y,
            "ne": lambda: not _compare(x, "eq", y),
            "lt": lambda: x < y, "le": lambda: x <= y,
            "gt": lambda: x > y, "ge": lambda: x >= y}[op]()


def _possible(predicates, fields):
    """Exact conjunction satisfiability over finite categories/binary64 numbers."""
    for key in {p["field"] for p in predicates}:
        ps = [p for p in predicates if p["field"] == key]
        field = fields[key]
        if field["type"] == "number":
            boundaries = [p["value"] for p in ps]
            candidates = boundaries + [math.nextafter(v, d) for v in boundaries
                                        for d in (-math.inf, math.inf)]
            candidates = [v for v in candidates if math.isfinite(v)]
        else:
            candidates = field["values"]
        if not any(all(_compare(x, p["op"], p["value"]) for p in ps) for x in candidates):
            return False
    return True


def _approval_contract(contract, fields):
    if not isinstance(contract, dict) or set(contract) != {"domain", "affordability", "required_gates"}:
        _fail("INVALID_POLICY", "Explicit domain, affordability applicability and required gates required")
    if contract["domain"] not in ("RETAIL_CREDIT", "WHOLESALE_CREDIT", "INVESTMENT", "COUNTERPARTY_LIMIT"):
        _fail("INVALID_POLICY", "Unsupported decision domain")
    a = contract["affordability"]
    if not isinstance(a, dict) or a.get("mode") not in ("REQUIRED", "NOT_APPLICABLE"):
        _fail("INVALID_POLICY", "Declare affordability applicability")
    if a["mode"] == "REQUIRED":
        if set(a) != {"mode"}:
            _fail("INVALID_POLICY", "Required affordability has no waiver metadata")
    else:
        if contract["domain"] not in ("INVESTMENT", "COUNTERPARTY_LIMIT") or set(a) != {"mode", "reason", "reference", "maximum_age_days"}:
            _fail("INVALID_POLICY", "Documented inapplicability is restricted to investment/counterparty domains")
        for key in ("reason", "reference"):
            _text(a[key], key)
        age = a["maximum_age_days"]
        if not _number(age) or age < 0 or age != math.floor(age):
            _fail("INVALID_POLICY", "Applicability evidence needs a nonnegative whole-day maximum age")
    gates = contract["required_gates"]
    if not isinstance(gates, list):
        _fail("INVALID_POLICY", "Required gates must be an array")
    seen = set()
    for gate in gates:
        if not isinstance(gate, dict) or set(gate) != {"field", "value"} or not isinstance(gate["field"], str):
            _fail("INVALID_POLICY", "Each gate requires exactly a field and categorical value")
        key = gate["field"]
        if key in seen or key not in fields or fields[key]["type"] != "category" or not isinstance(gate["value"], str) or gate["value"] not in fields[key]["values"]:
            _fail("INVALID_POLICY", "Distinct declared categorical approval gates required")
        seen.add(key)


def _applicability_findings(contract, inputs, as_of):
    a = contract["affordability"]
    if a["mode"] != "NOT_APPLICABLE":
        return []
    obj = inputs["affordability"]
    findings = []
    if obj.get("applicability_reference") != a["reference"] or obj.get("domain") != contract["domain"]:
        findings.append("affordability.APPLICABILITY_BINDING")
    if "as_of" in obj and (date.fromisoformat(as_of) - date.fromisoformat(obj["as_of"])).days > a["maximum_age_days"]:
        findings.append("affordability.STALE_APPLICABILITY")
    if obj.get("valid_until") is None:
        findings.append("affordability.MISSING_VALIDITY")
    else:
        until = _date(obj["valid_until"])
        if until <= obj.get("as_of", as_of) or not as_of < until:
            findings.append("affordability.EXPIRED_APPLICABILITY")
    return findings


def compile_credit_decision_policy(specification, *, context=None):
    """Validate a dated rule table and freeze its canonical content fingerprint.

    Smaller integer priorities execute first. Exceptions are explicit rules at
    higher priority. All conditions within a rule must hold. Missing required
    inputs and approval gates are evaluated before the ordered rule table.
    """
    if not isinstance(specification, dict):
        _fail("INVALID_POLICY", "Policy must be an object")
    s = deepcopy(specification)
    if s.get("schema_version") not in ("1.0", "1.1"):
        _fail("UNSUPPORTED_SCHEMA_VERSION", "Unsupported decision policy schema")
    if s["schema_version"] == "1.0" and "approval_contract" in s:
        _fail("INVALID_POLICY", "Approval contracts require specification schema 1.1")
    for k in ("policy_id", "version", "approval_reference"):
        _text(s.get(k), k)
    for k in ("known_at", "effective_from", "effective_until"):
        _date(s.get(k))
    if s["effective_from"] >= s["effective_until"]:
        _fail("INVALID_POLICY", "Effective interval must be nonempty and end-exclusive")
    if s.get("default_action") not in ("REFER", "DECLINE", "BLOCKED"):
        _fail("INVALID_POLICY", "Unmatched cases require an explicit nonapproval action")
    if s.get("missing_action") not in ("REFER", "BLOCKED"):
        _fail("INVALID_POLICY", "Missing inputs must refer or block")
    target = s.get("target")
    if not isinstance(target, dict) or any(k not in target for k in
            ("event", "horizon", "time_unit", "probability_measure", "conditioning")):
        _fail("INVALID_POLICY", "Declare the risk target used by this policy")
    if not _number(target["horizon"]) or target["horizon"] <= 0 or target["probability_measure"] != "PHYSICAL":
        _fail("INVALID_POLICY", "Credit policy requires a positive-horizon physical risk target")
    for k in ("event", "time_unit", "conditioning"):
        _text(target[k], k)
    if target["time_unit"] not in ("days", "months", "years"):
        _fail("INVALID_POLICY", "Target time unit must be days, months or years")
    fields = s.get("fields")
    if not isinstance(fields, dict) or not fields:
        _fail("INVALID_POLICY", "Declare typed source fields")
    for key, f in fields.items():
        if not isinstance(key, str) or len(key.split(".")) != 2 or key.split(".")[0] not in _SOURCES or not key.split(".")[1]:
            _fail("INVALID_POLICY", "Fields use source.name with four separate sources")
        if not isinstance(f, dict) or f.get("type") not in ("number", "category"):
            _fail("INVALID_POLICY", "Field type must be number or category")
        if f["type"] == "category":
            vs = f.get("values")
            if not isinstance(vs, list) or not vs or any(not isinstance(v, str) or not v for v in vs) or len(set(vs)) != len(vs):
                _fail("INVALID_POLICY", "Categories must enumerate distinct strings")
    if s["schema_version"] == "1.1":
        _approval_contract(s.get("approval_contract"), fields)
    rules = s.get("rules")
    if not isinstance(rules, list) or not rules:
        _fail("INVALID_POLICY", "Policy requires rules")
    ids = set()
    diagnostics = []
    for r in rules:
        if not isinstance(r, dict):
            _fail("INVALID_POLICY", "Rules must be objects")
        for k in ("rule_id", "reason_code"):
            _text(r.get(k), k)
        if r["rule_id"] in ids:
            _fail("INVALID_POLICY", "Duplicate rule id")
        ids.add(r["rule_id"])
        if not _number(r.get("priority")) or r["priority"] < 0 or r["priority"] != math.floor(r["priority"]) or r.get("action") not in _ACTIONS:
            _fail("INVALID_POLICY", "Rules require nonnegative integer priority and allowed action")
        if not isinstance(r.get("when"), list):
            _fail("INVALID_POLICY", "when must be an array of conjunctive predicates")
        for p in r["when"]:
            if not isinstance(p, dict) or not isinstance(p.get("field"), str) or p["field"] not in fields or p.get("op") not in ("eq", "ne", "lt", "le", "gt", "ge"):
                _fail("INVALID_POLICY", "Invalid field or predicate")
            f = fields[p["field"]]
            if f["type"] == "number":
                if not _number(p.get("value")):
                    _fail("INVALID_POLICY", "Numeric thresholds must be finite")
            elif p["op"] not in ("eq", "ne") or p.get("value") not in f["values"]:
                _fail("INVALID_POLICY", "Categorical predicates use declared equality values")
        if not _possible(r["when"], fields):
            _fail("UNREACHABLE_RULE", "Rule has contradictory predicates", {"rule_id": r["rule_id"]})
    rules.sort(key=lambda r: (r["priority"], r["rule_id"]))
    for i, r in enumerate(rules):
        for earlier in rules[:i]:
            overlap = _possible(earlier["when"] + r["when"], fields)
            if overlap and earlier["priority"] == r["priority"] and earlier["action"] != r["action"]:
                _fail("CONFLICTING_RULES", "Equal-priority rules can prescribe different actions",
                      {"rules": [earlier["rule_id"], r["rule_id"]]})
            if overlap and earlier["priority"] < r["priority"]:
                diagnostics.append({"earlier": earlier["rule_id"], "later": r["rule_id"], "finding": "PRIORITY_OVERLAP"})
                # A conjunction implies an earlier rule iff adding the negation
                # of any earlier predicate is impossible.
                negate = {"eq": "ne", "ne": "eq", "lt": "ge", "le": "gt", "gt": "le", "ge": "lt"}
                if all(not _possible(r["when"] + [dict(p, op=negate[p["op"]])], fields) for p in earlier["when"]):
                    diagnostics.append({"rule_id": r["rule_id"], "finding": "SHADOWED_RULE", "by": earlier["rule_id"]})
    artifact = {"schema_version": "1.0", "kind": "DecisionPolicy", "specification": s}
    artifact["sha256"] = _fingerprint(artifact)
    return _result(artifact, "decisioning.compile_credit_decision_policy", "policy", context,
                   diagnostics={"rule_findings": diagnostics},
                   limitations=("Approval references are supplied assertions, not institutional authorization.",))


def _checked(artifact, kind):
    if isinstance(artifact, CalcResult):
        artifact = artifact.value
    if not isinstance(artifact, dict):
        _fail("INVALID_ARTIFACT", "Expected a fingerprinted artifact")
    payload = deepcopy(artifact)
    digest = payload.pop("sha256", None)
    if payload.get("kind") != kind or digest != _fingerprint(payload):
        _fail("ARTIFACT_HASH_MISMATCH", "Artifact content differs from its fingerprint")
    return artifact


def apply_credit_decision_policy(policy, risk, affordability, eligibility, operations,
                                 *, as_of, known_at, context=None):
    """Return a local recommendation with complete rule trace and input hashes."""
    policy = _checked(policy, "DecisionPolicy")
    s = policy.get("specification")
    # Revalidate even if a caller supplies a newly hashed malformed artifact.
    if compile_credit_decision_policy(s).value["sha256"] != policy["sha256"]:
        _fail("INVALID_ARTIFACT", "Policy is not canonical")
    _date(as_of)
    _date(known_at)
    if known_at < as_of:
        _fail("INVALID_DATE", "Knowledge date cannot precede decision date")
    inputs = dict(risk=risk, affordability=affordability, eligibility=eligibility, operations=operations)
    inputs = {k: deepcopy(v.value if isinstance(v, CalcResult) else v) for k, v in inputs.items()}
    if any(not isinstance(v, dict) for v in inputs.values()):
        _fail("INVALID_INPUT", "Provide separate objects for all four information sources")
    findings = []
    if not s["effective_from"] <= as_of < s["effective_until"] or s["known_at"] > known_at:
        findings.append("POLICY_NOT_EFFECTIVE")
    for source, obj in inputs.items():
        if "as_of" not in obj or "known_at" not in obj:
            findings.append(source + ".MISSING_TIME")
        else:
            observed, available = _date(obj["as_of"]), _date(obj["known_at"])
            if available < observed:
                _fail("INVALID_DATE", "Input knowledge date cannot precede its observation date")
            if observed > as_of or available > known_at:
                findings.append(source + ".FUTURE_INPUT")
    if inputs["risk"].get("target") != s["target"]:
        findings.append("risk.TARGET_MISMATCH")
    if inputs["risk"].get("freshness") != "CURRENT":
        findings.append("risk.NOT_CURRENT")
    if s["schema_version"] == "1.1":
        findings.extend(_applicability_findings(s["approval_contract"], inputs, as_of))
    values = {}
    for key, f in s["fields"].items():
        source, name = key.split(".")
        x = inputs[source].get(name)
        if x is None:
            findings.append(key + ".MISSING")
        elif (f["type"] == "number" and not _number(x)) or (f["type"] == "category" and x not in f["values"]):
            findings.append(key + ".INVALID")
        else:
            values[key] = x
    trace = []
    for r in s["rules"]:
        results = [None if p["field"] not in values else _compare(values[p["field"]], p["op"], p["value"]) for p in r["when"]]
        matched = False if False in results else (None if None in results else True)
        trace.append({"rule_id": r["rule_id"], "priority": r["priority"], "action": r["action"],
                      "reason_code": r["reason_code"], "sources": sorted({p["field"].split(".")[0] for p in r["when"]}),
                      "predicate_results": results, "matched": matched, "selected": False})
    selected = next((r for r in trace if r["matched"] is True), None)
    action = s["missing_action"] if findings else (selected["action"] if selected else s["default_action"])
    if selected and not findings:
        selected["selected"] = True
    # Independent non-waivable approval gates. Their failure never alters PD.
    gates = {"affordability": "AFFORDABLE", "eligibility": "ELIGIBLE", "operations": "CLEAR"}
    gate_trace = []
    if s["schema_version"] == "1.1":
        contract = s["approval_contract"]
        if contract["affordability"]["mode"] == "NOT_APPLICABLE":
            gates["affordability"] = "NOT_APPLICABLE"
        checks = [dict(field=k + ".status", value=v) for k, v in gates.items()] + contract["required_gates"]
        for check in checks:
            source, field = check["field"].split(".")
            actual = inputs[source].get(field)
            gate_trace.append(dict(field=check["field"], required_value=check["value"], actual_value=actual, passed=actual == check["value"]))
    if action == "APPROVE":
        blocked = [k + ".APPROVAL_GATE" for k, required in gates.items() if inputs[k].get("status") != required]
        if s["schema_version"] == "1.1":
            blocked.extend(r["field"] + ".APPROVAL_GATE" for r in gate_trace[3:] if not r["passed"])
        if blocked:
            findings.extend(blocked)
            action = "REFER"
    result = {"schema_version": "1.0", "kind": "DecisionArtifact", "policy_id": s["policy_id"],
              "policy_version": s["version"], "policy_hash": policy["sha256"],
              "as_of": as_of, "known_at": known_at, "action": action,
              "local_recommendation": True, "input_hashes": {k: _fingerprint(v) for k, v in inputs.items()},
              "findings": sorted(set(findings)), "rule_trace": trace}
    if s["schema_version"] == "1.1":
        result["approval_gate_trace"] = gate_trace
        result["approval_domain"] = s["approval_contract"]["domain"]
    result["sha256"] = _fingerprint(result)
    return _result(result, "decisioning.apply_credit_decision_policy", "decision", context,
                   limitations=("Local recommendation only; no loan, limit, price or payment is executed.",))


def decompose_policy_decision(decision, reason_dictionary, *, context=None):
    """Explain the recorded evaluation without rerunning or inventing reasons."""
    d = _checked(decision, "DecisionArtifact")
    if (d.get("schema_version") != "1.0" or d.get("action") not in _ACTIONS
            or not isinstance(d.get("rule_trace"), list)
            or not isinstance(d.get("findings"), list)
            or any(not isinstance(x, str) for x in d["findings"])):
        _fail("INVALID_ARTIFACT", "Invalid decision schema")
    for row in d["rule_trace"]:
        if (not isinstance(row, dict) or not isinstance(row.get("rule_id"), str)
                or not isinstance(row.get("reason_code"), str)
                or not isinstance(row.get("sources"), list)
                or any(x not in _SOURCES for x in row["sources"])
                or type(row.get("selected")) is not bool
                or (row.get("matched") is not None and type(row["matched"]) is not bool)
                or row.get("action") not in _ACTIONS):
            _fail("INVALID_ARTIFACT", "Invalid recorded rule evaluation")
    if not isinstance(reason_dictionary, dict):
        _fail("INVALID_INPUT", "Reason dictionary must be an object")
    reasons = []
    for row in d["rule_trace"]:
        if row["matched"] is True:
            code = row["reason_code"]
            if not isinstance(reason_dictionary.get(code), str) or not reason_dictionary[code]:
                _fail("MISSING_REASON_CODE", "Missing reason description", {"reason_code": code})
            reasons.append({"rule_id": row["rule_id"], "reason_code": code,
                            "description": reason_dictionary[code], "sources": row["sources"],
                            "selected": row["selected"], "rule_action": row["action"]})
    return _result({"decision_hash": d["sha256"], "action": d["action"],
                    "findings": deepcopy(d["findings"]), "reasons": reasons},
                   "decisioning.decompose_policy_decision", "explanation", context)

from ._decisioning_economics import compute_limit_frontier, compute_risk_based_price_grid, evaluate_treatment_candidates

from ._decisioning_evaluation import evaluate_decision_grid, select_constrained_policy, assess_policy_robustness
