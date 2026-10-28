"""Dated account risk sets, payment allocation and empirical state dynamics."""
from calendar import monthrange
from datetime import date, timedelta
import math
import numpy as np

from .core import CalcResult, FitResult, _fail, _fingerprint, _model_artifact, _numeric, _result as _core_result
from .data import _rows, _date, _required, _unique, _days


def _result(value, *args, **kwargs):
    def check(x):
        if isinstance(x, dict):
            for v in x.values(): check(v)
        elif isinstance(x, list):
            for v in x: check(v)
        elif isinstance(x, (float, int)) and not math.isfinite(x):
            _fail("NUMERICAL_FAILURE", "Behavioral output exceeds finite numeric range")
    check(value)
    return _core_result(value, *args, **kwargs)


def _amount(x, name, nonnegative=True):
    a = _numeric(x, name)
    if a.ndim or (nonnegative and a < 0):
        _fail("BEHAVIORAL_INPUT", f"{name} must be a finite scalar" + (" >=0" if nonnegative else ""))
    return float(a)


def _array(x, name):
    if not isinstance(x, (list, tuple)) or any(not isinstance(v, str) or not v for v in x) or len(set(x)) != len(x):
        _fail("BEHAVIORAL_INPUT", f"{name} must contain distinct nonempty strings")
    return list(x)


def _month(d, n, end=False):
    total = d.year*12+d.month-1+n
    y, m = divmod(total, 12); m += 1
    return date(y, m, monthrange(y, m)[1] if end else min(d.day, monthrange(y, m)[1]))


def _events(events):
    rows = _rows(events, "events"); _unique(rows, "event_id", "events")
    for r in rows:
        _required(r, ["account_id", "event_type", "occurred_at", "known_at"])
        if _date(r["known_at"], "known_at") < _date(r["occurred_at"], "occurred_at"):
            _fail("DATA_DATE", "Event knowledge cannot precede occurrence")
    return rows


def compute_days_past_due(obligations, payments, *, as_of, known_at,
                          allocation_policy="fifo", currency, context=None):
    """Allocate known net payments to due obligations without materiality rules.

    Reversals identify the original payment and cannot exceed its amount.
    FIFO applies the net payment pool to due obligations, oldest first; explicit
    mode restricts each payment to its supplied obligation_id.
    """
    anchor, cutoff = _date(as_of, "as_of"), _date(known_at, "known_at")
    if allocation_policy not in ("fifo", "explicit") or not isinstance(currency, str) or not currency:
        _fail("BEHAVIORAL_INPUT", "Declare allocation policy and currency")
    obs = _rows(obligations, "obligations"); pays = _rows(payments, "payments")
    _unique(obs, "obligation_id", "obligations"); _unique(pays, "payment_id", "payments")
    accounts = set()
    for r in obs:
        _required(r, ["account_id", "due_at", "known_at", "amount", "currency"])
        _date(r["due_at"], "due_at"); _date(r["known_at"], "known_at")
        _amount(r["amount"], "obligation amount"); accounts.add(r["account_id"])
        if r["currency"] != currency: _fail("BEHAVIORAL_CURRENCY", "Currency mismatch")
    for r in pays:
        _required(r, ["account_id", "paid_at", "known_at", "amount", "currency", "kind"])
        if _date(r["known_at"], "known_at") < _date(r["paid_at"], "paid_at"):
            _fail("DATA_DATE", "Payment knowledge precedes occurrence")
        _amount(r["amount"], "payment amount"); accounts.add(r["account_id"])
        if r["currency"] != currency: _fail("BEHAVIORAL_CURRENCY", "Currency mismatch")
        if r["kind"] not in ("payment", "reversal"): _fail("BEHAVIORAL_INPUT", "Unknown payment kind")
    if len(accounts) > 1: _fail("BEHAVIORAL_INPUT", "DPD calculation requires one account")
    due = sorted([r for r in obs if _date(r["due_at"], "due_at") <= anchor and _date(r["known_at"], "known_at") <= cutoff], key=lambda r: (r["due_at"], r["obligation_id"]))
    visible = [r for r in pays if _date(r["paid_at"], "paid_at") <= anchor and _date(r["known_at"], "known_at") <= cutoff]
    credits = {r["payment_id"]: dict(r, net=float(r["amount"])) for r in visible if r["kind"] == "payment"}
    for r in visible:
        if r["kind"] == "reversal":
            original = credits.get(r.get("reverses_payment_id"))
            if original is None or original["paid_at"] > r["paid_at"]:
                _fail("BEHAVIORAL_REVERSAL", "Reversal lacks an earlier visible payment")
            original["net"] -= r["amount"]
            if original["net"] < 0: _fail("BEHAVIORAL_REVERSAL", "Reversals exceed original payment")
    available = sum(r["net"] for r in credits.values())
    pool = available; rows = []
    ids = {r["obligation_id"] for r in obs}
    if allocation_policy == "explicit" and any(r.get("obligation_id") not in ids for r in credits.values()):
        _fail("BEHAVIORAL_ALLOCATION", "Explicit payments need known obligation identifiers")
    for r in due:
        funds = pool if allocation_policy == "fifo" else sum(p["net"] for p in credits.values() if p["obligation_id"] == r["obligation_id"])
        applied = min(float(r["amount"]), funds)
        if allocation_policy == "fifo": pool -= applied
        unpaid = float(r["amount"])-applied
        rows.append({"obligation_id": r["obligation_id"], "due_at": r["due_at"], "amount": float(r["amount"]), "applied": applied,
                     "unpaid": unpaid, "dpd": (anchor-_date(r["due_at"], "due_at")).days if unpaid > 0 else 0})
    open_rows = [r for r in rows if r["unpaid"] > 0]
    past_due = sum(r["unpaid"] for r in rows if r["due_at"] < as_of)
    return _result({"account_id": next(iter(accounts)) if accounts else None, "as_of": as_of, "known_at": known_at,
                    "dpd": max((r["dpd"] for r in open_rows), default=0), "past_due_amount": past_due,
                    "unpaid_due_amount": sum(r["unpaid"] for r in rows),
                    "oldest_unpaid_obligation": open_rows[0]["obligation_id"] if open_rows else None,
                    "obligations": rows, "net_payments": available,
                    "unallocated_payments": available-sum(r["applied"] for r in rows), "currency": currency},
                   "behavioral.compute_days_past_due", "days", context,
                   limitations=("No materiality or grace threshold is applied; obligations must represent the historical contractual schedule.",))


def assign_delinquency_state(dpd, past_due_amount, exposure, events, state_profile, *, context=None):
    """Apply explicit event precedence, materiality and gap-free DPD buckets."""
    _days(dpd, "dpd"); arrears = _amount(past_due_amount, "past_due_amount"); ead = _amount(exposure, "exposure")
    if not isinstance(events, dict) or any(type(x) is not bool for x in events.values()) or not isinstance(state_profile, dict):
        _fail("BEHAVIORAL_PROFILE", "Supply event flags and state profile")
    p = state_profile
    if p.get("schema_version") != "1.0": _fail("UNSUPPORTED_SCHEMA_VERSION", "Unsupported state profile")
    _required(p, ["profile_id", "immaterial_state", "materiality", "buckets", "event_priority"])
    material = p["materiality"]
    if not isinstance(material, dict) or material.get("logic") not in ("and", "or"):
        _fail("BEHAVIORAL_PROFILE", "Declare materiality logic")
    absolute = _amount(material.get("absolute"), "absolute materiality")
    relative = _amount(material.get("relative"), "relative materiality")
    relative_met = arrears > 0 if ead == 0 else arrears/ead >= relative
    tests = [arrears >= absolute, relative_met]
    is_material = arrears > 0 and (all(tests) if material["logic"] == "and" else any(tests))
    buckets = _rows(p["buckets"], "buckets"); priorities = _rows(p["event_priority"], "event_priority")
    lower = 0; names = set()
    for i, b in enumerate(buckets):
        _required(b, ["state", "lower"]); _days(b["lower"], "lower")
        upper = b.get("upper")
        if b["state"] in names or b["lower"] != lower or (upper is None and i != len(buckets)-1):
            _fail("BEHAVIORAL_PROFILE", "Buckets must be distinct and gap-free")
        if upper is not None:
            _days(upper, "upper")
            if upper <= lower: _fail("BEHAVIORAL_PROFILE", "Empty DPD bucket")
        names.add(b["state"]); lower = upper
    if not buckets or buckets[-1].get("upper") is not None: _fail("BEHAVIORAL_PROFILE", "Final bucket is unbounded")
    seen = set()
    for r in priorities:
        _required(r, ["event", "state", "at_risk", "absorbing"])
        if r["event"] in seen or type(r["at_risk"]) is not bool or type(r["absorbing"]) is not bool:
            _fail("BEHAVIORAL_PROFILE", "Invalid event priority")
        seen.add(r["event"])
    if set(events)-seen: _fail("BEHAVIORAL_PROFILE", "Every supplied event needs declared precedence")
    for r in priorities:
        if events.get(r["event"], False):
            return _result({"state": r["state"], "trigger": r["event"], "material": is_material,
                            "at_risk": r["at_risk"], "absorbing": r["absorbing"], "profile_id": p["profile_id"]},
                           "behavioral.assign_delinquency_state", "state", context)
    rule = p.get("dpd_default_rule")
    if rule is not None:
        _required(rule, ["minimum_dpd", "state", "material_required"]); _days(rule["minimum_dpd"], "minimum_dpd")
        if type(rule["material_required"]) is not bool: _fail("BEHAVIORAL_PROFILE", "material_required must be boolean")
        if dpd >= rule["minimum_dpd"] and (is_material or not rule["material_required"]):
            return _result({"state": rule["state"], "trigger": "DECLARED_DPD_DEFAULT", "material": is_material,
                            "at_risk": False, "absorbing": True, "profile_id": p["profile_id"]},
                           "behavioral.assign_delinquency_state", "state", context)
    state = next(b["state"] for b in buckets if b["lower"] <= dpd and (b.get("upper") is None or dpd < b["upper"])) if is_material else p["immaterial_state"]
    return _result({"state": state, "trigger": "DPD_BUCKET" if is_material else "IMMATERIAL",
                    "material": is_material, "at_risk": True, "absorbing": False, "profile_id": p["profile_id"]},
                   "behavioral.assign_delinquency_state", "state", context,
                   limitations=("A delinquency bucket is not a regulatory default definition.",))


def build_account_month_panel(accounts, statements, account_events, *, observation_dates,
                               reporting_lag_days=0, obligations=None, payments=None, context=None):
    """Build calendar month-end snapshots using information available by each cutoff."""
    acc = _rows(accounts, "accounts"); _unique(acc, "account_id", "accounts")
    stmt = _rows(statements, "statements"); _unique(stmt, "statement_id", "statements")
    events = _events(account_events); dates = _array(observation_dates, "observation_dates")
    lag = _days(reporting_lag_days, "reporting_lag_days")
    for d in dates:
        dt = _date(d, "observation_date")
        if dt.day != monthrange(dt.year, dt.month)[1]: _fail("DATA_DATE", "Panel observations must be calendar month ends")
    aid = {a["account_id"] for a in acc}
    for a in acc:
        _required(a, ["customer_id", "product_id", "opened_at", "known_at", "currency"])
        if _date(a["known_at"], "known_at") < _date(a["opened_at"], "opened_at"): _fail("DATA_DATE", "Account known before opening")
    for s in stmt:
        _required(s, ["account_id", "observed_at", "known_at", "balance", "credit_limit", "dpd", "past_due_amount", "currency"])
        if s["account_id"] not in aid: _fail("BEHAVIORAL_ACCOUNT", "Unknown statement account")
        if _date(s["known_at"], "known_at") < _date(s["observed_at"], "observed_at"): _fail("DATA_DATE", "Statement known before observation")
        for k in ("balance", "credit_limit", "past_due_amount"): _amount(s[k], k)
        _days(s["dpd"], "dpd")
    for e in events:
        if e["account_id"] not in aid or e["event_type"] not in ("ACTIVE", "DEFAULT", "CHARGEOFF", "CURED", "CLOSED", "PREPAID", "SOLD"):
            _fail("BEHAVIORAL_EVENT", "Unknown account or lifecycle event")
    if (obligations is None) != (payments is None): _fail("BEHAVIORAL_INPUT", "Supply obligations and payments together")
    obs = None if obligations is None else _rows(obligations, "obligations")
    pays = None if payments is None else _rows(payments, "payments")
    output = []
    for anchor in sorted(dates):
        cutoff = (_date(anchor, "date")-timedelta(days=lag)).isoformat()
        for a in sorted(acc, key=lambda x: x["account_id"]):
            if a["opened_at"] > anchor or a["known_at"] > cutoff: continue
            cid = a["account_id"]
            history = sorted([e for e in events if e["account_id"] == cid and e["occurred_at"] <= anchor and e["known_at"] <= cutoff], key=lambda x: (x["occurred_at"], x["known_at"], x["event_id"]))
            if len({(e["occurred_at"], e["known_at"]) for e in history}) != len(history):
                _fail("BEHAVIORAL_AMBIGUITY", "Simultaneous lifecycle events require explicit resolution")
            terminal = [e for e in history if e["event_type"] in ("CLOSED", "PREPAID", "SOLD")]
            if terminal and any(e["occurred_at"] > terminal[0]["occurred_at"] and e["event_type"] in ("ACTIVE", "CURED") for e in history):
                _fail("BEHAVIORAL_EVENT", "Legal termination cannot be silently reopened")
            lifecycle = terminal[0]["event_type"] if terminal else history[-1]["event_type"] if history else "ACTIVE"
            ever_default = any(e["event_type"] in ("DEFAULT", "CHARGEOFF") for e in history)
            available = [s for s in stmt if s["account_id"] == cid and s["observed_at"] <= anchor and s["known_at"] <= cutoff]
            available.sort(key=lambda x: (x["observed_at"], x["known_at"], x["statement_id"]))
            s = available[-1] if available else None
            if len(available)>1 and (available[-1]["observed_at"], available[-1]["known_at"]) == (available[-2]["observed_at"], available[-2]["known_at"]):
                _fail("BEHAVIORAL_AMBIGUITY", "Ambiguous statement revision")
            if s and s["currency"] != a["currency"]: _fail("BEHAVIORAL_CURRENCY", "Statement currency mismatch")
            features = {k: s[k] if s else None for k in ("balance", "credit_limit", "dpd", "past_due_amount")}
            if obs is not None:
                dpd = compute_days_past_due([r for r in obs if r["account_id"] == cid], [r for r in pays if r["account_id"] == cid], as_of=anchor, known_at=cutoff, currency=a["currency"]).value
                features.update(dpd=dpd["dpd"], past_due_amount=dpd["past_due_amount"])
            features["utilization"] = None if features["balance"] is None or features["credit_limit"] in (None, 0) else features["balance"]/features["credit_limit"]
            opened = _date(a["opened_at"], "opened_at"); adate = _date(anchor, "date")
            row = {"account_id": cid, "customer_id": a["customer_id"], "product_id": a["product_id"],
                   "as_of": anchor, "known_at": cutoff, "opened_at": a["opened_at"], "currency": a["currency"],
                   "months_on_book": (adate.year-opened.year)*12+adate.month-opened.month,
                   "lifecycle_state": lifecycle, "state": lifecycle, "ever_default": ever_default,
                   "at_risk": lifecycle in ("ACTIVE", "CURED") and not ever_default,
                   "features": features, "statement_id": s["statement_id"] if s else None,
                   "statement_observed_at": s["observed_at"] if s else None,
                   "source_age_days": (adate-_date(s["observed_at"], "observed_at")).days if s else None,
                   "source_complete": all(v is not None for k, v in features.items() if k != "utilization")}
            row["snapshot_hash"] = _fingerprint(row); output.append(row)
    customers = []
    for key in sorted({(r["customer_id"], r["as_of"], r["currency"]) for r in output}):
        rows = [r for r in output if (r["customer_id"], r["as_of"], r["currency"]) == key]
        balances = [r["features"]["balance"] for r in rows if r["lifecycle_state"] not in ("CLOSED", "PREPAID", "SOLD")]
        customers.append({"customer_id": key[0], "as_of": key[1], "currency": key[2], "account_count": len(rows),
                          "at_risk_accounts": sum(r["at_risk"] for r in rows),
                          "active_balance": None if any(x is None for x in balances) else sum(balances)})
    return _result({"accounts": output, "customers": customers}, "behavioral.build_account_month_panel", "panel", context,
                   limitations=("Features preserve reporting cutoffs; target construction uses separate future event evidence.",
                                "Customer balance aggregation is not a customer PD. Closed accounts remain visible for lifecycle/recovery context."))


def _coverage(coverage, cutoff):
    rows = _rows(coverage, "coverage"); _unique(rows, "account_id", "coverage")
    output = {}
    for r in rows:
        _required(r, ["observed_through", "known_at"])
        end, known = _date(r["observed_through"], "observed_through"), _date(r["known_at"], "known_at")
        if known < end: _fail("DATA_DATE", "Outcome completeness cannot be known before observation end")
        if known <= cutoff: output[r["account_id"]] = min(end, cutoff)
    return output


def build_behavioral_targets(panel, events, *, horizon_months, target_event,
                              competing_events, exit_treatment, evaluation_cutoff,
                              coverage, context=None):
    """First-event labels on (snapshot date, monthly anniversary], never stale zeroes."""
    rows = _rows(panel, "panel"); ev = _events(events)
    horizon = _days(horizon_months, "horizon_months", 1)
    competing = _array(competing_events, "competing_events")
    _array([target_event], "target_event")
    if target_event in competing or exit_treatment not in ("competing", "censor"):
        _fail("BEHAVIORAL_TARGET", "Declare distinct competing events and competing/censor treatment")
    cutoff = _date(evaluation_cutoff, "evaluation_cutoff"); complete = _coverage(coverage, cutoff)
    output = []; seen = set()
    for r in rows:
        _required(r, ["account_id", "customer_id", "as_of", "at_risk", "ever_default"])
        key = (r["account_id"], r["as_of"])
        if key in seen: _fail("DATA_DUPLICATE", "Duplicate account snapshot")
        seen.add(key); anchor = _date(r["as_of"], "as_of"); end = _month(anchor, horizon)
        history = sorted([e for e in ev if e["account_id"] == r["account_id"] and _date(e["known_at"], "known_at") <= cutoff], key=lambda e: (e["occurred_at"], e["event_id"]))
        prior = any(e["event_type"] == target_event and _date(e["occurred_at"], "occurred_at") <= anchor for e in history)
        eligible = r["at_risk"] is True and not r["ever_default"] and not prior and anchor <= cutoff
        future = [e for e in history if anchor < _date(e["occurred_at"], "occurred_at") <= min(end, cutoff) and e["event_type"] in [target_event]+competing]
        if future and any(e["occurred_at"] == future[0]["occurred_at"] and e["event_type"] != future[0]["event_type"] for e in future):
            _fail("BEHAVIORAL_AMBIGUITY", "Same-day competing outcomes need explicit ordering outside this contract")
        first = future[0] if future else None
        observed = complete.get(r["account_id"])
        label, status = None, "NOT_AT_RISK"
        if eligible:
            if first and first["event_type"] == target_event: label, status = 1, "EVENT"
            elif first: label, status = (0 if exit_treatment == "competing" else None), ("COMPETING_EVENT" if exit_treatment == "competing" else "CENSORED_EXIT")
            elif observed is not None and observed >= end: label, status = 0, "MATURE_NO_EVENT"
            else: status = "OUTCOME_PENDING" if end <= cutoff else "IMMATURE"
        stop = min(end, cutoff, observed) if observed is not None else min(end, cutoff)
        if first: stop = _date(first["occurred_at"], "occurred_at")
        output.append({"account_id": r["account_id"], "customer_id": r["customer_id"], "as_of": r["as_of"],
                       "horizon_end": end.isoformat(), "target_event": target_event, "label": label,
                       "eligible": eligible, "status": status, "first_event_id": first["event_id"] if first and eligible else None,
                       "first_event_type": first["event_type"] if first and eligible else None,
                       "followup_days": max(0, (stop-anchor).days) if observed is not None or first is not None else None, "calendar_mature": end <= cutoff,
                       "outcome_complete": label is not None, "evaluation_cutoff": evaluation_cutoff})
    return _result(output, "behavioral.build_behavioral_targets", "labels", context,
                   limitations=("Overlapping account windows and customers are dependent; isolate customers/groups and time in fitting and evaluation.",
                                "A competing-exit zero means no first target before that exit, not hypothetical continuing-account performance."))


def estimate_retail_roll_rates(panel, *, states, from_date, to_date, group_by,
                                weighting, missing_successor, context=None):
    """Estimate dated empirical transitions and reconcile every origin count/mass."""
    states = _array(states, "states"); groups = _array(group_by, "group_by")
    if not states or weighting not in ("count", "opening_balance") or missing_successor not in ("error", "censor"):
        _fail("BEHAVIORAL_INPUT", "Declare states, weighting and missing successor policy")
    if _date(from_date, "from_date") >= _date(to_date, "to_date"): _fail("DATA_DATE", "Transition dates must increase")
    rows = _rows(panel, "panel"); lookup = {}; buckets = {}
    for r in rows:
        _required(r, ["account_id", "as_of", "state"])
        key = (r["account_id"], r["as_of"])
        if key in lookup: _fail("DATA_DUPLICATE", "Duplicate state snapshot")
        if r["state"] not in states: _fail("BEHAVIORAL_STATE", "Unknown state")
        lookup[key] = r
    for r in rows:
        if r["as_of"] != from_date: continue
        for g in groups: _required(r, [g])
        key = tuple(r[g] for g in groups)
        if any(not isinstance(x, (str, int, float)) or isinstance(x, bool) for x in key): _fail("BEHAVIORAL_INPUT", "Grouping values must be scalar")
        b = buckets.setdefault(key, {"counts": np.zeros((len(states), len(states)), dtype=int), "mass": np.zeros((len(states), len(states))),
                                     "origin_count": np.zeros(len(states), dtype=int), "origin_mass": np.zeros(len(states)), "censored_count": np.zeros(len(states), dtype=int), "censored_mass": np.zeros(len(states))})
        i = states.index(r["state"])
        weight = 1.0 if weighting == "count" else _amount(r.get("balance", r.get("features", {}).get("balance")), "opening balance")
        b["origin_count"][i] += 1; b["origin_mass"][i] += weight
        successor = lookup.get((r["account_id"], to_date))
        if successor is None:
            if missing_successor == "error": _fail("MISSING_SUCCESSOR", "Every origin requires an explicit destination, including exits")
            b["censored_count"][i] += 1; b["censored_mass"][i] += weight
        else:
            j = states.index(successor["state"]); b["counts"][i, j] += 1; b["mass"][i, j] += weight
    output = []
    for key, b in buckets.items():
        observed = b["mass"].sum(axis=1)
        probabilities = [[None]*len(states) if observed[i] == 0 else (b["mass"][i]/observed[i]).tolist() for i in range(len(states))]
        output.append({"group": dict(zip(groups, key)), "states": states, "transition_matrix": probabilities,
                       **{k: v.tolist() for k, v in b.items()}, "observed_mass": observed.tolist(),
                       "empty_rows": [states[i] for i, x in enumerate(observed) if x == 0]})
    return _result({"from_date": from_date, "to_date": to_date, "weighting": weighting, "groups": output},
                   "behavioral.estimate_retail_roll_rates", "transition_estimate", context,
                   limitations=("Censor mode estimates transitions conditional on observed successors, not an unbiased full-cohort matrix.",
                                "Opening-balance weights measure monetary flows, not account probabilities; zero balances retain origin counts."))


def project_retail_roll_rates(initial_distribution, transition_matrices, *, states,
                               default_states, closure_states, exposure_by_period, context=None):
    """Project first-default probabilities with separate supplied state exposures."""
    states = _array(states, "states"); defaults = _array(default_states, "default_states"); closures = _array(closure_states, "closure_states")
    if not states or not defaults or not set(defaults+closures) <= set(states) or set(defaults)&set(closures):
        _fail("BEHAVIORAL_STATE", "Declare disjoint default and closure states")
    v = _numeric(initial_distribution, "initial_distribution", vector_only=True)
    if len(v) != len(states) or np.any(v < 0) or abs(float(v.sum())-1)>1e-12:
        _fail("INVALID_PROBABILITY", "Initial probabilities must sum to one")
    if not isinstance(transition_matrices, list) or not transition_matrices:
        _fail("BEHAVIORAL_INPUT", "Supply one transition matrix per projection period")
    exposure = _numeric(exposure_by_period, "exposure_by_period")
    if exposure.shape != (len(transition_matrices), len(states)) or np.any(exposure < 0):
        _fail("INCOMPATIBLE_SHAPE", "Exposure requires nonnegative period/state amounts")
    default_idx = [states.index(s) for s in defaults]; absorbing = [states.index(s) for s in defaults+closures]
    prior_default = float(v[default_idx].sum()); initial_default = prior_default
    output = []
    for t, matrix in enumerate(transition_matrices):
        p = _numeric(matrix, "transition_matrix")
        if p.shape != (len(states),len(states)) or np.any(p<0) or np.any(np.abs(p.sum(axis=1)-1)>1e-12):
            _fail("INVALID_PROBABILITY", "Transition rows must be nonnegative and sum to one; unresolved rows cannot project")
        for i in absorbing:
            if not np.array_equal(p[i], np.eye(len(states))[i]): _fail("BEHAVIORAL_STATE", "First-default/closure states must be absorbing")
        if np.any(exposure[t, [states.index(s) for s in closures]] != 0): _fail("BEHAVIORAL_EXPOSURE", "Closed states have zero projected exposure")
        flows = v[:,None]*p
        v = v@p
        cumulative = float(v[default_idx].sum()); marginal = cumulative-prior_default
        output.append({"period": t+1, "state_probabilities": v.tolist(), "probability_flows": flows.tolist(),
                       "balance_flows": (flows*exposure[t][None,:]).tolist(),
                       "marginal_default": marginal, "cumulative_default": cumulative,
                       "new_cumulative_default": cumulative-initial_default,
                       "state_exposure": exposure[t].tolist(), "expected_balance": float(v@exposure[t]),
                       "expected_balance_by_state": (v*exposure[t]).tolist()})
        prior_default = cumulative
    return _result({"states": states, "initial_default_mass": initial_default, "periods": output},
                   "behavioral.project_retail_roll_rates", "projection", context,
                   limitations=("Exposure amounts are supplied separately for each state/period and must incorporate amortization, drawdowns and maturity.",
                                "Default is absorbing for first-event probability; operational cure needs a separate state model."))


def predict_behavioral_risk(snapshot, model, target, calibrator=None, offset=None, *, context=None):
    """Predict only for the frozen model's explicit account risk set and target."""
    from .classification import predict_pd
    from .calibration import apply_calibrator
    s = snapshot.value if isinstance(snapshot, CalcResult) else snapshot
    if not isinstance(s, dict): _fail("BEHAVIORAL_INPUT", "Snapshot must be an object")
    payload = dict(s); digest = payload.pop("snapshot_hash", None)
    if digest != _fingerprint(payload): _fail("INTEGRITY_ERROR", "Snapshot differs from its hash")
    m = model.model if isinstance(model, FitResult) else model.value if isinstance(model, CalcResult) else model
    m = _model_artifact(m); contract = m.get("behavioral_contract")
    if _date(m["training"]["known_at"], "training.known_at") > _date(s.get("known_at"), "snapshot.known_at"):
        _fail("BEHAVIORAL_LEAKAGE", "Model training evidence was not available at snapshot cutoff")
    if not isinstance(contract, dict) or contract.get("risk_unit") != "account" or contract.get("event_policy") != "first_default":
        _fail("BEHAVIORAL_MODEL", "Require account/first_default behavioral applicability contract")
    if s.get("product_id") not in contract.get("product_ids", []) or s.get("lifecycle_state") not in contract.get("allowed_states", []):
        _fail("BEHAVIORAL_MODEL", "Snapshot outside declared product/state support")
    if s.get("at_risk") is not True or s.get("ever_default") is not False:
        _fail("BEHAVIORAL_RISK_SET", "Default/terminated account requires a separate outcome model")
    max_age = _days(contract.get("maximum_source_age_days"), "maximum_source_age_days")
    if s.get("source_age_days") is None or s["source_age_days"] > max_age:
        _fail("BEHAVIORAL_STALE", "Behavioral evidence is missing or stale")
    if not isinstance(target, dict) or target.get("probability_measure") != "PHYSICAL": _fail("BEHAVIORAL_TARGET", "Physical target required")
    features = s.get("features", {})
    x = [[_amount(features.get(k), k, False) for k in m["feature_order"]]]
    raw = float(predict_pd(m, x, feature_order=m["feature_order"], expected_target=target, offset=offset, context=context).value[0])
    pd = raw
    if calibrator is not None:
        artifact = calibrator.value if isinstance(calibrator, CalcResult) else calibrator
        if isinstance(artifact, dict) and _date(artifact.get("metadata", {}).get("calibration", {}).get("known_at"), "calibration.known_at") > _date(s["known_at"], "snapshot.known_at"):
            _fail("BEHAVIORAL_LEAKAGE", "Calibration was not available at snapshot cutoff")
        if isinstance(artifact, dict) and artifact.get("calibration_type") == "isotonic" and artifact.get("metadata", {}).get("input_scale") != "probability":
            _fail("BEHAVIORAL_CALIBRATION", "Isotonic behavioral calibration must declare probability input scale")
        pd = float(apply_calibrator(calibrator, [raw], target=target, base_model_version=m["model_version"], context=context).value[0])
    return _result({"account_id": s["account_id"], "customer_id": s["customer_id"], "as_of": s["as_of"],
                    "known_at": s["known_at"], "target": target, "pd": pd, "unadjusted_pd": raw,
                    "calibrated": calibrator is not None, "model_id": m["model_id"], "model_version": m["model_version"],
                    "snapshot_hash": digest, "source_complete": s["source_complete"]},
                   "behavioral.predict_behavioral_risk", "probability", context,
                   limitations=("Zero account balance does not imply zero PD; action thresholds and future EAD are separate.",))


def build_vintage_performance(accounts, panel, events, *, months_on_book,
                               evaluation_cutoff, coverage, default_event,
                               delinquency_states, exit_events, context=None):
    """Fixed-origination-denominator vintage cells with explicit maturity and gaps."""
    acc = _rows(accounts, "accounts"); _unique(acc, "account_id", "accounts")
    rows = _rows(panel, "panel"); ev = _events(events)
    ages = list(months_on_book) if isinstance(months_on_book, (list, tuple)) else []
    if not ages or len(set(ages)) != len(ages): _fail("BEHAVIORAL_INPUT", "Supply distinct nonnegative MOB values")
    for n in ages: _days(n, "months_on_book")
    delinquent = _array(delinquency_states, "delinquency_states"); exits = _array(exit_events, "exit_events")
    _array([default_event], "default_event")
    if default_event in exits: _fail("BEHAVIORAL_TARGET", "Default and competing exits must differ")
    cutoff = _date(evaluation_cutoff, "evaluation_cutoff"); complete = _coverage(coverage, cutoff)
    lookup = {}
    for r in rows:
        _required(r, ["account_id", "as_of", "state"])
        key = (r["account_id"], r["as_of"])
        if key in lookup: _fail("DATA_DUPLICATE", "Duplicate vintage stock snapshot")
        lookup[key] = r
    cohorts = {}
    for a in acc:
        _required(a, ["opened_at", "known_at", "original_balance", "currency"])
        opened = _date(a["opened_at"], "opened_at")
        if _date(a["known_at"], "known_at") > cutoff or opened > cutoff: continue
        _amount(a["original_balance"], "original_balance")
        cohorts.setdefault((a["opened_at"][:7], a["currency"]), []).append(a)
    output = []
    for (cohort, currency), members in sorted(cohorts.items()):
        for age in sorted(ages):
            end = _month(date.fromisoformat(cohort+"-01"), age, True)
            prev = _month(date.fromisoformat(cohort+"-01"), age-1, True)
            counts = {"defaults": 0, "incidence": 0, "exits": 0, "complete_accounts": 0,
                      "stock_observed": 0, "delinquent_stock": 0}
            losses = 0.0
            for a in members:
                hist = sorted([e for e in ev if e["account_id"] == a["account_id"] and e["occurred_at"] >= a["opened_at"] and _date(e["occurred_at"], "occurred_at") <= min(end,cutoff) and _date(e["known_at"], "known_at") <= cutoff], key=lambda e:(e["occurred_at"],e["event_id"]))
                first = next((e for e in hist if e["event_type"] in [default_event]+exits), None)
                if first and any(e["occurred_at"] == first["occurred_at"] and e["event_type"] in [default_event]+exits and e["event_type"] != first["event_type"] for e in hist):
                    _fail("BEHAVIORAL_AMBIGUITY", "Same-day default/exit requires explicit ordering")
                if first:
                    counts["complete_accounts"] += 1
                    if first["event_type"] == default_event:
                        counts["defaults"] += 1
                        counts["incidence"] += _date(first["occurred_at"], "occurred_at") > prev
                    else: counts["exits"] += 1
                elif complete.get(a["account_id"], date.min) >= end: counts["complete_accounts"] += 1
                stock = lookup.get((a["account_id"], end.isoformat()))
                if stock is not None and _date(stock.get("known_at", stock["as_of"]), "known_at") <= cutoff:
                    counts["stock_observed"] += 1
                    counts["delinquent_stock"] += stock["state"] in delinquent
                for e in hist:
                    if e["event_type"] in ("LOSS", "RECOVERY"):
                        if e.get("currency") != currency: _fail("BEHAVIORAL_CURRENCY", "Loss cashflow currency mismatch")
                        losses += _amount(e.get("amount"), "loss cashflow")*(1 if e["event_type"] == "LOSS" else -1)
            n = len(members); mature = end <= cutoff and counts["complete_accounts"] == n
            denom = sum(a["original_balance"] for a in members)
            output.append({"cohort": cohort, "currency": currency, "months_on_book": age, "period_end": end.isoformat(),
                           "accounts": n, "original_balance": denom, **counts, "mature": mature,
                           "cumulative_default_rate": counts["defaults"]/n if mature else None,
                           "incidence_rate": counts["incidence"]/n if mature else None,
                           "delinquency_stock_rate": counts["delinquent_stock"]/n if end <= cutoff and counts["stock_observed"] == n else None,
                           "observed_net_loss": losses, "net_loss_rate": losses/denom if mature and denom>0 else None})
    return _result(output, "behavioral.build_vintage_performance", "vintage_table", context,
                   limitations=("Cohort denominators are fixed origination counts; first competing exit ends loan risk, not hypothetical continuing-customer risk.",
                                "Loss rates use supplied observed cashflows and do not infer ultimate workout loss.",
                                "Age + cohort = calendar period; this descriptive table does not identify independent causal effects."))
