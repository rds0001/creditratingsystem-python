"""Affordability arithmetic remains distinct from risk estimates and decisions."""
import math
import numpy as np

from .core import _fail, _numeric, _result, _fingerprint, _iso_date

_FREQUENCIES = {"weekly": 52, "biweekly": 26, "four_weekly": 13, "monthly": 12,
                "quarterly": 4, "annual": 1}


def _align(**values):
    arrays = {k: _numeric(v, k).reshape(-1) for k, v in values.items()}
    if any(np.asarray(v).ndim > 1 for v in values.values()):
        _fail("INVALID_INPUT", "Only scalar and one-dimensional amounts are supported")
    sizes = {len(a) for a in arrays.values() if len(a) != 1}
    if len(sizes) > 1:
        _fail("INCOMPATIBLE_SHAPE", "Only length-one expansion is supported")
    n = max(len(a) for a in arrays.values())
    return {k: np.broadcast_to(v, (n,)).copy() for k, v in arrays.items()}


def _context_units(period, currency, basis="net"):
    if not isinstance(period, str) or period not in _FREQUENCIES or not isinstance(currency, str) or not currency.strip() or basis not in ("net", "gross"):
        _fail("INVALID_UNITS", "Declare frequency, currency and net/gross income basis")


def _output(x):
    values = list(x)
    return values[0] if len(values) == 1 else values


def _ratios(numerator, denominator):
    ratios, status = [], []
    for n, d in zip(numerator, denominator):
        if d <= 0:
            ratios.append(None)
            status.append("NONPOSITIVE_INCOME")
        else:
            value = float(n / d)
            if not math.isfinite(value):
                _fail("NUMERICAL_FAILURE", "Ratio exceeds finite range")
            ratios.append(value)
            status.append("APPLICABLE")
    return {"ratio": _output(ratios), "denominator_status": _output(status),
            "applicable": _output([s == "APPLICABLE" for s in status])}


def debt_to_income(debt_stock, income, *, income_frequency, currency, ratio_period,
                   income_basis="net", context=None):
    """Debt stock relative to an explicitly annualized or supplied-period income."""
    _context_units(income_frequency, currency, income_basis)
    if ratio_period not in ("annual", "income_period"):
        _fail("INVALID_UNITS", "ratio_period must be annual or income_period")
    a = _align(debt_stock=debt_stock, income=income)
    if np.any(a["debt_stock"] < 0):
        _fail("INVALID_AMOUNT", "Debt stock must be nonnegative")
    multiplier = _FREQUENCIES[income_frequency] if ratio_period == "annual" else 1
    with np.errstate(over="ignore"):
        denominator = a["income"] * multiplier
    if not np.all(np.isfinite(denominator)):
        _fail("NUMERICAL_FAILURE", "Annualized income exceeds finite range")
    return _result(_ratios(a["debt_stock"], denominator), "affordability.debt_to_income", "ratio", context,
                   diagnostics={"currency": currency, "income_basis": income_basis,
                                "income_frequency": income_frequency, "ratio_period": ratio_period,
                                "annualization_factor": multiplier},
                   limitations=("Debt-to-income is not a default probability or debt-service ratio.",))


def debt_service_to_income(debt_service, income, *, period, currency,
                           income_basis="net", context=None):
    """Same-period debt service divided by income, with explicit denominator status."""
    _context_units(period, currency, income_basis)
    a = _align(debt_service=debt_service, income=income)
    if np.any(a["debt_service"] < 0):
        _fail("INVALID_AMOUNT", "Debt service must be nonnegative")
    return _result(_ratios(a["debt_service"], a["income"]), "affordability.debt_service_to_income", "ratio", context,
                   diagnostics={"currency": currency, "period": period, "income_basis": income_basis},
                   limitations=("Income and debt service must cover the same period and debt scope.",))


def residual_income(net_income, recurring_expenses, existing_debt_service, new_debt_service,
                    *, period, currency, context=None):
    """Keep the full signed income surplus after disjoint expense/service totals."""
    _context_units(period, currency)
    a = _align(net_income=net_income, recurring_expenses=recurring_expenses,
               existing_debt_service=existing_debt_service, new_debt_service=new_debt_service)
    if any(np.any(v < 0) for k, v in a.items() if k != "net_income"):
        _fail("INVALID_AMOUNT", "Expenses and debt-service amounts must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        remaining = a["net_income"] - a["recurring_expenses"] - a["existing_debt_service"] - a["new_debt_service"]
    if not np.all(np.isfinite(remaining)):
        _fail("NUMERICAL_FAILURE", "Residual arithmetic exceeds finite range")
    value = {"residual": _output([float(x) for x in remaining]),
             "negative_buffer": _output([bool(x < 0) for x in remaining]),
             "minimum": float(np.min(remaining))}
    return _result(value, "affordability.residual_income", currency, context,
                   diagnostics={"period": period, "currency": currency,
                                "components": {k: _output([float(x) for x in v]) for k, v in a.items()}},
                   limitations=("Expense and service inputs must be reconciled disjoint totals.",))


def assess_affordability(metrics, threshold_profile, evidence, *, context=None):
    """Apply supplied affordability thresholds, with a separate missing-evidence state."""
    if not isinstance(metrics, dict) or not isinstance(evidence, dict) or not isinstance(threshold_profile, dict):
        _fail("INVALID_INPUT", "Metrics, profile and evidence must be objects")
    p = threshold_profile
    if p.get("schema_version") != "1.0":
        _fail("UNSUPPORTED_SCHEMA_VERSION", "Unsupported affordability profile schema")
    for key in ("policy_id", "version"):
        if not isinstance(p.get(key), str) or not p[key].strip():
            _fail("INVALID_POLICY", f"{key} is required")
    required = p.get("required_evidence")
    rules = p.get("rules")
    if not isinstance(required, list) or any(not isinstance(x, str) or not x for x in required) or len(set(required)) != len(required):
        _fail("INVALID_POLICY", "required_evidence must be distinct names")
    if not isinstance(rules, list) or not rules:
        _fail("INVALID_POLICY", "At least one explicit affordability rule is required")
    if any(not isinstance(v, bool) for v in evidence.values()):
        _fail("INVALID_INPUT", "Evidence availability flags must be boolean")
    missing_evidence = [key for key in required if evidence.get(key) is not True]
    trace, breaches, missing, seen = [], [], [], set()
    operators = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
                 ">=": lambda a, b: a >= b, ">": lambda a, b: a > b}
    for rule in rules:
        if not isinstance(rule, dict) or any(k not in rule for k in ("rule_id", "metric", "operator", "threshold")):
            _fail("INVALID_POLICY", "Incomplete affordability rule")
        rid, metric, op, threshold = (rule[k] for k in ("rule_id", "metric", "operator", "threshold"))
        if not isinstance(rid, str) or not rid or rid in seen or not isinstance(metric, str) or not metric or op not in operators:
            _fail("INVALID_POLICY", "Invalid or duplicate affordability rule")
        seen.add(rid)
        threshold = _numeric(threshold, "threshold")
        if threshold.ndim != 0:
            _fail("INVALID_POLICY", "Rule threshold must be scalar")
        threshold = float(threshold)
        value = metrics.get(metric)
        if value is None:
            if metric not in missing:
                missing.append(metric)
            passed, headroom = None, None
        else:
            v = _numeric(value, metric)
            if v.ndim != 0:
                _fail("INVALID_INPUT", "Assessment metrics must be scalar per subject")
            value = float(v)
            passed = operators[op](value, threshold)
            headroom = threshold - value if op in ("<=", "<") else value - threshold
            if not math.isfinite(headroom):
                _fail("NUMERICAL_FAILURE", "Headroom arithmetic exceeds finite range")
            if not passed:
                breaches.append(rid)
        trace.append({"rule_id": rid, "metric": metric, "operator": op, "threshold": threshold,
                      "value": value, "passed": passed, "headroom": headroom})
    assessment = "INSUFFICIENT_EVIDENCE" if missing or missing_evidence else "NOT_AFFORDABLE" if breaches else "AFFORDABLE"
    return _result({"assessment": assessment, "rule_trace": trace, "breaches": breaches,
                    "missing_metrics": missing, "missing_evidence": missing_evidence,
                    "policy_id": p["policy_id"], "policy_version": p["version"]},
                   "affordability.assess_affordability", "affordability_assessment", context,
                   limitations=("This assessment does not approve credit and does not alter a PD estimate.",))


def _aff_scalar(x, name, nonnegative=False):
    a = _numeric(x, name)
    if a.ndim or (nonnegative and a < 0):
        _fail("INVALID_AMOUNT", f"{name} must be a finite scalar" + (" >= 0" if nonnegative else ""))
    return float(a)


def _aff_result(value, *args, **kwargs):
    def check(x):
        if isinstance(x, dict):
            for v in x.values(): check(v)
        elif isinstance(x, list):
            for v in x: check(v)
        elif isinstance(x, (int, float)) and not math.isfinite(x):
            _fail("NUMERICAL_FAILURE", "Affordability arithmetic exceeds finite range")
    check(value)
    return _result(value, *args, **kwargs)


def _aff_text(x, name):
    if not isinstance(x, str) or not x.strip():
        _fail("INVALID_INPUT", f"{name} must be nonempty text")
    return x


def _aff_rows(rows, name):
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        _fail("INVALID_INPUT", f"{name} must be an array of objects")
    return rows


def _aff_calendar(calendar, period):
    if not isinstance(calendar, dict) or period not in calendar:
        _fail("INVALID_UNITS", "Supply an explicit payments_per_year calendar and target period")
    for key, value in calendar.items():
        _aff_text(key, "frequency")
        if _aff_scalar(value, "payments_per_year") <= 0:
            _fail("INVALID_UNITS", "Frequency counts must be positive")
    return calendar


def _aff_convert(amount, frequency, calendar, period):
    if frequency not in calendar:
        _fail("INVALID_UNITS", "Payment frequency absent from supplied calendar")
    value = amount * (calendar[frequency] / calendar[period])
    if not math.isfinite(value):
        _fail("NUMERICAL_FAILURE", "Period normalization overflowed")
    return value


def normalize_income_period(components, *, target_period, target_basis, currency,
                            payments_per_year, variable_income_factor, context=None):
    """Normalize deduplicated economic income sources using a declared calendar.

    A gross/net bridge requires caller-supplied tax_amount in the same source
    period. Transfers contribute zero; negative variable income is not haircut.
    Missing included amounts leave total_income unknown, not zero.
    """
    _aff_text(currency, "currency")
    if target_basis not in ("net", "gross"):
        _fail("INVALID_UNITS", "Declare target net/gross basis")
    calendar = _aff_calendar(payments_per_year, target_period)
    factor = _aff_scalar(variable_income_factor, "variable_income_factor", True)
    if factor > 1:
        _fail("INVALID_INPUT", "Variable income inclusion factor must be in [0,1]")
    rows, seen, duplicates = [], {}, []
    for row in _aff_rows(components, "components"):
        eid = _aff_text(row.get("economic_id"), "economic_id")
        if row.get("currency") != currency or row.get("basis") not in ("net", "gross"):
            _fail("INVALID_UNITS", "Income currency/basis mismatch")
        kind = row.get("kind")
        if kind not in ("recurring", "variable", "internal_transfer"):
            _fail("INVALID_INPUT", "Declare income kind")
        if row.get("frequency") not in calendar:
            _fail("INVALID_UNITS", "Income frequency absent from calendar")
        signature = {k: row.get(k) for k in ("amount", "frequency", "basis", "currency", "kind", "tax_amount")}
        if eid in seen:
            if _fingerprint(signature) != seen[eid]:
                _fail("CONFLICTING_EVIDENCE", "Repeated income id has conflicting economics")
            duplicates.append(eid)
            continue
        seen[eid] = _fingerprint(signature)
        amount = row.get("amount")
        applied = 0.0 if kind == "internal_transfer" else factor if kind == "variable" else 1.0
        if kind == "internal_transfer":
            normalized, status = 0.0, "EXCLUDED_TRANSFER"
        elif amount is None:
            normalized, status = None, "MISSING_AMOUNT"
        else:
            amount = _aff_scalar(amount, "income amount")
            if row["basis"] != target_basis:
                if row.get("tax_amount") is None:
                    normalized, status = None, "MISSING_TAX_BRIDGE"
                    rows.append({"economic_id": eid, "normalized_amount": normalized, "status": status, "inclusion_factor": applied})
                    continue
                tax = _aff_scalar(row["tax_amount"], "tax_amount", True)
                amount += -tax if target_basis == "net" else tax
            # Excluding positive uncertain earnings must not erase a reported loss.
            if amount < 0:
                applied = 1.0
            normalized = _aff_convert(amount, row["frequency"], calendar, target_period) * applied
            status = "NEGATIVE_INCOME" if normalized < 0 else "ZERO_INCOME" if normalized == 0 else "INCLUDED"
        rows.append({"economic_id": eid, "normalized_amount": normalized, "status": status, "inclusion_factor": applied})
    missing = [r["economic_id"] for r in rows if r["normalized_amount"] is None]
    known = sum(r["normalized_amount"] for r in rows if r["normalized_amount"] is not None)
    return _aff_result({"total_income": None if missing else known, "known_income": known,
                    "period": target_period, "basis": target_basis, "currency": currency,
                    "components": rows, "missing_components": missing, "duplicate_ids": duplicates},
                   "affordability.normalize_income_period", currency, context,
                   limitations=("Payment counts and tax amounts are supplied inputs; no tax law is inferred.",))


def estimate_recurring_expenses(expenses, *, target_period, currency, payments_per_year,
                                household_profile, benchmarks=None, context=None):
    """Reconcile repeated evidence by economic id, then apply local category floors.

    Same economic obligation observed and contractually reported contributes the
    larger same-period amount once. Debt service is returned separately.
    """
    _aff_text(currency, "currency")
    calendar = _aff_calendar(payments_per_year, target_period)
    if not isinstance(household_profile, dict):
        _fail("INVALID_INPUT", "Household profile is required")
    profile_id = _aff_text(household_profile.get("profile_id"), "profile_id")
    required = household_profile.get("required_categories")
    if not isinstance(required, list) or any(not isinstance(x, str) or not x for x in required) or len(set(required)) != len(required):
        _fail("INVALID_INPUT", "Declare distinct required expense categories")
    groups, excluded = {}, []
    for row in _aff_rows(expenses, "expenses"):
        eid = _aff_text(row.get("economic_id"), "economic_id")
        category = _aff_text(row.get("category"), "category")
        kind = row.get("kind")
        if kind not in ("essential", "discretionary", "debt_service", "internal_transfer"):
            _fail("INVALID_INPUT", "Declare expense kind")
        if row.get("currency") != currency:
            _fail("INVALID_UNITS", "Expense currencies must agree")
        if kind == "internal_transfer":
            excluded.append(eid)
            continue
        amount = None if row.get("amount") is None else _aff_convert(_aff_scalar(row["amount"], "expense", True), row.get("frequency"), calendar, target_period)
        group = groups.setdefault(eid, {"category": category, "kind": kind, "amounts": []})
        if group["category"] != category or group["kind"] != kind:
            _fail("CONFLICTING_EVIDENCE", "Expense id changes category or obligation kind")
        group["amounts"].append(amount)
    categories, debt, reconciled, missing_ids = {}, 0.0, [], []
    for eid, g in groups.items():
        known = [x for x in g["amounts"] if x is not None]
        amount = max(known) if known else None
        if amount is None:
            missing_ids.append(eid)
        elif g["kind"] == "debt_service":
            debt += amount
        else:
            entry = categories.setdefault(g["category"], {"amount": 0.0, "kind": g["kind"], "benchmark_floor": None})
            if entry["kind"] != g["kind"]:
                _fail("CONFLICTING_EVIDENCE", "Category changes essential/discretionary meaning")
            entry["amount"] += amount
        reconciled.append({"economic_id": eid, "amount": amount, "kind": g["kind"], "evidence_count": len(g["amounts"])})
    used_benchmarks = set()
    for row in _aff_rows([] if benchmarks is None else benchmarks, "benchmarks"):
        if row.get("profile_id") != profile_id:
            continue
        category = _aff_text(row.get("category"), "category")
        if category in used_benchmarks or row.get("currency") != currency or row.get("kind") not in ("essential", "discretionary"):
            _fail("INVALID_INPUT", "Ambiguous benchmark or mismatched currency/kind")
        used_benchmarks.add(category)
        amount = _aff_convert(_aff_scalar(row.get("amount"), "benchmark amount", True), row.get("frequency"), calendar, target_period)
        entry = categories.setdefault(category, {"amount": 0.0, "kind": row["kind"], "benchmark_floor": None})
        if entry["kind"] != row["kind"]:
            _fail("CONFLICTING_EVIDENCE", "Benchmark category kind mismatch")
        entry.update(amount=max(entry["amount"], amount), benchmark_floor=amount)
    missing_categories = [x for x in required if x not in categories]
    known_total = sum(x["amount"] for x in categories.values())
    return _aff_result({"recurring_expenses": None if missing_categories or missing_ids else known_total,
                    "known_expenses": known_total,
                    "debt_service": None if any(groups[k]["kind"] == "debt_service" for k in missing_ids) else debt,
                    "known_debt_service": debt,
                    "essential": sum(x["amount"] for x in categories.values() if x["kind"] == "essential"),
                    "discretionary": sum(x["amount"] for x in categories.values() if x["kind"] == "discretionary"),
                    "categories": categories, "reconciled": reconciled, "excluded_transfers": excluded,
                    "missing_categories": missing_categories, "missing_obligations": missing_ids,
                    "period": target_period, "currency": currency, "profile_id": profile_id},
                   "affordability.estimate_recurring_expenses", currency, context,
                   limitations=("Max reconciliation and benchmark floors are explicit estimation rules, not evidence of complete expenditure coverage.",))


def combined_loan_to_value(claims, collateral, allocations, *, currency, context=None):
    """Partition claims across collateral and allocate rank coverage without reuse."""
    _aff_text(currency, "currency")
    cs, assets = {}, {}
    for rows, key, target, amount_name in ((claims, "claim_id", cs, "amount"), (collateral, "collateral_id", assets, "value")):
        for row in _aff_rows(rows, key):
            rid = _aff_text(row.get(key), key)
            if rid in target or row.get("currency") != currency:
                _fail("INVALID_INPUT", "Duplicate identifiers or currency mismatch")
            target[rid] = dict(row, **{amount_name: _aff_scalar(row.get(amount_name), amount_name, True)})
            if key == "collateral_id":
                _iso_date(row.get("valuation_date"), "valuation_date")
                if row.get("freshness") not in ("CURRENT", "STALE"):
                    _fail("INVALID_INPUT", "Declare valuation freshness")
    by_asset, used, pairs = {}, {k: 0.0 for k in cs}, set()
    for row in _aff_rows(allocations, "allocations"):
        cid, aid = row.get("claim_id"), row.get("collateral_id")
        if cid not in cs or aid not in assets or (cid, aid) in pairs:
            _fail("INVALID_ALLOCATION", "Unknown or repeated claim/collateral allocation")
        priority = _aff_scalar(row.get("priority"), "priority", True)
        if priority != int(priority):
            _fail("INVALID_ALLOCATION", "Lien priority must be integer")
        amount = _aff_scalar(row.get("amount"), "allocation amount", True)
        pairs.add((cid, aid)); used[cid] += amount
        by_asset.setdefault(aid, []).append({"claim_id": cid, "amount": amount, "priority": priority})
    if any(not math.isclose(used[k], row["amount"], rel_tol=1e-12, abs_tol=1e-12) for k, row in cs.items()):
        _fail("INVALID_ALLOCATION", "Allocations must partition each secured claim exactly once")
    rows, coverage = [], []
    for aid in sorted(by_asset):
        value, remaining = assets[aid]["value"], assets[aid]["value"]
        entries = by_asset[aid]
        total = sum(x["amount"] for x in entries)
        cumulative = 0.0
        for rank in sorted({x["priority"] for x in entries}):
            peers = sorted([x for x in entries if x["priority"] == rank], key=lambda x: x["claim_id"])
            mass = sum(x["amount"] for x in peers)
            secured = min(remaining, mass)
            cumulative += mass
            for x in peers:
                allocated = 0.0 if mass == 0 else secured * x["amount"] / mass
                coverage.append(dict(x, collateral_id=aid, covered_amount=allocated,
                                     uncovered_amount=x["amount"]-allocated,
                                     cumulative_ltv=None if value == 0 else cumulative/value))
            remaining -= secured
        rows.append({"collateral_id": aid, "cltv": None if value == 0 else total/value,
                     "value": value, "claims": total, "valuation_date": assets[aid]["valuation_date"],
                     "freshness": assets[aid]["freshness"], "denominator_status": "ZERO_VALUE" if value == 0 else "APPLICABLE"})
    total_value = sum(assets[k]["value"] for k in by_asset)
    total_claims = sum(x["amount"] for x in cs.values())
    return _aff_result({"cltv": None if total_value == 0 else total_claims/total_value,
                    "total_claims": total_claims, "allocated_collateral_value": total_value,
                    "collateral": rows, "priority_coverage": coverage, "currency": currency},
                   "affordability.combined_loan_to_value", "ratio", context,
                   limitations=("Rank coverage is a static proportional waterfall without realization costs; it is not an LGD estimate.",))


def stress_affordability(baseline, scenario, *, context=None):
    """Joint deterministic monthly cashflow stress with coupon resets and maturity.

    Scenario rows supply all FX, income, expense, collateral and rate mappings;
    no independent shocks, random draws or undocumented extrapolation are added.
    Coupons are nominal annual rates divided by twelve. Annuity payments solve
    PV(payments)+PV(terminal balloon)=opening principal at each reset.
    """
    from calendar import monthrange
    from datetime import date

    if not isinstance(baseline, dict) or not isinstance(scenario, dict):
        _fail("INVALID_INPUT", "Baseline and scenario must be objects")
    if scenario.get("schema_version") != "1.0":
        _fail("UNSUPPORTED_SCHEMA_VERSION", "Unsupported stress scenario")
    sid = _aff_text(scenario.get("scenario_id"), "scenario_id")
    currency = _aff_text(baseline.get("currency"), "currency")
    cltv_scope = baseline.get("cltv_scope")
    if cltv_scope not in ("all_contracts_secured_by_pool", "not_applicable"):
        _fail("INVALID_INPUT", "Declare all-contracts collateral pool scope or not_applicable")
    start = date.fromisoformat(_iso_date(baseline.get("as_of"), "as_of"))
    liquid = _aff_scalar(baseline.get("liquid_buffer"), "liquid_buffer", True)
    groups = {}
    for key in ("incomes", "expenses", "contracts", "collateral"):
        groups[key] = {}
        for row in _aff_rows(baseline.get(key), key):
            rid = _aff_text(row.get("economic_id"), "economic_id")
            if rid in groups[key]:
                _fail("DUPLICATE_ECONOMIC_ID", "Baseline must already reconcile each economic source")
            _aff_text(row.get("currency"), "currency")
            groups[key][rid] = dict(row)
            if key == "incomes":
                if row.get("basis") != "net":
                    _fail("INVALID_UNITS", "Stress requires normalized monthly net income")
                _aff_scalar(row.get("monthly_amount"), "monthly_amount")
            elif key == "expenses":
                _aff_scalar(row.get("monthly_amount"), "monthly_amount", True)
            elif key == "collateral":
                _aff_scalar(row.get("value"), "collateral value", True)
    if set(groups["expenses"]) & set(groups["contracts"]):
        _fail("DOUBLE_COUNTING", "A contract debt service cannot also be a living expense")
    schedules = {}
    for cid, c in groups["contracts"].items():
        principal = _aff_scalar(c.get("principal"), "principal", True)
        months = _aff_scalar(c.get("remaining_months"), "remaining_months", True)
        if months < 1 or months != int(months):
            _fail("INVALID_INPUT", "remaining_months must be positive integer")
        annual_rate = _aff_scalar(c.get("annual_rate"), "annual_rate", True)
        if c.get("repayment") not in ("annuity", "interest_only") or c.get("rate_type") not in ("fixed", "reset"):
            _fail("INVALID_INPUT", "Declare repayment and rate_type")
        balloon = _aff_scalar(c.get("balloon_amount"), "balloon_amount", True)
        if balloon > principal or (c["repayment"] == "interest_only" and balloon != principal):
            _fail("INVALID_INPUT", "Balloon must not exceed principal; interest-only repays full principal at maturity")
        resets = c.get("reset_dates")
        if not isinstance(resets, list) or len(set(resets)) != len(resets):
            _fail("INVALID_INPUT", "Declare unique reset_dates")
        for d in resets:
            if _iso_date(d, "reset_date") <= start.isoformat():
                _fail("INVALID_DATE", "Future reset dates must follow baseline")
        if c["rate_type"] == "fixed" and resets:
            _fail("INVALID_INPUT", "Fixed contracts have no reset dates")
        floor = _aff_scalar(c.get("rate_floor"), "rate_floor", True)
        cap = _aff_scalar(c.get("rate_cap"), "rate_cap", True)
        if not floor <= annual_rate <= cap:
            _fail("INVALID_INPUT", "Initial coupon must lie inside declared floor/cap")
        schedules[cid] = {"balance": principal, "rate": annual_rate, "payment": None, "resets": sorted(resets)}
    periods = _aff_rows(scenario.get("periods"), "periods")
    if not periods:
        _fail("EMPTY_INPUT", "Scenario requires at least one monthly period")
    thresholds = baseline.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {"minimum_residual", "maximum_dsti", "maximum_cltv"}:
        _fail("INVALID_INPUT", "Supply residual, DSTI and CLTV thresholds explicitly")
    for key, value in thresholds.items():
        _aff_scalar(value, key, key != "minimum_residual")
    outputs, first_exhaustion = [], None
    prior_date = start.isoformat()
    for index, row in enumerate(periods, 1):
        total_month = start.year*12 + start.month-1 + index
        year, month0 = divmod(total_month, 12)
        expected_date = date(year, month0+1, min(start.day, monthrange(year, month0+1)[1])).isoformat()
        if row.get("date") != expected_date:
            _fail("INVALID_DATE", "Scenario must use consecutive monthly baseline anniversaries")
        fx = row.get("fx_to_reporting")
        currencies = {currency} | {r["currency"] for group in groups.values() for r in group.values()}
        if not isinstance(fx, dict) or not currencies <= set(fx):
            _fail("MISSING_SCENARIO_MAPPING", "Every currency needs explicit FX in every period")
        for code in currencies:
            if _aff_scalar(fx[code], "FX", True) <= 0 or (code == currency and fx[code] != 1):
                _fail("INVALID_UNITS", "FX rates must be positive; reporting FX is one")
        for group, mapping in (("incomes", "income_factors"), ("expenses", "expense_factors"), ("collateral", "collateral_factors"), ("contracts", "rate_shifts")):
            factors = row.get(mapping)
            if not isinstance(factors, dict) or set(factors) != set(groups[group]):
                _fail("MISSING_SCENARIO_MAPPING", f"{mapping} must identify every economic source exactly")
            for factor in factors.values():
                _aff_scalar(factor, mapping, mapping != "rate_shifts")
        income = sum(c["monthly_amount"] * row["income_factors"][k] * fx[c["currency"]] for k, c in groups["incomes"].items())
        expenses = sum(c["monthly_amount"] * row["expense_factors"][k] * fx[c["currency"]] for k, c in groups["expenses"].items())
        collateral_value = sum(c["value"] * row["collateral_factors"][k] * fx[c["currency"]] for k, c in groups["collateral"].items())
        services, opening_debt, closing_debt, payments = 0.0, 0.0, 0.0, []
        for cid, c in groups["contracts"].items():
            state = schedules[cid]
            balance, annual_rate = state["balance"], state["rate"]
            if index > c["remaining_months"]:
                payments.append({"economic_id": cid, "payment": 0.0, "interest": 0.0, "principal_repaid": 0.0, "closing_balance": 0.0, "annual_rate": annual_rate, "reset": False})
                continue
            opening_debt += balance * fx[c["currency"]]
            reset = any(prior_date < d <= expected_date for d in state["resets"])
            if reset:
                # Only payment-boundary resets can be represented without day-count accrual.
                if any(prior_date < d < expected_date for d in state["resets"]):
                    _fail("INVALID_DATE", "Resets must align with monthly payment dates")
                annual_rate = min(c["rate_cap"], max(c["rate_floor"], c["annual_rate"] + row["rate_shifts"][cid]))
            rate = annual_rate / 12
            remaining = int(c["remaining_months"])-index+1
            if state["payment"] is None or reset:
                if c["repayment"] == "interest_only":
                    state["payment"] = balance * rate
                elif rate == 0:
                    state["payment"] = (balance-c["balloon_amount"])/remaining
                else:
                    discount = math.exp(-remaining*math.log1p(rate))
                    state["payment"] = (balance-c["balloon_amount"]*discount)*rate / (-math.expm1(-remaining*math.log1p(rate)))
            interest = balance*rate
            payment = state["payment"]
            # Maturity pays the actual remaining balance; no refinancing is assumed.
            if remaining == 1:
                payment = balance+interest
            principal_repaid = payment-interest
            closing = balance-principal_repaid
            if closing < -1e-9*max(1, c["principal"]):
                _fail("NUMERICAL_FAILURE", "Negative projected balance")
            closing = max(0.0, closing)
            state.update(balance=closing, rate=annual_rate)
            services += payment*fx[c["currency"]]
            closing_debt += closing*fx[c["currency"]]
            payments.append({"economic_id": cid, "payment": payment, "interest": interest,
                             "principal_repaid": principal_repaid, "closing_balance": closing,
                             "annual_rate": annual_rate, "reset": reset})
        residual = income-expenses-services
        liquid += residual
        if liquid < 0 and first_exhaustion is None:
            first_exhaustion = expected_date
        dsti = None if income <= 0 else services/income
        dti = None if income <= 0 else opening_debt/(12*income)
        cltv = None if cltv_scope == "not_applicable" or collateral_value <= 0 else closing_debt/collateral_value
        breaches = []
        if residual < thresholds["minimum_residual"]: breaches.append("RESIDUAL")
        if dsti is None: breaches.append("NONPOSITIVE_INCOME")
        elif dsti > thresholds["maximum_dsti"]: breaches.append("DSTI")
        if cltv is not None and cltv > thresholds["maximum_cltv"]: breaches.append("CLTV")
        if cltv_scope != "not_applicable" and collateral_value <= 0 and closing_debt > 0: breaches.append("ZERO_COLLATERAL_VALUE")
        outputs.append({"date": expected_date, "net_income": income, "expenses": expenses,
                        "debt_service": services, "residual": residual, "liquid_buffer": liquid,
                        "opening_debt": opening_debt, "closing_debt": closing_debt,
                        "collateral_value": collateral_value, "dsti": dsti, "annual_dti": dti,
                        "cltv": cltv, "breaches": breaches, "contracts": payments})
        prior_date = expected_date
    return _aff_result({"scenario_id": sid, "currency": currency, "periods": outputs,
                    "first_liquidity_exhaustion": first_exhaustion,
                    "minimum_residual": min(x["residual"] for x in outputs),
                    "scenario_hash": _fingerprint(scenario), "baseline_hash": _fingerprint(baseline)},
                   "affordability.stress_affordability", currency, context,
                   limitations=("Conditional monthly cashflow projection; no default probability, refinancing, prepayment or causal effect is inferred.",
                                "CLTV requires the declared common all-contracts collateral pool; use combined_loan_to_value for legal lien coverage."))
