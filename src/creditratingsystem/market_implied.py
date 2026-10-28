"""Local market quotes, reduced-form calibration and identified assumptions."""

from copy import deepcopy
from datetime import date as _date
from collections.abc import Mapping
import numpy as np
from scipy.optimize import brentq, least_squares
from .core import CreditRiskError, _fail, _numeric, _iso_date, _fingerprint
from ._market_pricing import (
    _scalar,
    _text,
    _curve,
    _time,
    _discount,
    _out,
    cds_expected_cashflow_legs,
    risky_bond_expected_cashflows,
)


def _records(rows, name, id_key):
    if (
        not isinstance(rows, (list, tuple))
        or not rows
        or any(not isinstance(r, Mapping) for r in rows)
    ):
        _fail("INVALID_INPUT", name + " must contain records")
    ids = [_text(r.get(id_key), id_key) for r in rows]
    if len(set(ids)) != len(ids):
        _fail("DUPLICATE_ID", name + " contains duplicate identifiers")
    return deepcopy(list(rows))


def build_credit_quote_snapshot(
    quotes, instruments, discount_curve, snapshot_contract, *, context=None
):
    """Normalize dated local quote units and retain explicit eligibility findings."""
    d = _curve(discount_curve)
    c = dict(snapshot_contract)
    as_of = _iso_date(c.get("as_of"), "as_of")
    known = _iso_date(c.get("known_at"), "known_at")
    if (
        known > as_of
        or d["known_at"] > known
        or d["as_of"] != c["as_of"]
        or d["currency"] != c.get("currency")
    ):
        _fail("CONTEXT_MISMATCH", "Snapshot valuation, knowledge and discount curve conflict")
    max_age = _scalar(c.get("maximum_age_days"), "maximum_age_days", 0)
    if (
        max_age != int(max_age)
        or c.get("stale_policy") not in ("EXCLUDE", "KEEP_FLAGGED")
        or c.get("bid_ask_policy") not in ("REQUIRED", "OPTIONAL")
    ):
        _fail("INVALID_INPUT", "Explicit integer quote age, stale and bid/ask policies required")
    ins = {r["instrument_id"]: r for r in _records(instruments, "instruments", "instrument_id")}
    output = []
    findings = []
    scale_map = {
        "CDS_SPREAD": {"BASIS_POINTS": 0.0001, "FRACTION": 1.0},
        "CDS_UPFRONT": {"PERCENT_NOTIONAL": 0.01, "FRACTION": 1.0},
        "BOND_CLEAN_PRICE": {"PER_100_PAR": 1.0},
    }
    groups = {}
    for q in sorted(_records(quotes, "quotes", "quote_id"), key=lambda r: r["quote_id"]):
        instrument = ins.get(q.get("instrument_id"))
        if instrument is None:
            _fail("UNKNOWN_ID", "Quote instrument absent")
        for key in ("issuer_id", "seniority", "currency"):
            _text(instrument.get(key), key)
        kind = q.get("quote_type")
        if kind not in scale_map or q.get("unit") not in scale_map[kind]:
            _fail("INVALID_UNIT", "Quote type and units must be explicit and compatible")
        if instrument.get("instrument_type") != ("BOND" if kind == "BOND_CLEAN_PRICE" else "CDS"):
            _fail("INVALID_QUOTE", "Quote does not match instrument type")
        observed = _iso_date(q.get("observed_at"), "observed_at")
        available = _iso_date(q.get("known_at"), "quote.known_at")
        if available < observed:
            _fail("INVALID_DATE", "Quote knowledge cannot precede observation")
        reasons = []
        if observed > as_of or available > known:
            reasons.append("FUTURE_INFORMATION")
        age = (_date.fromisoformat(as_of) - _date.fromisoformat(observed)).days
        if age > max_age:
            reasons.append("STALE")
        if instrument["currency"] != c["currency"]:
            reasons.append("CURRENCY_MISMATCH")
        factor = scale_map[kind][q["unit"]]
        mid = _scalar(q.get("value"), "quote.value") * factor
        if kind in ("CDS_SPREAD", "BOND_CLEAN_PRICE") and mid < 0:
            _fail("INVALID_QUOTE", "Spreads and bond prices must be nonnegative")
        bid = q.get("bid")
        ask = q.get("ask")
        if (bid is None) != (ask is None) or bid is None and c["bid_ask_policy"] == "REQUIRED":
            _fail("INVALID_QUOTE", "Complete bid/ask pair required by policy")
        if bid is not None:
            bid = _scalar(bid, "bid") * factor
            ask = _scalar(ask, "ask") * factor
            if not bid <= mid <= ask:
                reasons.append("CROSSED_OR_OUTSIDE_BID_ASK")
        eligible = not any(r != "STALE" or c["stale_policy"] == "EXCLUDE" for r in reasons)
        row = dict(
            quote_id=q["quote_id"],
            instrument_id=q["instrument_id"],
            issuer_id=instrument["issuer_id"],
            seniority=instrument["seniority"],
            currency=instrument["currency"],
            quote_type=kind,
            value=mid,
            bid=bid,
            ask=ask,
            normalized_unit="PER_100_PAR" if kind == "BOND_CLEAN_PRICE" else "FRACTION",
            observed_at=q["observed_at"],
            known_at=q["known_at"],
            age_days=age,
            eligible=eligible,
            findings=reasons,
            instrument=instrument,
        )
        output.append(row)
        if reasons:
            findings.append(dict(quote_id=q["quote_id"], findings=reasons))
        groups.setdefault(instrument["issuer_id"], set()).add(instrument["seniority"])
    vintage_groups = {}
    for row in output:
        vintage_groups.setdefault(
            (row["instrument_id"], row["quote_type"], row["observed_at"], row["known_at"]), []
        ).append(row)
    for group in vintage_groups.values():
        if len({r["value"] for r in group}) > 1:
            for row in group:
                row["eligible"] = False
                row["findings"].append("CONFLICTING_SAME_VINTAGE_QUOTES")
    findings = [
        dict(quote_id=r["quote_id"], findings=r["findings"]) for r in output if r["findings"]
    ]
    mixed = [
        dict(issuer_id=k, seniorities=sorted(v)) for k, v in sorted(groups.items()) if len(v) > 1
    ]
    value = dict(
        schema_version="1.0",
        kind="CreditQuoteSnapshot",
        as_of=c["as_of"],
        known_at=c["known_at"],
        currency=c["currency"],
        quotes=output,
        findings=findings,
        cross_seniority_groups=mixed,
        discount_curve=d,
        snapshot_contract=c,
        information_clock="CALENDAR_DATE",
    )
    value["snapshot_hash"] = _fingerprint(value)
    return _out("build_credit_quote_snapshot", value, context)


def _hazard(discount, knots, rates):
    h = {k: discount[k] for k in ("as_of", "known_at", "currency", "time_basis", "extrapolation")}
    h.update(knots=list(knots), rates=list(rates), probability_measure="RISK_NEUTRAL")
    return h


def _cal_quotes(quotes, discount, kind):
    rows = _records(quotes, "calibration quotes", "quote_id")
    for row in rows:
        if row.get("eligible") is not True:
            _fail("INELIGIBLE_QUOTE", "Only explicitly eligible quotes may calibrate")
        observed = _iso_date(row.get("observed_at"), "observed_at")
        known = _iso_date(row.get("known_at"), "quote.known_at")
        if observed > known or known > _iso_date(discount["as_of"], "curve.as_of"):
            _fail("LOOKAHEAD_BIAS", "Calibration quote knowledge must not exceed valuation date")
        instrument = row.get("instrument")
        if not isinstance(instrument, Mapping) or instrument.get("instrument_type") != kind:
            _fail("INVALID_QUOTE", "Full instrument contract of correct type required")
        if (
            instrument.get("as_of") != discount["as_of"]
            or instrument.get("currency") != discount["currency"]
        ):
            _fail("CONTEXT_MISMATCH", "Instrument and calibration curve conflict")
        for key in ("issuer_id", "seniority"):
            _text(instrument.get(key), key)
        if row.get("normalized_unit") != ("PER_100_PAR" if kind == "BOND" else "FRACTION"):
            _fail("INVALID_UNIT", "Calibration requires normalized quote units")
        _scalar(row.get("value"), "quote value")
    for key in ("issuer_id", "seniority"):
        if len({r["instrument"][key] for r in rows}) != 1:
            _fail("INCOMPATIBLE_QUOTES", "Cannot pool different issuers or seniorities")
    return rows


def _seal(value):
    value = deepcopy(value)
    value["calibration_hash"] = _fingerprint(value)
    return value


def _read_fit(value):
    if not isinstance(value, Mapping):
        _fail("INVALID_ARTIFACT", "Calibration contract required")
    v = deepcopy(dict(value))
    h = v.pop("calibration_hash", None)
    if h != _fingerprint(v):
        _fail("INTEGRITY_ERROR", "Market calibration changed after fitting")
    if v.get("kind") != "MarketCreditCalibration" or v.get("probability_measure") != "RISK_NEUTRAL":
        _fail("INVALID_ARTIFACT", "Expected market calibration")
    return v


def bootstrap_cds_hazard_curve(quotes, discount_curve, calibration_contract, *, context=None):
    """Bootstrap bounded piecewise Q intensity; inconsistent quotes remain infeasible."""
    d = _curve(discount_curve)
    c = deepcopy(dict(calibration_contract))
    rows = _cal_quotes(quotes, d, "CDS")
    recovery = _scalar(c.get("recovery"), "recovery", 0, 1)
    upper = _scalar(c.get("maximum_hazard"), "maximum_hazard", 0)
    tol = _scalar(c.get("price_tolerance"), "price_tolerance", 0)
    if upper <= 0 or tol <= 0:
        _fail("INVALID_INPUT", "Positive solver bounds and tolerance required")
    rows.sort(key=lambda r: _time(r["instrument"].get("maturity"), d["as_of"]))
    knots = [_time(r["instrument"].get("maturity"), d["as_of"]) for r in rows]
    if min(knots) <= 0 or np.any(np.diff(knots) <= 0):
        _fail("INVALID_QUOTE", "Distinct increasing CDS maturities required")
    rates = []
    residuals = []
    solver = []
    feasible = True
    for j, row in enumerate(rows):
        instrument = deepcopy(row["instrument"])
        instrument["recovery"] = recovery
        if row.get("quote_type") == "CDS_SPREAD":
            instrument["coupon"] = row["value"]
            instrument["upfront"] = 0.0
            if instrument.get("accrued_premium") != 0:
                _fail(
                    "INVALID_QUOTE", "Par spread bootstrap requires zero settlement accrued premium"
                )
        elif row.get("quote_type") == "CDS_UPFRONT":
            instrument["upfront"] = row["value"]
        else:
            _fail("INVALID_QUOTE", "CDS spread or upfront quote required")

        def objective(rate):
            return cds_expected_cashflow_legs(
                _hazard(d, knots[: j + 1], rates + [rate]), d, instrument
            ).value["upfront_residual_pv"]

        lo, hi = objective(0.0), objective(upper)
        if abs(lo) <= tol:
            rate = 0.0
            iterations = 0
        elif abs(hi) <= tol:
            rate = upper
            iterations = 0
        elif lo * hi > 0:
            feasible = False
            solver.append(
                dict(
                    quote_id=row["quote_id"],
                    status="NO_NONNEGATIVE_BRACKET",
                    residual_at_zero=lo,
                    residual_at_upper=hi,
                )
            )
            break
        else:
            rate, info = brentq(objective, 0.0, upper, xtol=1e-13, rtol=1e-13, full_output=True)
            iterations = info.iterations
        residual = objective(rate)
        rates.append(float(rate))
        residuals.append(float(residual))
        solver.append(
            dict(
                quote_id=row["quote_id"],
                status="BOUNDARY" if rate in (0.0, upper) else "INTERIOR",
                iterations=int(iterations),
                residual_at_zero=lo,
                residual_at_upper=hi,
            )
        )
    if any(abs(r) > tol for r in residuals):
        feasible = False
        for entry, residual in zip(solver, residuals):
            if abs(residual) > tol:
                entry["status"] = "RESIDUAL_TOLERANCE_NOT_MET"
    h = _hazard(d, knots[: len(rates)], rates) if rates else None
    if h is not None:
        h["known_at"] = max(d["known_at"], *[r["known_at"] for r in rows])
    # Quote and recovery derivatives use the actual pricing engine, conditional on preceding knots.
    sensitivities = []
    if feasible:
        for j, row in enumerate(rows):
            ins = deepcopy(row["instrument"])
            ins["recovery"] = recovery
            if row["quote_type"] == "CDS_SPREAD":
                ins["coupon"] = row["value"]
                ins["upfront"] = 0.0
            else:
                ins["upfront"] = row["value"]

            def pv(rr, rec):
                cc = deepcopy(ins)
                cc["recovery"] = rec
                return cds_expected_cashflow_legs(_hazard(d, knots, rr), d, cc).value[
                    "upfront_residual_pv"
                ]

            step = 1e-6
            minus = max(0.0, rates[j] - step)
            plus = min(upper, rates[j] + step)
            r0 = list(rates)
            r1 = list(rates)
            r0[j] = minus
            r1[j] = plus
            derivative = (pv(r1, recovery) - pv(r0, recovery)) / (plus - minus)
            a = max(0.0, recovery - step)
            b = min(1.0, recovery + step)
            dr = (pv(rates, b) - pv(rates, a)) / (b - a)
            legs = cds_expected_cashflow_legs(h, d, ins).value
            dq = (
                -legs["risky_annuity"]
                if row["quote_type"] == "CDS_SPREAD"
                else -_discount(d, _time(ins["settlement_date"], d["as_of"]))
            )
            sensitivities.append(
                dict(
                    quote_id=row["quote_id"],
                    hazard_price_derivative=derivative,
                    local_hazard_per_quote=-dq / derivative if abs(derivative) > 1e-14 else None,
                    local_hazard_per_recovery=-dr / derivative if abs(derivative) > 1e-14 else None,
                    scope="LOCAL_IMPLICIT_DERIVATIVE_PRECEDING_HAZARDS_FIXED",
                )
            )
    value = _seal(
        dict(
            schema_version="1.0",
            kind="MarketCreditCalibration",
            instrument_type="CDS",
            probability_measure="RISK_NEUTRAL",
            known_at=max(d["known_at"], *[r["known_at"] for r in rows]),
            feasible=feasible,
            hazard_curve=h,
            issuer_id=rows[0]["instrument"]["issuer_id"],
            seniority=rows[0]["instrument"]["seniority"],
            discount_curve=d,
            quotes=rows,
            calibration_contract=c,
            residuals=residuals,
            solver=solver,
            sensitivities=sensitivities,
            survival=[_discount(h, t) for t in h["knots"]] if h else [],
            assumptions=dict(
                recovery="FIXED", intensity="PIECEWISE_CONSTANT", probability_measure="RISK_NEUTRAL"
            ),
        )
    )
    value.pop("calibration_hash")
    value["curve_sensitivities"] = _cds_curve_sensitivity(value) if feasible else None
    return _out("bootstrap_cds_hazard_curve", _seal(value), context)


def _jacobian(fn, x, lower, upper):
    columns = []
    for j, v in enumerate(x):
        step = 1e-5 * max(1.0, abs(v))
        a = max(lower[j], v - step)
        b = min(upper[j], v + step)
        xx = np.array(x, dtype=float)
        yy = xx.copy()
        xx[j] = a
        yy[j] = b
        columns.append((np.asarray(fn(yy)) - np.asarray(fn(xx))) / (b - a))
    return np.column_stack(columns)


def _rank_diagnostics(jac):
    if not np.all(np.isfinite(jac)):
        _fail("NUMERICAL_FAILURE", "Nonfinite calibration Jacobian")
    scale = np.linalg.norm(jac, axis=0)
    standardized = jac / np.where(scale > 0, scale, 1.0)
    singular = np.linalg.svd(standardized, compute_uv=False)
    rank = int(np.sum(singular > max(1e-10, (singular[0] if len(singular) else 0) * 1e-8)))
    full = rank == jac.shape[1]
    return dict(
        rank=rank,
        parameter_count=int(jac.shape[1]),
        column_norms=scale.tolist(),
        scaled_singular_values=singular.tolist(),
        full_column_rank=full,
        scaled_condition_number=float(singular[0] / singular[-1]) if full else None,
        rank_relative_tolerance=1e-8,
    )


def calibrate_bond_hazard_curve(quotes, discount_curve, calibration_contract, *, context=None):
    """Fit bounded Q hazards to Gaussian price errors with fixed recovery/liquidity.

    Price standard errors are supplied measurement assumptions. Local intervals are
    conditional Wald intervals, unavailable at an active boundary or deficient rank.
    """
    d = _curve(discount_curve)
    c = deepcopy(dict(calibration_contract))
    rows = _cal_quotes(quotes, d, "BOND")
    knots = _numeric(c.get("hazard_knots"), "hazard_knots", vector_only=True)
    if np.any(knots <= 0) or np.any(np.diff(knots) <= 0):
        _fail("INVALID_CURVE", "Positive increasing hazard knots required")
    initial = _numeric(c.get("initial_hazards"), "initial_hazards", vector_only=True)
    upper = _scalar(c.get("maximum_hazard"), "maximum_hazard", 0)
    recovery = _scalar(c.get("recovery"), "recovery", 0, 1)
    liquidity = _scalar(c.get("liquidity_spread"), "liquidity_spread")
    if c.get("weight_interpretation") != "KNOWN_GAUSSIAN_PRICE_STANDARD_ERRORS":
        _fail("INVALID_INPUT", "Explicit price measurement-error assumptions required")
    if len(initial) != len(knots) or upper <= 0 or np.any(initial < 0) or np.any(initial > upper):
        _fail("INVALID_INPUT", "Initial hazards must match bounded knot parameters")
    max_eval = _scalar(c.get("max_evaluations"), "max_evaluations", 1)
    if max_eval != int(max_eval):
        _fail("INVALID_INPUT", "Integer max_evaluations required")
    se = np.array([_scalar(r.get("price_standard_error"), "price_standard_error", 0) for r in rows])
    if np.any(se <= 0):
        _fail("INVALID_WEIGHTS", "Strictly positive known price errors required")
    if any(r.get("quote_type") != "BOND_CLEAN_PRICE" for r in rows):
        _fail("INVALID_QUOTE", "Bond clean-price quotes required")
    observed = np.array([r["value"] for r in rows])

    def prices(rates):
        h = _hazard(d, knots, rates)
        output = []
        for row in rows:
            ins = deepcopy(row["instrument"])
            ins.update(recovery=recovery, liquidity_spread=liquidity)
            output.append(risky_bond_expected_cashflows(h, d, ins).value["clean_price_per_100"])
        return np.array(output)

    def residual(rates):
        return (prices(rates) - observed) / se

    fit = least_squares(
        residual,
        initial,
        bounds=(np.zeros(len(knots)), np.full(len(knots), upper)),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-10,
        max_nfev=int(max_eval),
        diff_step=1e-5,
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        _fail("NON_CONVERGENCE", "Bond hazard fit failed", {"solver_status": int(fit.status)})
    rates = fit.x
    jac = _jacobian(residual, rates, np.zeros(len(knots)), np.full(len(knots), upper))
    rank = _rank_diagnostics(jac)
    boundary = (rates < 1e-7) | (upper - rates < 1e-7)
    intervals = []
    covariance = None
    if rank["full_column_rank"] and not np.any(boundary):
        cov = np.linalg.inv(jac.T @ jac)
        covariance = cov.tolist()
        sd = np.sqrt(np.diag(cov))
        for j, v in enumerate(rates):
            intervals.append(
                dict(
                    parameter_index=j,
                    lower=float(v - 1.959963984540054 * sd[j]),
                    upper=float(v + 1.959963984540054 * sd[j]),
                    scope="UNTRUNCATED_LOCAL_WALD_CONDITIONAL_ON_RECOVERY_LIQUIDITY_AND_KNOWN_PRICE_ERRORS",
                )
            )
    h = _hazard(d, knots, rates)
    h["known_at"] = max(d["known_at"], *[r["known_at"] for r in rows])
    value = _seal(
        dict(
            schema_version="1.0",
            kind="MarketCreditCalibration",
            instrument_type="BOND",
            probability_measure="RISK_NEUTRAL",
            known_at=h["known_at"],
            feasible=True,
            hazard_curve=h,
            issuer_id=rows[0]["instrument"]["issuer_id"],
            seniority=rows[0]["instrument"]["seniority"],
            discount_curve=d,
            quotes=rows,
            calibration_contract=c,
            residuals=(prices(rates) - observed).tolist(),
            weighted_residuals=residual(rates).tolist(),
            objective=float(np.sum(residual(rates) ** 2)),
            fitted_prices=prices(rates).tolist(),
            jacobian=jac.tolist(),
            identifiability=rank,
            boundary_parameters=np.flatnonzero(boundary).tolist(),
            parameter_intervals=intervals,
            covariance=covariance,
            interval_status="AVAILABLE_LOCAL_APPROXIMATION"
            if intervals
            else "UNAVAILABLE_BOUNDARY_OR_RANK_DEFICIENCY",
            solver=dict(
                converged=True, evaluations=int(fit.nfev), optimality=float(fit.optimality)
            ),
            survival=[_discount(h, t) for t in knots],
            assumptions=dict(
                recovery="FIXED",
                liquidity_spread="FIXED",
                price_errors="KNOWN_GAUSSIAN",
                probability_measure="RISK_NEUTRAL",
            ),
        )
    )
    return _out("calibrate_bond_hazard_curve", value, context)


def _predicted_quotes(fit, hazards, recovery, liquidity):
    d = fit["discount_curve"]
    h = _hazard(d, fit["hazard_curve"]["knots"], hazards)
    values = []
    for row in fit["quotes"]:
        ins = deepcopy(row["instrument"])
        ins.update(recovery=recovery)
        if fit["instrument_type"] == "BOND":
            ins["liquidity_spread"] = liquidity
            value = risky_bond_expected_cashflows(h, d, ins).value["clean_price_per_100"]
        else:
            value = cds_expected_cashflow_legs(h, d, ins).value[
                "par_spread" if row["quote_type"] == "CDS_SPREAD" else "fair_upfront"
            ]
        values.append(value)
    return np.array(values)


def diagnose_market_credit_identifiability(
    calibration, alternative_assumptions, diagnostic_contract, *, context=None
):
    """Refit declared alternatives and measure joint hazard/recovery/liquidity rank."""
    fit = _read_fit(calibration)
    c = dict(diagnostic_contract)
    if not fit["feasible"]:
        _fail("INFEASIBLE_CALIBRATION", "Identifiability needs a feasible reference fit")
    scenarios = _records(alternative_assumptions, "alternative_assumptions", "scenario_id")
    tolerance = _scalar(c.get("equivalence_tolerance"), "equivalence_tolerance", 0)
    quote_scales = _numeric(
        c.get("quote_standard_errors"), "quote_standard_errors", vector_only=True
    )
    if len(quote_scales) != len(fit["quotes"]) or np.any(quote_scales <= 0):
        _fail("INVALID_WEIGHTS", "Explicit positive quote-error scales required")
    reference = np.array([r["value"] for r in fit["quotes"]])
    base = fit["calibration_contract"]
    rates = fit["hazard_curve"]["rates"]
    recovery = base["recovery"]
    liquidity = base.get("liquidity_spread", 0.0)
    include_liquidity = fit["instrument_type"] == "BOND"
    params = list(rates) + [recovery] + ([liquidity] if include_liquidity else [])
    rb = c.get("recovery_bounds")
    lb = c.get("liquidity_bounds")
    if (
        not isinstance(rb, (list, tuple))
        or len(rb) != 2
        or not 0 <= rb[0] < rb[1] <= 1
        or not rb[0] <= recovery <= rb[1]
    ):
        _fail("INVALID_INPUT", "Recovery profile bounds must contain reference")
    if include_liquidity and (
        not isinstance(lb, (list, tuple))
        or len(lb) != 2
        or not lb[0] < lb[1]
        or not lb[0] <= liquidity <= lb[1]
    ):
        _fail("INVALID_INPUT", "Liquidity profile bounds must contain reference")
    lower = [0.0] * len(rates) + [rb[0]] + ([lb[0]] if include_liquidity else [])
    upper = [base["maximum_hazard"]] * len(rates) + [rb[1]] + ([lb[1]] if include_liquidity else [])
    independent = c.get("independent_recovery_observation")
    if independent is not None:
        _scalar(independent.get("value"), "independent recovery", 0, 1)
        if (
            _scalar(independent.get("standard_error"), "independent recovery standard error", 0)
            <= 0
        ):
            _fail("INVALID_INPUT", "Positive recovery standard error required")
        _text(independent.get("evidence_id"), "recovery evidence_id")
        if _iso_date(independent.get("known_at"), "recovery known_at") > fit["known_at"]:
            _fail("LOOKAHEAD_BIAS", "Future independent recovery evidence")

    def response(p):
        values = (
            _predicted_quotes(
                fit, p[: len(rates)], p[len(rates)], p[-1] if include_liquidity else 0
            )
            - reference
        ) / quote_scales
        if independent is not None:
            values = np.r_[
                values, (p[len(rates)] - independent["value"]) / independent["standard_error"]
            ]
        return values

    jac = _jacobian(response, params, lower, upper)
    rank = _rank_diagnostics(jac)
    base_objective = float(np.sum(response(params) ** 2))
    profiles = []
    for scenario in scenarios:
        rr = _scalar(scenario.get("recovery"), "scenario recovery", rb[0], rb[1])
        ll = _scalar(
            scenario.get("liquidity_spread", 0),
            "scenario liquidity",
            lb[0] if include_liquidity else 0,
            lb[1] if include_liquidity else 0,
        )
        cc = deepcopy(base)
        cc["recovery"] = rr
        if include_liquidity:
            cc["liquidity_spread"] = ll
        try:
            alternative = (
                calibrate_bond_hazard_curve if include_liquidity else bootstrap_cds_hazard_curve
            )(fit["quotes"], fit["discount_curve"], cc).value
        except CreditRiskError as error:
            if error.code not in ("NON_CONVERGENCE", "NUMERICAL_FAILURE"):
                raise
            profiles.append(
                dict(
                    scenario_id=scenario["scenario_id"],
                    recovery=rr,
                    liquidity_spread=ll,
                    feasible=False,
                    calibration_hash=None,
                    hazards=None,
                    objective=None,
                    near_equivalent=False,
                    failure_code=error.code,
                )
            )
            continue
        row = dict(
            scenario_id=scenario["scenario_id"],
            recovery=rr,
            liquidity_spread=ll,
            feasible=alternative["feasible"],
            calibration_hash=alternative["calibration_hash"],
        )
        if alternative["feasible"]:
            pp = alternative["hazard_curve"]["rates"] + [rr] + ([ll] if include_liquidity else [])
            obj = float(np.sum(response(pp) ** 2))
            row.update(
                hazards=alternative["hazard_curve"]["rates"],
                objective=obj,
                near_equivalent=obj <= base_objective + tolerance,
            )
        else:
            row.update(hazards=None, objective=None, near_equivalent=False)
        profiles.append(row)
    value = dict(
        reference_calibration_hash=calibration["calibration_hash"],
        joint_identifiability=rank,
        parameter_order=["hazard_" + str(i) for i in range(len(rates))]
        + ["recovery"]
        + (["liquidity_spread"] if include_liquidity else []),
        jacobian=jac.tolist(),
        reference_objective=base_objective,
        profiles=profiles,
        equivalent_scenario_ids=[r["scenario_id"] for r in profiles if r["near_equivalent"]],
        recovery_evidence="INDEPENDENT_OBSERVATION" if independent else "ASSUMPTION_ONLY",
        scope="FINITE_DECLARED_SCENARIO_PROFILE_NOT_GLOBAL_IDENTIFIED_SET",
        diagnostic_contract=c,
    )
    return _out("diagnose_market_credit_identifiability", value, context)


from ._market_bridge import fit_market_to_physical_pd_bridge  # noqa: E402,F401


def compare_cds_bond_credit_basis(
    cds_calibration, bond_calibration, comparison_contract, *, context=None
):
    """Reprice matched Q curves under one supplied CDS contract and common conventions."""
    cds = _read_fit(cds_calibration)
    bond = _read_fit(bond_calibration)
    c = deepcopy(dict(comparison_contract))
    if (
        cds["instrument_type"] != "CDS"
        or bond["instrument_type"] != "BOND"
        or not cds["feasible"]
        or not bond["feasible"]
    ):
        _fail("INVALID_ARTIFACT", "Feasible CDS and bond calibrations required")
    findings = []
    for key in ("issuer_id", "seniority"):
        if cds[key] != bond[key]:
            findings.append(key.upper() + "_MISMATCH")
    for key in ("as_of", "currency", "known_at"):
        if cds["discount_curve"][key] != bond["discount_curve"][key]:
            findings.append(key.upper() + "_MISMATCH")
    if _fingerprint(cds["discount_curve"]) != _fingerprint(bond["discount_curve"]):
        findings.append("DISCOUNT_CONVENTION_MISMATCH")
    recovery = _scalar(c.get("recovery"), "comparison recovery", 0, 1)
    if (
        cds["calibration_contract"]["recovery"] != recovery
        or bond["calibration_contract"]["recovery"] != recovery
    ):
        findings.append("RECOVERY_ASSUMPTION_MISMATCH")
    if bond["calibration_contract"]["liquidity_spread"] != c.get("bond_liquidity_spread"):
        findings.append("LIQUIDITY_ASSUMPTION_MISMATCH")
    if c.get("funding_treatment") != "NO_ADDITIONAL_FUNDING_WEDGE":
        _fail("UNSUPPORTED_CONVENTION", "Explicit no-additional-funding convention required")
    maximum_age = _scalar(c.get("maximum_quote_age_days"), "maximum_quote_age_days", 0)
    if maximum_age != int(maximum_age):
        _fail("INVALID_INPUT", "Integer maximum quote age required")
    for name, fit in (("CDS", cds), ("BOND", bond)):
        if any(
            (
                _date.fromisoformat(fit["discount_curve"]["as_of"])
                - _date.fromisoformat(q["observed_at"])
            ).days
            > maximum_age
            for q in fit["quotes"]
        ):
            findings.append(name + "_STALE_QUOTES")
    ins = c.get("comparison_cds_contract")
    if not isinstance(ins, Mapping):
        _fail("INVALID_INPUT", "Common comparison CDS contract required")
    if ins.get("issuer_id") != cds["issuer_id"] or ins.get("seniority") != cds["seniority"]:
        findings.append("COMPARISON_INSTRUMENT_MISMATCH")
    if ins.get("recovery") != recovery:
        findings.append("COMPARISON_RECOVERY_MISMATCH")
    maturity = _time(ins.get("maturity"), cds["discount_curve"]["as_of"])
    if maturity > min(cds["hazard_curve"]["knots"][-1], bond["hazard_curve"]["knots"][-1]):
        findings.append("UNMATCHED_MATURITY_EXTRAPOLATION")
    value = dict(
        comparable=not findings,
        findings=findings,
        cds_calibration_hash=cds_calibration["calibration_hash"],
        bond_calibration_hash=bond_calibration["calibration_hash"],
        probability_measure="RISK_NEUTRAL",
        basis_sign="CDS_PAR_SPREAD_MINUS_BOND_IMPLIED_EQUIVALENT_CDS_PAR_SPREAD",
        interpretation="CONVENTION_CONDITIONAL_MARKET_BASIS_NOT_PURE_DEFAULT_SIGNAL",
        comparison_contract=c,
    )
    if findings:
        value.update(
            cds_par_spread=None,
            bond_implied_par_spread=None,
            basis=None,
            basis_basis_points=None,
            survival_difference=None,
        )
    else:
        a = cds_expected_cashflow_legs(cds["hazard_curve"], cds["discount_curve"], ins).value
        b = cds_expected_cashflow_legs(bond["hazard_curve"], bond["discount_curve"], ins).value
        basis = a["par_spread"] - b["par_spread"]
        value.update(
            cds_par_spread=a["par_spread"],
            bond_implied_par_spread=b["par_spread"],
            basis=basis,
            basis_basis_points=1e4 * basis,
            survival_difference=a["maturity_survival"] - b["maturity_survival"],
        )
    return _out("compare_cds_bond_credit_basis", value, context)


def _cds_curve_sensitivity(fit):
    rates = np.array(fit["hazard_curve"]["rates"])
    c = fit["calibration_contract"]
    recovery = c["recovery"]
    upper = c["maximum_hazard"]

    def fn(p):
        return _predicted_quotes(fit, p, recovery, 0.0)

    jac = _jacobian(fn, rates, np.zeros(len(rates)), np.full(len(rates), upper))
    rank = _rank_diagnostics(jac)
    a = max(0.0, recovery - 1e-6)
    b = min(1.0, recovery + 1e-6)
    dr = (_predicted_quotes(fit, rates, b, 0.0) - _predicted_quotes(fit, rates, a, 0.0)) / (b - a)
    boundary = bool(np.any((rates <= 1e-10) | (rates >= upper - 1e-10)))
    available = rank["full_column_rank"] and not boundary
    return dict(
        quote_order=[r["quote_id"] for r in fit["quotes"]],
        hazard_order=list(range(len(rates))),
        quote_hazard_jacobian=jac.tolist(),
        identifiability=rank,
        hazard_per_quote=np.linalg.solve(jac, np.eye(len(rates))).tolist() if available else None,
        hazard_per_recovery=np.linalg.solve(jac, -dr).tolist() if available else None,
        status="INTERIOR_IMPLICIT_FULL_CURVE"
        if available
        else "UNAVAILABLE_BOUNDARY_OR_RANK_DEFICIENCY",
    )
