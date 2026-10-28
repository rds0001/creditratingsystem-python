"""Deterministic reduced-form pricing with supplied contractual schedules.

These continuous-default integrals are not a claim of ISDA model equivalence.
"""

from copy import deepcopy
from datetime import date as _date
from collections.abc import Mapping
import math
import numpy as np
from .core import _fail, _numeric, _iso_date, _result, _fingerprint, _context_value


def _scalar(x, name, lower=None, upper=None):
    a = _numeric(x, name)
    if a.ndim or (lower is not None and a < lower) or (upper is not None and a > upper):
        _fail("INVALID_INPUT", name + " violates its scalar bounds")
    return float(a)


def _text(x, name):
    if not isinstance(x, str) or not x.strip():
        _fail("INVALID_INPUT", name + " must be nonempty text")
    return x


def _curve(curve, hazard=False):
    if not isinstance(curve, Mapping):
        _fail("INVALID_CURVE", "A curve contract is required")
    c = deepcopy(dict(curve))
    _iso_date(c.get("as_of"), "curve.as_of")
    if _iso_date(c.get("known_at"), "curve.known_at") > _iso_date(c["as_of"], "as_of"):
        _fail("LOOKAHEAD_BIAS", "Curve information is later than valuation")
    _text(c.get("currency"), "curve.currency")
    if c.get("time_basis") != "ACT/365F" or c.get("extrapolation") != "FLAT_LAST_RATE":
        _fail("INVALID_CURVE", "Explicit ACT/365F clock and FLAT_LAST_RATE extrapolation required")
    if hazard and c.get("probability_measure") != "RISK_NEUTRAL":
        _fail("MEASURE_MISMATCH", "Market hazard must be RISK_NEUTRAL")
    knots = _numeric(c.get("knots"), "knots", vector_only=True)
    rates = _numeric(c.get("rates"), "rates", vector_only=True)
    if (
        len(knots) != len(rates)
        or np.any(knots <= 0)
        or np.any(np.diff(knots) <= 0)
        or (hazard and np.any(rates < 0))
    ):
        _fail(
            "INVALID_CURVE",
            "Increasing positive interval ends and aligned admissible rates required",
        )
    c["knots"], c["rates"] = knots.tolist(), rates.tolist()
    return c


def _rate(c, t):
    return c["rates"][min(int(np.searchsorted(c["knots"], t, side="right")), len(c["rates"]) - 1)]


def _integrated(c, t):
    if t < 0:
        _fail("INVALID_DATE", "Curve evaluation before origin")
    result, start = 0.0, 0.0
    for end, rate in zip(c["knots"], c["rates"]):
        result += rate * max(0.0, min(t, end) - start)
        if t <= end:
            return result
        start = end
    return result + c["rates"][-1] * (t - start)


def _discount(c, t, spread=0.0):
    exponent = -_integrated(c, t) - spread * t
    if abs(exponent) > 700:
        _fail("NUMERICAL_FAILURE", "Discount or survival exponent exceeds stable range")
    return math.exp(exponent)


def _moments(q, length):
    z = q * length
    if not math.isfinite(z) or -z > 700:
        _fail("NUMERICAL_FAILURE", "Cashflow integration exponent exceeds stable range")
    if abs(z) < 1e-4:
        # Integrated exp(-q u) and u exp(-q u); stable at zero and negative rates.
        a = sum((-z) ** k / math.factorial(k) / (k + 1) for k in range(12)) * length
        b = sum((-z) ** k / math.factorial(k) / (k + 2) for k in range(12)) * length**2
    else:
        e = math.exp(-z)
        a = -math.expm1(-z) / q
        b = (1 - (1 + z) * e) / (q * q)
    return a, b


def _integral(
    h,
    d,
    start,
    end,
    *,
    spread=0.0,
    hazard_multiplier=1.0,
    default_density=True,
    linear_origin=None,
    linear_scale=1.0,
):
    points = sorted(set([start, end] + [t for c in (h, d) for t in c["knots"] if start < t < end]))
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        lam, r = _rate(h, (a + b) / 2), _rate(d, (a + b) / 2)
        q = r + spread + hazard_multiplier * lam
        i0, i1 = _moments(q, b - a)
        exponent = -_integrated(d, a) - spread * a - hazard_multiplier * _integrated(h, a)
        if abs(exponent) > 700:
            _fail("NUMERICAL_FAILURE", "Integral exponent exceeds stable range")
        value = i0 if linear_origin is None else linear_scale * ((a - linear_origin) * i0 + i1)
        total += math.exp(exponent) * (lam if default_density else 1.0) * value
    if not math.isfinite(total):
        _fail("NUMERICAL_FAILURE", "Nonfinite expected cashflow")
    return total


def _time(date, origin):
    return (
        _date.fromisoformat(_iso_date(date, "cashflow date"))
        - _date.fromisoformat(_iso_date(origin, "as_of"))
    ).days / 365.0


def _setup(hazard_curve, discount_curve, contract):
    h, d = _curve(hazard_curve, True), _curve(discount_curve)
    if not isinstance(contract, Mapping):
        _fail("INVALID_INPUT", "Instrument contract required")
    c = deepcopy(dict(contract))
    for key in ("as_of", "currency"):
        if h[key] != d[key] or c.get(key) != d[key]:
            _fail("CONTEXT_MISMATCH", "Instrument and curves must share " + key)
    if c.get("probability_measure") != "RISK_NEUTRAL":
        _fail("MEASURE_MISMATCH", "Expected market cashflows require RISK_NEUTRAL")
    if c.get("default_settlement") != "AT_DEFAULT":
        _fail("UNSUPPORTED_CONVENTION", "This engine requires recovery settlement at default")
    return h, d, c


def _out(method, value, context=None):
    if _context_value(context).get("probability_measure") not in (None, "RISK_NEUTRAL"):
        _fail("MEASURE_MISMATCH", "Market pricing context must be RISK_NEUTRAL")
    ctx = _context_value(context)
    valuation = value.get("discount_curve", value)
    for ck, vk in (("as_of", "as_of"), ("currency", "currency")):
        if ctx.get(ck) is not None and valuation.get(vk) is not None and ctx[ck] != valuation[vk]:
            _fail("CONTEXT_MISMATCH", "Calculation context conflicts with market " + vk)
    known = value.get("known_at", valuation.get("known_at"))
    if ctx.get("known_at") is not None and known is not None and ctx["known_at"] < known:
        _fail("LOOKAHEAD_BIAS", "Context knowledge precedes market input vintage")
    return _result(
        value,
        "market_implied." + method,
        context=context,
        provenance={"parameter_origin": "ESTIMATED"}
        if method in ("bootstrap_cds_hazard_curve", "calibrate_bond_hazard_curve")
        else None,
        limitations=(
            "Deterministic continuous-default valuation with supplied schedules; no ISDA-equivalence claim. Risk-neutral PD is not physical PD.",
        ),
    )


def cds_expected_cashflow_legs(hazard_curve, discount_curve, cds_contract, *, context=None):
    """Price unit-notional CDS premium, accrued-default and protection cashflows."""
    h, d, c = _setup(hazard_curve, discount_curve, cds_contract)
    recovery = _scalar(c.get("recovery"), "recovery", 0, 1)
    coupon = _scalar(c.get("coupon"), "coupon", 0)
    start, end = _time(c.get("protection_start"), c["as_of"]), _time(c.get("maturity"), c["as_of"])
    if start < 0 or end <= start:
        _fail("INVALID_DATE", "Positive future protection interval required")
    settlement = _time(c.get("settlement_date"), c["as_of"])
    if settlement < 0 or settlement > end:
        _fail("INVALID_DATE", "Settlement must lie between valuation and maturity")
    accrued = _scalar(c.get("accrued_premium"), "accrued_premium")
    upfront = _scalar(c.get("upfront"), "upfront")
    mode = c.get("premium_mode")
    accrual = 0.0
    if mode == "CONTINUOUS":
        premium = _integral(h, d, start, end, default_density=False)
        if accrued != 0:
            _fail("INVALID_INPUT", "Continuous premium has no discrete accrued settlement")
    elif mode == "SCHEDULED":
        rows = c.get("schedule")
        if not isinstance(rows, list) or not rows:
            _fail("INVALID_SCHEDULE", "Supplied CDS coupon schedule required")
        if c.get("day_count") not in ("ACT/360", "ACT/365F") or not isinstance(
            c.get("accrual_on_default"), bool
        ):
            _fail("INVALID_INPUT", "Explicit supported day count and accrual-on-default required")
        premium = 0.0
        previous = None
        for row in rows:
            a, b, p = (
                _time(row.get(k), c["as_of"])
                for k in ("accrual_start", "accrual_end", "payment_date")
            )
            if (
                b <= a
                or b <= start
                or p < b
                or p < 0
                or (previous is not None and abs(previous - a) > 1e-12)
            ):
                _fail(
                    "INVALID_SCHEDULE",
                    "Contiguous accrual periods with payment at or after accrual end required",
                )
            if previous is None and a > start or b > end + 1e-12:
                _fail("INVALID_SCHEDULE", "Coupon accrual must cover protection interval")
            previous = b
            scale = 365 / (360 if c["day_count"] == "ACT/360" else 365)
            # Default risk terminates at accrual end; payment lag only changes discounting.
            premium += (b - a) * scale * _discount(d, p) * _discount(h, b)
            if c["accrual_on_default"]:
                accrual += _integral(h, d, max(start, a), b, linear_origin=a, linear_scale=scale)
        if abs(previous - end) > 1e-12:
            _fail("INVALID_SCHEDULE", "Last coupon end must equal maturity")
    else:
        _fail("INVALID_INPUT", "premium_mode must be CONTINUOUS or SCHEDULED")
    protection = (1 - recovery) * _integral(h, d, start, end)
    annuity = premium + accrual
    settle_df = _discount(d, settlement)
    fair_upfront = (protection - coupon * annuity) / settle_df - accrued
    value = dict(
        as_of=c["as_of"],
        known_at=max(h["known_at"], d["known_at"]),
        currency=c["currency"],
        probability_measure="RISK_NEUTRAL",
        premium_annuity=premium,
        accrual_on_default_annuity=accrual,
        risky_annuity=annuity,
        premium_pv=coupon * premium,
        accrual_on_default_pv=coupon * accrual,
        protection_pv=protection,
        settlement_accrued_pv=accrued * settle_df,
        par_spread=protection / annuity if annuity > 0 else None,
        fair_upfront=fair_upfront,
        upfront_residual_pv=(fair_upfront - upfront) * settle_df,
        maturity_survival=_discount(h, end),
        pricing_contract_hash=_fingerprint(c),
    )
    return _out("cds_expected_cashflow_legs", value, context)


def risky_bond_expected_cashflows(hazard_curve, discount_curve, bond_contract, *, context=None):
    """Price supplied bond cashflows under three distinct recovery conventions.

    Liquidity is an exogenous valuation discount wedge, never a default intensity.
    Treasury recovery pays the risk-free value of all remaining promised cashflows.
    """
    h, d, c = _setup(hazard_curve, discount_curve, bond_contract)
    recovery = _scalar(c.get("recovery"), "recovery", 0, 1)
    liquidity = _scalar(c.get("liquidity_spread"), "liquidity_spread")
    convention = c.get("recovery_convention")
    if convention not in ("PAR", "TREASURY", "MARKET_VALUE"):
        _fail("UNSUPPORTED_CONVENTION", "Explicit PAR, TREASURY or MARKET_VALUE recovery required")
    if c.get("cashflow_assumption") not in ("CONTRACTUAL_NONOPTIONAL", "EXOGENOUS_OPTION_SCENARIO"):
        _fail("UNSUPPORTED_CONVENTION", "Optional cashflows require an explicit exogenous scenario")
    if c["cashflow_assumption"] == "EXOGENOUS_OPTION_SCENARIO":
        _text(c.get("option_scenario_id"), "option_scenario_id")
    rows = c.get("schedule")
    if not isinstance(rows, list) or not rows:
        _fail("INVALID_SCHEDULE", "Coupon/principal schedule required")
    times = []
    coupons = []
    principals = []
    for row in rows:
        t = _time(row.get("payment_date"), c["as_of"])
        if t <= 0 or times and t <= times[-1]:
            _fail("INVALID_SCHEDULE", "Strictly increasing future bond payments required")
        times.append(t)
        coupons.append(_scalar(row.get("coupon_amount"), "coupon_amount", 0))
        principals.append(_scalar(row.get("principal_amount"), "principal_amount", 0))
    face = _scalar(c.get("outstanding_principal"), "outstanding_principal", 0)
    if face <= 0 or abs(sum(principals) - face) > 1e-10 * max(1, face):
        _fail("INVALID_SCHEDULE", "Amortization must conserve outstanding principal")
    accrued = _scalar(c.get("accrued_interest"), "accrued_interest", 0)
    # Price is dirty value at as_of; only same-day settlement avoids hidden forward funding.
    if c.get("settlement_date") != c["as_of"]:
        _fail("UNSUPPORTED_CONVENTION", "Bond valuation uses explicitly same-day settlement")

    def price(spread):
        cp = sum(x * _discount(d, t, spread) * _discount(h, t) for x, t in zip(coupons, times))
        pp = sum(x * _discount(d, t, spread) * _discount(h, t) for x, t in zip(principals, times))
        if convention == "PAR":
            outstanding = face
            previous = 0.0
            rp = 0.0
            for t, p in zip(times, principals):
                rp += recovery * outstanding * _integral(h, d, previous, t, spread=spread)
                outstanding -= p
                previous = t
        elif convention == "MARKET_VALUE":
            total = sum(
                (x + p)
                * math.exp(-_integrated(d, t) - spread * t - (1 - recovery) * _integrated(h, t))
                for x, p, t in zip(coupons, principals, times)
            )
            rp = total - cp - pp
        else:
            zero = deepcopy(d)
            zero["rates"] = [0.0] * len(d["rates"])
            rp = sum(
                recovery * (x + p) * _discount(d, t) * _integral(h, zero, 0.0, t, spread=spread)
                for x, p, t in zip(coupons, principals, times)
            )
        return cp, pp, rp

    cp, pp, rp = price(liquidity)
    zero_liquidity = sum(price(0.0))
    risk_free = sum((x + p) * _discount(d, t) for x, p, t in zip(coupons, principals, times))
    dirty = cp + pp + rp
    value = dict(
        as_of=c["as_of"],
        known_at=max(h["known_at"], d["known_at"]),
        currency=c["currency"],
        probability_measure="RISK_NEUTRAL",
        recovery_convention=convention,
        coupon_pv=cp,
        principal_pv=pp,
        recovery_pv=rp,
        dirty_price=dirty,
        clean_price=dirty - accrued,
        clean_price_per_100=100 * (dirty - accrued) / face,
        risk_free_price=risk_free,
        credit_only_price=zero_liquidity,
        credit_price_effect=zero_liquidity - risk_free,
        liquidity_price_effect=dirty - zero_liquidity,
        maturity_survival=_discount(h, times[-1]),
        pricing_contract_hash=_fingerprint(c),
        liquidity_treatment="EXOGENOUS_DISCOUNT_WEDGE_ALL_DEFAULT_AND_PROMISED_CASHFLOWS",
    )
    if not all(math.isfinite(v) for v in value.values() if isinstance(v, float)):
        _fail("NUMERICAL_FAILURE", "Nonfinite bond price")
    return _out("risky_bond_expected_cashflows", value, context)
