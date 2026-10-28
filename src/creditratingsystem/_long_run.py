"""Observed annual default-rate aggregation with explicit assumptions."""
from collections.abc import Mapping
from datetime import date
import math
import numpy as np
from scipy.stats import beta, t
from .core import _fail, _numeric, _result
from ._ttc import _physical_horizon


def _day(value):
    try:
        result = date.fromisoformat(value)
        if result.isoformat() != value:
            raise ValueError
        return result
    except (TypeError, ValueError):
        _fail("INVALID_DATE", "Expected ISO calendar date")


def estimate_long_run_pd(cohorts, *, selected_cohort_ids, knowledge_date, weighting,
                         uncertainty, dependence_assumption, representativeness,
                         target, confidence_level=.95, context=None):
    """Aggregate selected complete annual cohorts; not automatically TTC."""
    if not isinstance(target, Mapping):
        _fail("INVALID_TARGET", "Explicit target required")
    for key in ("event", "default_definition", "conditioning"):
        if not isinstance(target.get(key), str) or not target[key].strip():
            _fail("INVALID_TARGET", f"target.{key} required")
    ctx, _ = _physical_horizon(target.get("horizon"), target.get("time_unit"), target.get("probability_measure"), context)
    if target["horizon"] != 1 or target["time_unit"] != "years":
        _fail("INCOMPATIBLE_HORIZON", "One-year cohorts required")
    cutoff = _day(knowledge_date)
    if weighting not in ("annual_mean", "pooled_obligor") or uncertainty not in ("none", "year_t", "clopper_pearson"):
        _fail("INVALID_INPUT", "Unsupported weighting or uncertainty")
    if dependence_assumption not in ("unspecified", "independent_years", "independent_homogeneous_obligors"):
        _fail("INVALID_INPUT", "Explicit supported dependence assumption required")
    c = _numeric(confidence_level, "confidence_level")
    if c.ndim or not 0 < c < 1:
        _fail("INVALID_INPUT", "confidence_level must lie in (0,1)")
    if not isinstance(representativeness, Mapping) or representativeness.get("assessment") not in ("representative", "not_representative", "unknown"):
        _fail("INVALID_REPRESENTATIVENESS", "Explicit assessment required")
    for key in ("rationale", "reference_period"):
        if not isinstance(representativeness.get(key), str) or not representativeness[key].strip():
            _fail("INVALID_REPRESENTATIVENESS", f"{key} required")
    if not isinstance(cohorts, (list, tuple)) or not cohorts:
        _fail("EMPTY_INPUT", "Nonempty cohort records required")
    ids = selected_cohort_ids
    if not isinstance(ids, (list, tuple)) or not ids or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        _fail("INVALID_COHORT_SELECTION", "Select unique cohort identifiers")
    registry = {}
    for row in cohorts:
        if not isinstance(row, Mapping) or not isinstance(row.get("cohort_id"), str) or not row["cohort_id"] or row["cohort_id"] in registry:
            _fail("INVALID_COHORT_SELECTION", "Cohort identifiers must be unique")
        registry[row["cohort_id"]] = row
    if any(x not in registry for x in ids):
        _fail("INVALID_COHORT_SELECTION", "Unknown selected cohort")
    intervals, counts, totals = [], [], []
    for identifier in ids:
        row = registry[identifier]
        start, end, known = (_day(row.get(k)) for k in ("period_start", "period_end", "known_at"))
        try:
            anniversary = start.replace(year=start.year+1)
        except ValueError:
            if start.month != 2 or start.day != 29:
                _fail("INVALID_DATE", "Anniversary outside supported calendar")
            anniversary = start.replace(year=start.year+1, day=28)
        if end != anniversary:
            _fail("INCOMPATIBLE_HORIZON", "Each cohort must cover one calendar year")
        if row.get("mature") is not True or end > cutoff or known > cutoff or known < end:
            _fail("IMMATURE_COHORT", "Selected cohorts must be mature and available at knowledge date")
        if row.get("default_definition") != target["default_definition"]:
            _fail("TARGET_MISMATCH", "Cohort default definition differs from target")
        d, n = (_numeric(row.get(k), k) for k in ("defaults", "denominator"))
        if d.ndim or n.ndim or d < 0 or n <= 0 or d > n or d != math.floor(d) or n != math.floor(n) or n > 2**53-1:
            _fail("INVALID_COUNTS", "Integer counts require 0 <= defaults <= positive denominator")
        intervals.append((start, end)); counts.append(int(d)); totals.append(int(n))
    rates = np.asarray(counts, float)/totals
    pooled_n, pooled_d = sum(totals), sum(counts)
    if pooled_n > 2**53-1:
        _fail("INVALID_COUNTS", "Total denominator exceeds exact integer range")
    estimate = float(np.mean(rates)) if weighting == "annual_mean" else pooled_d/pooled_n
    ordered = sorted(intervals)
    overlap = any(b[0] < a[1] for a,b in zip(ordered, ordered[1:]))
    lower = upper = None
    if uncertainty == "year_t":
        if weighting != "annual_mean" or dependence_assumption != "independent_years" or overlap or len(rates) < 2:
            _fail("INCOMPATIBLE_UNCERTAINTY", "Year-t needs annual mean, independent nonoverlapping years and at least two cohorts")
        delta = float(t.ppf((1+float(c))/2, len(rates)-1)*np.std(rates, ddof=1)/math.sqrt(len(rates)))
        lower, upper = max(0.,estimate-delta), min(1.,estimate+delta)
    if uncertainty == "clopper_pearson":
        if weighting != "pooled_obligor" or dependence_assumption != "independent_homogeneous_obligors" or overlap:
            _fail("INCOMPATIBLE_UNCERTAINTY", "Exact interval needs pooled independent homogeneous obligors and nonoverlapping periods")
        alpha = (1-float(c))/2
        lower = 0. if pooled_d == 0 else float(beta.ppf(alpha, pooled_d, pooled_n-pooled_d+1))
        upper = 1. if pooled_d == pooled_n else float(beta.ppf(1-alpha, pooled_d+1, pooled_n-pooled_d))
    return _result({"pd":estimate, "lower":lower, "upper":upper}, "calibration.estimate_long_run_pd", "probability", ctx,
        diagnostics={"selected_cohort_ids":list(ids), "cohort_rates":rates.tolist(), "defaults":pooled_d,"denominator":pooled_n,
                     "weighting":weighting,"uncertainty":uncertainty,"confidence_level":float(c),"dependence_assumption":dependence_assumption,
                     "overlapping_periods":overlap,"knowledge_date":knowledge_date,"target":dict(target),
                     "representativeness":dict(representativeness),"representativeness_verified":False,"automatically_ttc":False},
        limitations=("Representativeness and independence are user assertions, not verified from aggregate counts.",
                     "Year-t intervals are approximate and intersected with [0,1]; pooled binomial intervals require homogeneous independent observations.",
                     "Observed long-run rates are not automatically TTC probabilities; selection, missing defaults, repeated obligors and censoring require upstream assessment."))
