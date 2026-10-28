"""Selection mechanisms, assumption-labelled inference and honest evaluation scopes."""

from collections.abc import Mapping
from copy import deepcopy
import numpy as np
from scipy.special import expit, logit, ndtr
from scipy.optimize import minimize
from .core import (
    _fail,
    _numeric,
    _probability,
    _result,
    _fit_result,
    _model_artifact,
    _fingerprint,
    _iso_date,
)
from .classification import _design, fit_binary_glm, predict_pd
from .calibration import _target


def _out(method, value, context=None, diagnostics=None, synthetic=False):
    return _result(
        value,
        "selection_bias." + method,
        context=context,
        diagnostics=diagnostics,
        provenance={"data_origin": "SYNTHETIC"} if synthetic else None,
        limitations=(
            "Unobserved own-contract outcomes are not identified without the stated assumptions. Hypothetical labels are not validation evidence.",
        ),
    )


def _vec(value, n, name, probability=False):
    x = (
        _probability(value, "closed", vector_only=True)
        if probability
        else _numeric(value, name, vector_only=True)
    )
    if len(x) != n:
        _fail("INCOMPATIBLE_SHAPE", name + " must have one value per row")
    return x


def _sample(outcomes, observed, target_weights=None):
    s = _numeric(observed, "observed", vector_only=True)
    if len(s) == 0 or np.any(~np.isin(s, [0, 1])):
        _fail("INVALID_SELECTION", "observed must be nonempty numeric zero/one")
    n = len(s)
    if not isinstance(outcomes, (list, tuple, np.ndarray)):
        _fail(
            "INVALID_OUTCOME", "Outcomes must be an aligned vector with missing unobserved values"
        )
    y = np.zeros(n)
    if len(outcomes) != n:
        _fail("INCOMPATIBLE_SHAPE", "outcomes and observed must match")
    for i, v in enumerate(outcomes):
        if s[i] == 0:
            if v is not None and not (isinstance(v, (float, np.floating)) and np.isnan(v)):
                _fail("UNOBSERVED_OUTCOME", "Unobserved own-contract outcomes must be missing")
        else:
            z = _numeric(v, "outcome")
            if z.ndim or z not in (0, 1):
                _fail("INVALID_OUTCOME", "Observed outcomes must be binary")
            y[i] = z
    w = np.ones(n) if target_weights is None else _vec(target_weights, n, "target_weights")
    if np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
        _fail("INVALID_WEIGHTS", "Positive finite target mass required")
    return y, s, w


def _str(value, name):
    if not isinstance(value, str) or not value.strip():
        _fail("INVALID_INPUT", name + " must be nonempty text")
    return value


def _rows(rows, name):
    if not isinstance(rows, (list, tuple)) or any(not isinstance(r, Mapping) for r in rows):
        _fail("INVALID_INPUT", name + " must be records")
    ids = [_str(r.get("application_id"), "application_id") for r in rows]
    if len(set(ids)) != len(ids):
        _fail("DUPLICATE_ID", name + " has duplicate application IDs")
    return {i: dict(r) for i, r in zip(ids, rows)}


def build_selection_dataset(
    applications, decisions, bookings, outcomes, *, as_of, target_population, target, context=None
):
    """Join dated application stages without converting missing potential outcomes to goods."""
    cutoff = _iso_date(as_of, "as_of")
    _str(target_population, "target_population")
    target = _target(target)
    a = _rows(applications, "applications")
    d = _rows(decisions, "decisions")
    b = _rows(bookings, "bookings")
    o = _rows(outcomes, "outcomes")
    if not a:
        _fail("EMPTY_INPUT", "Applications required")
    if (set(d) | set(b) | set(o)) - set(a):
        _fail("UNKNOWN_ID", "Stage records require a submitted application")
    panel = []
    for key, app in a.items():
        submitted = _iso_date(app.get("submitted_at"), "submitted_at")
        entity = _str(app.get("entity_id"), "entity_id")
        if submitted > cutoff:
            _fail("FUTURE_INPUT", "Application is later than audit cutoff")
        stage = {}
        last = submitted
        for name, table in [("decision", d), ("booking", b), ("outcome", o)]:
            row = table.get(key)
            if row is None:
                continue
            event = _iso_date(row.get("as_of"), name + ".as_of")
            known = _iso_date(row.get("known_at"), name + ".known_at")
            if known < event or event < last:
                _fail("INVALID_DATE", "Stage dates must be causal and ordered")
            if known > cutoff:
                continue
            stage[name] = row
            last = event
        dec = stage.get("decision")
        book = stage.get("booking")
        out = stage.get("outcome")
        if dec:
            if dec.get("decision") not in ("offered", "rejected", "withdrawn", "not_decisionable"):
                _fail("INVALID_SELECTION", "Unknown decision")
            _str(dec.get("policy_version"), "policy_version")
        offered = dec is not None and dec["decision"] == "offered"
        if book and (
            not offered
            or book.get("taken_up") not in (True, False)
            or book.get("funded") not in (True, False)
            or (book["funded"] and not book["taken_up"])
        ):
            _fail("INVALID_SELECTION", "Booking requires offer, and funding requires uptake")
        funded = bool(book and book["funded"])
        observed = False
        y = None
        external = None
        if out:
            origin = out.get("outcome_scope")
            if origin == "external_contract":
                external = deepcopy(out)
            elif origin == "own_contract":
                if not funded:
                    _fail("UNOBSERVED_OUTCOME", "Own-contract outcome requires funding")
                if out.get("target") != target:
                    _fail("TARGET_MISMATCH", "Outcome target differs")
                if out.get("status") not in ("observed", "immature", "censored"):
                    _fail("INVALID_OUTCOME", "Declare outcome observation status")
                observed = out["status"] == "observed"
                if observed:
                    y = float(_numeric(out.get("outcome"), "outcome"))
                    if y not in (0, 1):
                        _fail("INVALID_OUTCOME", "Binary observed outcome required")
                elif out.get("outcome") is not None:
                    _fail("UNOBSERVED_OUTCOME", "Unobserved target must remain missing")
            else:
                _fail("INVALID_OUTCOME", "Declare own_contract or external_contract")
        if observed:
            stratum = "funded_observed"
        elif funded:
            stratum = "funded_" + (
                out["status"] if out and out.get("outcome_scope") == "own_contract" else "immature"
            )
        elif book and book["taken_up"]:
            stratum = "taken_up_not_funded"
        elif offered:
            stratum = "offered_not_taken_up"
        elif dec:
            stratum = dec["decision"]
        else:
            stratum = "pending_decision"
        panel.append(
            dict(
                application_id=key,
                entity_id=entity,
                submitted_at=submitted,
                decisionable=bool(dec and dec["decision"] != "not_decisionable"),
                offered=offered,
                taken_up=bool(book and book["taken_up"]),
                funded=funded,
                observed=int(observed),
                outcome=y,
                stratum=stratum,
                policy_version=dec["policy_version"] if dec else None,
                external_outcome=external,
                outcome_scope="own_contract",
                validation_eligible=observed,
            )
        )
    return _out(
        "build_selection_dataset",
        dict(
            rows=panel,
            target=target,
            target_population=target_population,
            as_of=cutoff,
            counts={
                k: sum(r["stratum"] == k for r in panel)
                for k in sorted(set(r["stratum"] for r in panel))
            },
        ),
        context,
    )


def _folds(folds, row_metadata, n):
    if (
        not isinstance(row_metadata, (list, tuple))
        or not isinstance(folds, (list, tuple))
        or any(not isinstance(r, Mapping) for r in row_metadata)
        or any(not isinstance(f, Mapping) for f in folds)
    ):
        _fail("INVALID_SPLIT", "Metadata and folds must be record arrays")
    if len(row_metadata) != n or not folds:
        _fail("INVALID_SPLIT", "Dated row metadata and folds required")
    meta = []
    for r in row_metadata:
        entity = _str(r.get("entity_id"), "entity_id")
        date = _iso_date(r.get("as_of"), "as_of")
        known = _iso_date(r.get("label_known_at"), "label_known_at")
        features = _iso_date(r.get("features_known_at"), "features_known_at")
        if features > date or known < date:
            _fail("LEAKAGE", "Features must be available at decision; labels cannot predate it")
        meta.append((entity, date, known))
    seen = set()
    plans = []
    for fold in folds:
        tr = _numeric(fold.get("train"), "train", vector_only=True)
        te = _numeric(fold.get("validation"), "validation", vector_only=True)
        if any(
            len(v) == 0
            or np.any(v != np.floor(v))
            or np.any((v < 0) | (v >= n))
            or len(set(v)) != len(v)
            for v in (tr, te)
        ):
            _fail("INVALID_SPLIT", "Nonempty unique zero-based fold indices required")
        tr = tr.astype(int)
        te = te.astype(int)
        if set(tr) & set(te) or seen & set(te):
            _fail("LEAKAGE", "Training and evaluation rows must be disjoint; evaluation unique")
        if {meta[i][0] for i in tr} & {meta[i][0] for i in te}:
            _fail("LEAKAGE", "Entities overlap training/evaluation")
        if max(meta[i][2] for i in tr) >= min(meta[i][1] for i in te):
            _fail("LEAKAGE", "All training outcomes must precede evaluation dates")
        seen.update(te)
        plans.append((tr, te))
    return plans


def fit_selection_propensity(
    X,
    selected,
    *,
    feature_order,
    mechanism,
    folds,
    row_metadata,
    target,
    training,
    model_id,
    model_version,
    context=None,
):
    """Fit a logit selection mechanism and genuine grouped forward out-of-fold predictions."""
    if mechanism not in ("decision", "takeup", "funding", "outcome_observation"):
        _fail("INVALID_SELECTION", "Declare a separate selection mechanism")
    x, _ = _design(X, feature_order, True)
    s = _vec(selected, len(x), "selected")
    if np.any(~np.isin(s, [0, 1])):
        _fail("INVALID_SELECTION", "Binary mechanism outcomes required")
    if target.get("event") != mechanism:
        _fail("TARGET_MISMATCH", "Propensity target event must equal mechanism")
    plans = _folds(folds, row_metadata, len(x))
    cutoff = _iso_date(training.get("known_at"), "training.known_at")
    if any(r["label_known_at"] > cutoff for r in row_metadata):
        _fail("LEAKAGE", "Training cutoff precedes mechanism labels")
    common = dict(
        link="logit",
        feature_order=feature_order,
        fit_intercept=True,
        target=target,
        model_id=model_id,
        model_version=model_version,
        context=context,
    )
    full = fit_binary_glm(x, s, training=training, **common)
    predictions = [None] * len(x)
    lineage = []
    for tr, te in plans:
        ft = dict(
            training,
            known_at=max(row_metadata[i]["label_known_at"] for i in tr),
            data_fingerprint=_fingerprint({"X": x[tr].tolist(), "selected": s[tr].tolist()}),
        )
        fit = fit_binary_glm(x[tr], s[tr], training=ft, **common)
        p = predict_pd(fit, x[te], feature_order=feature_order, expected_target=target).value
        for i, v in zip(te, p):
            predictions[i] = float(v)
        lineage.append(
            dict(train=tr.tolist(), validation=te.tolist(), model_hash=_fingerprint(fit.model))
        )
    diag = dict(
        full.diagnostics,
        mechanism=mechanism,
        out_of_fold_propensity=predictions,
        folds=lineage,
        row_metadata=list(row_metadata),
        warmup_indices=[i for i, v in enumerate(predictions) if v is None],
        positivity_min=min(v for v in predictions if v is not None),
        selection_assumptions=[
            "Conditional exchangeability is not established by a propensity fit.",
            "The input rows must be the eligible risk set for this mechanism.",
        ],
    )
    model = deepcopy(full.model)
    model["diagnostics"] = diag
    return _fit_result(model, "selection_bias.fit_selection_propensity", diag, context)


def selection_ipw_weights(
    observed, propensities, *, target_weights=None, stabilization=None, trim=None, context=None
):
    """Multiply sequential conditional selection probabilities and form observed-case IPW."""
    s = _numeric(observed, "observed", vector_only=True)
    n = len(s)
    if not n or np.any(~np.isin(s, [0, 1])):
        _fail("INVALID_SELECTION", "Binary observed indicator required")
    p = _numeric(propensities, "propensities")
    if p.ndim == 1:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] != n or p.shape[1] == 0 or np.any((p < 0) | (p > 1)):
        _fail("INVALID_PROPENSITY", "Rows by sequential mechanisms probabilities required")
    e = np.prod(p, axis=1)
    w = np.ones(n) if target_weights is None else _vec(target_weights, n, "target_weights")
    if np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
        _fail("INVALID_WEIGHTS", "Positive target mass required")
    keep = np.ones(n, dtype=bool)
    if trim is not None:
        t = _numeric(trim, "trim", vector_only=True)
        if len(t) != 2 or not 0 < t[0] <= t[1] <= 1:
            _fail("INVALID_TRIM", "Declare inclusive propensity support [lower,upper], lower>0")
        keep = (e >= t[0]) & (e <= t[1])
    if np.any((e == 0) & (w > 0) & keep):
        _fail(
            "POSITIVITY_FAILURE",
            "Zero probability target rows are not identified; explicitly restrict target",
        )
    if np.any((s == 1) & (e == 0)):
        _fail("INVALID_PROPENSITY", "Observed event cannot have zero selection probability")
    factor = 1.0
    if stabilization is not None:
        f = _probability(stabilization, "closed", vector_only=True)
        if np.any(f <= 0):
            _fail("INVALID_PROPENSITY", "Stabilization rates must lie in (0,1]")
        if len(f) != p.shape[1]:
            _fail("INCOMPATIBLE_SHAPE", "One marginal stabilization rate per mechanism")
        factor = float(np.prod(f))
    ipw = np.zeros(n)
    eligible = (s == 1) & keep
    ipw[eligible] = w[eligible] * factor / e[eligible]
    if not np.any(ipw > 0):
        _fail("EMPTY_SUPPORT", "No observed target mass remains")
    return _out(
        "selection_ipw_weights",
        dict(
            weights=ipw.tolist(),
            joint_propensity=e.tolist(),
            included=keep.tolist(),
            effective_sample_size=float(ipw.sum() ** 2 / (ipw @ ipw)),
            excluded_target_mass=float(w[~keep].sum()),
            retained_target_mass=float(w[keep].sum()),
            estimand_changed=bool(np.any((~keep) & (w > 0))),
            stabilization_factor=factor,
            assumptions=[
                "Sequential conditional MAR at every mechanism",
                "Positivity in the retained target population",
            ],
            estimand="retained_target_population",
        ),
        context,
    )


def fuzzy_reject_augmentation(
    outcomes, observed, baseline_pd, *, application_ids, augmentation_weight=1, context=None
):
    """Fractional good/bad training pairs; pseudo labels cannot become test truth."""
    y, s, w = _sample(outcomes, observed)
    p = _vec(baseline_pd, len(s), "baseline_pd", True)
    if len(application_ids) != len(s) or len(set(application_ids)) != len(s):
        _fail("DUPLICATE_ID", "One unique application ID per row required")
    mass = _numeric(augmentation_weight, "augmentation_weight")
    if mass.ndim or mass < 0:
        _fail("INVALID_WEIGHTS", "Nonnegative scalar augmentation mass required")
    rows = []
    for i, key in enumerate(application_ids):
        _str(key, "application_id")
        labels = (
            [(float(y[i]), 1.0)]
            if s[i]
            else [(0.0, float(mass * (1 - p[i]))), (1.0, float(mass * p[i]))]
        )
        for label, weight in labels:
            rows.append(
                dict(
                    application_id=key,
                    outcome=label,
                    weight=weight,
                    label_origin="OBSERVED" if s[i] else "HYPOTHETICAL",
                    validation_eligible=bool(s[i]),
                    baseline_pd=float(p[i]),
                )
            )
    return _out(
        "fuzzy_reject_augmentation",
        dict(
            rows=rows,
            role="TRAINING_ONLY",
            assumption="Missing applicant PD equals the frozen baseline PD; not identified from observed labels.",
        ),
        context,
    )


def parcel_rejected_applications(
    bands,
    accepted_defaults,
    accepted_counts,
    rejected_counts,
    *,
    log_odds_shift=0,
    uniforms=None,
    context=None,
):
    """Frozen-band fractional or explicitly sampled reject scenarios in log-odds space."""
    n = len(bands)
    if n == 0 or len(set(bands)) != n:
        _fail("INVALID_BANDS", "Distinct frozen band names required")
    d = _vec(accepted_defaults, n, "accepted_defaults")
    a = _vec(accepted_counts, n, "accepted_counts")
    r = _vec(rejected_counts, n, "rejected_counts")
    if (
        np.any(a <= 0)
        or np.any((d < 0) | (d > a))
        or np.any(r < 0)
        or any(np.any(v != np.floor(v)) for v in (d, a, r))
    ):
        _fail(
            "INVALID_COUNTS",
            "Accepted counts positive, defaults bounded, reject counts nonnegative integers",
        )
    shift = _numeric(log_odds_shift, "log_odds_shift")
    shift = np.full(n, float(shift)) if shift.ndim == 0 else shift
    if shift.shape != (n,):
        _fail("INCOMPATIBLE_SHAPE", "Scalar or per-band log-odds shifts required")
    base = d / a
    with np.errstate(divide="ignore"):
        risk = expit(logit(base) + shift)
    labels = None
    if uniforms is not None:
        if len(uniforms) != n:
            _fail("INCOMPATIBLE_SHAPE", "One shared-uniform vector per band")
        labels = []
        for i, draw in enumerate(uniforms):
            u = _numeric(draw, "uniforms", allow_empty=True, vector_only=True)
            if len(u) != r[i] or np.any((u < 0) | (u >= 1)):
                _fail("INVALID_DRAWS", "Exactly reject-count shared uniforms in [0,1) required")
            labels.append((u < risk[i]).astype(int).tolist())
    return _out(
        "parcel_rejected_applications",
        dict(
            bands=list(bands),
            accepted_defaults=d.tolist(),
            accepted_counts=a.tolist(),
            rejected_counts=r.tolist(),
            baseline_bad_rate=base.tolist(),
            assumed_reject_bad_rate=risk.tolist(),
            hypothetical_default_mass=(r * risk).tolist(),
            hypothetical_labels=labels,
            log_odds_shift=shift.tolist(),
            label_origin="HYPOTHETICAL",
            validation_eligible=False,
            role="TRAINING_SCENARIO",
        ),
        context,
    )


def reject_inference_bounds(
    outcomes, observed, *, target_weights=None, strata=None, risk_bounds=None, context=None
):
    """Sharp weighted missing-outcome bounds with explicit optional stratum restrictions."""
    y, s, w = _sample(outcomes, observed, target_weights)
    strata = ["all"] * len(s) if strata is None else list(strata)
    if len(strata) != len(s) or any(not isinstance(k, str) or not k for k in strata):
        _fail("INVALID_STRATA", "One stratum name per row")
    names = sorted(set(strata))
    restrictions = {} if risk_bounds is None else risk_bounds
    if not isinstance(restrictions, Mapping) or set(restrictions) - set(names):
        _fail("INVALID_BOUNDS", "Bounds must name existing strata")
    parts = []
    lower = upper = 0.0
    for key in names:
        k = np.array([v == key for v in strata])
        known = float((w[k] * s[k] * y[k]).sum())
        missing = float((w[k] * (1 - s[k])).sum())
        bounds = _numeric(restrictions.get(key, [0, 1]), "risk_bounds", vector_only=True)
        if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] <= 1:
            _fail("INVALID_BOUNDS", "Require 0 <= lower <= upper <= 1")
        lo = known + missing * bounds[0]
        hi = known + missing * bounds[1]
        lower += lo
        upper += hi
        parts.append(
            dict(
                stratum=key,
                observed_default_mass=known,
                unobserved_mass=missing,
                assumed_missing_risk_bounds=bounds.tolist(),
                lower_mass=float(lo),
                upper_mass=float(hi),
            )
        )
    return _out(
        "reject_inference_bounds",
        dict(
            lower=float(lower / w.sum()),
            upper=float(upper / w.sum()),
            strata=parts,
            assumptions=["Binary own-contract potential outcome"]
            + (["User-specified missing-risk restrictions"] if restrictions else []),
            identification="PARTIAL" if lower != upper else "POINT",
            target_mass=float(w.sum()),
        ),
        context,
    )


def reject_inference_sensitivity(
    baseline_pd,
    observed,
    outcomes,
    *,
    log_odds_shifts,
    target_weights=None,
    threshold=None,
    context=None,
):
    """Pattern-mixture log-odds sensitivity, retaining observed outcomes exactly."""
    y, s, w = _sample(outcomes, observed, target_weights)
    p = _vec(baseline_pd, len(s), "baseline_pd", True)
    shifts = _numeric(log_odds_shifts, "log_odds_shifts", vector_only=True)
    if len(shifts) == 0 or len(set(shifts)) != len(shifts):
        _fail("INVALID_GRID", "Unique explicit shifts required")
    if threshold is not None:
        z = _probability(threshold, "closed")
        if z.ndim:
            _fail("INVALID_INPUT", "Scalar risk threshold required")
    surface = []
    with np.errstate(divide="ignore"):
        for delta in sorted(shifts):
            risk = expit(logit(p) + delta)
            contribution = s * y + (1 - s) * risk
            surface.append(
                dict(
                    log_odds_shift=float(delta),
                    population_pd=float(w @ contribution / w.sum()),
                    hypothetical_pd=risk.tolist(),
                )
            )
    crossings = []
    if threshold is not None:
        for left, right in zip(surface, surface[1:]):
            if left["population_pd"] <= threshold <= right["population_pd"]:
                crossings.append([left["log_odds_shift"], right["log_odds_shift"]])
    return _out(
        "reject_inference_sensitivity",
        dict(
            surface=surface,
            crossover_brackets=crossings,
            observed_default_mass=float(w @ (s * y)),
            unobserved_target_mass=float(w @ (1 - s)),
            identified_component="Observed own-contract outcomes only",
            assumed_component="User-supplied log-odds shift for missing potential outcomes",
            baseline_boundary_count=int(np.sum((p == 0) | (p == 1))),
            validation_eligible=False,
        ),
        context,
    )


def estimate_population_risk_aipw(
    outcomes,
    observed,
    outcome_predictions,
    propensities,
    *,
    fold_lineage,
    target_weights=None,
    clusters=None,
    context=None,
):
    """Cross-fitted MAR population mean with empirical cluster influence uncertainty."""
    y, s, w = _sample(outcomes, observed, target_weights)
    n = len(s)
    m = _vec(outcome_predictions, n, "outcome_predictions", True)
    e = _vec(propensities, n, "propensities", True)
    if not isinstance(fold_lineage, Mapping):
        _fail("INVALID_SPLIT", "Explicit cross-fit lineage required")
    plans = _folds(fold_lineage.get("folds", []), fold_lineage.get("rows", []), n)
    covered = {i for _, te in plans for i in te}
    if any(w[i] > 0 and i not in covered for i in range(n)):
        _fail(
            "LEAKAGE",
            "Every positive target-weight row needs out-of-fold nuisance predictions; set warmup target weights zero",
        )
    if fold_lineage.get("nuisance_prediction_origin") != "OUT_OF_FOLD":
        _fail("LEAKAGE", "Both nuisance predictions must be out-of-fold")
    if np.any((e <= 0) & (w > 0)):
        _fail("POSITIVITY_FAILURE", "Positive target-weight propensities must exceed zero")
    correction = np.zeros(n)
    selected = (s == 1) & (w > 0)
    correction[selected] = (y[selected] - m[selected]) / e[selected]
    pseudo = m + correction
    estimate = float(w @ pseudo / w.sum())
    influence = w * (pseudo - estimate) / w.sum()
    groups = [r["entity_id"] for r in fold_lineage["rows"]] if clusters is None else list(clusters)
    if len(groups) != n:
        _fail("INCOMPATIBLE_SHAPE", "One cluster per row required")
    entity_clusters = {}
    for row, group in zip(fold_lineage["rows"], groups):
        entity_clusters.setdefault(row["entity_id"], set()).add(str(group))
    if any(len(v) > 1 for v in entity_clusters.values()):
        _fail("INVALID_CLUSTERS", "A repeated entity cannot be split across uncertainty clusters")
    names = sorted(set(str(g) for g, wi in zip(groups, w) if wi > 0))
    totals = [float(sum(v for v, g in zip(influence, groups) if str(g) == key)) for key in names]
    if len(names) < 2:
        _fail("INSUFFICIENT_CLUSTERS", "At least two positive-mass independent clusters required")
    variance = float(len(names) / (len(names) - 1) * np.dot(totals, totals))
    se = float(np.sqrt(variance))
    return _out(
        "estimate_population_risk_aipw",
        dict(
            population_pd=estimate,
            augmented_contributions=pseudo.tolist(),
            influence=influence.tolist(),
            standard_error=se,
            confidence_interval=[
                estimate - 1.959963984540054 * se,
                estimate + 1.959963984540054 * se,
            ],
            cluster_count=len(names),
            uncertainty="CLUSTER_EMPIRICAL_INFLUENCE_NORMAL",
            nuisance_lineage_hash=_fingerprint(fold_lineage),
            outside_probability_domain=bool(estimate < 0 or estimate > 1),
            assumptions=[
                "Conditional MAR",
                "Positivity",
                "Either outcome or selection nuisance correctly specified",
                "Independent clusters and adequate nuisance convergence",
            ],
            estimand="Population mean, not individual reject PD",
        ),
        context,
    )


def prepare_reject_evaluation(
    candidates, evaluation_rows, *, as_of, target, outcome_comparability=None, context=None
):
    """Declare evaluation evidence scopes, excluding generated labels from truth."""
    cutoff = _iso_date(as_of, "as_of")
    target = _target(target)
    if not isinstance(candidates, (list, tuple)) or not candidates:
        _fail("INVALID_INPUT", "Frozen candidate records required")
    for c in candidates:
        _str(c.get("model_id"), "model_id")
        _str(c.get("model_hash"), "model_hash")
        if not isinstance(c.get("training_entities"), (list, tuple)) or any(
            not isinstance(v, str) or not v for v in c["training_entities"]
        ):
            _fail(
                "INVALID_SPLIT",
                "Candidate training_entities must explicitly identify all training groups",
            )
        if _iso_date(c.get("training_known_at"), "training_known_at") > cutoff:
            _fail("LEAKAGE", "Candidate trained after evaluation cutoff")
    rows = _rows(evaluation_rows, "evaluation_rows")
    scopes = []
    flags = []
    for key, row in rows.items():
        date = _iso_date(row.get("as_of"), "as_of")
        known = _iso_date(row.get("known_at"), "known_at")
        if known < date or known > cutoff:
            _fail("INVALID_DATE", "Evaluation label unavailable at cutoff")
        entity = _str(row.get("entity_id"), "entity_id")
        origin = row.get("label_origin")
        contaminated = any(
            entity in c.get("training_entities", []) or date <= c["training_known_at"]
            for c in candidates
        )
        scope = "NOT_IDENTIFIED"
        eligible = False
        if origin == "OBSERVED_OWN_CONTRACT":
            if (
                row.get("target") != target
                or row.get("funded") is not True
                or row.get("mature") is not True
            ):
                _fail("TARGET_MISMATCH", "Comparable mature funded own-contract outcomes required")
            scope = "FUNDED_OBSERVED_TEST"
            eligible = True
        elif origin == "AUTHORIZED_EXPLORATION":
            if (
                not row.get("authorization_reference")
                or row.get("target") != target
                or row.get("funded") is not True
                or row.get("mature") is not True
            ):
                _fail(
                    "INVALID_EVIDENCE",
                    "Exploration needs local authorization and comparable observed target",
                )
            scope = "OBSERVED_EXPLORATION"
            eligible = True
        elif origin == "SYNTHETIC_TRUTH":
            if row.get("data_origin") != "SYNTHETIC" or row.get("target") != target:
                _fail("INVALID_EVIDENCE", "Declare synthetic origin and exact target")
            scope = "SYNTHETIC_ENGINEERING_ONLY"
            eligible = True
        elif origin == "EXTERNAL_CONTRACT":
            if outcome_comparability is None:
                flags.append(dict(application_id=key, reason="EXTERNAL_TARGET_NOT_TRANSPORTED"))
            else:
                if (
                    not isinstance(outcome_comparability, Mapping)
                    or not outcome_comparability.get("transport_assumption")
                    or outcome_comparability.get("target") != target
                ):
                    _fail("INVALID_EVIDENCE", "Explicit transport assumption and target required")
                scope = "EXTERNAL_PROXY_ASSUMED_TRANSPORT"
                eligible = True
        elif origin in ("HYPOTHETICAL", "PSEUDO", "UNOBSERVED"):
            flags.append(dict(application_id=key, reason="NO_OBSERVED_TRUTH"))
        else:
            _fail("INVALID_EVIDENCE", "Unknown label origin")
        if eligible:
            y = _numeric(row.get("outcome"), "outcome")
            if y.ndim or y not in (0, 1):
                _fail("INVALID_OUTCOME", "Observed/hidden synthetic binary label required")
        if contaminated:
            flags.append(dict(application_id=key, reason="TRAIN_EVALUATION_CONTAMINATION"))
            eligible = False
        scopes.append(
            dict(
                application_id=key,
                entity_id=entity,
                evidence_scope=scope,
                validation_eligible=eligible,
                label_origin=origin,
            )
        )
    synthetic = bool(scopes) and all(
        r["evidence_scope"] == "SYNTHETIC_ENGINEERING_ONLY" for r in scopes
    )
    return _out(
        "prepare_reject_evaluation",
        dict(
            rows=scopes,
            contamination_flags=flags,
            target=target,
            as_of=cutoff,
            validation_identified=any(r["validation_eligible"] for r in scopes),
            candidate_hashes=[c["model_hash"] for c in candidates],
            role="EVALUATION_ONLY",
            split_plan={
                "role": "EVALUATION_ONLY",
                "validation_indices": [i for i, r in enumerate(scopes) if r["validation_eligible"]],
                "excluded_indices": [
                    i for i, r in enumerate(scopes) if not r["validation_eligible"]
                ],
                "entity_disjoint": True,
                "cutoff": cutoff,
            },
            no_lending_action=True,
        ),
        context,
        synthetic=synthetic,
    )


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


def _bvn(a, b, rho):
    """Plackett correlation integral, deterministic 64-point Gauss-Legendre rule."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    t = rho * (_GL_NODES + 1) / 2
    q = 1 - t * t
    integrand = np.exp(
        -(a[..., None] ** 2 - 2 * t * a[..., None] * b[..., None] + b[..., None] ** 2) / (2 * q)
    ) / (2 * np.pi * np.sqrt(q))
    value = ndtr(a) * ndtr(b) + rho / 2 * (integrand @ _GL_WEIGHTS)
    return np.maximum(0, np.minimum(np.minimum(ndtr(a), ndtr(b)), value))


def _joint_objective(theta, d, z, s, y, fixed_rho):
    k = d.shape[1]
    j = z.shape[1]
    rho = theta[-1] if fixed_rho is None else fixed_rho
    a = d @ theta[:k]
    b = z @ theta[k : k + j]
    chosen = s == 1
    sign = 2 * y[chosen] - 1
    u = sign * a[chosen]
    v = b[chosen]
    r = sign * rho
    prob = np.empty(len(s))
    prob[~chosen] = ndtr(-b[~chosen])
    prob[chosen] = np.array([_bvn(aa, bb, rr) for aa, bb, rr in zip(u, v, r)])
    if np.any(prob <= 1e-14):
        return 1e15 + float(np.sum(np.maximum(0, 1e-14 - prob))) * 1e15, np.zeros(len(theta))
    phi = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
    den = np.sqrt(1 - r * r)
    score_a = sign * phi(u) * ndtr((v - r * u) / den) / prob[chosen]
    score_b = np.zeros(len(s))
    score_b[chosen] = phi(v) * ndtr((u - r * v) / den) / prob[chosen]
    score_b[~chosen] = -phi(b[~chosen]) / prob[~chosen]
    grad = np.r_[d[chosen].T @ score_a, z.T @ score_b]
    if fixed_rho is None:
        density = np.exp(-(u * u - 2 * r * u * v + v * v) / (2 * (1 - r * r))) / (2 * np.pi * den)
        grad = np.r_[grad, np.sum(sign * density / prob[chosen])]
    return -float(np.log(prob).sum()), -grad


def fit_joint_selection_model(
    X_outcome,
    X_selection,
    selected,
    outcomes,
    *,
    outcome_features,
    selection_features,
    target,
    training,
    model_id,
    model_version,
    rho_grid=None,
    selected_rho=None,
    exclusion=None,
    context=None,
):
    """Binary bivariate-probit selection likelihood, fixed-rho profiles or excluded-instrument MLE."""
    x, d = _design(X_outcome, outcome_features, True)
    z0, z = _design(X_selection, selection_features, True)
    y, s, w = _sample(outcomes, selected)
    if len(x) != len(s) or len(z) != len(s):
        _fail("INCOMPATIBLE_SHAPE", "Both designs and selection must have equal rows")
    if len(set(s)) != 2 or len(set(y[s == 1])) != 2:
        _fail(
            "INSUFFICIENT_CLASS_SUPPORT",
            "Need selected/nonselected and both selected outcome classes",
        )
    for name in set(outcome_features) & set(selection_features):
        if not np.array_equal(
            x[:, list(outcome_features).index(name)], z0[:, list(selection_features).index(name)]
        ):
            _fail("FEATURE_MISMATCH", "Shared feature names must mean identical values")
    target = _target(target)
    if target["conditioning"] != "POTENTIAL_OWN_CONTRACT":
        _fail(
            "TARGET_MISMATCH",
            "Joint outcome is unconditional potential own-contract PD, not observed funded risk",
        )
    mode = "FIXED_RHO_SENSITIVITY"
    if rho_grid is None:
        if (
            not isinstance(exclusion, Mapping)
            or not exclusion.get("features")
            or not exclusion.get("justification")
        ):
            _fail(
                "IDENTIFICATION_REQUIRED",
                "Estimated rho requires named exclusion and substantive justification; otherwise supply explicit fixed rho grid",
            )
        names = exclusion["features"]
        if len(set(names)) != len(names) or not set(names) <= set(selection_features) - set(
            outcome_features
        ):
            _fail("INVALID_EXCLUSION", "Exclusions must occur only in selection design")
        if np.linalg.matrix_rank(
            np.column_stack((d, z0[:, [list(selection_features).index(v) for v in names]]))
        ) < d.shape[1] + len(names):
            _fail("INVALID_EXCLUSION", "Excluded variables are redundant with outcome design")
        grid = [None]
        mode = "ESTIMATED_RHO_ASSUMED_EXCLUSION"
        if selected_rho is not None:
            _fail("INVALID_GRID", "selected_rho is only valid for sensitivity grids")
    else:
        vals = _numeric(rho_grid, "rho_grid", vector_only=True)
        if len(vals) == 0 or len(set(vals)) != len(vals) or np.any(np.abs(vals) > 0.95):
            _fail("INVALID_GRID", "Unique rho grid in [-0.95,0.95] required")
        grid = sorted(float(v) for v in vals)
        if selected_rho is None or selected_rho not in grid:
            _fail(
                "INVALID_GRID",
                "Explicit selected_rho must be a supplied scenario, not data-selected as validation",
            )
    common = dict(
        link="probit",
        fit_intercept=True,
        target=target,
        training=training,
        model_id=model_id,
        model_version=model_version,
    )
    outcome_fit = fit_binary_glm(x[s == 1], y[s == 1], feature_order=outcome_features, **common)
    selection_fit = fit_binary_glm(z0, s, feature_order=selection_features, **common)
    start = np.r_[
        outcome_fit.model["parameters"]["coefficients"],
        selection_fit.model["parameters"]["coefficients"],
    ]
    profiles = []
    fits = []
    for rho in grid:
        initial = np.r_[start, 0.0] if rho is None else start.copy()
        bounds = [(-20.0, 20.0)] * len(start) + ([(-0.95, 0.95)] if rho is None else [])
        fit = minimize(
            _joint_objective,
            initial,
            args=(d, z, s, y, rho),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-7, "maxls": 50},
        )
        objective, gradient = _joint_objective(fit.x, d, z, s, y, rho)
        if not fit.success or objective >= 1e14 or np.max(np.abs(gradient)) > 2e-3:
            _fail(
                "NONCONVERGENCE",
                "Bivariate-probit likelihood did not converge",
                {"message": str(fit.message), "max_score": float(np.max(np.abs(gradient)))},
            )
        fitted_rho = float(fit.x[-1]) if rho is None else rho
        boundary = bool(
            np.any(np.abs(fit.x[: len(start)]) > 19.999)
            or (rho is None and abs(fitted_rho) > 0.94999)
        )
        if boundary:
            _fail(
                "WEAK_IDENTIFICATION",
                "Optimum at coefficient/rho boundary; do not treat as identified finite fit",
            )
        profiles.append(
            dict(
                rho=fitted_rho,
                log_likelihood=-objective,
                model_population_pd=float(np.mean(ndtr(d @ fit.x[: d.shape[1]]))),
                risk_scope="MODEL_ASSUMED_POTENTIAL_OWN_CONTRACT",
                outcome_coefficients=fit.x[: d.shape[1]].tolist(),
                selection_coefficients=fit.x[d.shape[1] : len(start)].tolist(),
                maximum_absolute_score=float(np.max(np.abs(gradient))),
                iterations=int(fit.nit),
            )
        )
        fits.append(fit)
    index = 0 if mode.startswith("ESTIMATED") else grid.index(selected_rho)
    chosen = profiles[index]
    theta = fits[index].x
    h = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        step = 1e-4 * max(1, abs(theta[j]))
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += step
        minus[j] -= step
        h[:, j] = (
            _joint_objective(plus, d, z, s, y, grid[index])[1]
            - _joint_objective(minus, d, z, s, y, grid[index])[1]
        ) / (2 * step)
    h = (h + h.T) / 2
    eigen = np.linalg.eigvalsh(h)
    condition = float(eigen.max() / eigen.min()) if eigen.min() > 0 else None
    covariance = np.linalg.inv(h).tolist() if eigen.min() > 1e-8 else None
    a = d @ np.asarray(chosen["outcome_coefficients"])
    b = z @ np.asarray(chosen["selection_coefficients"])
    joint = _bvn(a, b, chosen["rho"])
    ps = ndtr(b)
    features = list(outcome_features) + [f for f in selection_features if f not in outcome_features]
    diag = dict(
        mode=mode,
        profiles=profiles,
        selected_rho=chosen["rho"],
        exclusion=exclusion,
        hessian_condition_number=condition,
        weak_identification=bool(condition is None or condition > 1e8),
        covariance=covariance,
        covariance_type="OBSERVED_INFORMATION_IID_CONDITIONAL_ON_FIXED_RHO"
        if rho_grid is not None
        else "OBSERVED_INFORMATION_IID",
        quadrature="PLACKETT_GAUSS_LEGENDRE_64",
        fitted_population_pd=ndtr(a).tolist(),
        fitted_selection_probability=ps.tolist(),
        fitted_selected_pd=(joint / ps).tolist(),
        fitted_unselected_pd=((ndtr(a) - joint) / (1 - ps)).tolist(),
        assumptions=[
            "Joint standard-normal latent disturbances",
            "Correct outcome and selection indices",
            "Exclusion validity is user asserted, not established by likelihood",
        ]
        if exclusion
        else [
            "Joint standard-normal latent disturbances",
            "Correct indices",
            "User-fixed latent selection correlation",
        ],
        role="TRAINING_AND_ASSUMPTION_SENSITIVITY_ONLY",
    )
    train = dict(
        training,
        data_fingerprint=_fingerprint(
            {
                "X_outcome": x.tolist(),
                "X_selection": z0.tolist(),
                "selected": s.tolist(),
                "outcomes": [float(v) if si else None for v, si in zip(y, s)],
            }
        ),
    )
    model = _model_artifact(
        dict(
            schema_version="1.0",
            model_id=model_id,
            model_version=model_version,
            method_id="selection_bias.fit_joint_selection_model",
            feature_order=features,
            parameters=dict(
                outcome_features=list(outcome_features),
                selection_features=list(selection_features),
                outcome_coefficients=chosen["outcome_coefficients"],
                selection_coefficients=chosen["selection_coefficients"],
                rho=chosen["rho"],
            ),
            preprocessing={"transformation": "identity"},
            target=target,
            training=train,
            diagnostics=diag,
            provenance={
                "data_origin": training.get("data_origin", "USER_SUPPLIED"),
                "parameter_origin": "ESTIMATED",
            },
        )
    )
    return _fit_result(model, "selection_bias.fit_joint_selection_model", diag, context)


def _predict_joint_pd(model, X, *, feature_order, expected_target, context=None):
    model = _model_artifact(model)
    if list(feature_order) != model["feature_order"]:
        _fail("FEATURE_MISMATCH", "Feature order differs from frozen joint model")
    if expected_target != model["target"]:
        _fail("TARGET_MISMATCH", "Expected potential own-contract target must match")
    x, _ = _design(X, feature_order, True)
    p = model["parameters"]
    if (
        model["preprocessing"] != {"transformation": "identity"}
        or not isinstance(p.get("outcome_features"), list)
        or not set(p["outcome_features"]) <= set(feature_order)
    ):
        _fail("INVALID_MODEL", "Invalid joint model feature mapping")
    rho = _numeric(p.get("rho"), "rho")
    if (
        rho.ndim
        or abs(rho) > 0.95
        or len(set(p["outcome_features"])) != len(p["outcome_features"])
        or model["target"]["conditioning"] != "POTENTIAL_OWN_CONTRACT"
    ):
        _fail("INVALID_MODEL", "Invalid joint correlation, mapping or target")
    coef = _numeric(p.get("outcome_coefficients"), "outcome_coefficients", vector_only=True)
    if len(coef) != len(p["outcome_features"]) + 1:
        _fail("INVALID_MODEL", "Outcome coefficient count differs")
    xx = x[:, [list(feature_order).index(v) for v in p["outcome_features"]]]
    pd = ndtr(coef[0] + xx @ coef[1:])
    return _result(
        pd,
        "classification.predict_pd",
        "probability",
        context,
        provenance=model["provenance"],
        diagnostics={
            "prediction_family": "joint_binary_selection",
            "conditioning": "POTENTIAL_OWN_CONTRACT",
            "rho": p["rho"],
        },
        limitations=(
            "Potential own-contract risk depends on the frozen selection and normality assumptions; it is not observed reject validation.",
        ),
    )
