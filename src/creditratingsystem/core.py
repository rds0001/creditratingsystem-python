"""Explicit results, domain checks, rating scales and local model interchange."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np


class CreditRiskError(ValueError):
    """A controlled failure with a stable, language-independent code."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        self.code = code
        self.details = deepcopy(details or {})
        super().__init__(f"{code}: {message}")


def _fail(code, message, details=None):
    raise CreditRiskError(code, message, details)


@dataclass(frozen=True)
class CalcResult:
    """Calculation value and its interpretation, without an implied approval."""

    value: Any
    unit: str
    method_id: str
    context: dict = field(default_factory=lambda: {"mode": "TIME_INDEPENDENT"})
    status: str = "OK"
    provenance: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Return finite JSON-compatible data; no executable object encoding."""
        return _json_data({name: getattr(self, name) for name in self.__dataclass_fields__})


@dataclass(frozen=True)
class FitResult(CalcResult):
    """A fitted model and its diagnostics, preserving the calculation contract."""

    @property
    def model(self):
        return self.value["model"]


def _json_data(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.ndarray):
        return _json_data(value.tolist())
    if isinstance(value, np.generic):
        return _json_data(value.item())
    if isinstance(value, numbers.Integral):
        if abs(value) > 2**53 - 1:
            _fail("INVALID_JSON", "Integers outside exact binary64 range must be strings")
        return int(value)
    if isinstance(value, numbers.Real):
        if not math.isfinite(value):
            _fail("INVALID_JSON", "Non-finite numbers cannot be serialized")
        return float(value)
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            _fail("INVALID_JSON", "Object keys must be strings")
        return {k: _json_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_data(v) for v in value]
    _fail("INVALID_JSON", "Unsupported value type", {"type": type(value).__name__})


def _context_value(context):
    if context is None:
        return {"mode": "TIME_INDEPENDENT"}
    if isinstance(context, CalcResult):
        if context.method_id != "core.build_calculation_context":
            _fail("INVALID_CONTEXT", "Expected a calculation context result")
        context = context.value
    if not isinstance(context, Mapping):
        _fail("INVALID_CONTEXT", "Context must be an object")
    return _json_data(context)


def _result(value, method_id, unit="dimensionless", context=None, diagnostics=None,
            limitations=(), provenance=None):
    ctx = _context_value(context)
    origin = {"data_origin": "USER_SUPPLIED", "parameter_origin": "USER_SUPPLIED",
              "validation_status": "NOT_VALIDATED", "method_version": "1.0"}
    manifest = ctx.get("data_manifest")
    if isinstance(manifest, dict):
        for key in ("data_origin", "parameter_origin"):
            if key in manifest:
                origin[key] = manifest[key]
    origin.update(provenance or {})
    return CalcResult(deepcopy(value), unit, method_id, ctx, "OK",
                      deepcopy(origin), deepcopy(diagnostics or {}), tuple(limitations))


def _fit_result(model, method_id, diagnostics=None, context=None):
    result = _result({"model": model}, method_id, "fitted_model", context,
                     diagnostics=diagnostics, provenance=model["provenance"])
    return FitResult(**result.__dict__)


def _numeric(value, name="value", allow_empty=False, vector_only=False):
    """Convert real numeric input without silently accepting bools or strings."""
    def inspect(item):
        if item is None:
            _fail("MISSING_INPUT", f"{name} contains missing input")
        if isinstance(item, (bool, np.bool_)):
            _fail("INVALID_INPUT", f"{name} must contain real numbers, not booleans")
        if isinstance(item, (list, tuple)):
            for child in item:
                inspect(child)
        elif isinstance(item, np.ndarray):
            if item.dtype.kind not in "iuf":
                _fail("INVALID_INPUT", f"{name} must be a real numeric array")
        elif not isinstance(item, numbers.Real):
            _fail("INVALID_INPUT", f"{name} must contain real numbers")
    inspect(value)
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        _fail("INVALID_INPUT", f"{name} is not a rectangular numeric input", {"reason": str(exc)})
    if vector_only and arr.ndim != 1:
        _fail("INVALID_INPUT", f"{name} must be a one-dimensional vector")
    if not allow_empty and arr.size == 0:
        _fail("EMPTY_INPUT", f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        _fail("NON_FINITE_INPUT", f"{name} contains NaN or infinity")
    return arr


def _probability(value, boundary_policy="closed", name="value", vector_only=False):
    if boundary_policy not in ("closed", "open"):
        _fail("INVALID_INPUT", "boundary_policy must be closed or open")
    arr = _numeric(value, name, allow_empty=True, vector_only=vector_only)
    valid = (arr >= 0) & (arr <= 1) if boundary_policy == "closed" else (arr > 0) & (arr < 1)
    if not np.all(valid):
        _fail("INVALID_PROBABILITY", f"{name} violates {boundary_policy} unit interval",
              {"locations": np.argwhere(~valid).tolist()})
    return arr


def validate_probability(value, boundary_policy="closed"):
    """Validate finite probabilities, preserving dimensions and never clipping."""
    arr = _probability(value, boundary_policy)
    return _result(float(arr) if arr.ndim == 0 else arr, "core.validate_probability",
                   "probability", diagnostics={"boundary_policy": boundary_policy, "violations": []})


def _scale_index(pd, scale):
    grades = scale["grades"]
    if pd == 1:
        return len(grades) - 1
    if pd == 0:
        return 0
    for i, row in enumerate(grades):
        if scale["boundary_convention"] == "left_closed":
            match = row["lower"] <= pd < row["upper"]
        else:
            match = row["lower"] < pd <= row["upper"]
        if match:
            return i
    _fail("INVALID_RATING_SCALE", "Scale does not cover input PD")


def _rating_scale(rating_scale, boundary_convention=None):
    if isinstance(rating_scale, CalcResult):
        if rating_scale.method_id != "core.validate_rating_scale":
            _fail("INVALID_RATING_SCALE", "Expected a validated scale")
        rating_scale = rating_scale.value
    if not isinstance(rating_scale, Mapping):
        _fail("INVALID_RATING_SCALE", "Scale must be a versioned object")
    scale = _json_data(rating_scale)
    required = ("schema_version", "scale_id", "version", "boundary_convention", "grades")
    if any(k not in scale for k in required):
        _fail("INVALID_RATING_SCALE", "Scale is missing required fields")
    if scale["schema_version"] != "1.0":
        _fail("INVALID_RATING_SCALE", "Unsupported scale schema")
    for key in ("scale_id", "version"):
        if not isinstance(scale[key], str) or not scale[key].strip():
            _fail("INVALID_RATING_SCALE", f"{key} must be a nonempty string")
    policy = scale["boundary_convention"]
    if policy not in ("left_closed", "right_closed") or (boundary_convention is not None and policy != boundary_convention):
        _fail("INVALID_RATING_SCALE", "Incompatible boundary convention")
    grades = scale["grades"]
    if not isinstance(grades, list) or not grades:
        _fail("INVALID_RATING_SCALE", "grades must be a nonempty ordered array")
    seen, previous_upper, previous_pd, defaults = set(), 0.0, -1.0, 0
    for i, row in enumerate(grades):
        if not isinstance(row, dict) or any(k not in row for k in ("grade", "lower", "upper", "representative_pd", "is_default")):
            _fail("INVALID_RATING_SCALE", "Incomplete grade record")
        grade = row["grade"]
        if not isinstance(grade, str) or not grade.strip() or grade in seen:
            _fail("INVALID_RATING_SCALE", "Grade IDs must be distinct nonempty strings")
        seen.add(grade)
        if not isinstance(row["is_default"], bool):
            _fail("INVALID_RATING_SCALE", "is_default must be boolean")
        values = [row[k] for k in ("lower", "upper", "representative_pd")]
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            _fail("INVALID_RATING_SCALE", "Grade boundaries and representatives must be probabilities")
        lower, upper, rep = values
        singleton = i == len(grades) - 1 and row["is_default"] and lower == upper == 1
        if lower != previous_upper or not (lower < upper or singleton) or rep <= previous_pd:
            _fail("INVALID_RATING_SCALE", "Intervals must be ordered, gap-free, with increasing representatives")
        if row["is_default"]:
            defaults += 1
            if i != len(grades) - 1 or defaults > 1:
                _fail("INVALID_RATING_SCALE", "A default label may occur only at the end")
        previous_upper, previous_pd = upper, rep
    if previous_upper != 1:
        _fail("INVALID_RATING_SCALE", "Scale must cover [0,1]")
    for i, row in enumerate(grades):
        if _scale_index(row["representative_pd"], scale) != i:
            _fail("INVALID_RATING_SCALE", "Representative PD is outside its mapped grade")
    return scale


def validate_rating_scale(rating_scale, boundary_convention=None):
    """Check a user-defined scale; grades are ordered from lower to higher risk."""
    scale = _rating_scale(rating_scale, boundary_convention)
    return _result(scale, "core.validate_rating_scale", "rating_scale",
                   diagnostics={"overlaps": [], "gaps": [], "direction": "increasing_risk"})


def map_pd_to_grade(pd, rating_scale, *, context=None):
    """Map PD to one interval before display rounding; never infer an event."""
    probabilities = _probability(pd)
    if probabilities.ndim > 1:
        _fail("INVALID_INPUT", "PD mapping accepts a scalar or vector")
    scale = _rating_scale(rating_scale)
    selected = [scale["grades"][_scale_index(float(p), scale)] for p in probabilities.reshape(-1)]
    labels = [row["grade"] for row in selected]
    value = labels[0] if len(labels) == 1 else labels
    return _result(value, "core.map_pd_to_grade", "grade", context,
                   diagnostics={"scale_id": scale["scale_id"], "scale_version": scale["version"],
                                "binding_intervals": selected},
                   limitations=("A mapped default grade is not an observed default event.",))


def _canonical(value):
    value = _json_data(value)
    def encode(x):
        if x is None:
            return b"n"
        if isinstance(x, bool):
            return b"t" if x else b"f"
        if isinstance(x, str):
            payload = x.encode("utf-8")
            return b"s" + str(len(payload)).encode("ascii") + b":" + payload
        if isinstance(x, (int, float)):
            number = float(x)
            number = 0.0 if number == 0 else number
            return b"d" + struct.pack(">d", number).hex().encode("ascii")
        if isinstance(x, list):
            return b"a" + str(len(x)).encode("ascii") + b":" + b"".join(encode(v) for v in x)
        if isinstance(x, dict):
            if any(not k or any(not 32 <= ord(c) <= 126 for c in k) for k in x):
                _fail("INVALID_JSON", "Canonical object keys must be nonempty printable ASCII")
            return b"o" + str(len(x)).encode("ascii") + b":" + b"".join(encode(k) + encode(x[k]) for k in sorted(x))
        _fail("INVALID_JSON", "Unsupported canonical value")
    return encode(value)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _iso_date(value, name):
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        _fail("INVALID_DATE", f"{name} must be an ISO calendar date")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        _fail("INVALID_DATE", f"{name} is not a valid calendar date")


def build_calculation_context(as_of_date=None, knowledge_date=None, *, horizon=None,
                              time_unit=None, currency=None, probability_measure=None,
                              conditioning=None, model_version=None, data_manifest=None,
                              scenario_id=None, policy_id=None):
    """Freeze explicit information dates and identities without reading a clock."""
    if (as_of_date is None) != (knowledge_date is None):
        _fail("MISSING_INPUT", "Both observation and knowledge dates are required")
    if (horizon is None) != (time_unit is None):
        _fail("MISSING_INPUT", "horizon and time_unit must be supplied together")
    if horizon is not None:
        numeric = _numeric(horizon, "horizon")
        if numeric.ndim != 0 or numeric <= 0 or time_unit not in ("days", "months", "years"):
            _fail("INCOMPATIBLE_HORIZON", "Positive scalar horizon and declared time unit required")
        horizon = float(numeric)
    if probability_measure not in (None, "PHYSICAL", "RISK_NEUTRAL"):
        _fail("INVALID_CONTEXT", "Unknown probability measure")
    optional = {"currency": currency, "conditioning": conditioning, "model_version": model_version,
                "scenario_id": scenario_id, "policy_id": policy_id}
    for key, value in optional.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            _fail("INVALID_CONTEXT", f"{key} must be a nonempty string")
    if data_manifest is not None and not isinstance(data_manifest, Mapping):
        _fail("INVALID_CONTEXT", "data_manifest must be an object")
    ctx = {"schema_version": "1.0", "mode": "TIME_INDEPENDENT" if as_of_date is None else "DATED",
           "as_of": _iso_date(as_of_date, "as_of_date") if as_of_date is not None else None,
           "known_at": _iso_date(knowledge_date, "knowledge_date") if knowledge_date is not None else None,
           "horizon": horizon, "time_unit": time_unit, "probability_measure": probability_measure,
           "data_manifest": _json_data(data_manifest), **optional}
    ctx["context_hash"] = _fingerprint(ctx)
    return _result(ctx, "core.build_calculation_context", "context")


def _model_artifact(model):
    if isinstance(model, FitResult):
        model = model.model
    elif isinstance(model, CalcResult):
        model = model.value
    if not isinstance(model, Mapping):
        _fail("INVALID_MODEL", "Model must be an object")
    model = _json_data(model)
    required = ("schema_version", "model_id", "model_version", "method_id", "feature_order", "parameters",
                "preprocessing", "target", "training", "diagnostics", "provenance")
    if any(k not in model for k in required):
        _fail("INVALID_MODEL", "Model is missing mandatory metadata")
    if model["schema_version"] != "1.0":
        _fail("UNSUPPORTED_SCHEMA_VERSION", "Unsupported model schema")
    for key in ("model_id", "model_version", "method_id"):
        if not isinstance(model[key], str) or not model[key].strip():
            _fail("INVALID_MODEL", f"{key} must be a nonempty string")
    features = model["feature_order"]
    if not isinstance(features, list) or any(not isinstance(v, str) or not v.strip() for v in features) or len(set(features)) != len(features):
        _fail("INVALID_MODEL", "feature_order must list distinct feature names")
    for key in ("parameters", "preprocessing", "target", "training", "diagnostics", "provenance"):
        if not isinstance(model[key], dict):
            _fail("INVALID_MODEL", f"{key} must be an object")
    target = model["target"]
    if any(k not in target for k in ("event", "horizon", "time_unit", "probability_measure", "conditioning")):
        _fail("INVALID_MODEL", "Incomplete target definition")
    for key in ("event", "conditioning"):
        if not isinstance(target[key], str) or not target[key].strip():
            _fail("INVALID_MODEL", f"target.{key} must be nonempty")
    if target["time_unit"] not in ("days", "months", "years") or target["probability_measure"] not in ("PHYSICAL", "RISK_NEUTRAL"):
        _fail("INVALID_MODEL", "Invalid target horizon or measure")
    h = target["horizon"]
    if isinstance(h, bool) or not isinstance(h, (float, int)) or not math.isfinite(h) or h <= 0:
        _fail("INVALID_MODEL", "target.horizon must be positive")
    for key in ("data_fingerprint", "split_id", "known_at"):
        if not isinstance(model["training"].get(key), str) or not model["training"][key].strip():
            _fail("INVALID_MODEL", f"training.{key} is required")
    _iso_date(model["training"]["known_at"], "training.known_at")
    if model["provenance"].get("data_origin") not in ("SYNTHETIC", "USER_SUPPLIED", "REFERENCE"):
        _fail("INVALID_MODEL", "Model requires a declared data_origin")
    if model["provenance"].get("parameter_origin") not in ("ESTIMATED", "METHOD_CONSTANT", "USER_SUPPLIED", "SYNTHETIC_REFERENCE"):
        _fail("INVALID_MODEL", "Model requires a declared parameter_origin")
    _canonical(model)
    return model


def _local_path(value):
    if not isinstance(value, (str, os.PathLike)):
        _fail("INVALID_PATH", "Expected an explicit local path")
    path = os.fspath(value)
    if not isinstance(path, str) or not path or "://" in path or path.startswith(("//", "\\\\")) or "\x00" in path:
        _fail("INVALID_PATH", "Remote or empty paths are not accepted")
    return Path(path)


def serialize_fitted_model(fitted_model, destination, schema_version="1.0"):
    """Write a new local, integrity-checked JSON artifact without overwriting."""
    if schema_version != "1.0":
        _fail("UNSUPPORTED_SCHEMA_VERSION", "Unsupported envelope schema")
    model = _model_artifact(fitted_model)
    path = _local_path(destination)
    digest = _fingerprint(model)
    envelope = {"format": "creditratingsystem-model", "schema_version": "1.0",
                "hash_format": "cr-canonical-v1", "model": model, "sha256": digest}
    text = json.dumps(envelope, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except FileExistsError:
        _fail("FILE_EXISTS", "Destination already exists")
    except OSError as exc:
        _fail("IO_ERROR", "Cannot write model", {"reason": str(exc)})
    return _result({"path": str(path), "sha256": digest, "schema_version": schema_version},
                   "core.serialize_fitted_model", "artifact_manifest")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail("INVALID_JSON", "Duplicate object key", {"key": key})
        value[key] = item
    return value


def load_fitted_model(manifest_path, allowed_schema_versions=("1.0",)):
    """Load finite declarative data, checking content hash and supported schema."""
    path = _local_path(manifest_path)
    if not isinstance(allowed_schema_versions, (tuple, list)) or any(not isinstance(x, str) for x in allowed_schema_versions):
        _fail("INVALID_INPUT", "allowed_schema_versions must be a list of strings")
    try:
        text = path.read_text(encoding="utf-8")
        envelope = json.loads(text, object_pairs_hook=_unique_object,
                              parse_constant=lambda x: _fail("INVALID_JSON", "Non-finite JSON number"))
    except (OSError, UnicodeError) as exc:
        _fail("IO_ERROR", "Cannot read model", {"reason": str(exc)})
    except json.JSONDecodeError:
        _fail("INVALID_JSON", "Malformed JSON")
    if not isinstance(envelope, dict) or envelope.get("format") != "creditratingsystem-model" or envelope.get("hash_format") != "cr-canonical-v1":
        _fail("INVALID_MODEL", "Unknown model envelope")
    if envelope.get("schema_version") != "1.0" or envelope["schema_version"] not in allowed_schema_versions:
        _fail("UNSUPPORTED_SCHEMA_VERSION", "Unsupported model envelope schema")
    model = _model_artifact(envelope.get("model"))
    digest = _fingerprint(model)
    if envelope.get("sha256") != digest:
        _fail("INTEGRITY_ERROR", "Model content does not match its recorded hash")
    return _result(model, "core.load_fitted_model", "model_artifact",
                   diagnostics={"sha256": digest, "integrity": "MATCH"},
                   provenance=model["provenance"],
                   limitations=("A matching content hash is not authenticated authorship or model approval.",))


def content_fingerprint(value, *, context=None):
    """Return the canonical content digest used to bind local data-only recipes.

    Object key order is immaterial; array order and scalar types are retained.
    This is a content identifier, not a signature or a schema validation result.
    """
    return _result(_fingerprint(value), 'core.content_fingerprint', 'sha256', context,
                   diagnostics={'encoding': 'creditratingsystem-canonical-v1'},
                   limitations=('Identifies canonical content only; it does not authenticate its author or validate financial contracts. Remove a digest field before hashing the payload it is intended to bind.',))
