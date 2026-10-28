# Core contracts

Every public calculation returns `CalcResult`: `value`, `unit`, `method_id`,
`context`, `status`, `provenance`, `diagnostics` and `limitations`. An `OK` calculation
is not approval of a model. Numeric input errors raise `CreditRiskError` with a
stable `code` and optional `details`. Probabilities are decimal fractions; there
is no clipping, string coercion or conversion of missing values into zero.

`validate_probability(value, boundary_policy="closed")` accepts finite real scalars
and rectangular arrays. `open` excludes 0 and 1. Shape is retained, including an
empty array. Booleans are not probabilities.

## Explicit information context

`build_calculation_context(as_of_date=None, knowledge_date=None, *, horizon=None,
time_unit=None, currency=None, probability_measure=None, conditioning=None,
model_version=None, data_manifest=None, scenario_id=None, policy_id=None)` returns
a context in `.value`. Dates are ISO calendar dates; both dates must be supplied
together. Neither is taken from the system clock. A historical replay can have a
knowledge date after the observation date. A forecast can use earlier information.
`horizon` must be positive and paired with `days`, `months` or `years`.

Absent dates explicitly mean `TIME_INDEPENDENT`. A context or its result can be
passed to calculations. `data_manifest` can declare `data_origin="SYNTHETIC"` and
`parameter_origin="SYNTHETIC_REFERENCE"`; this provenance is retained. It does not
change `validation_status="NOT_VALIDATED"`. Hashes use the supplied information,
not current environment variables or external lookups.

## User-defined rating scales

`validate_rating_scale(rating_scale, boundary_convention=None)` validates an object
with `schema_version="1.0"`, `scale_id`, `version`, `boundary_convention` and `grades`.
Each grade record has `grade`, `lower`, `upper`, `representative_pd`, `is_default`.
Grades are unique and ordered from low to high risk; intervals cover [0,1] exactly.
Representatives strictly increase and belong to their mapped intervals.

`left_closed` uses [lower,upper); the final interval includes 1. `right_closed`
uses (lower,upper]; the first interval includes 0. An optional final default
singleton [1,1] takes priority for PD=1 under either convention. A default flag
may only occur in the final grade. No proprietary or universal rating scale is
built in. `map_pd_to_grade(pd, rating_scale, context=None)` returns grade IDs and
binding intervals. A default grade is a scale label, not evidence of an event.

## Local model interchange

`serialize_fitted_model(fitted_model, destination, schema_version="1.0")` writes a
new local JSON file and refuses overwrite. `load_fitted_model(manifest_path,
allowed_schema_versions=("1.0",))` checks version, structure and SHA-256 integrity.
The artifact contains no executable deserialization or remote resource retrieval.
The content hash detects changes; it does not authenticate the model's author.

Model objects require `schema_version`, `model_id`, `model_version`, `method_id`,
`feature_order`, `parameters`, `preprocessing`, `target`, `training`, `diagnostics`
and `provenance`. Feature names are ordered and unique. Target includes event,
positive horizon, time unit, probability measure and conditioning. Training includes
data fingerprint, split ID and knowledge date. Provenance declares data and parameter
origins. Each actual estimator validates its parameter layout when predicting.
All arrays are inline in version 1.0; there is no external array-file dependency.

The wire envelope records `format="creditratingsystem-model"`, `schema_version`,
`hash_format="cr-canonical-v1"`, `model` and `sha256`. Nonfinite numbers, duplicate
JSON keys and unsupported types are rejected. Integers outside ±(2^53−1) must be
encoded as strings when integer identity matters.

Canonical hashing encodes null as `n`, boolean as `t`/`f`, strings as
`s<byte-length>:<UTF8 bytes>`, real numbers as `d<big-endian binary64 hex>`, arrays
as `a<count>:<encoded elements>`, and objects as `o<count>:<encoded key/value pairs>`.
Object keys are nonempty printable ASCII, sorted by bytes. Negative zero is
normalized. Strings retain exact Unicode text, without normalization. Objects and
arrays remain distinct. R uses named lists for objects and unnamed lists for
arrays, including length-one arrays; an empty object is a named empty list.

## Example

```python
from creditratingsystem import build_calculation_context, expected_loss

context = build_calculation_context(
    currency="EUR",
    data_manifest={"data_origin": "SYNTHETIC",
                   "parameter_origin": "SYNTHETIC_REFERENCE"},
)
result = expected_loss(0.02, 0.4, 1000, context=context)
assert result.value == 8
assert result.unit == "EUR"
```

The arithmetic uses fixed or suitably conditioned loss inputs. Arbitrary products
of mean loss severity and mean exposure do not capture their dependence.
