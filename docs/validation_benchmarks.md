# Frozen paired prediction and dated rating benchmarks

`benchmark_predictions(model_cohorts, *, reference_model_id, metrics,
matching_policy, multiple_comparisons, design, uncertainty,
cluster_resamples=None, confidence_level=.95, context=None)` compares frozen
models on the same mature cases. `model_cohorts` is a named mapping from model ID
to `{cohort, model_version, parameter_version, model_fingerprint,
training_known_at}`. Each cohort is the sealed output of
`build_monitoring_cohorts`; training must precede every forecast. The common
validation `design` declares model/version/parameter/data/population/split IDs,
HOLDOUT or EXTERNAL prediction origin, evaluation cutoff and physical performing
default target in days. Declared provenance is checked, not authenticated.

Matching is `strict` (identical case sets) or `intersection` (explicit unmatched
coverage). Entity, cluster, forecast and horizon dates, weights and known labels
must agree. A label enters comparisons only if every model's cohort marks the
same case eligible and mature. Negative labels require complete-through
coverage; an early default does not bypass maturity. Zero weights are excluded.
Metrics are `brier`, `log_loss`, `auc`, `gini`, `calibration_error` (observed minus
predicted), and `mean_pd`. Differences are candidate minus reference; lower
Brier/log loss and higher AUC/Gini are preferable. Calibration error and mean PD
have no universal monotone desirability. Boundary log loss is explicitly
infinite; one-class AUC and undefined differences are null with applicability.

`uncertainty='none'` or `'cluster_bootstrap'` is mandatory. Supplied draws each
sample the original number of distinct clusters with replacement and are shared
across all models. Percentile intervals are conditional on frozen models, without
refitting. Any undefined draw suppresses that metric's interval. Multiplicity is
`none` or `bonferroni`; the latter allocates the family error across every
requested candidate/metric comparison. Bootstrap coverage is approximate and
candidate selection is not corrected. There are no invented p-values, universal
acceptance thresholds or causal improvement claims.

`benchmark_ratings(internal_ratings, benchmark_records, *, scale_mapping,
matching_policy, design, uncertainty, cluster_resamples=None,
confidence_level=.95, context=None)` compares local, dated rating records.
Internal records carry the common validation fields `prediction_id`, `entity_id`,
`cluster_id`, `segment`, `as_of`, `forecast_known_at`, `horizon_end`, model/version/
parameter IDs; additionally `rating_unit_id`, `scale_id`, `scale_version`,
`rating_status`, `grade`, `rating_type`, `support_concept`, `currency`, and
`rating_philosophy`. External records carry `benchmark_id`, entity/unit/provider
IDs, the same four rating concepts, explicit target, scale IDs, `status`, `grade`,
`rating_at`, and `known_at`. Status is ACTIVE, WATCH, WITHDRAWN or MISSING.

The frozen mapping specifies `mapping_id`, `version`, `reference`, `known_at`,
best-to-worst `common_grades`, `internal` and `benchmark` scale objects with
`scale_id`, `scale_version`, `grades`, and complete `internal_to_common` and
`benchmark_to_common` maps. Both maps must preserve order; coarsening is allowed.
Mapping knowledge must precede forecasts. The matching policy specifies
`benchmark_origin` (EXTERNAL_JUDGEMENT, INTERNAL_CHALLENGER or
INTERNAL_DERIVED_MAPPING), `provider_id`, `method` (exact_date/latest_known),
`max_age_days`, `watch_policy` (include/exclude), and `kappa_weighting`
(unweighted/linear/quadratic).

Matching requires exact entity and rating unit plus matching target, currency,
issuer/issue type, support concept and rating philosophy. No issuer/issue,
standalone/supported, currency or PIT/TTC conversion is inferred. Only records
known by the forecast and effective by its date are considered. Latest effective
date then latest knowledge date resolves history; equal-date ambiguity errors.
The selected record is then tested for age, scale and status: a withdrawal,
excluded watch or changed scale never falls back to an older active rating.
Coverage and a per-record matching audit include unused benchmarks.

Outputs include confusion counts, exact/within-one-notch agreement, signed
(positive means internal worse) and absolute notch differences, weighted Cohen
kappa and Spearman correlation using tied midranks. Kappa is undefined when
chance agreement is one; correlation is undefined under zero rank variance.
Shared cluster bootstrap recomputes the complete weighted table and its margins.
An internally derived comparison is labelled INTERNAL_MAPPING_CHECK. Agreement
with any benchmark is not independent outcome calibration or ground truth.

Both functions return `CalcResult.value` containing a `ValidationResult` with
metrics, explicit uncertainty/applicability, design, coverage, diagnostics,
fingerprints and a NOT_ASSESSED decision. Rating agreement does not require an
outcome horizon to have matured, because it does not estimate outcome accuracy.

The kappa formulation follows Cohen (1968), *Psychological Bulletin* 70,
213–220, [doi:10.1037/h0026256](https://doi.org/10.1037/h0026256).
Bibliographic metadata were checked; the independent table oracle has quadratic
observed agreement .875, chance agreement .65625, kappa 7/11 and tied-rank
correlation .5. Paired score tests independently obtain Brier difference .06 and
shared-bootstrap Bonferroni bounds [.033,.087]. Seven full-payload Python/native
R replays cover exclusions, boundary and degenerate resampling cases.
