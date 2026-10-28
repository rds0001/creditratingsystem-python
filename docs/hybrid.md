# Frozen hybrid credit models

The four functions combine explicitly compatible default estimates, fit a real held-out meta-model, execute a typed frozen graph, and explain a result relative to supplied reference inputs. They do not infer independence, causal effects, or truthful provenance from a hash.

## Compatible probability signals

`combine_pd_signals(pd_signals, weights, combination_scale=..., combination_contract=...)`
requires each signal to contain `signal_id`, `model_id`, `model_version`, ordered `row_ids`, a vector `pd`, a target, `as_of`, and `known_at`. Targets contain exactly `event='default'`, positive `horizon`, `time_unit` (days/months/years), `probability_measure`, and `conditioning`. All signals have the same target, rows and dates; knowledge cannot follow the forecast date. An optional calculation context must agree.

* `linear`: global nonnegative weights, normalized to sum one; the contract declares `interpretation='estimator_pool'` and a reference.
* `log_odds`: the same weights combine log odds; the interpretation is `log_odds_pool`. Positive-weight zeros or ones give the appropriate limiting probability. A positive-weight zero and one on the same row are undefined and rejected. No hidden clipping occurs.
* `mixture`: an n-by-m matrix of supplied conditional regime weights, normalized within rows. Declare `interpretation='latent_regime_mixture'`, `conditional_regime_probabilities=True`, ordered `regime_ids` matching signal IDs and a reference. This is the law of total probability conditional on the supplied mutually exclusive regime design, not the probability of a union of default events.

Results include normalized weights, contributions on the stated scale, boundary flags, signal fingerprints and weighted disagreement. Mixtures report within-regime Bernoulli variance and between-regime mean variance, whose sum is p(1-p). Disagreement is not a confidence interval.

## Held-out stacking

`fit_stacked_pd(out_of_fold_base_predictions, outcomes, base_model_ids=..., split_plan=..., meta_model_spec=..., target=..., training=..., model_id=..., model_version=..., weights=None)` fits an actual penalized binomial logit to the declared held-out rows. Its returned fitted artifact is data-only. Prediction takes aligned base-PD columns in `base_model_ids` order through `predict_pd`; new base-model fitting is outside this API.

`split_plan` has `horizon_date_rule='calendar_clamp'`, `rows` and `folds`. Each row supplies `row_id`, `entity_id`, `group_id`, role `development`/`calibration`/`test`, `as_of`, `label_known_at`, `complete_through`, and `prediction_known_at`. A fold supplies `fold_id`, `train_row_ids`, `validation_row_ids`, `fit_known_at`, and a `base_lineage` object keyed by every base ID. Each lineage record contains matching `model_id`, `model_version`, `target`, `prediction_origin='HELD_OUT'`, `training_fingerprint`, `reference`, `known_at`, `fit_row_ids`, `preprocessing_row_ids` and `selection_row_ids`.

Base fit/preprocessing/selection rows must lie inside that fold's training rows. Training and held-out entities/groups cannot overlap. Outer development/calibration/test entities and groups cannot overlap. Base artifacts must exist by their declared fit and held-out forecast dates, and fit precedes each held-out forecast. All used training or meta-fit outcomes must be known and their full horizon mature by the corresponding fit cutoff, including early known defaults; a negative outcome additionally requires explicit complete-through coverage. Horizon ends use integer days or whole calendar months, clamping the final day at month end. This is a date contract, not an inferred observation convention.

Only the union of held-out development rows trains the meta-model. Initial burn-in and calibration/test predictions can be missing; test labels are never inspected or used. These checks verify declared lineage relationships; they cannot authenticate how externally supplied base predictions were produced. Repeated entities/groups across meta rows are counted, not treated as evidence for independent uncertainty.

`meta_model_spec` declares `input_scale='probability'` or `'log_odds'`, `penalty='ridge'/'lasso'/'elasticnet'`, `lambda_path`, `selected_lambda`, `fit_intercept`, `scale`, and optional `l1_ratio`, `max_iter`, `tolerance`, with the same likelihood/penalty conventions as `fit_penalized_glm`. Log-odds designs require interior probabilities. Standardization is fit only on used meta rows. Exactly constant columns on positive-weight rows are removed and frozen; if all bases are constant, the model genuinely fits an intercept. Prediction flags departures from these constant training values. Weighting and penalty are estimated by the existing native numerical backend, not by a preset average. The supplied selected penalty is not silently optimized on test data; selection provenance remains the caller's responsibility. No post-selection confidence interval is claimed.

## Typed frozen graph

`evaluate_model_graph(model_graph, input_data, parameter_snapshots, context=...)` is a pure evaluation API. It neither trains models nor accesses external state. Supply a dated calculation context, a graph with `graph_id`, `graph_version`, `nodes`, `output_node`, and named maps of input and parameter snapshots. Every node has `node_id`, `operation`, named `inputs` mapping ports to parent node IDs, `parameters`, and `output_contract`.

Contracts contain `unit`, `target`, ordered `row_ids`, `as_of`, and `known_at`. Units are `design`, `probability`, `loss_fraction`, or `money`. Only designs add `feature_order`; only money adds explicit `currency`. An input snapshot is `{value, contract}` with optional already-applied `effects`. Vectors must match rows exactly; there is no recycling/broadcasting. Designs are row-by-feature matrices, including valid zero-column designs. Native R uses named lists for JSON objects and unnamed lists for arrays; an empty object is `structure(list(), names=character())`.

Supported operations and ports:

| Operation | Ports | Parameters |
|---|---|---|
| `input` | none | `input_id` |
| `predict_pd` | `design` | `snapshot_id` naming a frozen fitted model |
| `apply_calibrator` | `pd` | `snapshot_id`, `base_model_version`, optional `groups` |
| `combine_pd_signals` | named signal ports | `signal_order`, `weights`, `combination_scale`, `combination_contract` |
| `log_odds_shift` | `pd` | aligned finite `shift` vector, `effect_id`, `reference` |
| `expected_loss` | `pd`, `lgd`, `ead` | `loss_assumption`, `reference` |

Models and calibrators must have been known by the graph cutoff and pass their normal target/version checks. Log-odds shifts preserve structural 0/1 boundaries. Calibration artifact identity and adjustment effect IDs are recorded through ancestry; repeating an effect is rejected, including effects placed on multiple loss legs. This guards declared duplication; renamed external claims are not semantically identified.

Expected loss requires performing-conditioned PD, default-conditioned LGD/EAD, matching event, horizon, measure, rows and dates, and nonnegative monetary EAD. Raw economic LGD may be negative or exceed one. Explicit `loss_assumption` must be `deterministic_conditional_amounts` or `conditional_independence_EAD_LGD`; otherwise a product of conditional means is not justified. A general dependent loss distribution requires a joint simulation outside this node.

The computed output contract must match the declaration. Unknown operations/parameters, cycles, currencies and incompatible edges are rejected. Cyclic financial networks must first be evaluated with their domain solver and supplied as typed inputs. Identical operations with identical parameters, inputs, snapshots and context reuse local content-keyed computations. Results retain node values, effects, cache keys, immutable input/parameter/context snapshots and a content seal. No global cache or persistence occurs.

## Reconstruction and interactions

`decompose_hybrid_prediction(hybrid_result, baseline_inputs, decomposition_scale='native')` checks the content seal and replays the same frozen graph. Numerical replay allows relative 1e-12 plus absolute 1e-14 differences across native backends; contracts and declared trace relationships must agree. The baseline supplies exactly the same inputs and contracts, changing values only. Optional `log_odds` scale requires interior final probabilities in every evaluated scenario and never clips endpoints.

Each reported input contribution changes that entire input from the common baseline to its actual value, keeping other inputs at baseline. Thus

`actual = baseline + sum(one-input changes) + interaction remainder`.

The remainder is explicit and can be nonzero; e.g. multiplying three inputs that each double produces a final eight times baseline, three one-input increments of one baseline, and an interaction remainder of four baselines. These are not Shapley values or causal effects. Local same-contract single-parent deltas expose calibration/adjustment steps on their native scale; they must not be summed across branches. All frozen graph steps and a reconstruction error are returned.

## References and verification

The held-out estimation distinction follows the documented stacked-generalization fit contract in [scikit-learn's StackingClassifier](https://sklearn.org/1.7/modules/generated/sklearn.ensemble.StackingClassifier.html); this implementation adds explicit temporal maturity and group lineage checks. Pooling and decomposition formulas are documented algebraic definitions, tested against independent arithmetic. Native R uses existing glmnet/statistical helpers for genuine meta-estimation and has no Python bridge. Default tests use tiny real fits, frozen calibration, typed graph execution, exact product interaction checks, and leakage/error cases with one numerical thread. Detailed build and cross-language evidence is maintained separately from the distributed libraries.
