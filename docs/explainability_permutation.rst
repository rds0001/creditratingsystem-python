Frozen held-out grouped permutation importance
==============================================

``credit_permutation_importance`` compares prediction loss before and after
explicit shared permutations, without refitting the predictor or selecting
features. X and y are held-out rows in frozen model feature order. feature_groups
partition every feature exactly once. permutations is a matrix of complete
zero-based row-index bijections; each plan is reused across groups. A block's
columns are permuted jointly. Optional strata labels constrain every donor to the
recipient's stratum. The original outcome and weight remain with the recipient.

The design declares split OUT_OF_TIME or HOLDOUT, training_end, data_cutoff,
train_row_ids, train_subject_ids, model_and_calibration_frozen True,
selection_use FORBIDDEN, and permutation_design FIXED_PERMUTATIONS or
IID_RANDOM_PERMUTATIONS. row_metadata supplies unique row_id, subject_id,
observation_time, target_end and label_known_time. Evaluation identities must not
appear in declared fitting identities. Targets must be mature and labels known by
data_cutoff. Out-of-time observations must follow training_end. Declared identities
and cutoffs are checked but are not independently authenticated provenance.

The versioned loss_contract has loss_id, version and kind BRIER, LOG_LOSS, MSE,
MAE or CUSTOM. Binary probability losses require binary outcomes and probability
outputs. CUSTOM also requires custom_loss(y, predictions), returning one finite
loss per row. Positive-weight infinite log loss raises an explicit non-estimability
error; probabilities are never silently clipped. Weights are nonnegative and
normalized, with positive total. Per-plan losses and paired increases are retained.

A declared IID random permutation design with at least two plans yields sample
standard deviation of loss increases divided by sqrt(number of plans). Fixed
plans receive no standard error. Neither result is a confidence interval for
population, fitted-model or feature-selection uncertainty. Do not reuse this
held-out result for unaccounted feature selection.

Permutation may create implausible combinations of correlated features. Weighted
reference correlations are reported (null for constant features). Optional local
support_evaluator(X) returns one boolean per row and requires a support_contract
with support_id and version. Unsupported row IDs and mass are reported for baseline
and each permutation; those rows remain in all loss denominators. Without such a
callback, joint support is NOT_ASSESSED. Grouping preserves within-block observed
combinations but does not preserve dependence between blocks or justify a causal
interpretation of the reported importance.
