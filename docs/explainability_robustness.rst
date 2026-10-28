Finite counterfactual robustness scenarios
==========================================

``assess_counterfactual_robustness`` takes candidate rows and IDs, their baseline
record, named local model_variants, weighted scenarios, a feasibility contract and
a model-output or full-policy target. Every model variant supplies predictor and
model_contract; policy targets additionally require policy_evaluator and
policy_contract. Variants must share feature order, output scale and target.
nominal_variant_id identifies the unperturbed reference evaluation.

Each scenario declares scenario_id, model_variant_id, an additive delta vector
in feature order and a nonnegative weight. At least one weight must be positive.
Weights are normalized across the complete supplied set. The same delta is added
to each candidate and the baseline: immutable means no action on that feature,
not an assertion that the measurement has no uncertainty. All structural bounds
and identities are checked again after perturbation. Shared additive perturbation
is the explicit supported measurement law; a nonlinear measurement model must be
translated into a separately justified candidate/scenario design.

Every scenario retains its candidate results, feasibility findings, prediction,
optional full-policy decision, target success and margin. A failed model/policy
evaluation remains present with error_code and false target success. It is never
removed from the denominator. When a whole batch callback fails, that scenario is
unevaluable for every candidate; it is not silently retried on selected rows.
The nominal evaluation is reported separately and is not an extra weighted draw.

For model-output targets, margin is threshold minus prediction for <=, prediction
minus threshold for >=, and the minimum distance to either closed interval edge
for INTERVAL. Positive margins meet the target. If any scenario margin is
unavailable, worst/best margins are null rather than extrema of an incomplete
subset. Policy targets have no invented numeric margin. Feasibility lower/upper
summaries assign unknown feasibility to failure/success respectively. Failed
scenario IDs and their weights remain explicit. All-scenario success includes
zero-weight supplied scenarios, whereas weighted fractions use their declared
mass. This deliberate distinction separates enumerated robustness from weighted
summaries.

Feature reports show perturbation ranges and failure co-occurrence, not causal
importance. A prediction of -0.01 against target <=0 fails under +0.1 measurement
error; -1 survives that same error. Adding an unevaluable variant with half the
scenario mass caps the conservative success fraction at one half even for the
second candidate. These fractions are not probabilities that real interventions
succeed and do not establish worst-case robustness outside the finite scenarios.
