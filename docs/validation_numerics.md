# Simulation precision and evaluated frozen-model sensitivity

`simulation_convergence(runs, shared_innovations, *, sampling_contract, precision,
design, context=None)` diagnoses supplied path-level simulation results.
The contract declares `sampling_unit='iid_paths_with_independent_replicates'`,
`metric='mean'/'bernoulli_probability'`, `reference`, `finest_time_step`, and
`time_unit` matching the target. Optional `analytic_reference` contains a finite
`value` and `source`. The common validation design supplies frozen model/data/
population/split identities, target and evaluation cutoff.

Shared innovations map replicate IDs to ordered unique `path_ids` and finite
path-by-innovation `values`. Each run provides `replicate_id`, positive
`time_step`, integer `sample_size >= 2`, prefix `path_ids`, path-level `values` and
`innovation_fingerprint` (the library's canonical SHA256 of the corresponding
innovation prefix). Sample sizes must use exact prefixes; time steps must be
integer multiples of the finest step. Every reported resolution/sample-size
combination must contain all declared replicates. These checks establish supplied
input identity, not that an external simulator actually consumed those inputs.
They do not validate an undisclosed coarse-grid transformation algorithm.

Path means are pooled only under the declared independence assumption; MCSE is
sample standard deviation divided by square root of total path count. General
mean intervals use a normal approximation. Bernoulli probabilities use Wilson
score intervals, which retain positive upper uncertainty with zero observed
defaults even though the empirical standard error is zero. The precision contract
provides `confidence_level`, `mc_half_width` and `grid_difference_tolerance`.
MC precision compares the maximum radius around the point estimate to the first
threshold. Consecutive grid resolutions at the same sample size use paired
coarse-minus-fine path differences and their own MCSE. Their interval's maximum
absolute endpoint is compared to the second threshold. These unadjusted
intervals are diagnostics; no continuous-time bias bound or convergence approval
is claimed. More paths cannot remove time-discretization bias.

`sensitivity_analysis(model, baseline, perturbations, *, method, shared_samples,
design, context=None)` actually evaluates frozen model predictions. A PD evaluator
is `{kind:'pd_model', fitted_model:...}`; baseline coordinates must cover the
complete trained feature order. Each perturbation specifies `name`,
`location:'feature'`, `lower`, `upper`, `step`.

A graph evaluator is `{kind:'model_graph', model_graph, input_data,
parameter_snapshots, execution_context, output_row:0}`. Graph coordinates have
`location:'input'` and a path such as `['input_id','value',0]`, or
`location:'parameter'` and a path such as
`['nodes',2,'parameters','weights',0]`. Numeric array indices are **zero based in
both Python and R**. Only value leaves or declared graph-node parameter leaves
can change. Numeric baselines must equal those frozen leaves; graph topology,
contracts, effect lineage and fitted parameter snapshots remain intact. Parameter
interventions are explicit scenario assumptions, not refits or parameter
uncertainty estimates. The final scalar output must have the declared target and
cutoff. All evaluations pass through public frozen prediction/graph semantics.

`finite_difference` requires `shared_samples=None`. It evaluates central
second-order differences where both sides are supported, otherwise second-order
one-sided differences, shrinking steps to respect bounds. Returned derivatives
use half the reported coarse step and report the absolute step-halving
difference; unrepresentable steps error. Elasticity is undefined for zero baseline
output. Nonsmooth scores may have zero or unstable local derivatives.

`variance_based` accepts `shared_samples={A,B,distribution_contract}`. A/B are
equal finite matrices with at least two rows; their columns follow coordinate
order. The distribution contract requires `independent_inputs:true` and a
`reference`. Correlated-input Sobol interpretation is rejected. Every A/B row and
one-/two-column hybrid is genuinely evaluated. With C_j equal to A except B's
column j, total index is mean((f(A)−f(C_j))²)/(2V), first index is
1−mean((f(B)−f(C_j))²)/(2V), where V is the population variance of pooled A/B
outputs. Pairwise closed indices use the two-column hybrid, then subtract both
first indices. Zero variance is NOT_APPLICABLE; finite-sample indices are not
clipped. Global uncertainty is explicitly NOT_ESTIMATED, and independence is a
caller declaration rather than an empirical proof. Common input designs and a
full evaluation trace permit replay; these deterministic evaluators introduce no
fresh Monte Carlo noise. Neither local nor global sensitivity is causal.

Independent tests use exact Cartesian integration of .25*x+.75*y to obtain
first/total shares .1/.9 and zero interaction, local slopes .25/.75, a graph
weight derivative −.45, and a real grouped logistic MLE's analytic derivative.
A simulation oracle has constant coarse-minus-fine difference .5 with zero paired
sampling error, while marginal MCSE remains positive.

References: [Jansen (1999), sections 1–3](https://openturns.github.io/openturns/papers/jansen1999.pdf)
for independent-input variance components and shared designs;
[NIST confidence interval formulas](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
for Wilson score inversion. The source formulas do not establish independence of
user-supplied paths or model validity.
