# Portable tabular and sequence neural PD models

`fit_neural_pd` fits a dense tabular MLP. `fit_sequence_pd` fits a unidirectional,
single recurrent layer of arbitrary positive width using GRU or LSTM gates,
followed by a scalar logistic output. Both Python and R implement actual forward
propagation, analytic backpropagation and Adam updates on CPU. Their artifacts
contain numeric weights, architecture and training-only scaling; inference needs
no neural backend or Python bridge. These are likelihood prediction models:
empirical calibration and model uncertainty intervals are not inferred from a
training loss curve.

Tabular `architecture` contains `hidden_units` and matching `activations` chosen
from tanh/relu/sigmoid. An empty hidden-layer list gives a logistic output directly.
Every layer has explicit `initial_weights` entries `layer0_weight`, `layer0_bias`,
`layer1_weight`, etc. Weight matrices are input-width by output-width; biases are
one-dimensional output-width arrays. Hidden activation choices do not change the
final scalar logit. Arbitrary depth is supported by the same backpropagation chain.

Sequence initial parameters are `input_weight`, `hidden_weight`, `input_bias`,
`hidden_bias`, `output_weight`, `output_bias`. For input width p and hidden width h,
the first two shapes are (p,3h)/(h,3h) for GRU or (p,4h)/(h,4h) for LSTM, with the
corresponding gate-width biases. Output weight is (h,1), output bias length one.
When time gaps are included, p is raw feature count plus one. Gates are concatenated
r,z,n for GRU and i,f,g,o for LSTM. GRU uses the reset-after-hidden-affine convention:

    r = sigmoid(input_r + hidden_r)
    z = sigmoid(input_z + hidden_z)
    n = tanh(input_n + r * hidden_n)
    h = (1-z) * n + z * previous_h

Input and hidden affine terms each include their own bias. LSTM uses
c=f*previous_c+i*g, h=o*tanh(c), with sigmoid i/f/o and tanh g. Hidden and cell
states start at zero for every sequence, including every prediction call. Weights
use the transpose of PyTorch's stored gate matrices; the equations and both bias
vectors match its CPU GRU/LSTM reference. Full BPTT updates gate, recurrent and
output parameters; this is not a fixed feature extractor with only an output fit.

Each sequence record supplies `values`, `observation_times`, `known_times`, numeric
zero/one `mask`, and `forecast_time`, on one absolute clock measured in the target's
time_unit. Known time cannot precede observation time. Only mask-one observations
with observation and known time no later than forecast are used. The retained
observations must be strictly ordered, and at least one must be available. Padding,
future observations and reports not yet known do not enter recurrent state or
scaling. All supplied numeric fields, including padding, must still be finite;
use finite padding values. The target horizon is the fixed output horizon, not the
sequence length. Optional time-gap input is zero at the first retained observation
and subsequent elapsed observation gaps divided by positive `time_gap_scale`.

Training labels additionally supply `label_known_times` and an explicit numeric
`training_as_of_time` on that clock. No label can be known after the training cutoff
or before its forecast. A nondefault label must be known no earlier than forecast
plus full horizon; observed early default labels can be used earlier. This contract
prevents treating immature uncensored nondefaults as negatives. It does not
reconstruct default evidence: callers must provide correct observed labels. Late
reports are handled by their known times, not backdated into forecasts.

Scaling uses population mean/SD of tabular training rows or the pooled *retained*
training sequence observations, including optional time-gap features. A constant
SD becomes one. This convention weights each retained sequence observation equally;
longer sequences contribute more to scaling. The fitted values are frozen. State
resets and variable-length loops make padding unnecessary for computation, although
masks are accepted as an explicit data contract.

`optimizer_spec` requires `algorithm='adam'`, positive learning_rate, batch_size,
and `epoch_orders`: a list of complete zero-based permutations of all training
rows, one per requested epoch. Optional beta1/beta2 default to .9/.999, epsilon to
1e-8, l2 to zero. Adam uses bias-corrected first/second moments, with L2 added to
weight gradients only (coupled L2, not AdamW); biases are unpenalized. Optional
positive `gradient_clip_norm` clips the global gradient after L2, and clipped
updates are counted. The loss is mean Bernoulli negative log likelihood. Initial
weight fingerprints and executed epoch orders make the supplied training design
reviewable; no initialization or shuffling is hidden.

Inner validation supplies a distinct split_id, `role='inner_validation'`, unique
row_ids disjoint from train_row_ids, and an information cutoff no later than the
model cutoff. It supplies X/y for MLP, or sequences/y/label_known_times for sequence
models, with the same sequence maturity checks. `early_stopping_patience` requires
this explicit validation set. The artifact retains the best validation epoch,
including epoch zero if every update harms validation loss. Without validation it
retains the final epoch. A separate final test set must not be relabeled inner
validation. Row identifiers alone do not establish borrower-group independence;
split repeated borrowers appropriately before calling the estimator.

Diagnostics report complete training/validation loss paths, selected/executed
epochs, Adam update count, gradient norm and optimization status. Completing an
epoch budget is not called convergence to a global optimum. Models remain suitable
for review even when training choices perform poorly; a low loss on a small
synthetic fixture does not certify portfolio discrimination or calibration.

Primary references: [Adam](https://arxiv.org/abs/1412.6980),
[PyTorch GRU equations](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html),
and [PyTorch LSTM equations](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
Tests compare complete gated gradients with independent CPU PyTorch autograd and
all-parameter central differences, verify fixed-weight forward arithmetic, actual
loss reduction, explicit early stopping, label maturity, row-order state resets,
and invariant predictions after padding/future observations are appended. Public
model JSON and shared initialization/training orders reproduce native R and Python
predictions to floating-point precision.

Default tests retain small genuine fits and frozen predictions, including both Bayesian links, MLP, GRU and LSTM. Full posterior recovery against quadrature and the optional independent Torch autograd oracle run when `CREDITRATINGSYSTEM_EXTENDED_TESTS=true`. All-parameter finite-difference gradient checks run by default. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for test runs.
