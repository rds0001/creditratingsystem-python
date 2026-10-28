Accumulated local effects with explicit support
===============================================

``compute_credit_ale`` accepts a frozen local predictor, reference matrix X and
model contract, one or two selected_features, explicit bins, feature_types and
optional nonnegative weights. Selected features must be declared continuous.
Numeric category codes are not silently treated as a continuous axis. Bin edges
must be finite, strictly increasing and cover every reference value. Bins are
left-closed/right-open, with the final upper endpoint included. min_bin_count
counts positive-weight observations; zero-weight records cannot create support.

For one feature, each populated bin receives the weighted mean prediction
change from setting that feature to the bin's upper instead of lower boundary,
leaving every other feature at its observed value. The differences are accumulated
from zero at the first boundary. This implementation linearly interpolates the
accumulated effect within each supported bin and subtracts its weighted mean over
the original reference observations. It returns both the centered boundary grid
and each reference observation's centered effect. This interpolation/centering
convention is explicit; it differs from an unlabelled step-function display.

For two features, each supported cell uses the four-corner mixed difference
f(high,high)-f(high,low)-f(low,high)+f(low,low), weighted over observations in that
cell. Double accumulation produces a boundary surface. Both first-order effects
of this accumulated surface are then computed using conditional reference-bin
weights and removed. The remaining bilinear surface is centered over the original
weighted reference. The subtracted first-axis and second-axis accumulations are
returned so the purification is reproducible. For independent symmetric x,y and
f=x*y, the result is x*y; an additive f(x)+g(y) yields zero interaction.

If any required bin/cell has insufficient positive-weight count or zero mass,
status is NON_ESTIMABLE_SUPPORT and the global centered grid/reference values are
null. Available local differences, cell counts, masses and zero-based unsupported
cell indices remain visible. No holes are assigned a zero effect or interpolated
across. This conservative rule avoids inventing a global surface from disconnected
support; users may explicitly choose different bins and reassess support.

Weighted selected-feature correlations are descriptive (null for constant axes).
Even populated cells do not prove that their boundary interventions lie on a
realistic data manifold. These are expected frozen model responses on the declared
output scale, not causal effects or real credit actions. Wide bins approximate
within-bin behavior through finite differences and interpolation; bin sensitivity
requires a separately declared comparison.

Methodological reference: Apley and Zhu, Visualizing the Effects of Predictor
Variables in Black Box Supervised Learning Models,
https://arxiv.org/abs/1612.08468. The precise finite-bin interpolation and
empirical-centering convention used here is specified above.
