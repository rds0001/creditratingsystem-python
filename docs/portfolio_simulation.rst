Shared portfolio default and migration paths
===========================================

``prepare_credit_portfolio`` binds dated entity relationships, legal default
policies and exposure lots to a sealed portfolio. Each position identifies its
claim, facility and entity, reporting currency, EAD and complete facility EAD,
sector and region. Lots must reconcile to the facility total. Already-netted
exposure requires effective dated evidence covering its facility. Collateral
recovery is represented once in conditional LGD. Relationship and legal evidence
are supplied by the user; the package does not authenticate them.

The portfolio contract names its ID/version, valuation and knowledge dates,
currency, horizon and time unit. The target is PHYSICAL first default conditional
on initially performing groups. Each default group receives exactly one PD;
facilities inherit their group's common event. ``simulate_credit_portfolio``
accepts either supplied dependent uniforms or Gaussian systematic factors plus
idiosyncratic normals. The Gaussian contract contains ordered group/factor IDs,
a PSD factor correlation matrix and loadings with systematic variance at most
one. Asset correlation and default-indicator correlation are distinct.

Shared innovations contain ordered scenario/group IDs, normalized scenario
weights, sampling design and the required random coordinates. Conditional EAD
and raw LGD paths are scenario-by-position arrays with matching identities and
``driver_hash = hash(dependence_contract, shared_innovations)`` using the
package's canonical content hash. This binds their declared scenario origin;
it does not independently establish their statistical generation law. Loss is
computed pathwise as default indicator times EAD times LGD. Negative raw LGD
is preserved. Weighted loss distributions use the documented exact discrete
expected-shortfall convention.

Guarantee allocations come from ``allocate_guarantee_coverage``. Reusing an
allocation's guarantee or exceeding a position's EAD is rejected. Guaranteed
claim maturity must equal the declared portfolio horizon. A contract explicitly
selects first-loss capped protection or covered-EAD pro-rata protection. Both
pay only positive remaining loss when the borrower defaults and the guarantor
survives the horizon. The payoff is capped by the reserved monetary coverage.
These terminal assumptions do not model interim settlement or regulatory
capital substitution. Collateral and guarantee benefits must not already be
embedded twice in supplied loss paths.

``stress_portfolio_factor_structure`` replays the same random coordinates for
baseline and all stress cases. Every case declares fixed EAD/LGD paths or
replaced conditional paths and binds them to its new driver hash. The output
contains paired changes in loss, VaR and ES. Identical marginal PD parameters
do not guarantee identical default counts in a finite simulation sample.

``simulate_portfolio_migrations`` takes homogeneous or time-varying physical
transition matrices and supplied scenario-by-period-by-group uniforms. Its
contract declares sequential conditional uniformity; arbitrary temporal draw
dependence cannot establish the Markov law. The initial state, each subsequent
state and first-default time are retained. Default is absorbing unless cure is
explicitly permitted; cure never resets first-default time. Empirical terminal
frequencies and matrix-product terminal probabilities are reported separately.
An optional frozen valuation table produces initial mark minus terminal mark
minus cash distributions. It preserves gains and does not convert physical
migration probabilities to pricing probabilities.

Analytical checks include four equiprobable independent Bernoulli states,
shared-obligor facilities, perfectly common Gaussian factors, first-loss versus
pro-rata guarantees, exact two-period Markov enumeration, identity transitions,
cure and marked gains/losses. For two exposures of 100, PD 0.5 and LGD 0.5,
independent losses are 0, 50, 50, 100; EL is 50 and ES at 50% is 75. Perfectly
shared defaults preserve EL while raising that ES to 100.
