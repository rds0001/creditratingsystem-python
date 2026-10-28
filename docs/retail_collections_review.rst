Collections review for one observed default claim
================================================

Run ``examples/retail_collections_review.py`` or source the native R example and
call ``run_retail_collections_review``. The packaged independent synthetic data
bind one account, first-default claim, observed episode history, March1 offer,
repayment plans and open-workout valuation. No plan is accepted or performed.
Later observed recovery is not attributed causally to the offer.

Five genuinely prior 2022 accounts start in default. One exits at day15, one is
administratively censored at30, two achieve validated cures at45 and60, and the
last remains observed in default through90. A cure requires ten consecutive
covered clean days. Raw clean records therefore occur at35 and50, not at the
modeled cure date. The first cured account defaults again at75. The same actual
histories produce the first-cure cause-specific fit, Fine–Gray fit, and native
Aalen–Johansen multistate fit. Training availability2022-04-02 precedes the
held-out2024 default and every forecast. These are real fits, with small explicit
risk sets that permit hand calculations; they are not bank calibrations.

At first-default origin the cause-specific exit hazard is1/5; cure hazards are
1/3 and1/2. Proper joint integration gives cure CIF exp(-.2)*(1-exp(-1/3)) and
exp(-.2)*(1-exp(-5/6)). Fine–Gray's increments are1/3.75 and1/2.75. It estimates
another first-cure CIF; the two estimates are never added. Both first-default
forecasts are retained separately from the later landmark forecast.

The held-out account defaults2024-02-15 and pays8.333333333333334 on February29
against EAD100. The offer and open-workout valuation occur March1, while the
account remains in default. The later clean record March6 satisfies probation
on March16; operational separation closes the first spell March21. The April15
redefault opens a new linked episode. It is not retained as the original open
workout at an invented later valuation. Removing coverage leaves cure targets
unknown, even though a clean state was recorded. Historical default is preserved.

At the March1 landmark, model time15 and stateD have prior risk-set support after
the earlier exits. Aalen–Johansen predicts first cure2/3 and cure-then-redefault
1/3 by original day90. Explicit augmented states D,C,R,X distinguish never-cured,
cured without later default, redefault after cure, and terminal exit. Conditional
on the supported landmark, three mutually exclusive histories each have mass1/3.
The transition-probability interpretation assumes time-inhomogeneous Markov
behavior within the declared state representation and observation support.
Actions label observational support; none of these estimates identifies a
counterfactual treatment effect. No additional cure after redefault is inferred
outside the declared transition graph.

The single remaining claim is100 minus the observed prior recovery. Principal
arrears20 are part of that claim, not added again. The existing plan requests
half the residual in each of the first two periods. The offered plan defers
these installments to periods two and three. Explicit available cash20,50,50
shows the initial baseline shortfall and candidate feasibility. Receipt fractions
change realized receipts without reducing the contractual affordability burden.
The offered plan does not modify the legal contract or default status.

Conditional remaining receipts are full principal for cure without redefault,
half for cure followed by redefault, and one quarter for no cure. These are
explicit synthetic monetary assumptions, not estimated causal recoveries.
Consequently the expected accepted gross receipt is(7/12)*remaining_principal.
The dated future cost5 comes from the same open-workout input ledger and enters
once, in its actual planned period. Decline recovery20, uptake.25 and offer cost2
are separately declared assumptions. Economic discount is explicitly zero for
all cash, prior and future, so all figures share one valuation basis.
Expected future net recovery is

    .25*((7/12)*remaining_principal - 5) + .75*20 - 2.

Uptake zero gives18 exactly. Acceptance is not inferred from an offered action.
Prior cash is added once when presenting the assumption-based ultimate LGD.
The separate frozen recovery-hurdle estimate values the very same residual claim
using its own prior fitted recovery distribution and explicit tail assumption;
it is an alternative estimate, not another amount to add to the action value.
The open ledger contains only cash known by March1 and has no known final
resolution. Future scenario amounts are not realized workout labels.

The action-history panel excludes a higher available-cash revision logged March2
from the March1 decision. The actual historical action remains offered only.
Tests reject future model availability and attempts to move this first-workout
valuation after validated cure. Independent oracles cover all hazards, landmark
probabilities, mutually exclusive cash mixture, uptake, prior-cash reconciliation,
missing probation and immutable contractual/default status. Original fitted
models and exact value/arrangement inputs remain available with
``return_artifacts=True`` (R ``TRUE``). All nine collections recipe methods execute
on this linked history, with supporting multistate fitting and curve prediction.

Method references: Fine and Gray(1999), JASA94:496–509,
DOI10.1080/01621459.1999.10474144; Aalen and Johansen(1978), An empirical transition
matrix for non-homogeneous Markov chains based on censored observations,
Scandinavian Journal of Statistics5:141–150. All monetary coefficients here are
explicit independent synthetic assumptions, not proprietary calibrations.
