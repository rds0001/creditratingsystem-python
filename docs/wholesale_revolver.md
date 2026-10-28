# Synthetic wholesale revolving commitments

Run `examples/wholesale_revolver.py`, or source the installed R example and call `run_wholesale_revolver()`. The packaged JSON contains dated observed training cohorts, evidence, declared cash budget, contracts and separate hidden truth. Both implementations fit a logit PD model, hierarchical holdout calibration, fractional LGD and conditional CCF regression. The demonstrative 30-day PD is deliberately high (10%); this stress cohort is not a market benchmark.

All four facilities are active with drawn balance zero. Working-capital, cancellable and uncancellable limits are 100; the contingent commitment limit is 200. Observed historical defaults have CCFs 0.25, 0.50 and 0.75, giving mean CCF 0.50. Conditional EAD is therefore 50 or 100, despite zero current balance. Observed closed-workout losses 0.1, 0.3, 0.5 and 0.7 give mean LGD 0.4 and expected loss 2 or 4. These are distinct from the explicitly specified future draw scenarios of 60 or 80.

For the cancellable case only, a separately executed contractual cancellation on January 10 cuts the limit to zero before any draw. The associated counterfactual schedule excludes the cancelled draw. Rating evidence by itself does not change the contract. No cancellation schedule is generated for uncancellable terms. Each variant has its own route, legal cancellation flag and missing-financial-statements referral.

A documented expert Beta(2,18) distribution is an independent elicited benchmark; it is not inserted as observed outcomes or used for validation. A verified management assessment scores 20 under an explicit rubric. Unverified evidence refers the qualitative assessment and fails the decision eligibility gate. The declared reviewed cash budget has available cashflow 20 against service 10; operational clearance is a supplied dated assumption. Credit policy is synthetic and local.

Typed assessments keep unconditional PD, conditional LGD, conditional EAD and unconditional expected loss in separate target/variant components. Equal calibration group rates put hierarchical variance near its declared lower boundary; this example does not claim evidence of group heterogeneity. Hidden truth is used only by independent checks.

This establishes the fifteen-operation wholesale SME component recipe and four variants. Full eight-phase lifecycle acceptance remains pending. Every native output is compared; any numerical exceptions to the strict comparison policy are individually listed in the work evidence report.
