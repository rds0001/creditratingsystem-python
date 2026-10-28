# Covered-bond product examples

Run `python examples/covered_bond_products.py`, or source the installed R
`examples/covered_bond_products.R` and call `run_covered_bond_products()`.
The identical packaged `covered_bond_products.json` contains three distinct
contracts: mortgage cover, public-sector cover and legally declared conditional
maturity extension. Replace the data argument and disable reference verification
when using another local portfolio; the included numerical assertions belong
to these synthetic analytical inputs.

The workflow creates distinct issuer/issue/pool subjects, builds the eligible
cover pool, simulates its dated cash, and uses a credit-portfolio scenario engine
to derive the issuer default indicator. Issuer and cover share the two supplied
joint states: common uniform 0.25 triggers both at 50% PD, while 0.75 triggers
neither. These are fixed weighted assumptions, not an empirical estimate or IID
sample. The issuer portfolio's standalone expected loss is 25; it is retained
as an issuer diagnostic and is never added to the covered-bond loss.

In the joint default state, cover recovery 40 less workout cost 5 leaves 35.
For the original principal claim 100, a 50% residual estate recovery pays 32.5,
leaving final shortfall 32.5. In the surviving state the issuer pays 100 and no
estate claim is collected. The covered bond's two-state PD is therefore 0.5
and its expected loss is 16.25. Both mortgage and public-sector cover cases
exercise this full cash chain with their own typed collateral contract and
separate invalid-eligibility case.

The extension case moves principal to 1 July 2024 only after the supplied legal
trigger and issuer failure. It adds an explicit coupon 5. The default-state
claim is 105, cover still pays 35 and the estate pays 35, leaving shortfall 35;
expected loss is 17.5. The original-date principal deferral remains 100. The
surviving issuer neither extends nor owes the extension coupon. An attempted
extension without issuer default is rejected.

A subsequent account-bank stress takes the 45 already collected in the
performing cover state and models its delayed remittance. At 50% account-bank
recovery it loses 22.5. This conditional operational scenario is displayed
separately; it is not automatically added to the original joint-state covered
bond PD or expected loss. The final issue-level assessment binds its explicit
ultimate contractual-shortfall definition, actual settlement horizon and
source artifacts. Fixed weighted scenarios receive no IID confidence interval.

These examples demonstrate the full covered-bond recipe and distinct economic
variants under supplied legal conventions. They do not certify national legal
eligibility or complete every longitudinal governance/monitoring phase; those
phase-specific workflows need separate evidence.
