# Trade receivable product examples

Run `examples/trade_receivable_products.py`; native R exposes `run_trade_receivable_products()` in the corresponding packaged example. The JSON data are complete dated synthetic analytical references. They are deliberately small hand-checkable scenarios, not fitted probabilities or a representative empirical portfolio.

Each variant builds its invoice pool and dated borrowing base, simulates shared debtor/seller/servicer events, calculates the weighted net investor loss distribution, stresses an account bank and composes a typed portfolio assessment. Four equally weighted contractual scenarios represent debtor default alone, joint debtor/seller default, pure 20% dilution and full payment. Scenario probabilities are supplied assumptions.

A single invoice has principal 100 and debtor recovery 40. With seller recourse, its residual credit claim 60 is capped at performing seller capacity 30; when the seller also defaults, estate recovery is capped at 20. Net losses are therefore 30, 40, 20 and 0, with mean 22.5. Without recourse they are 60, 60, 20 and 0, with mean 35. The two-invoice pool shares one debtor default group; performing seller capacity 50 and default estate capacity 20 apply once to the aggregate claims. Its losses are 70, 100, 40 and 0, mean 52.5. Dilution remains a commercial loss and does not become a debtor default label.

Borrowing bases use a 50% shared debtor-group cap, 90% recourse or 80% nonrecourse advance rate, and a separate reserve of 5. Resulting bases are 40, 35 and 85. Splitting invoices does not evade the group cap.

The account-bank stress begins after the fully performing path has collected cash: May remittance of the April balance suffers 50% recovery. This is a subsequent conditional custody scenario, shown separately; its loss is not added to the origination scenario mean. Assessment EL includes credit loss, dilution and any modeled operating costs under an explicit net-investor-shortfall definition. It is not a borrower PD and does not claim a regulatory loss definition.

Each variant also rejects an invalid calendar or inconsistent party mapping. Both language examples verify hand references and full output parity. These examples cover the product recipe and variants; separate lifecycle evidence is required for all origination-through-exit governance phases.
