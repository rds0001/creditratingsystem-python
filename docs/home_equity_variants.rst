Home equity line variants
=========================

Run ``home_equity_variants.py`` or source installed ``home_equity_variants.R``
and call ``run_home_equity_variants()``. Both accept replacement local JSON inputs.
Reference assertions apply only to the supplied synthetic assumptions.

A 40,000 drawn line with limit100,000 accrues annual interest2,400 at6% before a
20,000 end-period draw. Default exposure after these events is62,400. Transition
into repayment freezes the principal commitment at60,000. The next annual period
accrues3,600 interest; paying all6,000 accrued interest and10,000 principal leaves
50,000. The limit amortizes to50,000 too, so a further1,000 draw fails with
EXPOSURE_LIMIT. The supplied dated limit calendar represents the contract; no
legal phase transition is inferred from a product name. Interest remains separate
from the principal commitment limit.

Both lien variants have total claims90,000 against value100,000, so CLTV is0.9.
One collateral sale produces80,000. A first-ranking50,000 home-equity claim is fully
paid; behind the other40,000 claim it receives40,000 and loses20%. Identical CLTV
therefore does not imply identical loss severity. Default is supplied separately
and is not ruled out by collateral value. Real priority and enforceability must
come from the bank's legal input contract. These are variant component scenarios;
product lifecycle and empirical model validation require separate evidence.
