"""Dated credit pools, tranches and receivables calculations."""

from ._structured_pool import build_credit_asset_pool, tranche_loss_from_pool, receivables_borrowing_base

from ._structured_paths import simulate_asset_pool

from ._structured_waterfall import run_tranche_waterfall

from ._structured_metrics import estimate_tranche_credit_metrics, stress_tranche_structure

from ._structured_recourse import covered_bond_dual_recourse

from ._structured_trade import simulate_trade_receivables

from ._structured_counterparties import stress_pool_counterparties
