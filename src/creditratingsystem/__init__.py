"""Local credit risk methods with explicit assumptions and provenance."""
from .lgd import (fit_fractional_lgd, fit_beta_lgd, fit_zero_one_inflated_lgd, predict_lgd)
from .transfer import (assess_transfer_applicability, transfer_event_default_probability, combine_default_transfer, transfer_loss_components)
from .core import (
    CalcResult, FitResult, CreditRiskError, build_calculation_context, load_fitted_model,
    map_pd_to_grade, serialize_fitted_model, validate_probability, validate_rating_scale,
)
from .losses import expected_loss
from .pd_term_structure import cumulative_pd_from_conditional, marginal_pd_from_conditional, pit_ttc_transform
from .data import (
    read_local_dataset, validate_credit_dataset, join_as_of, build_observation_windows,
    build_recovery_episodes, make_grouped_temporal_splits,
)
from .affordability import (
    debt_to_income, debt_service_to_income, residual_income, assess_affordability,
)
from .classification import fit_binary_glm, predict_pd
from .calibration import calibrate_intercept, calibrate_logistic, calibrate_isotonic, apply_calibrator, estimate_long_run_pd
from .application import (
    build_application_snapshot, assemble_joint_application, build_application_targets,
    predict_application_risk, compare_offer_risk,
)
from .decisioning import compile_credit_decision_policy, apply_credit_decision_policy, decompose_policy_decision

from .structural import (
    merton_equity_value, infer_merton_assets, distance_to_default, merton_default_probability, first_passage_pd, simulate_structural_paths, calibrate_distance_pd_mapping
)

from .rating_system import (
    build_rating_subjects, route_rating_profile, assess_rating_applicability, register_rating_model_version, select_effective_model, compose_risk_assessment, check_assessment_freshness, transition_assessment_state, rate_subject_batch, apply_assessment_event, replay_assessment_history
)

from .affordability import (
    normalize_income_period, estimate_recurring_expenses, combined_loan_to_value, stress_affordability
)

from .pd_term_structure import (
    conditional_pd_from_cumulative, hazard_to_pd, estimate_transition_matrix, project_transition_matrix
)

from .validation import (
    auc_gini, ks_discrimination, brier_score, log_loss, binomial_pd_test
)

from .monitoring import (
    assign_model_role, rollback_model_assignment, classify_monitoring_findings
)

from .features import (
    fit_preprocessor, transform_features, fit_monotone_bins, compute_woe_iv, build_financial_ratios, build_distributed_lags
)

from .scorecard import (
    fit_scorecard, score_scorecard, scale_score_points, apply_qualitative_adjustments, scorecard_reason_codes
)

from .behavioral import (
    compute_days_past_due, assign_delinquency_state, build_account_month_panel, build_behavioral_targets, estimate_retail_roll_rates, project_retail_roll_rates, predict_behavioral_risk, build_vintage_performance
)

from .validation import (
    calibration_intercept_slope, calibration_curve, spiegelhalter_test, poisson_binomial_pd_test, hosmer_lemeshow_test, dependent_pd_backtest, default_count_backtest
)

from .losses import (
    expected_loss_from_paths, lifetime_expected_loss, event_conditioned_loss, counterparty_expected_loss, aggregate_portfolio_loss
)

from .monitoring import (
    build_monitoring_cohorts, monitor_rating_cohort, detect_credit_feature_drift, detect_pd_calibration_drift, compare_challenger_cohorts, diagnose_fairness_slices
)

from .collections import (
    build_collections_episodes, build_cure_targets, build_collections_action_history, simulate_repayment_arrangement, expected_collections_value
)

from .lgd import discount_recoveries, workout_lgd

from .ead import exposure_schedule, realized_ccf

from .survival import (
    estimate_survival_curve, fit_cox_ph, fit_aft_model, fit_discrete_hazard, fit_cause_specific_hazards, fit_fine_gray, predict_event_curves, estimate_multistate_transitions
)
from .collections import predict_cure_and_redefault

__version__ = "0.1.0"
__all__ = [
    "CalcResult", "FitResult", "CreditRiskError", "build_calculation_context", "load_fitted_model",
    "map_pd_to_grade", "serialize_fitted_model", "validate_probability", "validate_rating_scale",
    "expected_loss", "cumulative_pd_from_conditional", "marginal_pd_from_conditional",
    "read_local_dataset", "validate_credit_dataset", "join_as_of", "build_observation_windows",
    "build_recovery_episodes", "make_grouped_temporal_splits",
    "debt_to_income", "debt_service_to_income", "residual_income", "assess_affordability",
    "fit_binary_glm", "predict_pd",
    "calibrate_intercept", "calibrate_logistic", "calibrate_isotonic", "apply_calibrator",
    "build_application_snapshot", "assemble_joint_application", "build_application_targets",
    "predict_application_risk", "compare_offer_risk",
    "pit_ttc_transform", "estimate_long_run_pd",
    "compile_credit_decision_policy", "apply_credit_decision_policy", "decompose_policy_decision",
    'merton_equity_value',
    'infer_merton_assets',
    'distance_to_default',
    'merton_default_probability',
    'first_passage_pd',
    'simulate_structural_paths',
    'calibrate_distance_pd_mapping',
    'build_rating_subjects',
    'route_rating_profile',
    'assess_rating_applicability',
    'register_rating_model_version',
    'select_effective_model',
    'compose_risk_assessment',
    'check_assessment_freshness',
    'transition_assessment_state',
    'rate_subject_batch',
    'apply_assessment_event',
    'replay_assessment_history',
    'normalize_income_period',
    'estimate_recurring_expenses',
    'combined_loan_to_value',
    'stress_affordability',
    'conditional_pd_from_cumulative',
    'hazard_to_pd',
    'estimate_transition_matrix',
    'project_transition_matrix',
    'auc_gini',
    'ks_discrimination',
    'brier_score',
    'log_loss',
    'binomial_pd_test',
    'assign_model_role',
    'rollback_model_assignment',
    'classify_monitoring_findings',
    'fit_preprocessor',
    'transform_features',
    'fit_monotone_bins',
    'compute_woe_iv',
    'build_financial_ratios',
    'build_distributed_lags',
    'fit_scorecard',
    'score_scorecard',
    'scale_score_points',
    'apply_qualitative_adjustments',
    'scorecard_reason_codes',
    'compute_days_past_due',
    'assign_delinquency_state',
    'build_account_month_panel',
    'build_behavioral_targets',
    'estimate_retail_roll_rates',
    'project_retail_roll_rates',
    'predict_behavioral_risk',
    'build_vintage_performance',
    'calibration_intercept_slope',
    'calibration_curve',
    'spiegelhalter_test',
    'poisson_binomial_pd_test',
    'hosmer_lemeshow_test',
    'dependent_pd_backtest',
    'default_count_backtest',
    'expected_loss_from_paths',
    'lifetime_expected_loss',
    'event_conditioned_loss',
    'counterparty_expected_loss',
    'aggregate_portfolio_loss',
    'build_monitoring_cohorts',
    'monitor_rating_cohort',
    'detect_credit_feature_drift',
    'detect_pd_calibration_drift',
    'compare_challenger_cohorts',
    'diagnose_fairness_slices',
    'build_collections_episodes',
    'build_cure_targets',
    'build_collections_action_history',
    'simulate_repayment_arrangement',
    'expected_collections_value',
    'discount_recoveries',
    'workout_lgd',
    'exposure_schedule',
    'realized_ccf',
    'estimate_survival_curve',
    'fit_cox_ph',
    'fit_aft_model',
    'fit_discrete_hazard',
    'fit_cause_specific_hazards',
    'fit_fine_gray',
    'predict_event_curves',
    'estimate_multistate_transitions',
    'predict_cure_and_redefault',
    'assess_transfer_applicability',
    'transfer_event_default_probability',
    'combine_default_transfer',
    'transfer_loss_components',
    'fit_fractional_lgd',
    'fit_beta_lgd',
    'fit_zero_one_inflated_lgd',
    'predict_lgd',
]

from ._classification_models import (fit_discriminant_pd, fit_naive_bayes_pd, fit_svm_score, predict_model_score, fit_firth_logit, fit_penalized_glm, fit_ordinal_model)
__all__ += ['fit_discriminant_pd', 'fit_naive_bayes_pd', 'fit_svm_score', 'predict_model_score', 'fit_firth_logit', 'fit_penalized_glm', 'fit_ordinal_model']

from .dependence import (validate_dependence_matrix, fit_copula, simulate_copula, joint_default_probability, make_shared_innovations)
from .ead import (aggregate_netting_set_exposure, simulate_collateral_margin, closeout_exposure_at_default)
from ._classification_flexible import (fit_gam_pd, fit_random_forest_pd, fit_gradient_boosting_pd)
__all__ += ['validate_dependence_matrix', 'fit_copula', 'simulate_copula', 'joint_default_probability', 'make_shared_innovations', 'aggregate_netting_set_exposure', 'simulate_collateral_margin', 'closeout_exposure_at_default', 'fit_gam_pd', 'fit_random_forest_pd', 'fit_gradient_boosting_pd']

from .synthetic import (generate_macro_reference_world, generate_obligor_panel, generate_facility_workouts, generate_challenge_cases, validate_synthetic_world, generate_retail_reference_world, generate_asset_pool_reference_world, generate_system_event_reference_world)
__all__ += ['generate_macro_reference_world', 'generate_obligor_panel', 'generate_facility_workouts', 'generate_challenge_cases', 'validate_synthetic_world', 'generate_retail_reference_world', 'generate_asset_pool_reference_world', 'generate_system_event_reference_world']

from .ead import fit_ead_model, predict_ead
from .lgd import estimate_downturn_lgd
__all__ += ["fit_ead_model", "predict_ead", "estimate_downturn_lgd"]

from ._classification_bayesian import fit_bayesian_pd
from ._classification_neural import fit_neural_pd, fit_sequence_pd
__all__ += ["fit_bayesian_pd", "fit_neural_pd", "fit_sequence_pd"]

from .links import (build_entity_graph, resolve_parent_links, assign_default_groups, apply_parent_support, allocate_guarantee_coverage, apply_guarantee_risk, apply_rating_constraints, apply_override, solve_contagion_network)
__all__ += ['build_entity_graph', 'resolve_parent_links', 'assign_default_groups', 'apply_parent_support', 'allocate_guarantee_coverage', 'apply_guarantee_risk', 'apply_rating_constraints', 'apply_override', 'solve_contagion_network']

from ._calibration_models import (fit_stepwise, fit_regularization_path, resample_fit_uncertainty, calibrate_beta, calibrate_grade_pd, low_default_bound, fit_hierarchical_calibration, calibrate_lgd_level, estimate_margin_of_conservatism, population_reweighting, fit_model_selection_protocol)
from .lgd import fit_recovery_hurdle, estimate_open_workout_lgd
__all__ += ['fit_stepwise', 'fit_regularization_path', 'resample_fit_uncertainty', 'calibrate_beta', 'calibrate_grade_pd', 'low_default_bound', 'fit_hierarchical_calibration', 'calibrate_lgd_level', 'estimate_margin_of_conservatism', 'population_reweighting', 'fit_model_selection_protocol', 'fit_recovery_hurdle', 'estimate_open_workout_lgd']

from .expert import (fit_elicited_beta_prior, update_expert_beta_binomial, pool_expert_probability_distributions, update_odds_from_evidence, check_expert_preference_consistency, assess_expert_rating_agreement, compile_discrete_bayesian_network, infer_discrete_credit_network)
__all__ += ['fit_elicited_beta_prior', 'update_expert_beta_binomial', 'pool_expert_probability_distributions', 'update_odds_from_evidence', 'check_expert_preference_consistency', 'assess_expert_rating_agreement', 'compile_discrete_bayesian_network', 'infer_discrete_credit_network']
from .simulation import detect_cashflow_default
__all__ += ["detect_cashflow_default"]

from .simulation import build_debt_schedule, apply_cash_waterfall
__all__ += ["build_debt_schedule", "apply_cash_waterfall"]
from .retail_features import (validate_retail_dataset, build_customer_household_graph, aggregate_household_obligations)
__all__ += ['validate_retail_dataset', 'build_customer_household_graph', 'aggregate_household_obligations']

from .expert import compile_qualitative_rubric, evaluate_qualitative_evidence
__all__ += ['compile_qualitative_rubric', 'evaluate_qualitative_evidence']

from .selection_bias import (build_selection_dataset, fit_selection_propensity, selection_ipw_weights, fuzzy_reject_augmentation, parcel_rejected_applications, fit_joint_selection_model, estimate_population_risk_aipw, reject_inference_bounds, reject_inference_sensitivity, prepare_reject_evaluation)
__all__ += ['build_selection_dataset', 'fit_selection_propensity', 'selection_ipw_weights', 'fuzzy_reject_augmentation', 'parcel_rejected_applications', 'fit_joint_selection_model', 'estimate_population_risk_aipw', 'reject_inference_bounds', 'reject_inference_sensitivity', 'prepare_reject_evaluation']

from .retail_features import build_bureau_features, build_thin_file_features, reconcile_income_evidence
from .simulation import simulate_operating_cashflows, simulate_asset_values
__all__ += ['build_bureau_features', 'build_thin_file_features', 'reconcile_income_evidence', 'simulate_operating_cashflows', 'simulate_asset_values']

from .retail_features import build_payment_history_features, build_revolving_features
from .simulation import simulate_contract_counterparties
__all__ += ['build_payment_history_features', 'build_revolving_features', 'simulate_contract_counterparties']

from .retail_features import build_transaction_features
__all__ += ["build_transaction_features"]

from .retail_features import build_bnpl_obligation_features
from .simulation import estimate_simulation_risk
__all__ += ["build_bnpl_obligation_features", "estimate_simulation_risk"]

from .hybrid import fit_stacked_pd, combine_pd_signals, evaluate_model_graph, decompose_hybrid_prediction
from .retail_features import build_retail_sme_features
__all__ += ['fit_stacked_pd', 'combine_pd_signals', 'evaluate_model_graph', 'decompose_hybrid_prediction', 'build_retail_sme_features']

from .simulation import importance_sample_defaults
__all__ += ["importance_sample_defaults"]

from .simulation import simulate_recovery_waterfall
__all__ += ["simulate_recovery_waterfall"]

from .retail_features import fit_retail_risk_pools, assign_retail_risk_pool, estimate_pool_risk_parameters
__all__ += ['fit_retail_risk_pools', 'assign_retail_risk_pool', 'estimate_pool_risk_parameters']

from .simulation import simulate_transactions
__all__ += ["simulate_transactions"]

from .portfolio import credit_loss_distribution, allocate_expected_shortfall, incremental_credit_risk, credit_concentration_measures, asymptotic_factor_loss
__all__ += ['credit_loss_distribution', 'allocate_expected_shortfall', 'incremental_credit_risk', 'credit_concentration_measures', 'asymptotic_factor_loss']

from .validation import lgd_backtest, ead_backtest, survival_backtest, population_stability, rating_migration_backtest
__all__ += ['lgd_backtest', 'ead_backtest', 'survival_backtest', 'population_stability', 'rating_migration_backtest']

from .macro import transform_macro_history, fit_var, fit_vecm, fit_regime_switching, fit_dynamic_factor, block_bootstrap_scenarios, simulate_macro_scenarios, condition_macro_paths, fit_satellite_model, apply_satellite_model, reverse_stress_scenario
__all__ += ['transform_macro_history', 'fit_var', 'fit_vecm', 'fit_regime_switching', 'fit_dynamic_factor', 'block_bootstrap_scenarios', 'simulate_macro_scenarios', 'condition_macro_paths', 'fit_satellite_model', 'apply_satellite_model', 'reverse_stress_scenario']

from .decisioning import compute_limit_frontier, compute_risk_based_price_grid, evaluate_treatment_candidates
__all__ += ['compute_limit_frontier', 'compute_risk_based_price_grid', 'evaluate_treatment_candidates']

from .decisioning import evaluate_decision_grid, select_constrained_policy, assess_policy_robustness
__all__ += ['evaluate_decision_grid', 'select_constrained_policy', 'assess_policy_robustness']

from .portfolio import prepare_credit_portfolio, simulate_credit_portfolio, simulate_portfolio_migrations, stress_portfolio_factor_structure
__all__ += ['prepare_credit_portfolio', 'simulate_credit_portfolio', 'simulate_portfolio_migrations', 'stress_portfolio_factor_structure']

from .explainability import explain_linear_credit_prediction, estimate_credit_shapley_values
__all__ += ['explain_linear_credit_prediction', 'estimate_credit_shapley_values']

from .validation import benchmark_predictions, benchmark_ratings
__all__ += ['benchmark_predictions', 'benchmark_ratings']

from .explainability import rank_credit_reason_codes
__all__ += ["rank_credit_reason_codes"]

from .structured_credit import build_credit_asset_pool, tranche_loss_from_pool, receivables_borrowing_base
__all__ += ['build_credit_asset_pool', 'tranche_loss_from_pool', 'receivables_borrowing_base']

from .explainability import compute_credit_ice
__all__ += ["compute_credit_ice"]

from .explainability import find_feasible_credit_counterfactual
__all__ += ["find_feasible_credit_counterfactual"]

from .validation import override_monitoring, support_transfer_backtest, simulation_convergence, sensitivity_analysis
__all__ += ['override_monitoring', 'support_transfer_backtest', 'simulation_convergence', 'sensitivity_analysis']

from .structured_credit import simulate_asset_pool
__all__ += ['simulate_asset_pool']

from .explainability import assess_counterfactual_robustness
__all__ += ["assess_counterfactual_robustness"]

from .market_implied import build_credit_quote_snapshot, cds_expected_cashflow_legs, bootstrap_cds_hazard_curve, risky_bond_expected_cashflows, calibrate_bond_hazard_curve, diagnose_market_credit_identifiability, fit_market_to_physical_pd_bridge, compare_cds_bond_credit_basis
__all__ += ['build_credit_quote_snapshot', 'cds_expected_cashflow_legs', 'bootstrap_cds_hazard_curve', 'risky_bond_expected_cashflows', 'calibrate_bond_hazard_curve', 'diagnose_market_credit_identifiability', 'fit_market_to_physical_pd_bridge', 'compare_cds_bond_credit_basis']

from .explainability import credit_permutation_importance
__all__ += ['credit_permutation_importance']

from .explainability import compute_credit_ale
__all__ += ["compute_credit_ale"]

from .structured_credit import run_tranche_waterfall
__all__ += ['run_tranche_waterfall']

from .structured_credit import estimate_tranche_credit_metrics, stress_tranche_structure
__all__ += ['estimate_tranche_credit_metrics', 'stress_tranche_structure']

from .validation import validate_retail_cohort, roll_rate_backtest, vintage_backtest
__all__ += ['validate_retail_cohort', 'roll_rate_backtest', 'vintage_backtest']

from .structured_credit import covered_bond_dual_recourse
__all__ += ["covered_bond_dual_recourse"]

from .structured_credit import simulate_trade_receivables
__all__ += ["simulate_trade_receivables"]

from .validation import collections_backtest
__all__ += ["collections_backtest"]

from .structured_credit import stress_pool_counterparties
__all__ += ["stress_pool_counterparties"]

from .validation import validate_system_coverage, release_candidate_assessment, validation_report
__all__ += ['validate_system_coverage', 'release_candidate_assessment', 'validation_report']

from .validation import validate_source_lineage, replay_consistency
__all__ += ['validate_source_lineage', 'replay_consistency']

from .validation import counterparty_exposure_backtest
__all__ += ['counterparty_exposure_backtest']

from .ead import reconcile_card_authorizations
__all__ += ['reconcile_card_authorizations']

from .validation import affordability_stress_backtest
__all__ += ['affordability_stress_backtest']

from .core import content_fingerprint
__all__ += ['content_fingerprint']

from .validation import selection_bias_sensitivity
__all__ += ['selection_bias_sensitivity']

from .validation import decision_policy_backtest
__all__ += ["decision_policy_backtest"]

from .validation import retail_pool_diagnostics
__all__ += ["retail_pool_diagnostics"]

from .validation import asset_pool_backtest
__all__ += ['asset_pool_backtest']
