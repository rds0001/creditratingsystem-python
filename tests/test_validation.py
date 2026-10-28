"""Independent arithmetic oracles for discrimination, scoring and exact tests."""
import math
import numpy as np
import pytest
from creditratingsystem.core import CreditRiskError
from creditratingsystem.validation import auc_gini, ks_discrimination, brier_score, log_loss, binomial_pd_test


def test_weighted_pair_and_tie_counts_and_reversal():
    y, score, weights = [0,1,0,1], [0,1,1,2], [2,2,3,4]
    r = auc_gini(y,score,weights,score_direction="higher_is_risk").value
    assert r["auc"] == pytest.approx(27/30)
    assert r["gini"] == pytest.approx(.8)
    assert r["pair_count"] == 4 and r["tied_pair_count"] == 1
    assert r["tied_pair_mass"] == pytest.approx(6/30)
    assert auc_gini(y,score,weights,score_direction="lower_is_risk").value["auc"] == pytest.approx(.1)
    r = ks_discrimination(y,score,weights,score_direction="higher_is_risk").value
    assert r["ks"] == pytest.approx(2/3) and r["threshold"] == 1


def test_random_weighted_ties_against_explicit_pair_enumeration():
    rng = np.random.default_rng(161)
    y = rng.integers(0,2,80); s = rng.integers(0,7,80); w = rng.uniform(0,5,80)
    numerator = denominator = 0.
    for i in range(len(y)):
        for j in range(len(y)):
            if y[i] == 1 and y[j] == 0:
                denominator += w[i]*w[j]
                numerator += w[i]*w[j]*(float(s[i]>s[j])+.5*float(s[i]==s[j]))
    r=auc_gini(y,s,w,score_direction="higher_is_risk").value
    assert r["auc"] == pytest.approx(numerator/denominator,abs=2e-15)
    perm=rng.permutation(len(y))
    assert auc_gini(y[perm],s[perm],w[perm],score_direction="higher_is_risk").value["auc"] == pytest.approx(r["auc"],abs=2e-15)


def test_perfect_constant_single_class_and_zero_weight():
    assert auc_gini([0,1],[0,1],score_direction="higher_is_risk").value["auc"] == 1
    assert ks_discrimination([0,1],[0,1],score_direction="higher_is_risk").value["ks"] == 1
    assert auc_gini([0,1],[2,2],score_direction="higher_is_risk").value["auc"] == .5
    assert ks_discrimination([0,1],[2,2],score_direction="higher_is_risk").value["ks"] == 0
    for fn in (auc_gini,ks_discrimination):
        r=fn([0,1],[0,1],[1,0],score_direction="higher_is_risk").value
        assert r["applicability"] == "NOT_APPLICABLE" and r["reason"] == "ONE_CLASS"


def test_brier_arithmetic_contributions_and_baseline():
    r=brier_score([0,1],[.25,.75],[1,3]).value
    assert r["brier"] == .0625
    assert r["contributions"] == [.015625,.046875]
    assert r["outcome_variance_baseline"] == .1875
    assert brier_score([0,1],[0,1]).value["brier"] == 0


def test_log_loss_exact_boundaries_and_disclosed_clipping():
    assert log_loss([0,1],[0,1]).value["log_loss"] == 0
    r=log_loss([0,1],[0,0]).value
    assert r["loss_status"] == "INFINITE" and r["log_loss"] is None
    assert r["contributions"] == [0,None] and r["infinite_loss_flags"] == [False,True]
    assert log_loss([0,1],[0,0],[1,0]).value["log_loss"] == 0
    r=log_loss([0,1],[0,0],boundary_policy="clip",epsilon=.01)
    assert r.value["log_loss"] == pytest.approx((math.log(100)-math.log(.99))/2)
    assert r.diagnostics["clipped_count"] == 2
    assert log_loss([1],[1e-300]).value["log_loss"] == pytest.approx(300*math.log(10))
    with pytest.raises(CreditRiskError,match="INVALID_BOUNDARY_POLICY"):
        log_loss([0,1],[0,1],boundary_policy="clip",epsilon=1e-300)


@pytest.mark.parametrize("alternative,expected,upper", [
    ("less",1/16,1-.05**.25),
    ("two-sided",2/16,1-.025**.25),
    ("greater",1.,1.),
])
def test_binomial_enumeration_and_zero_count_interval(alternative,expected,upper):
    r=binomial_pd_test(4,0,.5,alternative=alternative,independence_assumption="independent_common_probability").value
    assert r["p_value"] == pytest.approx(expected,abs=1e-14)
    assert r["upper"] == pytest.approx(upper,abs=1e-12)
    assert r["lower"] == 0


def test_binomial_empty_cohort_and_assumptions():
    r=binomial_pd_test(0,0,.02,alternative="greater",independence_assumption="independent_common_probability").value
    assert r["p_value"] is None and r["applicability"] == "NOT_APPLICABLE"
    with pytest.raises(CreditRiskError,match="INCOMPATIBLE_ASSUMPTIONS"):
        binomial_pd_test(4,0,.5,alternative="greater",independence_assumption="correlated")


@pytest.mark.parametrize("fn",[brier_score,log_loss])
def test_invalid_scoring_inputs_are_not_silently_removed(fn):
    for y,p,w in [([0,2],[.1,.2],None),([0,1],[.1],None),([0,1],[.1,.2],[0,0]),
                  ([0,1],[.1,2],None),([0,float('nan')],[.1,.2],None)]:
        with pytest.raises(CreditRiskError): fn(y,p,w)


from copy import deepcopy
from creditratingsystem.core import _fingerprint
from creditratingsystem.validation import (calibration_intercept_slope,calibration_curve,
    spiegelhalter_test,poisson_binomial_pd_test,hosmer_lemeshow_test,dependent_pd_backtest,default_count_backtest)


def validation_design(p,origin='HOLDOUT',ids=None):
    result={'forecast_fingerprint':_fingerprint(list(p)),'prediction_origin':origin,'forecast_known_at':'2023-12-31','evaluation_at':'2025-01-01','split_id':'test','target':{'event':'first_default','horizon':1,'time_unit':'years','probability_measure':'PHYSICAL','conditioning':'PIT'}}
    if ids is not None:result['observation_ids']=ids
    return result


def calibration_reference():
    return [.25]*4+[.5]*4+[.75]*4,[0,0,0,1,0,0,1,1,0,1,1,1]


def test_diagnostic_logistic_mle_iid_and_hand_cluster_sandwich():
    p,y=calibration_reference();d=validation_design(p)
    r=calibration_intercept_slope(p,y,design=d,covariance_policy='iid').value
    assert r['converged'] and r['intercept']==pytest.approx(0,abs=1e-12) and r['slope']==pytest.approx(1,abs=1e-10)
    assert r['covariance'][0][0]==pytest.approx(.4,abs=1e-10)
    assert r['covariance'][1][1]==pytest.approx(1/(1.5*math.log(3)**2),abs=1e-10)
    c=calibration_intercept_slope(p,y,design=d,covariance_policy='cluster_cr1',cluster_ids=['A','B','C','D']*3).value
    # Cluster score sums are (-1.5,-l/2),(-.5,l/2),(.5,l/2),(1.5,-l/2), l=log(3).
    assert c['covariance'][0][0]==pytest.approx(88/75,abs=1e-10)
    assert c['covariance'][1][1]==pytest.approx(88/(135*math.log(3)**2),abs=1e-10)
    assert c['covariance'][0][1]==pytest.approx(0,abs=1e-10)
    assert p==calibration_reference()[0]


def test_calibration_diagnostic_identifiability_boundary_and_separation():
    for p,y,reason in [([.2,.2],[0,1],'CONSTANT_PD'),([0,.5],[0,1],'BOUNDARY_PD'),([.2,.8],[0,1],'SEPARATION')]:
        out=calibration_intercept_slope(p,y,design=validation_design(p),covariance_policy='iid').value
        assert out['reason']==reason and out['slope'] is None
    p,y=calibration_reference();d=validation_design(p);d['forecast_fingerprint']='changed'
    with pytest.raises(CreditRiskError,match='FORECAST_INTEGRITY'):
        calibration_intercept_slope(p,y,design=d,covariance_policy='none')


def test_prespecified_calibration_bins_include_boundaries_and_empty_bins():
    p=[0,.25,.5,1];y=[0,1,0,1]
    r=calibration_curve(p,y,[0,.25,.5,.75,.9,1],design=validation_design(p),interval_method='clopper_pearson',dependence_assumption='independent_common_probability_within_bins').value
    assert r['assignments']==[1,2,3,5] and r['bins'][3]['status']=='EMPTY'
    assert r['bins'][3]['observed_rate'] is None and r['bins'][3]['lower'] is None
    assert sum(b['count']*b['observed_rate'] for b in r['bins'] if b['count'])==sum(y)
    assert r['bins'][0]['upper']==pytest.approx(.975,abs=1e-12)


def test_spiegelhalter_formula_and_degenerate_variance():
    p=[.1,.2];y=[0,1]
    r=spiegelhalter_test(p,y,design=validation_design(p),alternative='two-sided',independence_assumption='independent_fixed_bernoulli').value
    assert r['numerator']==pytest.approx(.4) and r['variance']==pytest.approx(.1152)
    assert r['z']==pytest.approx(.4/math.sqrt(.1152))
    r=spiegelhalter_test([.5,.5],y,design=validation_design([.5,.5]),alternative='greater',independence_assumption='independent_fixed_bernoulli').value
    assert r['reason']=='ZERO_VARIANCE' and r['p_value'] is None


def test_poisson_binomial_polynomial_and_binomial_limit():
    p=[.2,.5,.8]
    r=poisson_binomial_pd_test(p,0,design=validation_design(p),alternative='two-sided',independence_assumption='independent_fixed_bernoulli').value
    assert r['probability_mass']==pytest.approx([.08,.42,.42,.08]) and r['p_value']==pytest.approx(.16)
    p=[.2]*12
    for k in (0,2,8,12):
        a=poisson_binomial_pd_test(p,k,design=validation_design(p),alternative='two-sided',independence_assumption='independent_fixed_bernoulli').value
        b=binomial_pd_test(12,k,.2,alternative='two-sided',independence_assumption='independent_common_probability').value
        assert a['p_value']==pytest.approx(b['p_value'],abs=1e-12)
        assert sum(a['probability_mass'])==pytest.approx(1,abs=1e-14)
    p=[0,1]
    assert poisson_binomial_pd_test(p,0,design=validation_design(p),alternative='less',independence_assumption='independent_fixed_bernoulli').value['p_value']==0


def test_hl_hand_statistic_design_df_and_small_cells():
    p=[.25]*20+[.75]*20;y=[1]*4+[0]*16+[1]*16+[0]*4
    r=hosmer_lemeshow_test(p,y,[0,.5,1],design=validation_design(p),degrees_of_freedom_policy='external_common_bin_pd',independence_assumption='independent_fixed_bernoulli').value
    assert r['statistic']==pytest.approx(8/15) and r['degrees_of_freedom']==2
    assert r['p_value']==pytest.approx(math.exp(-4/15))
    small=hosmer_lemeshow_test([.1,.9],[0,1],[0,.5,1],design=validation_design([.1,.9]),degrees_of_freedom_policy='external_common_bin_pd',independence_assumption='independent_fixed_bernoulli').value
    assert small['reason']=='SMALL_EXPECTED_CELL' and small['p_value'] is None
    with pytest.raises(CreditRiskError,match='INVALID_VALIDATION_DESIGN'):
        hosmer_lemeshow_test(p,y,[0,.5,1],design=validation_design(p),degrees_of_freedom_policy='fitted_logistic_approximation',independence_assumption='independent_fixed_bernoulli')


def dependent_reference(rho=1):
    p=[.5,.5];d=validation_design(p,ids=['O1','O2'])
    null={'family':'gaussian_factor','asset_correlation':rho,'factor_correlation':[[1]],'parameter_uncertainty':'conditional_fixed'}
    innovations={'draw_assumption':'independent_standard_normal_rows','observation_ids':['O1','O2'],'factor_keys':[{'cluster_id':'G','time_id':'2024'}],'factor_normals':[[-2],[-.5],[.5],[2]],'idiosyncratic_normals':[[-1,-1],[-1,1],[1,-1],[1,1]]}
    return dict(predicted_pd=p,outcomes=[1,1],cluster_ids=['G','G'],time_ids=['2024','2024'],design=d,null_model=null,innovations=innovations,alternative='greater')


def test_dependent_shared_cluster_paths_and_independent_boundary():
    args=dependent_reference();r=dependent_pd_backtest(**args).value
    assert r['null_counts']==[2,2,0,0] and r['p_value']==3/5
    assert r['null_count_interval']==[0,2]
    independent=dependent_pd_backtest(**dependent_reference(0)).value
    assert independent['null_counts']==[2,1,1,0] and independent['p_value']==2/5
    args['innovations']['observation_ids'].reverse()
    with pytest.raises(CreditRiskError,match='INNOVATION_ALIGNMENT'):dependent_pd_backtest(**args)


def test_dependent_parameter_paths_preserve_explicit_predictive_uncertainty():
    args=dependent_reference(0);args['null_model']['parameter_uncertainty']='supplied_predictive_pd_draws';args['null_model']['parameter_draw_source']='REFERENCE: four fixed parameter scenarios'
    args['innovations']['parameter_pd_draws']=[[0,0],[1,1],[0,1],[1,0]]
    out=dependent_pd_backtest(**args)
    assert out.value['null_counts']==[0,2,1,1]
    assert out.diagnostics['parameter_uncertainty']=='supplied_predictive_pd_draws'


def count_reference():
    base={'entity_id':'O','as_of':'2024-01-01','performance_start':'2024-01-01','outcome_end':'2024-01-31','eligible':True,'prior_default':False,'default_definition':'first_default','target':1,'outcome_known_at':'2024-01-15','grade':'B'}
    rows=[dict(base,observation_id='a',facility_id='F1'),dict(base,observation_id='b',facility_id='F2'),dict(base,observation_id='c',entity_id='Q',facility_id='F3',target=0,outcome_known_at='2024-01-31',grade='A'),dict(base,observation_id='d',entity_id='R',facility_id='F4',target=None,outcome_known_at=None)]
    forecasts=[dict(observation_id=r['observation_id'],pd=.1 if r['entity_id']=='O' else .2,forecast_at='2023-12-31',known_at='2023-12-31',default_definition='first_default',horizon_days=30,unit='obligor',probability_measure='PHYSICAL',model_id='M',model_version='1') for r in rows]
    return dict(observation_panel=rows,frozen_forecasts=forecasts,group_by=['grade'],unit='obligor',horizon_days=30,default_definition='first_default',redefault_policy='exclude_previous_default',evaluation_cutoff='2024-02-01')


def test_default_cohorts_deduplicate_obligor_and_match_mature_denominators():
    args=count_reference();out=default_count_backtest(**args).value
    assert out['deduplicated_rows']==1 and out['overall']['observed_count']==2 and out['overall']['observed_defaults']==1
    assert out['overall']['expected_defaults']==pytest.approx(.3) and out['overall']['observed_expected_ratio']==pytest.approx(10/3)
    assert out['excluded']==[{'observation_id':'d','reason':'MISSING_OUTCOME'}]
    assert sum(g['observed_count'] for g in out['groups'])==2
    args['evaluation_cutoff']='2024-01-20'
    immature=default_count_backtest(**args).value
    assert immature['overall']['observed_count']==0 and immature['overall']['expected_defaults']==0
    assert all(e['reason']=='IMMATURE' for e in immature['excluded'])


def test_default_cohort_conflicting_duplicate_and_leaked_forecast_fail():
    args=count_reference();args['frozen_forecasts'][1]['pd']=.5
    with pytest.raises(CreditRiskError,match='DUPLICATE_UNIT_CONFLICT'):default_count_backtest(**args)
    args=count_reference();args['frozen_forecasts'][0]['known_at']='2024-01-02'
    with pytest.raises(CreditRiskError,match='FORECAST_LEAKAGE'):default_count_backtest(**args)


def test_scalar_one_row_equivalence_and_matrix_rejection():
    assert brier_score(1,.2,2).value==brier_score([1],[.2],[2]).value
    assert log_loss(0,.1).value==log_loss([0],[.1]).value
    assert auc_gini(1,3,score_direction='higher_is_risk').value['reason']=='ONE_CLASS'
    for fn in (brier_score,log_loss):
        with pytest.raises(CreditRiskError):fn([[0,1]],[.1,.2])
        with pytest.raises(CreditRiskError):fn([0,1],[[.1,.2]])
        with pytest.raises(CreditRiskError):fn([0,1],[.1,.2],[[1,1]])


def test_temporal_factor_null_variance_matches_bivariate_normal_arcsine_identity():
    # For p=.5 and latent correlation c, Var(Y1+Y2)=.5+asin(c)/pi.
    rng=np.random.default_rng(305);B=25000;args=dependent_reference(1)
    args['time_ids']=['2023','2024']
    args['null_model']['factor_correlation']=[[1,.6],[.6,1]]
    args['innovations']['factor_keys']=[{'cluster_id':'G','time_id':'2023'},{'cluster_id':'G','time_id':'2024'}]
    args['innovations']['factor_normals']=rng.normal(size=(B,2)).tolist()
    args['innovations']['idiosyncratic_normals']=rng.normal(size=(B,2)).tolist()
    out=dependent_pd_backtest(**args).value
    assert np.mean(out['null_counts'])==pytest.approx(1,abs=.025)
    assert np.var(out['null_counts'])==pytest.approx(.5+math.asin(.6)/math.pi,abs=.025)


def test_poisson_tail_underflow_is_visible_and_zero_expected_ratio_is_not_finite():
    p=[.5]*1100
    out=poisson_binomial_pd_test(p,0,design=validation_design(p),alternative='less',independence_assumption='independent_fixed_bernoulli')
    assert out.value['reason']=='NUMERICAL_UNDERFLOW' and out.value['p_value'] is None
    assert out.diagnostics['underflowed_support_cells']>0
    args=count_reference()
    for forecast in args['frozen_forecasts']:forecast['pd']=0
    out=default_count_backtest(**args).value
    assert out['overall']['ratio_status']=='INFINITE' and out['overall']['observed_expected_ratio'] is None
