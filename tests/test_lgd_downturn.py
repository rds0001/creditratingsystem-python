from copy import deepcopy
import math
import pytest
from creditratingsystem.core import CreditRiskError
from creditratingsystem.lgd import fit_fractional_lgd,estimate_downturn_lgd
TARGET=dict(event='loss',horizon=5,time_unit='years',probability_measure='PHYSICAL',conditioning='given_default')
TRAIN=dict(data_fingerprint='group-oracle',split_id='train',known_at='2024-01-01',data_origin='SYNTHETIC',workout_population='closed_defaulted_episodes')
def case():
    fit=fit_fractional_lgd([[0],[0],[1],[1]],[0,.5,.5,1],feature_order=['stress'],link='logit',covariance='HC0',target=TARGET,training=TRAIN,model_id='lgd',model_version='1')
    return dict(fitted_lgd_model=fit,baseline_covariates=[[0],[0]],adverse_covariates=[[1],[1]],aggregation_weights=[1e308,1e308],feature_order=['stress'],expected_target=TARGET,downturn_definition=dict(definition_id='explicit',version='1',selection_basis='User stress indicator 0 to1',baseline_scenario_id='base',adverse_scenario_id='adverse',known_at='2024-01-01',as_of='2024-01-01',unit_ids=['a','b'],extrapolation_policy='error'))
def test_explicit_group_effect_and_same_scenario_no_uplift():
    a=case();r=estimate_downturn_lgd(**a);assert r.value['baseline_lgd']==pytest.approx(.25);assert r.value['downturn_lgd']==pytest.approx(.75);assert r.value['difference']==pytest.approx(.5);assert r.value['uncertainty']['status']=='NOT_ESTIMATED'
    a['adverse_covariates']=a['baseline_covariates'];assert estimate_downturn_lgd(**a).value['difference']==0
    a=case();a['baseline_covariates'],a['adverse_covariates']=a['adverse_covariates'],a['baseline_covariates'];assert estimate_downturn_lgd(**a).value['difference']==pytest.approx(-.5)
def test_support_policy_and_paired_draw_uncertainty():
    a=case();a['adverse_covariates']=[[2],[2]]
    with pytest.raises(CreditRiskError,match='DOWNTURN_SUPPORT'):estimate_downturn_lgd(**a)
    a['downturn_definition']['extrapolation_policy']='report';assert all(estimate_downturn_lgd(**a).diagnostics['adverse_outside_marginal_range'])
    a=case();m=a['fitted_lgd_model'].model;draws=[deepcopy(m),deepcopy(m)];draws[0]['parameters']['coefficients']=[0,0];draws[1]['parameters']['coefficients']=[-math.log(3),2*math.log(3)];a.update(model_draws=draws,uncertainty_contract=dict(source='user_supplied',interpretation='Two illustrative paired parameter draws'),interval_level=.8)
    u=estimate_downturn_lgd(**a).value['uncertainty'];assert u['uplift_interval']==pytest.approx([.05,.45]);assert u['draw_count']==2
