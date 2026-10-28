import hashlib
import numpy as np
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.dependence import validate_dependence_matrix,make_shared_innovations


def test_psd_singular_and_explicit_spectral_repair():
    result=validate_dependence_matrix([[1,1],[1,1]],['a','b'],matrix_kind='correlation')
    assert result.diagnostics['numerical_rank']==1
    assert not result.diagnostics['strictly_positive_definite']
    with pytest.raises(CreditRiskError,match='NON_PSD_MATRIX'):
        validate_dependence_matrix([[1,2],[2,1]],['a','b'],matrix_kind='correlation')
    result=validate_dependence_matrix([[1,2],[2,1]],['a','b'],matrix_kind='correlation',repair_policy='spectral')
    np.testing.assert_allclose(result.value['matrix'],[[1,1],[1,1]],atol=1e-15)
    assert result.diagnostics['frobenius_change']==pytest.approx(2**.5)
    with pytest.raises(CreditRiskError,match='NON_PSD_MATRIX'):
        validate_dependence_matrix(np.array([[1,2],[2,1]])*1e-20,['a','b'],matrix_kind='covariance')


def test_tolerance_does_not_silently_repair_input():
    a=np.array([[1,.2],[.2+1e-12,1]])
    out=validate_dependence_matrix(a,['a','b'],matrix_kind='correlation')
    np.testing.assert_array_equal(out.value['matrix'],a)
    assert out.diagnostics['frobenius_change']==0


def test_addressing_and_files_are_stable_across_chunks_and_row_order(tmp_path):
    specs=[dict(stream_id='common',distribution='uniform'),dict(stream_id='idiosyncratic',distribution='normal')]
    results=[]
    for name,ids,chunk in [('a',['one','two'],1),('b',['one','two'],20),('c',['two','one'],2)]:
        results.append(make_shared_innovations(ids,[0.,1.],['X','Y'],specs,'experiment',output_directory=tmp_path/name,chunk_rows=chunk))
    for i in range(2):
        a,b,c=[np.fromfile(tmp_path/name/f'stream-{i:04d}.f64le',dtype='<f8').reshape(2,2,2) for name in ['a','b','c']]
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c[::-1])
        assert np.isfinite(a).all()
        if i==0:assert ((a>0)&(a<1)).all()
        assert hashlib.sha256((tmp_path/'a'/f'stream-{i:04d}.f64le').read_bytes()).hexdigest()==results[0].value['manifest']['streams'][i]['sha256']
    assert results[0].provenance['data_origin']=='SYNTHETIC'
    with pytest.raises(CreditRiskError,match='OUTPUT_EXISTS'):
        make_shared_innovations(['one'],[0],['X'],specs,'experiment',output_directory=tmp_path/'a')

from creditratingsystem.dependence import fit_copula,simulate_copula,joint_default_probability
from creditratingsystem.core import _model_artifact
TARGET=dict(event='joint_default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
TRAIN=dict(data_fingerprint='copula-oracle',split_id='train',known_at='2024-01-01',data_origin='SYNTHETIC')
META=dict(target=TARGET,training=TRAIN,model_id='copula',model_version='1')

def frozen(family,**parameters):
    return _model_artifact(dict(schema_version='1.0',model_id='analytic-copula',model_version='1',method_id='dependence.fit_copula',feature_order=['a','b'],parameters=dict(family=family,dimension=2,**parameters),preprocessing=dict(transformation='identity'),target=TARGET,training=TRAIN,diagnostics={},provenance=dict(data_origin='SYNTHETIC',parameter_origin='SYNTHETIC_REFERENCE')))

@pytest.mark.parametrize('family,parameters,expected',[
    ('gaussian',dict(correlation=[[1,.5],[.5,1]]),1/3),
    ('t',dict(correlation=[[1,.5],[.5,1]],df=4),1/3),
    ('clayton',dict(theta=1),1/3),
    ('gumbel',dict(theta=2),2**(-2**.5)),
])
def test_copula_joint_analytic_oracles(family,parameters,expected):
    m=frozen(family,**parameters)
    r=joint_default_probability(.5,.5,event_order=['a','b'],dependence_model=m)
    assert r.value==pytest.approx(expected,abs=2e-10)
    assert joint_default_probability(0,.5,event_order=['a','b'],dependence_model=m).value==0
    assert joint_default_probability(1,.3,event_order=['a','b'],dependence_model=m).value==.3


def test_t_zero_correlation_is_not_independence_and_tiny_clayton_limit():
    m=frozen('t',correlation=[[1,0],[0,1]],df=3)
    assert joint_default_probability(.01,.01,event_order=['a','b'],dependence_model=m).value>.001
    m=frozen('clayton',theta=1e-15)
    assert joint_default_probability(.1,.2,event_order=['a','b'],dependence_model=m).value==pytest.approx(.02,rel=1e-12)


def test_rank_fit_known_tau_and_t_profile():
    x=[[1,1],[2,3],[3,2],[4,4]]
    for family,theta in [('clayton',4),('gumbel',3)]:
        fit=fit_copula(x,variable_ids=['a','b'],family=family,input_kind='observations',tie_policy='error',**META)
        assert fit.model['parameters']['theta']==pytest.approx(theta)
        assert fit.diagnostics['mean_kendall_tau']==pytest.approx(2/3)
    fit=fit_copula(x,variable_ids=['a','b'],family='gaussian',input_kind='observations',tie_policy='error',**META)
    assert fit.model['parameters']['correlation'][0][1]==pytest.approx(np.sin(np.pi/3))
    rng=np.random.default_rng(180);z=rng.multivariate_normal([0,0],[[1,.5],[.5,1]],size=250);z/=np.sqrt(rng.chisquare(4,250)/4)[:,None]
    fit=fit_copula(z,variable_ids=['a','b'],family='t',input_kind='observations',tie_policy='error',df_bounds=[1,30],**META)
    assert 1<=fit.model['parameters']['df']<=30
    assert np.isfinite(fit.diagnostics['conditional_log_likelihood'])
    with pytest.raises(CreditRiskError,match='TIED_OBSERVATIONS'):
        fit_copula([[1,1],[1,2],[2,3]],variable_ids=['a','b'],family='gaussian',input_kind='observations',tie_policy='error',**META)


def test_copula_transforms_uniform_marginals_and_kendall_tau():
    from scipy.stats import kendalltau
    rng=np.random.default_rng(981);u=rng.uniform(size=(16000,2));aux=rng.uniform(size=(16000,2))
    for family,parameters,expected_tau in [('gaussian',dict(correlation=[[1,.5],[.5,1]]),1/3),('t',dict(correlation=[[1,.5],[.5,1]],df=4),1/3),('clayton',dict(theta=2),.5),('gumbel',dict(theta=2),.5)]:
        extra=None if family=='gaussian' else aux if family=='gumbel' else aux[:,:1]
        out=simulate_copula(frozen(family,**parameters),u,variable_ids=['a','b'],auxiliary_uniforms=extra).value
        assert np.max(abs(out.mean(axis=0)-.5))<.015
        assert np.max(abs((out<.1).mean(axis=0)-.1))<.012
        assert kendalltau(out[:,0],out[:,1]).statistic==pytest.approx(expected_tau,abs=.025)
    for family,parameters in [('clayton',dict(theta=0)),('gumbel',dict(theta=1))]:
        np.testing.assert_array_equal(simulate_copula(frozen(family,**parameters),u,variable_ids=['a','b']).value,u)
    singular=simulate_copula(frozen('gaussian',correlation=[[1,1],[1,1]]),u,variable_ids=['a','b']).value
    np.testing.assert_array_equal(singular[:,0],singular[:,1])

def test_extreme_parameters_never_return_successful_nonfinite_diagnostics():
    with pytest.raises(CreditRiskError,match='NUMERICAL_FAILURE'):
        validate_dependence_matrix([[1e308,1e308],[1e308,1e308]],['a','b'],matrix_kind='covariance')
    out=joint_default_probability(.1,.2,event_order=['a','b'],dependence_model=frozen('clayton',theta=1e308))
    assert out.value==pytest.approx(.1)
    with pytest.raises(CreditRiskError,match='NUMERICAL_FAILURE'):
        simulate_copula(frozen('clayton',theta=1e5),[[.2,.8]],variable_ids=['a','b'],auxiliary_uniforms=[[.5]])
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):
        simulate_copula(frozen('clayton',theta=0),[[.2,.8]],variable_ids=['a','b'],context={'probability_measure':'RISK_NEUTRAL'})
