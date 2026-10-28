import math
import pytest
from creditratingsystem._validation_stability import population_stability
from creditratingsystem.core import CreditRiskError

def fixture():
 d=dict(model_id='monitor',model_version='1',parameter_version='1',data_fingerprint='synthetic-populations',population_id='applicants',split_id='future',prediction_origin='EXTERNAL',evaluation_cutoff='2024-01-01',reference_as_of='2020-01-01',current_as_of='2021-01-01',target=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing'))
 bins=dict(binning_id='frozen',version='1',known_at='2019-01-01',kind='numeric',cutpoints=[1],reference_support=[0,2]);return d,bins

def test_distances_exact_hand_values_missingness_and_zero_cells():
 d,bins=fixture();r=population_stability([0,0,2,2],[0,2,2,2],binning=bins,metrics=['psi','jsd','wasserstein'],smoothing=0,design=d).value
 assert r['metrics']['psi']['value']==pytest.approx(.25*math.log(3));assert r['metrics']['wasserstein']['value']==pytest.approx(.5)
 expected=.5*(.5*math.log(.5/.375)+.5*math.log(.5/.625)+.25*math.log(.25/.375)+.75*math.log(.75/.625));assert r['metrics']['jsd']['value']==pytest.approx(expected)
 r=population_stability([0,None],[2,2],binning=bins,metrics=['psi','jsd','wasserstein'],smoothing=0,design=d).value
 assert r['metrics']['psi']['status']=='INFINITE'and r['metrics']['psi']['value']is None;assert r['coverage']['missing_share_change']==-.5;assert r['metrics']['jsd']['value']==pytest.approx(math.log(2));assert r['metrics']['wasserstein']['value']==2

def test_identical_categorical_unseen_and_unsupported_populations():
 d,bins=fixture();bins.update(kind='categorical',levels=['a','b']);r=population_stability(['a',None,'new'],['a',None,'new'],binning=bins,metrics=['psi','jsd'],smoothing=.5,design=d).value
 assert all(v['value']==0 for v in r['metrics'].values());assert r['coverage']['current_novel_count']==1
 r=population_stability([],['a'],binning=bins,metrics=['psi','jsd'],smoothing=.5,design=d).value;assert r['applicability']=='NOT_APPLICABLE'
 with pytest.raises(CreditRiskError):population_stability(['a'],['b'],binning=bins,metrics=['wasserstein'],smoothing=0,design=d)
 bins['known_at']='2020-02-01'
 with pytest.raises(CreditRiskError):population_stability(['a'],['b'],binning=bins,metrics=['psi'],smoothing=0,design=d)
