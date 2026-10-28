import math
from copy import deepcopy
import numpy as np
import pytest
from creditratingsystem.features import (fit_preprocessor,transform_features,fit_monotone_bins,compute_woe_iv,build_financial_ratios,build_distributed_lags)
from creditratingsystem.core import CreditRiskError,serialize_fitted_model,load_fitted_model
TARGET=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
TRAINING=dict(data_fingerprint='training-only-test',split_id='train',known_at='2025-01-01',data_origin='SYNTHETIC')
META=dict(target=TARGET,training=TRAINING,model_id='features',model_version='1')
ROWS=[{'x':1,'category':'b'},{'x':None,'category':None},{'x':3,'category':'a'}]
SCHEMA=[dict(name='x',type='numeric',imputation='median',scaling='standard',missing_indicator=True),dict(name='category',type='categorical',missing_policy='separate',unknown_policy='separate',drop_reference='a')]

def test_preprocessing_train_only_batch_single_and_unknown():
 fitted=fit_preprocessor(ROWS,SCHEMA,**META)
 before=deepcopy(fitted.model)
 query=[{'x':100,'category':'never_seen'},{'x':None,'category':None}]
 result=transform_features(query,fitted).value
 np.testing.assert_allclose(result['design_matrix'],[[98/math.sqrt(2/3),0,0,0,1],[0,1,0,1,0]])
 np.testing.assert_array_equal(np.vstack([transform_features([row],fitted).value['design_matrix'] for row in query]),result['design_matrix'])
 assert fitted.model==before
 assert fitted.model['parameters']['feature_specs'][0]['fill_value']==2
 with pytest.raises(CreditRiskError,match='MISSING_FEATURE'):transform_features([{'x':1}],fitted)

def test_constant_robust_scaler_and_missing_category():
 fit=fit_preprocessor([{'x':2},{'x':2}],[dict(name='x',type='numeric',imputation='error',scaling='robust')],**META)
 assert fit.model['parameters']['feature_specs'][0]['constant_training_scale']
 assert transform_features([{'x':3}],fit).value['design_matrix'][0,0]==1
 with pytest.raises(CreditRiskError,match='NO_OBSERVED_VALUES'):fit_preprocessor([{'x':None}],[dict(name='x',type='numeric',imputation='median',scaling='none')],**META)
 fit=fit_preprocessor([{'c':None}],[dict(name='c',type='categorical',missing_policy='separate',unknown_policy='separate')],**META)
 np.testing.assert_array_equal(transform_features([{'c':None},{'c':'new'}],fit).value['design_matrix'],[[1,0],[0,1]])

def test_model_serialization_replays_frozen_features(tmp_path):
 fitted=fit_preprocessor(ROWS,SCHEMA,**META);path=tmp_path/'preprocessor.json'
 serialize_fitted_model(fitted,path);loaded=load_fitted_model(path)
 np.testing.assert_array_equal(transform_features(ROWS,loaded).value['design_matrix'],transform_features(ROWS,fitted).value['design_matrix'])

def test_monotone_bins_ties_pava_missing_and_support():
 fit=fit_monotone_bins([1,1,2,3,4,5,None],[0,0,1,1,0,1,1],feature_name='x',direction='increasing',min_bin_count=1,max_prebins=6,**META)
 assert fit.model['parameters']['upper_boundaries']==[1.,4.,None]
 assert [b['default_rate'] for b in fit.model['parameters']['bins']]==[0,2/3,1,1]
 np.testing.assert_array_equal(transform_features([{'x':-100},{'x':1},{'x':4},{'x':100},{'x':None}],fit).value['design_matrix'].ravel(),[0,0,1,2,3])
 support=fit_monotone_bins(list(range(8)),[1,0,1,0,1,0,1,0],feature_name='x',direction='decreasing',min_bin_count=3,max_prebins=8,**META)
 bins=support.model['parameters']['bins'][:-1]
 assert all(b['count']>=3 for b in bins)
 assert all(a['default_rate']>=b['default_rate'] for a,b in zip(bins,bins[1:]))

def test_woe_iv_arithmetic_and_sign():
 rows=[dict(bin_id='a',defaults=1,nondefaults=3),dict(bin_id='b',defaults=3,nondefaults=1)]
 first=compute_woe_iv(rows,smoothing=0,woe_sign='nondefault_over_default').value
 second=compute_woe_iv(rows,smoothing=0,woe_sign='default_over_nondefault').value
 assert first['iv']==pytest.approx(math.log(3))
 assert first['bins'][0]['woe']==pytest.approx(math.log(3))
 assert second['iv']==first['iv']
 assert second['bins'][0]['woe']==-first['bins'][0]['woe']
 with pytest.raises(CreditRiskError,match='ZERO_CLASS_CELL'):compute_woe_iv([dict(bin_id='a',defaults=0,nondefaults=3),dict(bin_id='b',defaults=3,nondefaults=1)],smoothing=0,woe_sign='nondefault_over_default')

def test_ratios_denominator_status_and_units():
 rows=[dict(entity_id='a',as_of='2024-12-31',period_start='2024-01-01',period_end='2024-12-31',currency='EUR',values={'cf':12,'interest':2,'principal':4},units={'cf':'EUR','interest':'EUR','principal':'EUR'})]
 specs=[dict(name='dscr',numerator={'cf':1},denominator={'interest':1,'principal':1},denominator_policy='positive',output_unit='dimensionless')]
 assert build_financial_ratios(rows,specs).value[0]['ratios']['dscr']['value']==2
 rows[0]['values'].update(interest=0,principal=0)
 assert build_financial_ratios(rows,specs).value[0]['ratios']['dscr']['status']=='ZERO_DENOMINATOR'
 rows[0]['units']['principal']='USD'
 with pytest.raises(CreditRiskError,match='INCOMPATIBLE_UNIT'):build_financial_ratios(rows,specs)

def test_exact_calendar_lags_and_vintage_no_lookahead():
 series=[dict(entity_id='a',feature='x',observed_at='2024-02-29',available_at='2024-03-01',known_at='2024-03-01',value=10),dict(entity_id='a',feature='x',observed_at='2024-02-29',available_at='2024-04-02',known_at='2024-04-02',value=999),dict(entity_id='a',feature='x',observed_at='2024-03-31',available_at='2024-03-31',known_at='2024-03-31',value=12)]
 observations=[dict(entity_id='a',as_of='2024-03-31',known_at='2024-03-31')]
 specs=[dict(name='lag1',feature='x',lag=1,operation='level'),dict(name='change',feature='x',lag=1,operation='relative_change'),dict(name='gap',feature='x',lag=2,operation='level')]
 result=build_distributed_lags(series,observations,specs,frequency='months').value[0]
 assert result['features']['lag1']==10
 assert result['features']['change']==pytest.approx(.2)
 assert result['features']['gap'] is None
 assert not result['eligible']
