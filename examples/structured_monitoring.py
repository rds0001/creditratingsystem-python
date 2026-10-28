"""Separate obligor default counts, claim losses and legal exposure concentrations."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('structured_monitoring.json').read_text())
    data=deepcopy(data);out={}
    for product,c in data['variants'].items():
        graph=cr.build_entity_graph(c['entities'],[],dict(policy_id=product+'-legal',version='1',edge_types=['CROSS_DEFAULT']),as_of='2024-01-16',known_at='2024-01-16')
        policy=dict(policy_id=product+'-group',version='1',edge_types=['CROSS_DEFAULT'],direction_policy='mutual_only',transitive=True)
        contract=dict(portfolio_id=product+'-holdings',version='1',currency='EUR',as_of='2024-01-16',known_at='2024-01-16',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='initially_performing',event='first_default')
        portfolio=cr.prepare_credit_portfolio(c['positions'],graph,policy,contract).value
        concentration=cr.credit_concentration_measures(c['positions'],weight_field='ead',group_fields=['entity_id'],weight_unit='EUR').value
        cohort=cr.build_monitoring_cohorts(**c['cohort_input']).value
        counts=cr.default_count_backtest(c['observation_panel'],c['frozen_forecasts'],['segment'],unit='obligor',horizon_days=30,default_definition=c['default_definition'],redefault_policy='exclude_previous_default',evaluation_cutoff='2024-05-02').value
        immature=cr.default_count_backtest(c['observation_panel'],c['frozen_forecasts'],['segment'],unit='obligor',horizon_days=30,default_definition=c['default_definition'],redefault_policy='exclude_previous_default',evaluation_cutoff='2024-02-02').value
        lgd=cr.lgd_backtest(c['loss_records'],design=c['loss_design'],workout_policy=dict(method='closed_only',reference='Explicit closed-claim descriptive cohort; unfinished workouts excluded'),uncertainty='none').value
        ead=None
        if c['ead_applicability']=='APPLICABLE':
            ead=cr.ead_backtest(c['loss_records'][:1],design=c['loss_design'],weighting_policy=dict(method='equal',reference='Fully funded claim balance at actual default'),uncertainty='none').value
        out[product]=dict(portfolio=portfolio,concentration=concentration,cohort=cohort,default_counts=counts,immature_default_counts=immature,lgd_backtest=lgd,ead_backtest=ead,ead_applicability=c['ead_applicability'],ead_inapplicability_reason=c['ead_inapplicability_reason'])
        if verify_reference:
            assert concentration['dimensions'][0]['hhi']==.25 and concentration['dimensions'][1]['hhi']==.375
            assert counts['deduplicated_rows']==1 and counts['overall']['observed_count']==3 and counts['overall']['observed_defaults']==2
            assert counts['overall']['expected_defaults']==1.5 and math.isclose(counts['overall']['observed_expected_ratio'],4/3)
            assert immature['overall']['observed_count']==0 and immature['overall']['expected_defaults']==0
            assert lgd['coverage']['included_count']==1 and lgd['coverage']['exclusion_counts']=={'OPEN_WORKOUT':1}
            assert math.isclose(lgd['metrics']['overall']['count_weighted']['bias'],c['expected_lgd_bias'],abs_tol=1e-12)
            if ead is not None:
                assert ead['metrics']['ead']['overall']['selected_weighting']['bias']==0
                assert ead['diagnostics']['ccf_undefined_count']==1
    return dict(data_origin=data['data_origin'],version=data['version'],product=data['product'],variants=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
