from copy import deepcopy
import pytest
from creditratingsystem._validation_system import validate_system_coverage, _CELL_EVIDENCE
from creditratingsystem._validation_release import release_candidate_assessment
from creditratingsystem._validation_report import validation_report
from creditratingsystem.core import _fingerprint, CreditRiskError


def coverage_fixture():
 axes=dict(customer_segment='household',product='mortgage',lifecycle='application',method='scorecard',granularity='household')
 evidence=[dict(evidence_id=kind+'/'+lang,cell_id='c',kind=kind,language=lang,status='PASS',reference='independent-unit-fixture',content_hash='a'*64)for kind in _CELL_EVIDENCE for lang in (['both']if kind=='parity'else['python','r'])]
 cell=dict(cell_id='c',axes=axes,applicability='REQUIRED',contracts={k:'explicit '+k for k in ['data','target','result','measure','horizon','model_profile','policy_profile']},function_ids=['fit'],journey_ids=['j'],evidence_ids=[e['evidence_id']for e in evidence])
 return dict(cells=[cell],function_inventory=[dict(function_id='fit',status='IMPLEMENTED',languages=['python','r'])],journeys=[dict(journey_id='j',function_ids=['fit'])],evidence=evidence,scope=dict(scope_id='declared',version='1',reference='test-obligations',axes={k:[v]for k,v in axes.items()},required_cell_ids=['c']))


def release_fixture():
 candidate=dict(candidate_id='candidate1',artifact_hashes=dict(python_wheel='a'*64,python_sdist='b'*64,native_r='c'*64),resource_hashes={'data':'d'*64})
 requirements=dict(requirements_id='gate',version='1',reference='tiny-gate',required_cell_ids=['c'],required_journey_ids=['j'],required_generator_ids=['g'],required_profile_ids=['p'],resource_hashes={'data':'d'*64},limits=dict(max_runtime_seconds=10,max_peak_memory_mb=128))
 evidence=[]
 def add(scope,identity,kind,language):
  evidence.append(dict(evidence_id='/'.join([scope,identity,kind,language]),scope_type=scope,scope_id=identity,kind=kind,language=language,status='PASS',reference='illustrative-unit-evidence-not-real-release',content_hash='e'*64,data_origin='SYNTHETIC',**deepcopy(candidate),executed=True,installed=True,offline=True,source_tree_access=False,runtime_seconds=.1,peak_memory_mb=32))
 for kind in _CELL_EVIDENCE:
  for lang in (['both']if kind=='parity'else['python','r']):add('CELL','c',kind,lang)
 for scope,identity in [('JOURNEY','j'),('GENERATOR','g'),('PROFILE','p')]:
  for runtime in candidate['artifact_hashes']:add(scope,identity,'installed_execution',runtime)
 for runtime in candidate['artifact_hashes']:add('PACKAGE',runtime,'installed_execution',runtime)
 return dict(candidate=candidate,evidence=evidence,requirements=requirements)


def report_fixture():
 manifests=dict(evaluation_started_at='2024-01-02',model_manifest={'model':'a'*64},data_manifest={'data':'b'*64},split_manifest={'split':'c'*64})
 result=dict(artifact_type='ValidationResult',applicability='APPLICABLE',decision={'status':'NOT_ASSESSED'},metrics=dict(brier=.125,auc=None))
 rows=[dict(result_id='v',result=result,reference='independent-two-observation-brier',data_origin='SYNTHETIC',manifest_hashes={k:_fingerprint(manifests[k])for k in ['model_manifest','data_manifest','split_manifest']})]
 policy=dict(policy_id='predeclared',version='1',reference='fixture-policy',declared_at='2024-01-01',required_result_ids=['v'],criteria=[dict(criterion_id='brier',result_id='v',path=['metrics','brier'],operator='LE',threshold=.2,rationale='Test threshold only')])
 return dict(results=rows,acceptance_policy=policy,manifests=manifests)


def test_explicit_coverage_missing_native_negative_routes_and_no_waiver():
 kw=coverage_fixture();r=validate_system_coverage(**kw).value
 assert r['decision']['status']=='PASS'and r['metrics']['complete_required_cells']==1 and not r['cartesian_completeness_claimed']
 kw['evidence']=[e for e in kw['evidence']if e['evidence_id']!='negative_routing/r'];r=validate_system_coverage(**kw).value
 assert r['decision']['status']=='BLOCKED'and r['violation_counts']['MISSING_PASS_EVIDENCE']==1
 kw=coverage_fixture();kw['cells'][0].update(applicability='NOT_APPLICABLE',applicability_decision=dict(reason='unimplemented',reason_code='NOT_IMPLEMENTED',version='1',authority='owner',reference='x'));r=validate_system_coverage(**kw).value
 assert set(r['violation_counts'])=={'REQUIRED_SCOPE_DOWNGRADED','INVALID_NONAPPLICABILITY_REASON'}
 kw=coverage_fixture();kw['scope']['axes']['granularity'].append('account_month');r=validate_system_coverage(**kw).value
 assert r['violation_counts']['UNCONSIDERED_AXIS_LEVEL']==1


def test_release_exact_binding_installed_execution_budgets_and_separate_axes():
 kw=release_fixture();r=release_candidate_assessment(**kw).value
 assert r['decision']['status']=='PASS'and r['metrics']['required_gate_count']==27
 assert r['assessment_axes']['empirical_model_performance']=='NOT_ASSESSED'and not r['publication_performed']and not r['tests_executed_by_assessor']
 kw['evidence'][-1].update(source_tree_access=True,peak_memory_mb=129);kw['evidence'][0]['artifact_hashes']['python_wheel']='f'*64;r=release_candidate_assessment(**kw).value
 assert {'WRONG_CANDIDATE_EVIDENCE','MISSING_REQUIRED_EVIDENCE','INVALID_INSTALLED_EXECUTION','EXECUTION_BUDGET_UNPROVEN'}<=set(r['violation_counts'])
 kw=release_fixture();kw['evidence'][0]['status']='NOT_APPLICABLE';assert release_candidate_assessment(**kw).value['decision']['status']=='BLOCKED'


def test_report_predeclared_threshold_missing_na_and_manifest_binding():
 kw=report_fixture();r=validation_report(**kw).value
 assert r['decision']['status']=='PASS'and r['metrics']['passed_criteria']==1 and not r['p_values_combined']
 kw['acceptance_policy']['criteria'][0]['path']=['metrics','auc'];r=validation_report(**kw).value
 assert r['findings'][0]['status']=='NOT_ESTIMABLE'and r['decision']['status']=='BLOCKED'
 kw=report_fixture();kw['results'][0]['result']['applicability']='NOT_APPLICABLE';kw['results'][0]['manifest_hashes']['data_manifest']='f'*64;r=validation_report(**kw).value
 assert {'NONPASS_REQUIRED_RESULT','RESULT_MANIFEST_MISMATCH'}<=set(r['violation_counts'])
 kw=report_fixture();kw['acceptance_policy']['declared_at']='2024-01-03'
 with pytest.raises(CreditRiskError):validation_report(**kw)


def test_common_currency_design_guard():
 from creditratingsystem._validation_common import _vb_design
 d=dict(model_id='m',model_version='1',parameter_version='1',data_fingerprint='x',population_id='p',split_id='s',prediction_origin='HOLDOUT',evaluation_cutoff='2024-01-01',currency='EUR',target=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing'))
 assert _vb_design(d,{'currency':'EUR'})['currency']=='EUR'
 with pytest.raises(CreditRiskError):_vb_design(d,{'currency':'USD'})
