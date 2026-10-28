"""Dated model/rating journeys with hand-defined semantic and numerical oracles."""
from copy import deepcopy
import math
import unittest

from creditratingsystem.core import CreditRiskError
from creditratingsystem import rating_system as rs


def model(version='v1', coefficient=0):
    return {'schema_version':'1.0','model_id':'M','model_version':version,'method_id':'classification.fit_binary_glm','feature_order':['x'],'parameters':{'link':'logit','fit_intercept':True,'coefficients':[coefficient,1],'offset_required':False},'preprocessing':{'transformation':'identity'},'target':{'event':'first_default','horizon':1,'time_unit':'years','probability_measure':'PHYSICAL','conditioning':'PIT'},'training':{'data_fingerprint':'reference','split_id':'train','known_at':'2023-12-01'},'diagnostics':{},'provenance':{'data_origin':'REFERENCE','parameter_origin':'USER_SUPPLIED'}}


def registry():
    return rs.register_rating_model_version(None,{'model':model(),'dependencies':{},'configuration':{}},{'model_id':'M','model_version':'v1','registered_at':'2023-12-15'}).value


def snapshot():
    return {'subject_id':'O','subject_type':'obligor','as_of':'2024-01-01','known_at':'2024-01-02','attributes':{'income':{'value':2000,'observed_at':'2023-12-31','known_at':'2024-01-01'},'product':{'value':'consumer','observed_at':'2024-01-01','known_at':'2024-01-01'}}}


def rules():
    base={'profile_version':'1','priority':1,'effective_from':'2024-01-01','effective_to':'2025-01-01','known_at':'2023-12-01'}
    return [dict(base,rule_id='low',profile_id='P-low',conditions=[{'field':'income','operator':'lt','value':2000}]),dict(base,rule_id='high',profile_id='P',conditions=[{'field':'income','operator':'ge','value':2000}])]


def composition():
    return {'assessment_id':'A','subject':{'subject_id':'O','subject_type':'obligor'},'components':[{'component_id':'PD-base','quantity':'PD','value':0.1,'subject_id':'O','subject_type':'obligor','variant':'standalone','definition':model()['target']}],'binding':{'profile_id':'P','profile_version':'1','model_id':'M','model_version':'v1','model_hash':registry()['entries'][0]['model_hash'],'input_hash':'input-content-hash'},'as_of':'2024-01-01','known_at':'2024-01-02','calculated_at':'2024-01-03'}


def request(identity='row1'):
    return {'row_id':identity,'known_at':'2024-01-02','model_id':'M','model_version':'v1','model_hash':registry()['entries'][0]['model_hash'],'recipe':{'method':'binary_glm','version':'1.0','parameters':{'expected_target':model()['target']}},'input':{'X':[[0],[1]],'feature_order':['x']}}


class RatingSystemTests(unittest.TestCase):
    def code(self,code,fn,*args,**kwargs):
        with self.assertRaises(CreditRiskError) as caught:fn(*args,**kwargs)
        self.assertEqual(caught.exception.code,code)

    def test_subject_graph_keeps_customer_obligor_and_facility_distinct(self):
        base={'version':'1','effective_from':'2023-01-01','effective_to':None,'known_at':'2023-01-01','attributes':{}}
        tables={'customer':[dict(base,subject_id='C')],'obligor':[dict(base,subject_id='O1'),dict(base,subject_id='O2')],'facility':[dict(base,subject_id='F')]}
        edge={'effective_from':'2023-01-01','effective_to':None,'known_at':'2023-01-01','relation':'legal_obligor','from_type':'customer','from_id':'C','to_type':'obligor'}
        links=[dict(edge,link_id='1',to_id='O1'),dict(edge,link_id='2',to_id='O2'),dict(edge,link_id='3',to_id='MISSING')]
        out=rs.build_rating_subjects(tables,links,'2024-01-01','2024-01-02').value
        self.assertEqual(len(out['subjects']),4);self.assertEqual(len(out['links']),2);self.assertEqual(out['unresolved_links'],[{'link_id':'3','code':'UNRESOLVED_ENDPOINT'}])
        tables['obligor'].append(dict(base,subject_id='O1',version='2'))
        out=rs.build_rating_subjects(tables,links,'2024-01-01','2024-01-02').value
        self.assertEqual(out['conflicts'][0]['code'],'OVERLAPPING_VERSIONS')
        self.assertFalse(any(n['subject_id']=='O1' for n in out['subjects']))

    def test_route_exact_threshold_dates_ties_and_future_features(self):
        out=rs.route_rating_profile(snapshot(),rules(),'2024-01-01','2024-01-02').value
        self.assertEqual(out['profile_id'],'P');self.assertEqual(out['candidate_rule_ids'],['high'])
        tied=rules()+[dict(rules()[1],rule_id='tie')]
        self.assertEqual(rs.route_rating_profile(snapshot(),tied,'2024-01-01','2024-01-02').value['status'],'AMBIGUOUS')
        future=snapshot();future['attributes']['income']['known_at']='2024-01-03'
        self.assertEqual(rs.route_rating_profile(future,rules(),'2024-01-01','2024-01-02').value['status'],'NO_ROUTE')
        expired=snapshot();expired['as_of']='2025-01-01';expired['known_at']='2025-01-01'
        self.assertEqual(rs.route_rating_profile(expired,rules(),'2025-01-01','2025-01-01').value['status'],'NO_ROUTE')

    def test_applicability_missing_is_refer_and_unsupported_product_inapplicable(self):
        c={'contract_id':'C','version':'1','required_features':['income'],'allowed_values':{'product':['consumer']},'support_bounds':{'income':[1000,5000]},'maximum_age_days':{'income':30}}
        self.assertEqual(rs.assess_rating_applicability(snapshot(),c,'2024-01-01','2024-01-02').value['status'],'APPLICABLE')
        s=snapshot();s['attributes']['income']['value']=None
        self.assertEqual(rs.assess_rating_applicability(s,c,'2024-01-01','2024-01-02').value['status'],'REFER')
        s['attributes']['product']['value']='mortgage'
        self.assertEqual(rs.assess_rating_applicability(s,c,'2024-01-01','2024-01-02').value['status'],'NOT_APPLICABLE')

    def test_registry_content_integrity_idempotence_and_conflict(self):
        r=registry();copy=deepcopy(r);bundle=r['entries'][0]['bundle'];registration=r['entries'][0]['registration']
        out=rs.register_rating_model_version(r,bundle,registration)
        self.assertTrue(out.diagnostics['idempotent']);self.assertEqual(out.value,r)
        changed=deepcopy(bundle);changed['model']['parameters']['coefficients'][0]=10
        self.code('RATING_VERSION_CONFLICT',rs.register_rating_model_version,r,changed,registration)
        r['entries'][0]['bundle']['model']['parameters']['coefficients'][0]=10
        self.code('RATING_INTEGRITY',rs.select_effective_model,r,[],'P','PD','2024-01-01','2024-01-02')
        self.assertEqual(copy['entries'][0]['bundle']['model']['parameters']['coefficients'][0],0)

    def test_effective_assignment_is_explicit_not_latest_registered(self):
        r=registry();r=rs.register_rating_model_version(r,{'model':model('v2',2),'dependencies':{},'configuration':{}},{'model_id':'M','model_version':'v2','registered_at':'2024-01-01'}).value
        entry=r['entries'][0]
        a={'assignment_id':'old','profile_id':'P','purpose':'PD','model_id':'M','model_version':'v1','model_hash':entry['model_hash'],'effective_from':'2024-01-01','effective_to':'2024-06-01','known_at':'2023-12-15'}
        self.assertEqual(rs.select_effective_model(r,[a],'P','PD','2024-01-01','2024-01-02').value['selected']['model_version'],'v1')
        self.assertEqual(rs.select_effective_model(r,[a],'P','PD','2024-06-01','2024-06-01').value['status'],'EXPIRED_MODEL')
        self.assertEqual(rs.select_effective_model(r,[a],'P','PD','2023-12-20','2023-12-20').value['status'],'NOT_YET_EFFECTIVE')

    def test_composition_preserves_variants_and_definition_conflicts(self):
        args=composition();c=deepcopy(args['components'][0]);c.update(component_id='PD-support',variant='supported',value=0.05);args['components'].append(c)
        args['effects']=[{'effect_id':'support','kind':'support','order':1,'level':'obligor','input_component_id':'PD-base','output_component_id':'PD-support','reason':'Explicit separate support calculation'}]
        a=rs.compose_risk_assessment(**args).value
        self.assertEqual(a['composition_status'],'CONSISTENT');self.assertEqual(a['components'][0]['value'],0.1)
        bad=deepcopy(args['components'][0]);bad.update(component_id='EL',quantity='EL',value=100,currency='EUR');bad['definition']['horizon']=2;args['components'].append(bad)
        a=rs.compose_risk_assessment(**args).value
        self.assertEqual(a['composition_status'],'CONFLICT');self.assertIn('TARGET_DEFINITION_CONFLICT',[c['code'] for c in a['conflicts']])
        args['effects'].append({'effect_id':'cycle','kind':'override','order':2,'level':'obligor','input_component_id':'PD-support','output_component_id':'PD-base','reason':'Invalid cycle'})
        self.code('RATING_EFFECT_CONFLICT',rs.compose_risk_assessment,**args)

    def test_freshness_age_boundary_and_late_known_event(self):
        a=rs.compose_risk_assessment(**composition()).value;p={'maximum_age_days':30,'maximum_information_age_days':30,'trigger_types':['income_change']}
        self.assertEqual(rs.check_assessment_freshness(a,p,[],'2024-01-31','2024-01-31').value['status'],'CURRENT')
        self.assertEqual(rs.check_assessment_freshness(a,p,[],'2024-02-01','2024-02-01').value['status'],'STALE')
        e={'event_id':'E','subject_id':'O','type':'income_change','occurred_at':'2023-12-31','known_at':'2024-01-05'}
        self.assertEqual(rs.check_assessment_freshness(a,p,[e],'2024-01-04','2024-01-04').value['status'],'CURRENT')
        self.assertEqual(rs.check_assessment_freshness(a,p,[e],'2024-01-05','2024-01-05').value['status'],'STALE')

    def test_lifecycle_gates_preserve_prior_assessment_hash(self):
        a=rs.compose_risk_assessment(**composition()).value
        transitions=[{'from_state':'DRAFT','to_state':'REVIEWED','event_type':'review','required_gates':['checked']},{'from_state':'REVIEWED','to_state':'FINAL','event_type':'finalize','required_gates':['approved_local']},{'from_state':'FINAL','to_state':'SUPERSEDED','event_type':'replace','required_gates':[]}]
        e={'event_id':'E','type':'review','occurred_at':'2024-01-04','known_at':'2024-01-04','reason':'Source evidence checked','gates':{'checked':False}}
        self.code('RATING_GATE',rs.transition_assessment_state,a,e,transitions)
        e['gates']['checked']=True;b=rs.transition_assessment_state(a,e,transitions).value
        self.assertEqual(a['state'],'DRAFT');self.assertEqual(b['state'],'REVIEWED');self.assertEqual(b['history'][0]['previous_assessment_hash'],a['assessment_hash'])
        e.update(event_id='F',type='finalize',gates={'approved_local':True});c=rs.transition_assessment_state(b,e,transitions).value
        e.update(event_id='S',type='replace',gates={});d=rs.transition_assessment_state(c,e,transitions).value
        self.assertEqual(c['state'],'FINAL');self.assertEqual(d['state'],'SUPERSEDED');self.assertEqual(len(d['history']),3)

    def test_batch_stable_order_errors_and_analytic_pd(self):
        one=request('B');two=request('A');bad=request('C');bad['model_hash']='wrong'
        a=rs.rate_subject_batch([one,two,bad],registry(),'batch','2024-01-03').value
        b=rs.rate_subject_batch([bad,two,one],registry(),'batch','2024-01-03').value
        self.assertEqual(a,b);self.assertEqual([r['row_id'] for r in a['rows']],['A','B','C']);self.assertEqual(a['rows'][0]['value'][0],0.5)
        self.assertAlmostEqual(a['rows'][0]['value'][1],1/(1+math.exp(-1)),places=15)
        self.assertEqual(a['rows'][2]['error_code'],'RATING_INTEGRITY');self.assertIsNone(a['rows'][2]['value'])

    def test_application_recipe_uses_frozen_snapshot_and_explicit_model(self):
        from creditratingsystem.application import build_application_snapshot, predict_application_risk
        m=model();m['target']['performance_origin']='first_draw_at'
        m['application_contract']={'product_ids':['consumer'],'population_ids':['new'],'performance_origin':'first_draw_at','term_stage':'requested','term_features':{}}
        r=rs.register_rating_model_version(None,{'model':m,'dependencies':{},'configuration':{}},{'model_id':'M','model_version':'v1','registered_at':'2023-12-15'}).value
        apps=[{'application_id':'A','customer_id':'C','product_id':'consumer','population_id':'new','application_at':'2024-01-01','information_cutoff':'2024-01-01','requested_terms':{}}]
        facts=[{'application_id':'A','feature':'x','value':0,'observed_at':'2024-01-01','known_at':'2024-01-01','source_id':'reference'}]
        s=build_application_snapshot(apps,facts).value[0]
        q={'row_id':'A','known_at':'2024-01-01','model_id':'M','model_version':'v1','model_hash':r['entries'][0]['model_hash'],'input':{'snapshot':s},'recipe':{'method':'application_risk','version':'1.0','parameters':{'target':m['target']}}}
        batch=rs.rate_subject_batch([q],r,'applications','2024-01-02').value
        self.assertEqual(batch['rows'][0]['value'],predict_application_risk(s,m,m['target']).value)
        self.assertEqual(batch['rows'][0]['value']['pd'],0.5)
        q['known_at']='2023-12-31'
        self.assertEqual(rs.rate_subject_batch([q],r,'bad','2024-01-02').value['rows'][0]['error_code'],'RATING_REQUEST_UNAVAILABLE')

    def test_event_idempotence_and_explicit_correction(self):
        e={'event_id':'payment','revision':'1','occurred_at':'2024-01-01','known_at':'2024-01-02','dependencies':[{'type':'facility','id':'F'}],'payload':{'amount':100}}
        deps=[{'assessment_id':'A','dependency_type':'facility','dependency_id':'F'},{'assessment_id':'B','dependency_type':'facility','dependency_id':'G'}]
        a=rs.apply_assessment_event(None,e,deps,'2024-01-02').value
        self.assertEqual(a['affected_assessment_ids'],['A'])
        b=rs.apply_assessment_event(a['state'],e,deps,'2024-01-02').value
        self.assertEqual(a['state'],b['state']);self.assertEqual(b['status'],'DUPLICATE')
        wrong=deepcopy(e);wrong['payload']['amount']=10
        self.code('RATING_EVENT_CONFLICT',rs.apply_assessment_event,a['state'],wrong,deps,'2024-01-02')
        wrong.update(revision='2',corrects_revision='1',known_at='2024-01-03')
        c=rs.apply_assessment_event(a['state'],wrong,deps,'2024-01-03').value
        self.assertEqual(len(c['state']['processed']),2);self.assertEqual(c['state']['processed'][0]['event']['payload']['amount'],100)

    def test_replay_pinned_model_survives_champion_change_missing_model_explicit(self):
        r=registry();q=request();batch=rs.rate_subject_batch([q],r,'original','2024-01-03').value
        newer=rs.register_rating_model_version(r,{'model':model('v2',10),'dependencies':{},'configuration':{}},{'model_id':'M','model_version':'v2','registered_at':'2024-06-01'}).value
        replay=rs.replay_assessment_history([q],newer,batch['rows'],'2024-01-03').value
        self.assertEqual(replay['status'],'REPRODUCED');self.assertEqual(replay['batch']['rows'][0]['value'],batch['rows'][0]['value'])
        missing=rs.replay_assessment_history([q],None,batch['rows'],'2024-01-03').value
        self.assertEqual(missing['status'],'NOT_REPRODUCIBLE');self.assertEqual(missing['findings'][0]['code'],'RATING_MISSING_MODEL')
        changed=deepcopy(q);changed['input']['X'][0][0]=2
        replay=rs.replay_assessment_history([changed],r,batch['rows'],'2024-01-03').value
        self.assertEqual(replay['findings'][0]['fields'],['input_hash','result_hash'])

if __name__=='__main__':unittest.main()
