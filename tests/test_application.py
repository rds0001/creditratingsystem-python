"""Application business cases with independently calculated expectations."""
from copy import deepcopy
import math
import unittest

from creditratingsystem.core import CreditRiskError, _fingerprint
from creditratingsystem.application import (build_application_snapshot,assemble_joint_application,
    build_application_targets,predict_application_risk,compare_offer_risk)


def app(i='A'):
    return {'application_id':i,'customer_id':'C','product_id':'consumer','population_id':'new_customer','application_at':'2024-01-01','information_cutoff':'2024-01-01','requested_terms':{'amount':1000,'currency':'EUR'}}


def target():
    return {'event':'first_default','horizon':30,'time_unit':'days','probability_measure':'PHYSICAL','conditioning':'funded','performance_origin':'first_draw_at'}


def model():
    return {'schema_version':'1.0','model_id':'conditional-logit','model_version':'v1','method_id':'classification.fit_binary_glm','feature_order':['income','offer_amount'],'parameters':{'link':'logit','fit_intercept':True,'coefficients':[0,-1,0.001],'offset_required':False},'preprocessing':{'transformation':'identity'},'target':target(),'training':{'data_fingerprint':'reference','split_id':'train','known_at':'2023-12-31'},'diagnostics':{},'provenance':{'data_origin':'REFERENCE','parameter_origin':'USER_SUPPLIED'},'application_contract':{'product_ids':['consumer'],'population_ids':['new_customer'],'performance_origin':'first_draw_at','term_stage':'offered','term_features':{'amount':'offer_amount'}}}


def snapshot():
    features=[{'application_id':'A','feature':'income','value':2,'observed_at':'2023-12-01','known_at':'2023-12-02','source_id':'payroll'}]
    terms=[{'application_id':'A','stage':'offered','terms':{'amount':1000},'effective_at':'2024-01-01','known_at':'2024-01-01'}]
    return build_application_snapshot([app()],features,terms,['income']).value[0]


class ApplicationTests(unittest.TestCase):
    def assert_code(self,code,fn,*args,**kwargs):
        with self.assertRaises(CreditRiskError) as caught:fn(*args,**kwargs)
        self.assertEqual(caught.exception.code,code)

    def test_snapshot_freezes_known_features_and_contract_stages(self):
        features=[{'application_id':'A','feature':'bureau','value':500,'observed_at':'2023-12-20','known_at':'2023-12-21','source_id':'v1'}, {'application_id':'A','feature':'bureau','value':800,'observed_at':'2023-12-20','known_at':'2024-01-02','source_id':'revision'}]
        terms=[{'application_id':'A','stage':'booked','terms':{'amount':9999},'effective_at':'2024-01-02','known_at':'2024-01-02'}]
        s=build_application_snapshot([app()],features,terms,['bureau','income']).value[0]
        self.assertEqual(s['features'],{'bureau':500})
        self.assertIsNone(s['terms']['booked'])
        self.assertEqual(s['terms']['requested']['amount'],1000)
        self.assertEqual(s['missing_features'],['income'])
        features[0]['value']=0
        self.assertEqual(s['features']['bureau'],500)
        self.assertEqual(s['snapshot_hash'],_fingerprint({k:v for k,v in s.items() if k!='snapshot_hash'}))
        self.assert_code('APPLICATION_AMBIGUOUS_FEATURE',build_application_snapshot,[app()],[features[0],dict(features[0],source_id='duplicate')])

    def test_joint_debts_and_income_counted_once_guarantor_excluded(self):
        parties=[{'party_id':'P','role':'borrower','liability_share':1,'household_id':'H'},{'party_id':'Q','role':'co_borrower','liability_share':1,'household_id':'H'},{'party_id':'G','role':'guarantor','liability_share':0.5}]
        incomes=[{'economic_id':'salary','party_id':'P','amount_monthly':3000,'currency':'EUR'},{'economic_id':'salary','party_id':'Q','amount_monthly':3000,'currency':'EUR'},{'economic_id':'guarantor_salary','party_id':'G','amount_monthly':10000,'currency':'EUR'}]
        debts=[{'economic_id':'mortgage','party_id':p,'amount_monthly':900,'currency':'EUR'} for p in ['P','Q']]
        v=assemble_joint_application('A',parties,incomes,debts,'EUR').value
        self.assertEqual(v['monthly_income'],3000)
        self.assertEqual(v['monthly_debt_service'],900)
        self.assertTrue(v['dependency_flags']['shared_obligation'])
        self.assertEqual(v['parties'][2]['role'],'guarantor')
        self.assertEqual(len(v['excluded_items']),1)
        debts[1]['amount_monthly']=901
        self.assert_code('APPLICATION_AMBIGUOUS_ECONOMIC_ITEM',assemble_joint_application,'A',parties,incomes,debts,'EUR')

    def test_selection_stages_and_first_funding_origin(self):
        apps=[app(i) for i in ['R','N','B','D','U','P']]
        decisions=[{'decision_id':'d'+i,'application_id':i,'status':'rejected' if i=='R' else 'approved','decided_at':'2024-01-02','known_at':'2024-01-02',**({'accepted_at':'2024-01-03','accepted_known_at':'2024-01-03'} if i not in ['R','N'] else {})} for i in ['R','N','B','D','U','P']]
        bookings=[{'application_id':i,'facility_id':'F'+i,'booked_at':'2024-01-04','known_at':'2024-01-04','first_draw_at':draw,'first_draw_known_at':draw,**({'outcome_available_at':'2024-03-15'} if i=='P' else {})} for i,draw in [('D','2024-01-10'),('U','2024-02-01'),('P','2024-01-10')]]
        events=[{'event_id':'default','facility_id':'FD','default_at':'2024-02-09','known_at':'2024-02-09'}]
        v=build_application_targets(apps,decisions,bookings,events,[],30,'2024-02-10').value
        self.assertEqual([r['selection_status'] for r in v],['rejected','approved_not_taken_up','accepted_not_booked','funded','funded','funded'])
        self.assertEqual([r['target'] for r in v],[None,None,None,1,None,None])
        self.assertEqual(v[3]['outcome_end'],'2024-02-09')
        self.assertEqual(v[3]['information_cutoff'],'2024-01-01')
        self.assertEqual(v[5]['label_status'],'outcome_pending')
        self.assertFalse(v[0]['label_observed'])

    def test_prepayment_not_nondefault_and_invalid_selection(self):
        a=[app()];d=[{'decision_id':'D','application_id':'A','status':'approved','decided_at':'2024-01-02','known_at':'2024-01-02','accepted_at':'2024-01-03','accepted_known_at':'2024-01-03'}]; b=[{'application_id':'A','facility_id':'F','booked_at':'2024-01-04','known_at':'2024-01-04','first_draw_at':'2024-01-05','first_draw_known_at':'2024-01-05'}]
        v=build_application_targets(a,d,b,[],[{'facility_id':'F','exit_at':'2024-01-20','known_at':'2024-01-20'}],30,'2024-03-01').value[0]
        self.assertIsNone(v['target']);self.assertEqual(v['label_status'],'exit')
        d[0]['status']='rejected'
        self.assert_code('APPLICATION_SELECTION_CONFLICT',build_application_targets,a,d,b,[],[],30,'2024-03-01')

    def test_prediction_contract_and_integrity(self):
        s=snapshot();m=model()
        r=predict_application_risk(s,m,target()).value
        self.assertAlmostEqual(r['pd'],1/(1+math.e),places=14) # eta = -2 + 1 = -1
        self.assertFalse(r['calibrated'])
        bad=deepcopy(m);bad['application_contract']['product_ids']=['mortgage']
        self.assert_code('APPLICATION_APPLICABILITY',predict_application_risk,s,bad,target())
        wrong=target();wrong['horizon']=365
        self.assert_code('APPLICATION_TARGET',predict_application_risk,s,m,wrong)
        altered=deepcopy(s);altered['features']['income']=100
        self.assert_code('APPLICATION_SNAPSHOT_INTEGRITY',predict_application_risk,altered,m,target())
        empty=build_application_snapshot([app()],[],[{'application_id':'A','stage':'offered','terms':{'amount':1000},'effective_at':'2024-01-01','known_at':'2024-01-01'}]).value[0]
        self.assert_code('APPLICATION_MISSING_FEATURES',predict_application_risk,empty,m,target())

    def test_calibration_pipeline_and_chronology(self):
        from creditratingsystem.calibration import calibrate_intercept
        meta={'target':target(),'base_model_version':'v1','population_definition':'new_customer','outcome_policy':'complete funded outcomes','weight_type':'frequency','data_origin':'REFERENCE','calibration':{'start':'2023-01-01','end':'2023-06-30','known_at':'2023-12-31','split_id':'cal','prediction_origin':'EXTERNAL_TARGET'}}
        base=1/(1+math.e)
        calibrator=calibrate_intercept([base,base],0.4,metadata=meta)
        risk=predict_application_risk(snapshot(),model(),target(),calibrator).value
        self.assertAlmostEqual(risk['pd'],0.4,places=11)
        self.assertAlmostEqual(risk['unadjusted_pd'],base,places=14)
        self.assertTrue(risk['calibrated'])
        future=app();future['application_at']='2025-01-01';future['information_cutoff']='2025-01-01'
        self.assert_code('APPLICATION_COHORT_CUTOFF',build_application_targets,[future],[],[],[],[],30,'2024-03-01')
        old_decision={'decision_id':'D','application_id':'A','status':'rejected','decided_at':'2023-12-31','known_at':'2023-12-31'}
        self.assert_code('APPLICATION_DATE_ORDER',build_application_targets,[app()],[old_decision],[],[],[],30,'2024-03-01')

    def test_unavailable_information_cutoff_and_nonapproved_acceptance(self):
        a=app();a['information_cutoff']='2024-04-01'
        self.assert_code('APPLICATION_COHORT_CUTOFF',build_application_targets,[a],[],[],[],[],30,'2024-03-01')
        d={'decision_id':'D','application_id':'A','status':'rejected','decided_at':'2024-01-02','known_at':'2024-01-02','accepted_at':'2024-01-03','accepted_known_at':'2024-01-03'}
        self.assert_code('APPLICATION_SELECTION_CONFLICT',build_application_targets,[app()],[d],[],[],[],30,'2024-03-01')
        self.assert_code('APPLICATION_SCHEMA',predict_application_risk,snapshot(),model(),target(),calibrator=[])

    def test_offer_comparison_fixed_customer_and_extrapolation(self):
        s=snapshot();before=deepcopy(s)
        offers=[{'offer_id':'one','terms':{'amount':1000}},{'offer_id':'same','terms':{'amount':1000}},{'offer_id':'outside','terms':{'amount':3000}}]
        v=compare_offer_risk(s,offers,model(),target(),{'amount':[500,2000]},lgd=0.4).value
        self.assertEqual(v[0]['pd'],v[1]['pd'])
        self.assertAlmostEqual(v[0]['expected_loss'],400/(1+math.e),places=12)
        self.assertTrue(v[2]['extrapolation'])
        self.assertAlmostEqual(v[2]['pd'],1/(1+math.exp(-1)),places=14)
        self.assertEqual(s,before)
        self.assert_code('APPLICATION_OFFER_MODEL',compare_offer_risk,s,[{'offer_id':'bad','terms':{'amount':1000,'unmodelled_term':30}}],model(),target(),{'amount':[500,2000]})

if __name__=='__main__':unittest.main()
