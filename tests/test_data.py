"""Hand-derived boundary cases: no model-generated expected labels."""
import json
from pathlib import Path
import unittest
import pytest
from creditratingsystem.core import CreditRiskError
from creditratingsystem.data import (read_local_dataset, validate_credit_dataset, join_as_of,
    build_observation_windows, build_recovery_episodes, make_grouped_temporal_splits)


class DataContractTests(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def temporary_fixture_directory(self, tmp_path):
        self.fixture_directory = tmp_path

    def assert_code(self, code, fn, *args, **kwargs):
        with self.assertRaises(CreditRiskError) as caught:
            fn(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_local_typed_import_and_hash(self):
        # Writable test storage is configured by pytest --basetemp; source is read-only.
        path = self.fixture_directory / 'data_import_fixture.csv'
        schema = {'version':'1.0','columns':{'id':{'type':'string'},'amount':{'type':'number','unit':'EUR'},'active':{'type':'boolean'}},'primary_key':'id'}
        try:
            path.write_text('id;amount;active\n001;12,50;true\n',encoding='utf-8')
            value=read_local_dataset(path,schema,delimiter=';',decimal_mark=',')
            self.assertEqual(value.value,[{'id':'001','amount':12.5,'active':True}])
            self.assertEqual(len(value.provenance['source_sha256']),64)
            self.assert_code('DATA_TYPE',read_local_dataset,path,schema,delimiter=';')
            self.assert_code('DATA_REMOTE_PATH',read_local_dataset,'https://example.org/x.csv',schema)
            path.write_text('id;amount;active\n',encoding='utf-8')
            self.assert_code('DATA_EMPTY',read_local_dataset,path,schema,delimiter=';')
            schema['allow_empty']=True
            self.assertEqual(read_local_dataset(path,schema,delimiter=';').value,[])
        finally:
            path.unlink(missing_ok=True)

    def test_import_json_null_and_safe_id(self):
        path=self.fixture_directory/'data_import_fixture.json'
        schema={'version':'1.0','columns':{'id':{'type':'string'},'value':{'type':'number','unit':'ratio','nullable':True}}}
        try:
            path.write_text('[{"id":"9007199254740999","value":null}]')
            self.assertIsNone(read_local_dataset(path,schema).value[0]['value'])
            path.write_text('[{"id":"A","id":"B","value":0}]')
            self.assert_code('DATA_DUPLICATE',read_local_dataset,path,schema)
            path.write_text('[{"id":9007199254740999,"value":0}]')
            self.assert_code('DATA_TYPE',read_local_dataset,path,schema)
        finally:
            path.unlink(missing_ok=True)

    def test_relational_violations_are_not_silently_dropped(self):
        rows={'obligors':[{'entity_id':'A'}], 'observations':[{'entity_id':'A','observed_at':'2024-01-01','available_at':'2024-01-02'}]*2,
              'facilities':[{'entity_id':'A','facility_id':'F','ead':-1}],
              'defaults':[{'entity_id':'A','episode_id':'D','default_at':'2024-03-01'}],
              'recoveries':[{'cashflow_id':'C','episode_id':'unknown','cashflow_at':'2024-03-02','amount':1,'kind':'recovery'}]}
        v=validate_credit_dataset(rows).value
        self.assertFalse(v['valid'])
        self.assertEqual({r['code'] for r in v['violations']},{'DUPLICATE_KEY','INVALID_AMOUNT','FOREIGN_KEY'})

    def test_bitemporal_revision_and_tie(self):
        left=[{'entity_id':'A','as_of':'2024-02-01','known_at':'2024-02-01'}, {'entity_id':'A','as_of':'2024-02-01','known_at':'2024-03-01'}]
        right=[{'entity_id':'A','observed_at':'2024-01-01','available_at':'2024-01-05','release_id':'old','score':1}, {'entity_id':'A','observed_at':'2024-01-01','available_at':'2024-02-15','release_id':'revision','score':2}, {'entity_id':'A','observed_at':'2024-03-01','available_at':'2024-03-01','release_id':'future','score':9}]
        self.assertEqual([r['score'] for r in join_as_of(left,right).value],[1,2])
        self.assert_code('DATA_AMBIGUOUS_JOIN',join_as_of,left,right+[dict(right[0],release_id='tie')])
        self.assertIsNone(join_as_of([dict(left[0],entity_id='B')],right).value[0]['score'])

    def test_windows_hand_derived_leap_year_and_censoring(self):
        ratings=[{'observation_id':i,'entity_id':i,'as_of':'2024-01-01'} for i in ['A','B','C','D','E','F']]
        ratings[1]['outcome_available_at']='2024-03-01'
        ratings[2]['followup_end']='2024-01-15'; ratings[3]['funded']=False
        events=[{'event_id':'a','entity_id':'A','default_at':'2024-01-31','known_at':'2024-01-31'}, {'event_id':'b','entity_id':'B','default_at':'2024-01-31','known_at':'2024-03-01'}, {'event_id':'f','entity_id':'F','default_at':'2024-01-01','known_at':'2024-01-01'}]
        exits=[{'entity_id':'E','exit_at':'2024-01-20','known_at':'2024-01-20'}]
        v=build_observation_windows(ratings,events,exits,30,'2024-02-01').value
        self.assertEqual([r['target'] for r in v],[1,None,None,None,None,None])
        self.assertEqual([r['label_status'] for r in v],['default','outcome_pending','right_censored','not_funded','exit','prior_default'])
        self.assertEqual(v[0]['outcome_end'],'2024-01-31')
        self.assertEqual(build_observation_windows([{'observation_id':'L','entity_id':'L','as_of':'2024-02-28'}],[],[],1,'2024-02-29').value[0]['outcome_end'],'2024-02-29')

    def test_reopening_and_cashflow_unique_allocation(self):
        defaults=[{'episode_id':ep,'entity_id':'A','default_at':d,'known_at':d,'ead':100,'currency':'EUR'} for ep,d in [('d1','2024-01-01'),('d2','2024-03-01'),('d3','2024-12-01')]]
        cures=[{'episode_id':'d1','cure_at':'2024-02-01','known_at':'2024-02-01'}, {'episode_id':'d2','cure_at':'2024-04-01','known_at':'2024-04-01','closed_at':'2024-05-01','closed_known_at':'2024-05-01'}]
        flows=[{'cashflow_id':cf,'episode_id':ep,'cashflow_at':d,'known_at':d,'amount':v,'kind':kind,'currency':'EUR'} for cf,ep,d,v,kind in [('c1','d1','2024-01-11',30,'recovery'),('c2','d2','2024-03-11',5,'cost')]]
        v=build_recovery_episodes(defaults,flows,cures,90,'2024-12-31').value
        self.assertEqual(len(v['episodes']),2)
        self.assertEqual(v['episodes'][0]['default_ids'],['d1','d2'])
        self.assertEqual(v['episodes'][0]['net_recovery'],25)
        self.assertEqual(v['episodes'][0]['ead_at_first_default'],100)
        self.assertEqual(v['episodes'][1]['status'],'open')
        self.assertEqual([r['days_since_first_default'] for r in v['cashflows']],[10,70])
        self.assert_code('DATA_DUPLICATE',build_recovery_episodes,defaults,flows+[flows[0]],cures)
        self.assertEqual(len(build_recovery_episodes(defaults,flows,cures,90,'2024-02-01').value['episodes']),1)

    def test_closure_knowledge_and_bad_identifiers(self):
        defaults=[{'episode_id':'D','entity_id':'A','default_at':'2024-01-01','known_at':'2024-01-01','ead':100,'currency':'EUR'}]
        cures=[{'episode_id':'D','cure_at':'2024-02-01','known_at':'2024-02-01','closed_at':'2024-03-01','closed_known_at':'2024-04-01'}]
        early=build_recovery_episodes(defaults,[],cures,evaluation_cutoff='2024-03-15').value['episodes'][0]
        late=build_recovery_episodes(defaults,[],cures,evaluation_cutoff='2024-04-01').value['episodes'][0]
        self.assertEqual(early['status'],'open')
        self.assertIsNone(early['closed_at'])
        self.assertEqual(late['status'],'closed')
        del cures[0]['closed_known_at']
        self.assert_code('DATA_SCHEMA',build_recovery_episodes,defaults,[],cures)
        self.assert_code('DATA_SCHEMA',build_observation_windows,[{'observation_id':[], 'entity_id':'A','as_of':'2024-01-01'}],[],[],30,'2024-02-01')
        violations=validate_credit_dataset({'obligors':[{'entity_id':[]}]}).value['violations']
        self.assertEqual(violations[0]['code'],'INVALID_KEY')

    def test_split_transitive_groups_and_embargo(self):
        panel=[{'observation_id':str(i),'entity_id':e,'group_id':g,'observed_at':d,'outcome_end':end} for i,e,g,d,end in [(1,'A','G','2024-01-01','2024-01-15'),(2,'B','G','2024-04-10','2024-04-20'),(3,'B','H','2024-07-10','2024-07-20'),(4,'C','H','2024-01-10','2024-01-20'),(5,'D','Z','2024-01-10','2024-02-10'),(6,'E','Y','2024-03-31','2024-04-05'),(7,'F','X','2024-04-02','2024-04-04')]]
        v=make_grouped_temporal_splits(panel,'2024-03-31','2024-06-30','2024-09-30',5).value
        self.assertEqual([a['split'] for a in v['assignments']],['excluded','excluded','test','excluded','train','excluded','excluded'])
        self.assertEqual(v['assignments'][5]['reason'],'outcome_overlap')
        self.assertEqual(v['assignments'][6]['reason'],'embargo')
        self.assertFalse(v['group_overlap'])


if __name__=='__main__': unittest.main()


def test_integral_json_days_and_invalid_numeric_values():
    assert build_recovery_episodes([], [], reopening_days=90.0).value == build_recovery_episodes([], [], reopening_days=90).value
    for value in (True, 90.5, float('nan'), float('inf'), '90', -1):
        with pytest.raises(CreditRiskError, match='DATA_SCHEMA'):
            build_recovery_episodes([], [], reopening_days=value)


def test_legal_closure_without_cure_and_terminal_controls():
    default = dict(episode_id='D', entity_id='A', default_at='2024-01-01', known_at='2024-01-02', ead=100, currency='EUR')
    cash = dict(cashflow_id='R', episode_id='D', cashflow_at='2024-03-01', known_at='2024-03-02', amount=60, kind='recovery', currency='EUR')
    closure = dict(closure_id='C', episode_id='D', closed_at='2024-03-01', known_at='2024-03-05', closure_reason='SETTLEMENT')
    def run(ds=None, cs=None, ts=None, **kw):
        return build_recovery_episodes(ds or [default], [cash] if cs is None else cs, closures=[closure] if ts is None else ts, **kw).value
    value=run(); ep=value['episodes'][0]
    assert (ep['status'],ep['net_recovery'],ep['closure_reason'],ep['closed_at'])==('closed',60,'SETTLEMENT','2024-03-01')
    assert run(evaluation_cutoff='2024-03-04')['episodes'][0]['status']=='open'
    cases=[('DATA_CLOSED_EPISODE',dict(cs=[dict(cash,cashflow_at='2024-03-02')])),
      ('DATA_DUPLICATE',dict(ts=[closure,closure])),
      ('DATA_FOREIGN_KEY',dict(ts=[dict(closure,episode_id='missing')])),
      ('DATA_DATE_ORDER',dict(ts=[dict(closure,known_at='2024-02-01')])),
      ('DATA_SCHEMA',dict(ts=[dict(closure,closure_reason='CURE')])),
      ('DATA_DATE_ORDER',dict(ds=[default,dict(default,episode_id='D2',default_at='2024-04-01',known_at='2024-04-02')])),
      ('DATA_TERMINAL_CONFLICT',dict(ds=[default,dict(default,episode_id='D2',default_at='2024-02-01',known_at='2024-02-02')],ts=[closure,dict(closure,episode_id='D2',closure_id='C2')])),
      ('DATA_TERMINAL_CONFLICT',dict(cures=[dict(episode_id='D',cure_at='2024-02-01',known_at='2024-02-02',closed_at='2024-03-01',closed_known_at='2024-03-05')]))]
    for code,kw in cases:
        with pytest.raises(CreditRiskError) as caught: run(**kw)
        assert caught.value.code==code
