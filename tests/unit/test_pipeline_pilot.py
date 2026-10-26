"""Prospective pilot contracts, disclosure and failure accounting (stdlib host)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from oeis_learn.pipeline_validation.identity import seal,digest


class Pilot(unittest.TestCase):
    def test_recipe_normalization_without_importing_model_dependencies(self):
        import subprocess,sys
        # Only InputMask's constructor is replaced. Real pinned tests check the
        # feature implementation; this isolates closed recipe/policy validation.
        code="""import sys,types
sys.path.insert(0,'src')
m=types.ModuleType('oeis_learn.encoder.ablation');m.InputMask=lambda **kw:kw
sys.modules[m.__name__]=m
from oeis_learn.pipeline_validation.pilot_recipe import recipe
from oeis_learn.fresh.bigint_config import resolve
r=recipe();assert resolve(r)==r
assert r['learning']['batch_size']==16 and r['data']['training']==32768
old=resolve({'version':'bigint-bootstrap/v1'});assert old['learning']['max_updates']==3000
"""
        result=subprocess.run([sys.executable,'-S','-c',code],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_profile_counts_roster_and_reservations(self):
        from oeis_learn.pipeline_validation.pilot_contract import SPEC,parse_profile,PROFILE
        self.assertEqual(parse_profile(PROFILE),PROFILE)
        self.assertEqual(SPEC['evaluation_rows'],1776)
        self.assertEqual(SPEC['scoring_outcomes'],10656)
        self.assertEqual(sum(SPEC['reservations'].values()),172800)
        for field,value in [('training',512),('batch',32),('curriculum',True),('final',True)]:
            bad=copy.deepcopy(PROFILE);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):parse_profile(seal(bad))

    def test_cost_uses_complete_three_observations_and_full_schedule(self):
        from oeis_learn.pipeline_validation.pilot_cost import forecast
        rows=[seal({'complete':True,'seconds':100,'generation_seconds':60,'scoring_seconds':40,'projected_all_selected_scoring_seconds':40,'rows':296,'source':digest('s'),'work':digest('w'),'checkpoint':digest(i)}) for i in range(3)]
        archives=[seal(dict(source=digest('s'),complete=True,seconds=1.,bytes=16<<20,members=4096,filesystem_device=1,ordinal=i)) for i in range(3)]
        r=forecast(rows,source=digest('s'),work=digest('w'),archives=archives)
        self.assertEqual(r['evaluation_seconds'],1200) # six snapshots * twice largest full observation
        self.assertTrue(r['within_budget'])
        for bad in [rows[:2],rows+[rows[0]],[dict(x,rows=32) for x in rows],
                    [dict(x,complete=False) for x in rows],[dict(x,source=digest('old')) for x in rows]]:
            self.assertFalse(forecast(bad,source=digest('s'),work=digest('w'),archives=archives)['within_budget'])
        slow=[seal(dict(x,seconds=5000,generation_seconds=4000,scoring_seconds=1000,projected_all_selected_scoring_seconds=1000)) for x in rows]
        self.assertFalse(forecast(slow,source=digest('s'),work=digest('w'),archives=archives)['within_budget'])

    def test_worker_counter_resume_cannot_reset_or_change_policy(self):
        from oeis_learn.fresh.pilot_control import Control,configuration
        with tempfile.TemporaryDirectory() as folder:
            clock=[10.];selection={}
            cfg=configuration(source=digest('s'),work=digest('w'),purpose='preflight',ordinal=0,seconds=0,cadence_tokens=128)
            settings={'pilot':cfg};first=Control(settings,selection,Path(folder),clock=lambda:clock[0])
            first.checkpoint(selection);first.finish()
            cfg=configuration(source=digest('s'),work=digest('w'),purpose='preflight',ordinal=1,seconds=900,cadence_tokens=128)
            control=Control({'pilot':cfg},selection,Path(folder),clock=lambda:clock[0])
            control.update([{'tokens':[1,2]}],1);control.checkpoint(selection)
            self.assertEqual(control.state['tokens'],2)
            self.assertEqual(control.state['updates'],1)
            with self.assertRaises(ValueError):Control({'pilot':cfg},selection,Path(folder),clock=lambda:clock[0])
            changed=configuration(source=digest('changed'),work=digest('w'),purpose='preflight',ordinal=2,seconds=900,cadence_tokens=128)
            with self.assertRaises(ValueError):Control({'pilot':changed},selection,Path(folder),clock=lambda:clock[0])

    def test_hidden_or_duplicate_learner_rows_rejected(self):
        from oeis_learn.pipeline_validation.pilot_data import learner_view
        row={'id':'a','family':'f','stage':'A','indices':list(map(str,range(20))),
             'values':list(map(str,range(20))),'tokens':[1,2],'source':'training target'}
        examples={'training':[row],'parameter_development':[dict(row,id='b')],'structural_development':[dict(row,id='c')]}
        self.assertEqual(learner_view(examples)['training'][0]['values'],row['values'])
        for bad in [dict(row,values=list(map(str,range(100)))),dict(row,hidden_values=['1']),dict(row,truth=['1'])]:
            with self.assertRaises(ValueError):learner_view(dict(examples,training=[bad]))
        with self.assertRaises(ValueError):learner_view(dict(examples,structural_development=[row]))

    def test_long_run_cannot_be_admitted_by_a_plan_or_forecast_alone(self):
        from oeis_learn.pipeline_validation.pilot_contract import parse_decision,decision
        record=decision(source=digest('s'))
        self.assertEqual(parse_decision(record)['status'],'BLOCKED')
        forged=seal(dict(record,state='ADMITTED'))
        self.assertEqual(parse_decision(forged)['status'],'BLOCKED')
        with self.assertRaises(ValueError):parse_decision(seal(dict(record,final_plan={'path':'private/final'})))

    def test_pilot_ledger_does_not_refund_or_renew_after_interruption(self):
        from oeis_learn.pipeline_validation.pilot_ledger import PilotLedger
        from oeis_learn.pipeline_validation.pilot_contract import decision
        from oeis_learn.pipeline_validation.identity import write
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);clock=[100.];d=decision(source=digest('s'))
            ledger=PilotLedger(root/'ledgers',d,clock=lambda:clock[0],boot='fixture')
            role=root/'role';role.mkdir();write(role/'cleanup.json',{'within_bound':True})
            ledger.reserve('train1',seconds=5400,category='learner',output=role)
            ledger.recover('train1')
            with self.assertRaises(ValueError):ledger.reserve('train1',seconds=5400,category='learner',output=role)
            again=PilotLedger(root/'ledgers',d,clock=lambda:clock[0],boot='fixture')
            self.assertEqual(again.value['slots']['train1']['seconds'],5400)
            self.assertEqual(again.value['expires'],172900)
            with self.assertRaises(ValueError):PilotLedger(root/'ledgers',d,boot='changed')
            clock[0]=172901
            with self.assertRaises(ValueError):again.reserve('train2',seconds=1,category='learner',output=role)

    def test_sampling_and_archive_forecast_bounds_are_prospective(self):
        from oeis_learn.pipeline_validation.pilot_contract import SPEC
        # Mandatory worst-case evidence, without assuming compression.
        upper=SPEC['max_attempt_events']*2048+120000*768+(6*8+6*4+32)*(1<<20)
        self.assertLess(upper,(512-16)<<20)

    def test_evidence_retains_all_unstarted_attempts_and_hashes_long_tokens(self):
        from oeis_learn.pipeline_validation.pilot_evidence import Evidence
        with tempfile.TemporaryDirectory() as folder:
            with Evidence(Path(folder),[{'id':'a'}]) as e:
                e('a','correct',0,'STARTED',{})
                e('a','correct',0,'GENERATED',{'tokens':[9999]*1024})
                e('a','correct',0,'INCOMPLETE',{'complete':False})
            r=json.loads((Path(folder)/'attempt-index.json').read_text())
            self.assertEqual(r['expected'],18);self.assertEqual(r['states']['UNSTARTED'],17)
            self.assertEqual(r['states']['INCOMPLETE'],1)
            self.assertLess(max(len(x) for x in (Path(folder)/'attempts.jsonl').read_bytes().splitlines()),2048)


class LearningCurves(unittest.TestCase):
    def rows(self):
        return [{'id':str(i),'family':'f'+str(i//2),'success':i==0,'complete':i<3} for i in range(4)]

    def test_complete_negatives_and_unknowns_have_distinct_bounds(self):
        from oeis_learn.pipeline_validation.pilot_analysis import coverage
        r=coverage(['0','1','2','3'],self.rows())
        self.assertEqual((r['lower'],r['upper']),(.25,.5))
        self.assertEqual((r['complete'],r['unresolved']),(3,1))
        with self.assertRaises(ValueError):coverage(['0','1','2','3'],self.rows()+[self.rows()[0]])
        with self.assertRaises(ValueError):coverage(['0','1'],[dict(self.rows()[0],complete=False)])

    def test_paired_regressions_not_discarded_and_missing_arms_not_zero(self):
        from oeis_learn.pipeline_validation.pilot_analysis import paired
        before=self.rows();after=[dict(r,complete=True,success=r['id']=='1') for r in before]
        r=paired(before,after)
        self.assertEqual((r['gained'],r['lost'],r['net_known']),(1,1,0))
        with self.assertRaises(ValueError):paired(before,after[:-1])

    def test_no_asymptote_from_a_short_or_censored_curve(self):
        from oeis_learn.pipeline_validation.pilot_analysis import extrapolate
        self.assertEqual(extrapolate([{'tokens':1,'accuracy':0.,'coverage':1.}])['status'],'UNKNOWN')
        points=[{'tokens':2**i,'accuracy':i/10,'coverage':1.} for i in range(5)]
        r=extrapolate(points)
        self.assertEqual(r['status'],'CONDITIONAL')
        self.assertAlmostEqual(r['scenarios']['2x']['recent_trend'],.5)
        self.assertIsNone(r['guaranteed_capacity_upper_bound'])
        self.assertEqual(extrapolate([dict(x,coverage=.5) for x in points])['status'],'UNKNOWN')
        points[-1]['accuracy']=0.
        self.assertEqual(extrapolate(points)['status'],'UNKNOWN')

    def test_zero_success_is_a_valid_measured_learning_outcome(self):
        from oeis_learn.pipeline_validation.pilot_analysis import coverage,extrapolate
        r=coverage(['a','b'],[{'id':k,'family':'f','success':False,'complete':True} for k in ['a','b']])
        self.assertEqual((r['accuracy'],r['upper'],r['coverage']),(0.,0.,1.))
        points=[{'tokens':2**i,'accuracy':0.,'coverage':1.} for i in range(5)]
        self.assertEqual(extrapolate(points)['status'],'CONDITIONAL')
