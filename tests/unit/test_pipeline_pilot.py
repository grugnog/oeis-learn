"""Prospective pilot contracts, disclosure and failure accounting (stdlib host)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from oeis_learn.pipeline_validation.identity import seal,digest


class Pilot(unittest.TestCase):
    def test_pretty_serialization_and_event_stream_caps_are_real_byte_limits(self):
        from oeis_learn.pipeline_validation.pilot_runtime import bounded_write
        from oeis_learn.pipeline_validation.pilot_evidence import Evidence
        from oeis_learn.pipeline_validation.runner import ResourceBreach
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);value={'rows':[{'a':i} for i in range(20)]}
            compact=len(json.dumps(value).encode())
            with self.assertRaises(ResourceBreach):bounded_write(root/'too-large.json',value,compact)
            self.assertFalse((root/'too-large.json').exists())
            with Evidence(root/'events',[{'id':'a'}]) as evidence:
                evidence.bytes=16<<20
                with self.assertRaisesRegex(ValueError,'16MiB'):evidence('a','correct',0,'STARTED',{})

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

    def test_exact_public_authority_cost_images_and_receipts_precede_private_inputs(self):
        from oeis_learn.pipeline_validation.pilot_contract import decision,decision_payload,parse_decision,PROFILE,qualification_requirements
        from oeis_learn.pipeline_validation.pilot_cost import forecast
        from oeis_learn.pipeline_validation.identity import write,file_hash
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=digest('source');work=digest('work')
            observations=[seal(dict(complete=True,seconds=100,generation_seconds=60,scoring_seconds=40,projected_all_selected_scoring_seconds=40,rows=296,source=source,work=work,checkpoint=digest(i))) for i in range(3)]
            archives=[seal(dict(source=source,complete=True,seconds=1.,bytes=16<<20,members=4096,filesystem_device=1,ordinal=i)) for i in range(3)]
            write(root/'cost.json',forecast(observations,source=source,work=work,archives=archives))
            required=sorted(qualification_requirements('podman'))
            q=seal(dict(version='pipeline-pilot-qualification/v1',state='PASS',source=source,profile=PROFILE['identity'],work=work,
                engine='podman',cpus=16,resources=digest('resources'),image={'gpu':digest('gpu'),'qualification':digest('cpu')},
                required=required,receipts={k:dict(state='PASS',identity=digest(k)) for k in required}))
            write(root/'q.json',q);write(root/'prepared.json',{'identity':work})
            ref=lambda n:dict(path=n,sha256=file_hash(root/n))
            d=decision(source=source,prepared=ref('prepared.json'),qualification=ref('q.json'),cost=ref('cost.json'))
            write(root/'authority.json',seal(dict(version='pipeline-study-authority/v1',decision_payload=decision_payload(d),approved=True,reviewed_by='fixture operator')))
            d=seal(dict(d,state='ADMITTED',authority=ref('authority.json')))
            self.assertEqual(parse_decision(d,root=root,source=source)['status'],'ADMITTED')
            # Even a newly sealed decision cannot use missing controls to open private inputs.
            q['receipts'][required[0]]['state']='PARTIAL';write(root/'q.json',seal(q),exclusive=False)
            bad=seal(dict(d,prepared={'path':'private/never-exists','sha256':digest('missing')},qualification=ref('q.json')))
            write(root/'authority.json',seal(dict(version='pipeline-study-authority/v1',decision_payload=decision_payload(bad),approved=True,reviewed_by='fixture operator')),exclusive=False)
            bad=seal(dict(bad,authority=ref('authority.json')))
            self.assertIn('qualification_receipts',parse_decision(bad,root=root,source=source)['missing'])

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
        # Events96, role samples<90, selections48, score24, snapshots36,
        # cumulative ledger40, report16, journals8, bins12, other core20MiB.
        upper=(96+90+48+24+36+40+16+8+12+20)*(1<<20)
        self.assertEqual(upper,390<<20)
        self.assertLess(upper+ (16<<20),512<<20)

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

    def test_evidence_accepts_real_generator_count_only_callback(self):
        from oeis_learn.pipeline_validation.pilot_evidence import Evidence
        with tempfile.TemporaryDirectory() as folder:
            with Evidence(Path(folder),[{'id':'a'}]) as e:
                e('a','correct',0,'STARTED',{'seed':1})
                e('a','correct',0,'GENERATED',dict(tokens=1024,source_sha256=digest('source'),generation_seconds=1.,complete=True))
                e('a','correct',0,'COMMITTED',{'complete':True})
            generated=json.loads((Path(folder)/'attempts.jsonl').read_text().splitlines()[1])
            self.assertEqual(generated['payload']['tokens_count'],1024)
            self.assertIsNone(generated['payload']['tokens_sha256'])


class LearningCurves(unittest.TestCase):
    def test_report_keeps_zero_success_missing_controls_and_overfitting_distinct(self):
        from oeis_learn.pipeline_validation.pilot import learning_report
        from oeis_learn.pipeline_validation.pilot_contract import ARMS
        prompts=[dict(id=k,family='f',strata={'length':'short'}) for k in ('a','b')]
        manifest={'roster':{'structural_development':['a','b']}}
        snapshots=[]
        for i in range(6):
            outcomes={k:{arm:dict(success=False,complete=True) for arm in ARMS} for k in ('a','b')}
            if i==5:outcomes['b'].pop('absent')
            splits={k:dict(complete=True,loss=.1 if k=='training_panel' else .5+i/10,token_accuracy=.9 if k=='training_panel' else .5)
                for k in ('training_panel','structural_development')}
            snapshots.append(dict(ordinal=i,tokens=0 if i==0 else 2**i,updates=i,examples=i*16,unique_examples=i*16,
                learner_role_seconds=i*100,teacher_forcing={'splits':splits},outcomes=outcomes))
        report=learning_report(manifest,prompts,snapshots,purpose='longrun')
        raw=report['series']['structural_development:raw']
        self.assertEqual(raw['points'][-1]['measurement']['coverage'],1.)
        self.assertEqual(raw['points'][-1]['measurement']['upper'],0.)
        self.assertEqual(raw['extrapolation']['status'],'UNKNOWN')
        absent=report['series']['structural_development:absent']['points'][-1]
        self.assertEqual(absent['measurement']['upper'],.5)
        self.assertEqual(report['series']['structural_development:equal_budget']['points'][-1]['correct_minus_controls']['absent']['unresolved_pairs'],1)
        self.assertGreater(report['teacher_forcing_gaps'][-1]['gaps']['structural_development']['loss_gap'],0)
        self.assertEqual(len(report['actual_snapshots']),6)

    def test_further_compute_is_conditional_and_zero_gain_has_no_cost_ratio(self):
        from oeis_learn.pipeline_validation.pilot_analysis import extrapolate,compute_estimates
        points=[dict(tokens=2**i,accuracy=i/10,coverage=1.,learner_role_seconds=2**i*3) for i in range(5)]
        result=compute_estimates(points,extrapolate(points),128)
        self.assertEqual(result['status'],'CONDITIONAL')
        self.assertEqual(result['scenarios']['2x']['additional_learner_seconds_range'],[48.,48.])
        self.assertGreater(result['scenarios']['2x']['recent_trend_seconds_per_additional_sequence_range'][0],0)
        zero=[dict(x,accuracy=0.) for x in points]
        self.assertIsNone(compute_estimates(zero,extrapolate(zero),128)['scenarios']['2x']['recent_trend_seconds_per_additional_sequence_range'])
        points[-1]['learner_role_seconds']=None
        self.assertEqual(compute_estimates(points,extrapolate(points),128)['status'],'UNKNOWN')

    def test_late_or_unacknowledged_slots_cannot_be_resumed(self):
        from oeis_learn.pipeline_validation.pilot_ledger import PilotLedger
        from oeis_learn.pipeline_validation.pilot_contract import decision
        from oeis_learn.pipeline_validation.runner import ResourceBreach
        from oeis_learn.pipeline_validation.identity import write
        with tempfile.TemporaryDirectory() as folder:
            clock=[100.];root=Path(folder);role=root/'role';role.mkdir()
            ledger=PilotLedger(root/'ledger',decision(source=digest('s')),clock=lambda:clock[0],boot='fixture')
            ledger.reserve('pilot.train-1',seconds=10,category='learner',output=role)
            with self.assertRaises(ValueError):ledger.recover('pilot.train-1')
            clock[0]=111.
            with self.assertRaises(ResourceBreach):ledger.finish('pilot.train-1','COMPLETE')
            self.assertEqual(ledger.value['slots']['pilot.train-1']['state'],'INTERRUPTED')
            self.assertTrue(ledger.value['slots']['pilot.train-1']['result']['fatal'])
            write(role/'cleanup.json',{'within_bound':True});write(role/'summary.json',{'fatal':False})
            with self.assertRaises(ValueError):ledger.recover('pilot.train-1')

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
        changed=[dict(x,complete_roster=digest(i%2)) for i,x in enumerate(points)]
        self.assertEqual(extrapolate(changed)['status'],'UNKNOWN')
        points[-1]['accuracy']=0.
        self.assertEqual(extrapolate(points)['status'],'UNKNOWN')

    def test_zero_success_is_a_valid_measured_learning_outcome(self):
        from oeis_learn.pipeline_validation.pilot_analysis import coverage,extrapolate
        r=coverage(['a','b'],[{'id':k,'family':'f','success':False,'complete':True} for k in ['a','b']])
        self.assertEqual((r['accuracy'],r['upper'],r['coverage']),(0.,0.,1.))
        points=[{'tokens':2**i,'accuracy':0.,'coverage':1.} for i in range(5)]
        self.assertEqual(extrapolate(points)['status'],'CONDITIONAL')
