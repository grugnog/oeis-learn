"""Closed replay authority and complete baseline-only admission, without Torch."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.pipeline_validation.contracts import parse
from oeis_learn.pipeline_validation.registry import replay_registry, resolve
from oeis_learn.pipeline_validation.replay import replay_plan, replay_report

ROOT=Path(__file__).resolve().parents[2]


class Replay(unittest.TestCase):
    def test_manifest_has_distinct_closed_scope(self):
        normal=json.loads((ROOT/'configs/foundation/pipeline_validation.json').read_text())
        narrow=json.loads((ROOT/'configs/foundation/pipeline_replay_validation.json').read_text())
        parse(normal);parse(narrow)
        self.assertNotEqual(normal['identity'],narrow['identity'])
        for field,value in [('workloads',[]),('comparisons',['hoisted/v1']),('phases',[1,2,3,4,5]),('scope','engineering')]:
            bad=copy.deepcopy(narrow);bad[field]=value
            with self.assertRaises(ValueError):parse(seal(bad))
        bad=copy.deepcopy(narrow);bad['authority']['training']=True
        with self.assertRaises(ValueError):parse(seal(bad))

    def test_static_graph_has_five_bounded_trials_without_studies_or_candidates(self):
        stages=resolve(replay_registry(('podman',),'podman'),'all-ready')
        trials=[s for s in stages if s.adapter=='replay-trial']
        self.assertEqual([s.id for s in trials],['replay.cold','replay.warmup','replay.calibration-0','replay.calibration-1','replay.calibration-2'])
        self.assertEqual(sum(s.seconds for s in trials),3000)
        self.assertTrue(all(s.seconds==600 for s in trials))
        self.assertFalse({s.adapter for s in stages}&{'native','joint','cost','inference','study','baseline'})
        self.assertEqual(trials[1].prerequisites,('replay.cold',))

    def test_plan_freezes_tokens_mapping_and_baseline_policy(self):
        plan=replay_plan(1)
        self.assertEqual(plan['positions'],[1,32,128,513,1024])
        self.assertEqual(plan['batches'],[1,8])
        self.assertEqual(plan['bookkeeping'],'baseline')
        self.assertEqual(plan['mixed_order'],[6,1,6])
        self.assertEqual(len(plan['tokens']),8)
        self.assertTrue(all(row==[1]*1024 for row in plan['tokens']))
        self.assertEqual(plan,replay_plan(1))

    def test_report_needs_all_five_trials_and_matching_reference(self):
        ids=['replay.cold','replay.warmup',*[f'replay.calibration-{i}' for i in range(3)]]
        rows={key:dict(state='PASS',observations={'result':dict(status='COMPLETE',reference='ref',workload='work',verified_seconds=2,identity=key,resources={'qualified':True,'ceiling_bytes':16<<30},device={'hip':'test','name':'gpu'})}) for key in ids}
        good=replay_report(rows,'ref','work')
        self.assertEqual(good['status'],'COMPLETE')
        self.assertEqual(len(good['calibration']['baseline_receipts']),3)
        self.assertFalse(good['optimization_adoption'])
        self.assertFalse(good['research_cost_admission'])
        from oeis_learn.pipeline_validation.costs import forecast_trials
        forecast=forecast_trials([good]*3,'diverse32',updates=3000,cadence=500,budget=7200)
        self.assertFalse(forecast['forecast']['within_budget'])
        for key in ids:
            for field,value in [('status','PARTIAL'),('reference','other'),('workload','other')]:
                changed=copy.deepcopy(rows);changed[key]['observations']['result'][field]=value
                self.assertEqual(replay_report(changed,'ref','work')['status'],'INCONCLUSIVE')
            changed=copy.deepcopy(rows);changed.pop(key)
            self.assertEqual(replay_report(changed,'ref','work')['status'],'INCONCLUSIVE')
        changed=copy.deepcopy(rows);changed[ids[-1]]['observations']['result']['resources']['qualified']=False
        self.assertEqual(replay_report(changed,'ref','work')['status'],'INCONCLUSIVE')
        changed=copy.deepcopy(rows);changed[ids[-1]]['observations']['result']['resources']['ceiling_bytes']=8<<30
        self.assertEqual(replay_report(changed,'ref','work')['status'],'INCONCLUSIVE')

    def test_public_scope_report_is_stdlib_and_baseline_only(self):
        result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/foundation_run.py'),'--engine','podman','validate','--suite','pipeline','--manifest','configs/foundation/pipeline_replay_validation.json','--phase','all-ready','--engines','selected','--scope-report'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        scope=json.loads(result.stdout)
        self.assertIn('replay.reference',[s['id'] for s in scope['selected']])
        self.assertNotIn('baseline.calibration',[s['id'] for s in scope['selected']])
        self.assertEqual(scope['scope'],'engineering-replay')

    def test_visible_parent_preparation_never_reads_hidden_truth(self):
        import tempfile
        import importlib.util
        # Reuse the existing parent fixture without importing pytest/Torch.
        import ast
        source=ROOT/'tests/unit/test_pipeline_profile.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parent_fixture')
        namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        spec=importlib.util.spec_from_file_location('replay_prepare_test',ROOT/'scripts/profile_bigint_pipeline.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run,inputs,output=namespace['parent_fixture'](root)
            (inputs/'private/synthetic-truth.json').unlink()
            result=module.prepare(run,inputs,output,image='image',source='source',render_node='/dev/dri/renderD128',replay=True)
            payload=json.loads((output/'input/work.json').read_text())
            self.assertEqual(set(payload),{'manifest','recipe','prompts'})
            self.assertEqual(len(payload['prompts']),8)
            self.assertEqual(set(payload['prompts'][0]),{'id','values'})
            self.assertFalse((output/'private').exists())
            self.assertEqual(payload['manifest'],result)

    def test_replay_role_options_cannot_extend_trial_or_smuggle_authority(self):
        from oeis_learn.pipeline_validation.replay import role_deadline
        options=dict(seconds=570,memory_ceiling=1<<30,render_node=None)
        role_deadline({'options':options})
        for changed in [dict(options,seconds=601),dict(options,seconds=0),dict(options,training=True)]:
            with self.assertRaises(ValueError):role_deadline({'options':changed})

    def test_orchestrator_separates_reference_and_trials_and_rejects_mutated_reference(self):
        import tempfile
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from oeis_learn.pipeline_validation.replay import ReplayStages
        from oeis_learn.pipeline_validation.registry import Stage
        from oeis_learn.pipeline_validation.identity import write,file_hash
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            host=SimpleNamespace(policy={'render_node':'/dev/dri/renderD128','memory_bytes':16<<30,'cpus':16},select_gpu=Mock(),gpu_lock=lambda:nullcontext())
            a=SimpleNamespace(root=root,args=SimpleNamespace(run='parent',inputs='data',engine='podman'),contract={'identity':'contract','weights_ref':None,'resource_policy':'workstation90'},images=SimpleNamespace(images={('podman','gpu'):'image'}),host=lambda _:host,receipts={})
            def prepare(run,data,output,**kw):
                self.assertTrue(kw['replay']);inputs=output/'input';inputs.mkdir()
                (inputs/'weights.pt').write_bytes(b'weights')
                m=seal({'binding':{'source':'current'}});write(inputs/'work.json',{'manifest':m});return m
            def container(h,image,inputs,output,**kw):
                calls.append(kw);output.mkdir();self.assertEqual(kw['seconds'],600)
                expected=json.loads((inputs/'expected.json').read_text());self.assertEqual(expected['options']['seconds'],570)
                if kw['mode']=='replay-reference':
                    (output/'private').mkdir();(output/'private/reference.pt').write_bytes(b'frozen')
                    ref=seal({'workload':'work'});write(output/'reference.json',ref)
                else:
                    self.assertEqual((inputs/'reference.pt').read_bytes(),b'frozen')
                    ref=json.loads((inputs/'reference.json').read_text())
                write(output/'summary.json',dict(status='COMPLETE',reference=ref['identity'],workload='work',identity=kw['mode'],verified_seconds=1,resources={'qualified':True,'ceiling_bytes':16<<30},device={'hip':'test','name':'gpu'}))
            module=SimpleNamespace(prepare=prepare,container_stage=container)
            observations=dict(physical=32<<30,available=24<<30,cpus=16,limits=[],quota=None,cpuset=None)
            stages=ReplayStages(a)
            with patch('oeis_learn.pipeline_validation.benchmarks.legacy',return_value=module),patch('oeis_learn.pipeline_validation.resources.observe',return_value=observations),patch('oeis_learn.experiments.source_snapshot.collect_snapshot',return_value={}):
                ref=root/'reference';ref.mkdir();stages(Stage('replay.reference',1,seconds=600,adapter='replay-reference'),ref)
                for name in ['cold','warmup','calibration-0','calibration-1','calibration-2']:
                    out=root/name;out.mkdir();result=stages(Stage('replay.'+name,2,seconds=600,adapter='replay-trial'),out)
                    a.receipts['replay.'+name]={'state':'PASS','identity':name,'observations':{'result':result}}
                out=root/'report';out.mkdir();report=stages(Stage('replay.report',2,adapter='replay-report'),out)
                self.assertEqual(report['status'],'COMPLETE')
                (ref/'role/reference.json').write_text(json.dumps(seal({'workload':'changed'})))
                out=root/'mutated';out.mkdir()
                with self.assertRaisesRegex(ValueError,'reference changed'):
                    stages(Stage('replay.cold',2,seconds=600,adapter='replay-trial'),out)
            self.assertEqual([c['gpu'] for c in calls],[False,True,True,True,True,True])
            self.assertTrue(all('oeis_learn.pipeline_validation.role' in c['argv'] for c in calls))

    def test_replay_preparation_rejects_changed_visible_denominator(self):
        import tempfile
        from oeis_learn.pipeline_validation.replay import prepare_inputs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);weights=root/'weights.pt';weights.write_bytes(b'frozen')
            rows=[{'id':str(i),'values':['0']*20} for i in range(8)]
            for number,changed in enumerate([rows[:7],[*rows[:7],rows[0]],[*rows[:7],dict(rows[7],values=['0']*21)]]):
                output=root/str(number);output.mkdir()
                with self.assertRaisesRegex(ValueError,'twenty-visible-term'):
                    prepare_inputs({},changed,weights,output,source='source')
                self.assertFalse((output/'input').exists())

    def test_partial_boundary_ledger_keeps_every_planned_outcome(self):
        import tempfile
        from oeis_learn.pipeline_validation.replay import BoundaryLedger
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(TimeoutError):
                with BoundaryLedger(root) as ledger:
                    ledger.queued('b1.p1');ledger.verified('b1.p1',{'hash':'fixture'})
                    ledger.queued('b1.p32')
                    raise TimeoutError('bounded fixture')
            report=json.loads((root/'boundary-status.json').read_text())
            self.assertEqual(report['status'],'PARTIAL')
            self.assertEqual(report['expected'],12)
            self.assertEqual(report['states']['b1.p1'],'COMPLETE')
            self.assertEqual(report['states']['b1.p32'],'INCOMPLETE')
            self.assertEqual(sum(v=='UNSTARTED' for v in report['states'].values()),10)
            self.assertEqual(report['error'],'TimeoutError')
            rows=[json.loads(s) for s in (root/'boundary-progress.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['event'],'PLAN')
            self.assertEqual(len(rows[0]['boundaries']),12)
