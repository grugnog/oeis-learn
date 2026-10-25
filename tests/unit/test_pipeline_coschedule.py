"""Closed component co-scheduling, failure stops and complete-work-only ratios."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from oeis_learn.pipeline_validation.identity import seal

ROOT=Path(__file__).resolve().parents[2]


class Coschedule(unittest.TestCase):
    def test_manifest_cpu_gpu_graph_and_phase_budget(self):
        from oeis_learn.pipeline_validation.contracts import parse
        from oeis_learn.pipeline_validation.registry import coschedule_registry,resolve
        manifest=json.loads((ROOT/'configs/foundation/pipeline_coschedule.json').read_text());parse(manifest)
        stages=resolve(coschedule_registry(('podman',),'podman'),'all-ready')
        self.assertIn('hip.preflight',[s.id for s in stages]);self.assertIn('coschedule.reference',[s.id for s in stages])
        self.assertLessEqual(sum(s.seconds for s in stages if s.phase==4),18000)
        for key,value in [('phases',[1,2]),('workloads',[]),('comparisons',[])]:
            with self.assertRaises(ValueError):parse(seal(dict(manifest,**{key:value})))

    def trial(self,inputs,output,config,*,schedule,observed,seconds,**kwargs):
        from oeis_learn.pipeline_validation.coschedule import work_identity
        result=seal(dict(version='pipeline-coschedule-trial/v1',status='COMPLETE',source=config['source'],contract=config['contract'],
            reference='gpu-ref',native_reference=getattr(self,'native_ref','native-ref'),workload=work_identity('gpu-ref',getattr(self,'native_ref','native-ref')),schedule=schedule,
            observed=observed,primary_seconds=80 if schedule=='serial' else 45,
            gpu=dict(status='COMPLETE',boundaries=640,reference='gpu-ref',batch_size=128,bookkeeping='hoisted/v1',measurement_mode='ordinary/v1',resources={'qualified':True,'ceiling_bytes':32<<30},device={'hip':'actual-fixture','name':'gpu'},torch_threads={'intraop':2,'interop':1}),
            native=dict(status='COMPLETE',cycles=8,rows=144,cycle_receipts=[f'cycle{i}' for i in range(8)]),
            component_seconds=dict(gpu=38,native=40),gpu_memory={'qualified':True,'ceiling_bytes':32<<30},
            interpretation='fixture'))
        return result

    def run_study(self,trial=None):
        from oeis_learn.pipeline_validation.coschedule import study,STUDY_SLOTS,BASIS
        from oeis_learn.pipeline_validation.identity import write
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);inputs=root/'input';inputs.mkdir();out=root/'out';out.mkdir()
            ref=seal(dict(identity='ignored',source='source',contract='contract',corpus='corpus',semantics=[],cycle_seconds=8,
                replay_reference='gpu-ref',weights='weights',basis=BASIS,version='pipeline-coschedule-native-reference/v1',trial='trial'))
            write(inputs/'native-reference.json',ref);self.native_ref=ref['identity']
            with patch('oeis_learn.pipeline_validation.coschedule.read_reference',return_value=ref),patch('oeis_learn.pipeline_validation.coschedule.work_identity',return_value='work'),patch('oeis_learn.pipeline_validation.coschedule.trial',trial or self.trial):
                report=study(inputs,out,dict(source='source',contract='contract',options=dict(seconds=12470,memory_ceiling=32<<30,render_node=None)))
            receipts={p.parent.name:json.loads(p.read_text()) for p in (out/'trials').glob('*/receipt.json')}
            return report,receipts

    def test_twenty_slots_complete_and_observations_excluded_from_pairs(self):
        result,receipts=self.run_study();self.assertEqual(result['status'],'COMPLETE');self.assertEqual(len(receipts),20)
        self.assertTrue(all(r['state']=='PASS' for r in receipts.values()))
        self.assertEqual(result['comparison']['selected_schedule'],'overlap');self.assertEqual(len(result['comparison']['pairs']),2)
        self.assertEqual(len(result['repeatability']['trials']),10);self.assertFalse(result['optimization_adoption'])

    def test_incomplete_baseline_blocks_candidate_and_repeatability(self):
        def incomplete(inputs,output,config,**kw):
            r=self.trial(inputs,output,config,**kw)
            return seal(dict(r,status='PARTIAL')) if output.name=='calibration-1' else r
        result,receipts=self.run_study(incomplete);self.assertNotEqual(result['status'],'COMPLETE')
        self.assertEqual(result['comparison']['pairs'],[])
        self.assertEqual(sum(r['state']=='BLOCKED' for r in receipts.values()),15)

    def test_incomplete_pair_cannot_select_overlap_or_start_repeatability(self):
        def incomplete(inputs,output,config,**kw):
            r=self.trial(inputs,output,config,**kw)
            return seal(dict(r,status='PARTIAL')) if output.name=='pair0-B' else r
        result,receipts=self.run_study(incomplete)
        self.assertEqual(result['comparison']['pairs'],[]);self.assertIsNone(result['comparison']['selected_schedule'])
        self.assertTrue(all(receipts[f'repeat-{i:02d}']['state']=='BLOCKED' for i in range(10)))

    def test_complete_negative_selects_serial_for_registered_ten_repeats(self):
        def negative(inputs,output,config,**kw):return seal(dict(self.trial(inputs,output,config,**kw),primary_seconds=80))
        result,receipts=self.run_study(negative);self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['comparison']['selected_schedule'],'serial');self.assertEqual(len(result['repeatability']['trials']),10)

    def test_changed_prospective_child_blocks_next_slot(self):
        def mutate(inputs,output,config,**kw):
            r=self.trial(inputs,output,config,**kw)
            if output.name=='candidate-warmup':(output.parents[1]/'comparison-contract.json').write_text('{}')
            return r
        result,receipts=self.run_study(mutate)
        self.assertEqual(receipts['pair0-A']['state'],'BLOCKED');self.assertEqual(result['comparison']['pairs'],[])

    def test_wrong_mode_resource_or_work_never_produces_ratio(self):
        for field,value in [('schedule','overlap'),('observed',True),('workload','changed'),('gpu_memory',{'qualified':False})]:
            def altered(inputs,output,config,**kw):
                r=self.trial(inputs,output,config,**kw)
                return seal(dict(r,**{field:value})) if output.name=='pair0-A' else r
            with self.subTest(field=field):
                result,_=self.run_study(altered);self.assertEqual(result['comparison']['pairs'],[])

    def test_resource_failure_and_interrupt_preserve_all_planned_slots(self):
        from oeis_learn.pipeline_validation.coschedule import study
        for error in (MemoryError('shared'),KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__),tempfile.TemporaryDirectory() as f:
                root=Path(f)
                with patch('oeis_learn.pipeline_validation.coschedule.read_reference',return_value={'identity':'native-ref','replay_reference':'gpu-ref'}),patch('oeis_learn.pipeline_validation.coschedule.trial',side_effect=error),self.assertRaises(type(error)):
                    study(root,root,dict(source='source',contract='contract',options=dict(seconds=12470,memory_ceiling=32<<30,render_node=None)))
                rows=[json.loads(p.read_text()) for p in root.glob('trials/*/receipt.json')]
                self.assertEqual(len(rows),20);self.assertTrue(all(r['state'] not in {'PENDING','READY','RUNNING'} for r in rows))
                self.assertTrue((root/'coschedule-report.json').is_file())

    def test_serial_order_and_deliberate_overlap_share_one_owner(self):
        import threading
        import time
        from oeis_learn.pipeline_validation.coschedule import trial,finish_native
        for schedule in ('serial','overlap'):
            with self.subTest(schedule=schedule),tempfile.TemporaryDirectory() as f:
                begun=threading.Event();gpu_begun=threading.Event();gpu_done=threading.Event();owners=[]
                def gpu(inputs,output,config,seconds):
                    owners.append(threading.get_ident());gpu_begun.set()
                    self.assertEqual(seconds,270)
                    if schedule=='serial':self.assertFalse(begun.is_set())
                    else:self.assertTrue(begun.wait(1))
                    time.sleep(.01);gpu_done.set();return {'status':'COMPLETE','resources':{'qualified':True}}
                def native(inputs,output,config,reference,deadline,control,*,ledgers):
                    begun.set()
                    if schedule=='serial':self.assertTrue(gpu_done.is_set())
                    else:self.assertTrue(gpu_begun.wait(1))
                    for ledger in ledgers.values():ledger.transition('READY');ledger.transition('RUNNING');ledger.transition('PASS')
                    return finish_native(output,ledgers)
                root=Path(f);config=dict(source='source',contract='contract',options=dict(memory_ceiling=32<<30,render_node=None))
                with patch('oeis_learn.pipeline_validation.coschedule.read_reference',return_value={'identity':'native','replay_reference':'gpu'}):
                    r=trial(root,root/'out',config,schedule=schedule,observed=False,seconds=400,gpu_executor=gpu,native_executor=native)
                self.assertEqual(r['status'],'COMPLETE');self.assertEqual(owners,[threading.get_ident()])
                self.assertEqual(len(list((root/'out/native').glob('cycle-*/receipt.json'))),8)

    def test_gpu_failure_cancels_active_native_peer_and_preserves_cycle_roster(self):
        from oeis_learn.pipeline_validation.coschedule import trial
        from types import SimpleNamespace
        import threading
        started=threading.Event();cancelled=threading.Event()
        def gpu(*args):
            self.assertTrue(started.wait(1));raise ValueError('parity failed')
        def native(inputs,output,config,reference,deadline,control,**kw):
            control.bind(SimpleNamespace(worker=SimpleNamespace(cancel_all=cancelled.set)));started.set()
            self.assertTrue(cancelled.wait(1));raise InterruptedError('peer cancelled')
        with tempfile.TemporaryDirectory() as f,patch('oeis_learn.pipeline_validation.coschedule.read_reference',return_value={'identity':'native','replay_reference':'gpu'}):
            root=Path(f)
            with self.assertRaisesRegex(ValueError,'parity'):
                trial(root,root/'out',dict(source='s',options=dict(render_node=None)),schedule='overlap',observed=False,seconds=400,gpu_executor=gpu,native_executor=native)
            self.assertTrue(cancelled.is_set())
            self.assertTrue(json.loads((root/'out/trial-progress.json').read_text())['peer_drained'])
            rows=[json.loads(p.read_text()) for p in (root/'out/native').glob('*/receipt.json')]
            self.assertEqual(len(rows),8);self.assertTrue(all(r['state']=='BLOCKED' for r in rows))

    def test_native_peer_resource_error_reaches_owning_role(self):
        from oeis_learn.pipeline_validation.coschedule import trial
        def native(*a,**kw):raise MemoryError('native RSS')
        with tempfile.TemporaryDirectory() as f,patch('oeis_learn.pipeline_validation.coschedule.read_reference',return_value={'identity':'native','replay_reference':'gpu'}):
            root=Path(f)
            with self.assertRaisesRegex(MemoryError,'native RSS'):
                trial(root,root/'out',dict(source='s',options=dict(render_node=None)),schedule='overlap',observed=False,seconds=400,
                    gpu_executor=lambda *a:{'status':'COMPLETE'},native_executor=native)

    def test_gpu_child_resource_failure_is_not_downgraded_to_ordinary_error(self):
        from oeis_learn.pipeline_validation.coschedule import gpu_run
        from oeis_learn.pipeline_validation.runner import ResourceBreach
        from oeis_learn.pipeline_validation.identity import write
        def failed(command,output,**kw):
            self.assertEqual(command[3],'coschedule-gpu')
            self.assertEqual(kw['seconds'],270)
            write(output/'summary.json',dict(status='FAIL',fatal='RESOURCE_BREACH',error='GPU cap'))
            raise RuntimeError('child failed')
        with tempfile.TemporaryDirectory() as f,patch('oeis_learn.pipeline_validation.supervision.launch',failed):
            root=Path(f);inputs=root/'input';inputs.mkdir()
            for name in ('weights.pt','work.json','reference.json','reference.pt'):(inputs/name).write_bytes(b'fixture')
            with self.assertRaisesRegex(ResourceBreach,'GPU cap'):
                gpu_run(inputs,root/'gpu',dict(source='source',contract='contract',options=dict(memory_ceiling=32<<30,render_node=None)),270)

    def test_native_cycle_ids_are_distinct_and_never_renew_parent(self):
        import time
        from oeis_learn.pipeline_validation.coschedule import native_run,Cancellation
        from oeis_learn.pipeline_validation.role import native_corpus
        from oeis_learn.pipeline_validation.native_capacity import semantics
        from tests.unit.test_pipeline_native_capacity import NativeCapacityTests
        helper=NativeCapacityTests();corpus=native_corpus();base=helper.trial(Path('unused'),workers=4,corpus=corpus,contract='contract',cap=8,admission='slot-waves/v1')
        calls=[]
        def actual_shape(output,**kw):
            calls.append(kw);kw.pop('on_client');return helper.trial(output,**kw)
        with tempfile.TemporaryDirectory() as f,patch('oeis_learn.pipeline_validation.role.native_trial',actual_shape):
            r=native_run(None,Path(f),{'source':'source','contract':'contract'},dict(identity='native',semantics=semantics(base,corpus)),time.monotonic()+400,Cancellation())
        self.assertEqual(r['status'],'COMPLETE');self.assertEqual(r['rows'],144);self.assertEqual(len(calls),8)
        self.assertEqual(len({row['id'] for kw in calls for row in kw['corpus']['rows']}),144)
        self.assertTrue(all(kw['cap']==8 and kw['workers']==4 for kw in calls))

    def test_cancelled_controller_rejects_new_native_client_before_work(self):
        from oeis_learn.pipeline_validation.coschedule import Cancellation
        from unittest.mock import Mock
        from types import SimpleNamespace
        control=Cancellation();control.cancel();worker=Mock()
        with self.assertRaises(InterruptedError):control.bind(SimpleNamespace(worker=worker))
        worker.cancel_all.assert_called_once()

    def test_gpu_and_native_error_drain_do_not_acquire_two_cleanup_allowances(self):
        import threading
        import time
        from types import SimpleNamespace
        from oeis_learn.pipeline_validation.coschedule import trial
        from oeis_learn.synthesis.worker_pool import NonQuiescentWorker
        started=threading.Event();release=threading.Event();ended=threading.Event()
        def native(*args,**kw):
            control=args[5];control.bind(SimpleNamespace(worker=SimpleNamespace(cancel_all=lambda:None)))
            started.set()
            try:release.wait(1)
            finally:ended.set()
            return {'status':'PARTIAL'}
        def failed(command,output,**kw):
            self.assertTrue(started.wait(1));kw['cleanup_state']['deadline']=time.monotonic()-1
            raise RuntimeError('GPU cleanup allowance already exhausted')
        with tempfile.TemporaryDirectory() as f,patch('oeis_learn.pipeline_validation.coschedule.read_reference',return_value={'identity':'native','replay_reference':'gpu'}),patch('oeis_learn.pipeline_validation.supervision.launch',failed):
            root=Path(f);inputs=root/'input';inputs.mkdir()
            for name in ('weights.pt','work.json','reference.json','reference.pt'):(inputs/name).write_bytes(b'fixture')
            began=time.monotonic()
            try:
                with self.assertRaises(NonQuiescentWorker):
                    trial(inputs,root/'out',dict(source='s',contract='c',options=dict(render_node=None,memory_ceiling=32<<30)),schedule='overlap',observed=False,seconds=400,native_executor=native)
                self.assertLess(time.monotonic()-began,.5)
            finally:release.set();self.assertTrue(ended.wait(1))

    def test_missing_repeat_does_not_produce_favorable_subset_summary(self):
        def partial(inputs,output,config,**kw):
            r=self.trial(inputs,output,config,**kw)
            return seal(dict(r,status='PARTIAL')) if output.name=='repeat-04' else r
        result,receipts=self.run_study(partial)
        self.assertEqual(result['status'],'INCONCLUSIVE');self.assertFalse(result['repeatability']['complete'])
        self.assertEqual(len(result['repeatability']['trials']),10)
        self.assertIsNone(result['repeatability']['median']);self.assertIsNone(result['repeatability']['maximum'])
