"""Stdlib sizing authority, fixed work, accounting and complete-only reports."""
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from oeis_learn.pipeline_validation.identity import seal

ROOT=Path(__file__).resolve().parents[2]


class Sizing(unittest.TestCase):
    def test_exact_profiles_counts_and_closed_authority(self):
        from oeis_learn.pipeline_validation.model_sizes import PROFILES,parameters,validate_profile
        self.assertEqual([p['width'] for p in PROFILES.values()],[256,512,768])
        self.assertEqual(parameters(PROFILES['m8'])['total'],8118294)
        for p in PROFILES.values():
            self.assertEqual(validate_profile(p),p)
            with self.assertRaises(ValueError):validate_profile(seal(dict(p,width=1024)))
            with self.assertRaises(ValueError):validate_profile(seal(dict(p,precision='bfloat16')))

    def test_closed_manifest_and_fifteen_slots_fit_phases(self):
        from oeis_learn.pipeline_validation.contracts import parse
        from oeis_learn.pipeline_validation.registry import model_sizing_registry,resolve
        from oeis_learn.pipeline_validation.model_sizes import SLOTS
        m=json.loads((ROOT/'configs/foundation/pipeline_model_sizing.json').read_text());parse(m)
        stages=resolve(model_sizing_registry(('podman',),'podman'),'all-ready')
        self.assertEqual(len(SLOTS),15)
        self.assertEqual([s.id for s in stages if s.adapter=='sizing-trial'],SLOTS)
        self.assertLessEqual(sum(s.seconds for s in stages if s.phase==4),18000)
        for key,v in [('workloads',[]),('phases',[1,4]),('authority',dict(m['authority'],training=True))]:
            with self.assertRaises(ValueError):parse(seal(dict(m,**{key:v})))

    def test_resource_reservation_grows_with_model_and_is_not_weight_storage(self):
        from oeis_learn.pipeline_validation.model_sizes import PROFILES,parameters,peak_estimate
        peaks=[peak_estimate(p) for p in PROFILES.values()]
        self.assertEqual(sorted(peaks),peaks)
        self.assertTrue(all(peak_estimate(p)>parameters(p)['total']*16 for p in PROFILES.values()))

    def test_all_rows_planned_and_partial_rows_not_program_success(self):
        from oeis_learn.pipeline_validation.model_sizing import attempt_plan,generation_counts
        prompts=[dict(id=str(i),values=['0']*20,indices=list(map(str,range(20)))) for i in range(32)]
        plan=attempt_plan(prompts)
        self.assertEqual(len(plan),64);self.assertEqual(len({r['id'] for r in plan}),64)
        rows=[dict(r,state='PARTIAL',tokens=[1,2],decoded=False,native=None) for r in plan]
        c=generation_counts(rows,10)
        self.assertEqual(c['decoded'],0);self.assertEqual(c['verified'],0);self.assertEqual(c['scheduled'],64)
        self.assertEqual(c['emitted_tokens'],128)

    def test_partial_calibration_cannot_produce_favorable_aggregate(self):
        from oeis_learn.pipeline_validation.model_sizing import summarize_profiles
        from oeis_learn.pipeline_validation.model_sizes import SLOTS,PROFILES
        receipts={k:seal(dict(state='PASS',observations={'result':self.result(PROFILES[k.split('.')[1]])})) for k in SLOTS}
        receipts['sizing.m100.calibration-1']=seal(dict(receipts['sizing.m100.calibration-1'],state='PARTIAL'))
        report=summarize_profiles(receipts)
        self.assertIsNone(report['profiles']['m100']['training_tokens_per_second_median'])
        self.assertNotIn('m100',report['hardware_eligible_profiles'])
        self.assertFalse(report['optimization_adoption']);self.assertFalse(report['research_cost_admission'])
        self.assertEqual(report['model_selection'],'NOT_DECIDED')

    def result(self,p):
        from oeis_learn.pipeline_validation.model_sizes import parameters
        return seal(dict(status='COMPLETE',profile=p,counts=parameters(p),training_tokens=100,
            training_updates=32,validation_rows=32,training_tokens_per_second=10,training_updates_per_second=1,
            whole_seconds=100,resources={'qualified':True,'ceiling_bytes':32<<30,'breach':None},
            numerical_parity=True,changed_parameters=True,torch_threads={'intraop':2,'interop':1},device={'hip':'fixture'},
            source='source',contract='contract',workload='work',reference='ref',initial_parameters='initial',trained_parameters='trained',
            generation={'scheduled':64,'started':64,'decoded':0,'verified':0}))

    def test_wrong_work_threads_resource_or_unsealed_trial_never_qualifies(self):
        from oeis_learn.pipeline_validation.model_sizing import summarize_profiles
        from oeis_learn.pipeline_validation.model_sizes import PROFILES,SLOTS
        for field,value in [('training_updates',31),('torch_threads',{'intraop':8,'interop':1}),
            ('reference','different'),('resources',{'qualified':False}),('whole_seconds',1081)]:
            receipts={k:seal(dict(state='PASS',observations={'result':self.result(PROFILES[k.split('.')[1]])})) for k in SLOTS}
            k='sizing.m32.calibration-2';v=seal(dict(receipts[k]['observations']['result'],**{field:value}))
            receipts[k]=seal(dict(receipts[k],observations={'result':v}))
            with self.subTest(field=field):self.assertNotIn('m32',summarize_profiles(receipts)['hardware_eligible_profiles'])

    def test_undispatched_report_retains_all_fifteen_slots(self):
        from oeis_learn.pipeline_validation.model_sizing import summarize_profiles
        r=summarize_profiles({})
        self.assertEqual(sum(len(p['slots']) for p in r['profiles'].values()),15)
        self.assertEqual(r['hardware_eligible_profiles'],[])
        self.assertTrue(all(v['training_tokens_per_second_median'] is None for v in r['profiles'].values()))

    def test_preparation_rejects_duplicate_prompts_changed_learning_or_long_targets(self):
        from oeis_learn.pipeline_validation.model_sizing import prepare_inputs
        prompts=[dict(id=str(i),values=['0']*20,indices=list(map(str,range(20)))) for i in range(32)]
        recipe=dict(model={'seed':20261016},inputs={'profile':'hybrid-chunks/v1','mask':{k:True for k in ('magnitude','modular','difference','exact_bytes')}},learning={'batch_size':8,'learning_rate':.0001})
        training=[dict(values=['0']*20,tokens=[2]) for _ in range(128)];validation=training[:32]
        with tempfile.TemporaryDirectory() as f:
            m=prepare_inputs(recipe,prompts,training,validation,Path(f),source='source')
            work=json.loads((Path(f)/'input/work.json').read_text())
            self.assertNotIn('weights.pt',[p.name for p in (Path(f)/'input').iterdir()])
            self.assertNotIn('truth',work);self.assertEqual(m['updates'],32)
        with tempfile.TemporaryDirectory() as f,self.assertRaises(ValueError):prepare_inputs(recipe,prompts[:-1]+prompts[:1],training,validation,Path(f))
        with tempfile.TemporaryDirectory() as f,self.assertRaises(ValueError):prepare_inputs(dict(recipe,learning={'batch_size':32,'learning_rate':.0001}),prompts,training,validation,Path(f))
        training[0]=dict(training[0],tokens=[1]*1025)
        with tempfile.TemporaryDirectory() as f,self.assertRaises(ValueError):prepare_inputs(recipe,prompts,training,validation,Path(f))

    def test_native_generated_jobs_use_admission_clock_unique_debits_and_safe_paths(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        import time
        from oeis_learn.pipeline_validation.model_sizing import fit_generated
        from oeis_learn.pipeline_validation.model_sizes import PROFILES
        calls=[]
        class Client:
            worker=SimpleNamespace(affinity=lambda _:nullcontext(),cancel_all=lambda:None)
            def __init__(self,*a,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def fit(self,source,prompt,**kw):calls.append(kw['deadline']);return dict(complete=True,outcome='verified',admitted=[{'fixture':True}])
        rows=[dict(id=f'b8-../prompt{i}',prompt=str(i),seed=20261016+i,decoded=True,native=None,source='(program (return $index))') for i in range(8)]
        prompts=[dict(id=str(i),indices=['0'],values=['0']) for i in range(8)]
        with tempfile.TemporaryDirectory() as f,patch('oeis_learn.synthesis.client.SynthesisClient',Client):
            fit_generated(Path(f),rows,prompts,dict(contract='fixture',options={'profile':PROFILES['m8']}),time.monotonic()+30)
            ledger=json.loads((Path(f)/'native-jobs.json').read_text())
            self.assertEqual(len(ledger['records']),8);self.assertEqual(ledger['boundary']['consumed'],8)
            for record in ledger['records'].values():
                clock=ledger['timing'][record['job']['prompt_id']]
                self.assertEqual(clock['deadline'],clock['admitted']+4)
                self.assertEqual(record['deadline'],clock['deadline']);self.assertLess(record['observed'],clock['deadline'])
                self.assertEqual(record['state'],'COMMITTED')
            self.assertTrue(all(r['native']['verified'] for r in rows));self.assertEqual(len(list(Path(f).glob('*-native.json'))),8)

    def test_public_adapter_full_graph_fresh_references_shared_quota_and_export(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock
        from oeis_learn.pipeline_validation.model_sizing import SizingStages
        from oeis_learn.pipeline_validation.model_sizes import PROFILES
        from oeis_learn.pipeline_validation.registry import model_sizing_registry,resolve
        from oeis_learn.pipeline_validation.runner import execute
        from oeis_learn.pipeline_validation.identity import write,file_hash
        from oeis_learn.pipeline_validation.bundle import verify
        contract=json.loads((ROOT/'configs/foundation/pipeline_model_sizing.json').read_text())
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);host=SimpleNamespace(policy=dict(memory_bytes=64<<30,cpus=16,render_node='/dev/dri/renderD128'),select_gpu=Mock(),gpu_lock=lambda:nullcontext())
            a=SimpleNamespace(root=root,args=SimpleNamespace(engine='podman',run='parent',inputs='data'),contract=contract,host=lambda _:host,images=SimpleNamespace(images={('podman','gpu'):'fixture-image'}),receipts={})
            calls=[];inodes={}
            def prepare(run,data,output,**kw):
                self.assertIs(kw['sizing'],True);(output/'input').mkdir();write(output/'input/work.json',{'fixture':True})
                return seal(dict(binding={'parent_weights':'fixture'},version='fixture'))
            def container(selected,image,inputs,output,**kw):
                output.mkdir();cfg=json.loads((inputs/'expected.json').read_text());p=cfg['options']['profile'];calls.append(cfg['mode'])
                self.assertEqual(selected.policy['memory_bytes'],64<<30);self.assertEqual(selected.policy['cpus'],16)
                if cfg['mode']=='sizing-reference':
                    self.assertFalse(kw['gpu']);(output/'private').mkdir();(output/'private/weights.pt').write_bytes(b'frozen');(output/'private/oracle.pt').write_bytes(b'oracle')
                    write(output/'reference.json',seal(dict(workload='work',version='fixture')))
                    inodes[p['name']]=(output/'private/weights.pt').stat().st_ino
                    result=dict(status='COMPLETE',resources={'qualified':True})
                else:
                    self.assertTrue(kw['gpu']);self.assertEqual(kw['seconds'],1110)
                    self.assertEqual((inputs/'weights.pt').stat().st_ino,inodes[p['name']])
                    result=seal(dict(self.result(p),source=cfg['source'],contract=cfg['contract'],reference=json.loads((inputs/'reference.json').read_text())['identity'],resources={'qualified':True,'ceiling_bytes':64<<30,'breach':None}))
                    write(output/'host-timing.json',{'seconds_through_cleanup':101})
                    write(output/'attempts.json',[dict(state='PARTIAL',tokens=[1],decoded=False,native=None) for _ in range(64)])
                write(output/'summary.json',seal(result))
            sizing=SizingStages(a)
            class Adapter:
                receipts={}
                def __call__(self,stage,output):
                    a.receipts=self.receipts
                    return sizing(stage,output) if stage.adapter.startswith('sizing-') else {'fixture':True}
            with patch('oeis_learn.pipeline_validation.benchmarks.legacy',return_value=SimpleNamespace(prepare=prepare,container_stage=container)),patch('oeis_learn.experiments.source_snapshot.collect_snapshot',return_value={'fixture':'bytes'}),patch('oeis_learn.pipeline_validation.resources.observe',return_value=dict(physical=128<<30,available=128<<30,limits=[],cpus=16)):
                code=execute(resolve(model_sizing_registry(('podman',),'podman'),'all-ready'),root/'out',adapter=Adapter(),binding={'source':'fixture'},cap=86400,reserve=1200,contract=contract)
            self.assertEqual(code,0);self.assertEqual(calls.count('sizing-reference'),3);self.assertEqual(calls.count('sizing-trial'),15)
            self.assertEqual(verify(root/'out.zip')['summary_state'],'PASS')

    def test_compact_telemetry_stops_at_declared_sample_bound(self):
        from oeis_learn.tracking.pipeline_profile import Recorder
        with tempfile.TemporaryDirectory() as f:
            recorder=Recorder(Path(f),instrumented=True,host_counters=False,sample_bytes=1024)
            with patch.object(recorder,'capacity',return_value={}),patch.object(recorder,'snapshot',return_value={'time':0,'payload':'x'*2048}):
                with recorder:pass
            report=json.loads((Path(f)/'telemetry.json').read_text())
            self.assertEqual(report['samples'],0)
            self.assertIn('prospective byte cap',report['collector_error'])
            self.assertEqual((Path(f)/'samples.jsonl').stat().st_size,0)
