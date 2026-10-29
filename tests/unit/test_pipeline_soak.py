"""Host-only soak authority, work, selection and cumulative recovery contracts."""
import copy
import json
from pathlib import Path
import unittest

from oeis_learn.pipeline_validation.identity import seal


class SoakContracts(unittest.TestCase):
    def test_launch_error_is_not_replaced_by_missing_device_file(self):
        import tempfile,types
        from unittest.mock import Mock,patch
        from oeis_learn.pipeline_validation.benchmarks import legacy
        from oeis_learn.pipeline_validation.runner import ResourceBreach
        module=legacy(Path(__file__).resolve().parents[2])
        inspection=[dict(State={'OOMKilled':False},HostConfig=dict(NetworkMode='none',ReadonlyRootfs=True,
            Privileged=False,Devices=[],DeviceRequests=[]),Mounts=[dict(Type='bind',Destination='/work/input',RW=False),
            dict(Type='bind',Destination='/work/output',RW=True)])]
        for fatal in (False,True):
            with self.subTest(fatal=fatal),tempfile.TemporaryDirectory() as td:
                root=Path(td);inputs=root/'input';inputs.mkdir();out=root/'role';out.mkdir()
                if fatal:(out/'summary.json').write_text(json.dumps(dict(fatal='RESOURCE_BREACH',error='original allocation failure')))
                engine=Mock();engine.command.return_value=['fixture-engine'];engine.call.return_value=types.SimpleNamespace(stdout=json.dumps(inspection))
                host=types.SimpleNamespace(engine=engine,policy=dict(memory_bytes=1<<30,cpus=2))
                with patch('oeis_learn.pipeline_validation.supervision.launch',side_effect=RuntimeError('original launch failure')):
                    with self.assertRaisesRegex(ResourceBreach if fatal else RuntimeError,'original allocation failure' if fatal else 'original launch failure'):
                        module.container_stage(host,'image',inputs,out,mode='fixture',seconds=1,gpu=False,argv=['python','-c','fixture'])
                self.assertEqual(json.loads((out/'isolation-audit.json').read_text())['status'],'INCOMPLETE')
                self.assertTrue(json.loads((out/'cleanup.json').read_text())['within_bound'])

    def test_uid_publication_overrides_private_default_file_modes(self):
        import os,tempfile,errno,shutil
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.soak_operations import fixture_code
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'input';source.mkdir();output=root/'output';output.mkdir()
            code=fixture_code(os.getuid(),inputs=source,output=output,owner_mapped=True)
            original_write=Path.write_text;original_open=Path.open;original_copy=shutil.copyfile
            def readonly(path,*args,**kwargs):
                if path==source/'forbidden':raise OSError(errno.EROFS,'labelled read-only mount simulation')
                return original_write(path,*args,**kwargs)
            def private_create(path,mode='r',*args,**kwargs):
                stream=original_open(path,mode,*args,**kwargs)
                if 'w' in mode or 'x' in mode:os.fchmod(stream.fileno(),0o600)
                return stream
            def private_copy(src,dst,*args,**kwargs):
                result=original_copy(src,dst,*args,**kwargs);Path(dst).chmod(0o600);return result
            before=os.umask(0o077)
            try:
                with patch.object(Path,'write_text',readonly),patch.object(Path,'open',private_create),patch('shutil.copyfile',side_effect=private_copy):
                    exec(compile(code,'uid-export-private-file-fixture','exec'),{})
            finally:os.umask(before)
            for path in [output/'devices.json',output/'ready.json',output/'uid-export.json',*(output/'failure-export-evidence').glob('*.json')]:
                self.assertEqual(path.stat().st_mode&0o777,0o644,path)

    def test_uid_export_executes_recovery_under_restrictive_umask(self):
        import os,tempfile,errno
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.soak_operations import fixture_code
        with tempfile.TemporaryDirectory() as td:
            # This is also the real container TMPDIR layout. Broad string
            # replacement corrupts an input path containing /work/output.
            root=Path(td)/'work/output/.runtime/tmp/case';root.mkdir(parents=True)
            source=root/'input';source.mkdir();output=root/'output';output.mkdir()
            code=fixture_code(os.getuid(),inputs=source,output=output,owner_mapped=True)
            original=Path.write_text
            def readonly(path,*args,**kwargs):
                if path==source/'forbidden':raise OSError(errno.EROFS,'labelled read-only mount simulation')
                return original(path,*args,**kwargs)
            before=os.umask(0o077)
            try:
                with patch.object(Path,'write_text',readonly):exec(compile(code,'uid-export-fixture','exec'),{})
            finally:os.umask(before)
            receipt=json.loads((output/'uid-export.json').read_text())
            self.assertTrue(receipt['healthy']);self.assertTrue(receipt['expected_failed_export_verified'])
            # Namespace-mapped host owner must read the foreign UID's public proof.
            for p in [output/'uid-export.json',*(output/'failure-export-evidence').glob('*.json')]:
                self.assertEqual(p.stat().st_mode&0o044,0o044,p)
            self.assertEqual((output/'devices.json').stat().st_mode&0o044,0o044)
            self.assertFalse(list(output.rglob('*.zip')))

    def test_worker_initialization_crosses_readonly_bind_mount_by_copy(self):
        import tempfile,errno
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.soak_runtime import run_worker
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inputs=root/'input';inputs.mkdir();(inputs/'initial.pt').write_bytes(b'frozen weights')
            output=root/'output'
            with patch('oeis_learn.pipeline_validation.soak_runtime.worker_settings',return_value=({'seconds':1},{})),patch('oeis_learn.pipeline_validation.supervision.launch',side_effect=RuntimeError('fixture stop before training')),patch('os.link',side_effect=OSError(errno.EXDEV,'distinct bind mounts')):
                with self.assertRaisesRegex(RuntimeError,'fixture stop'):
                    run_worker(inputs,output,None,None,{'options':{'seconds':3}})
            self.assertEqual((output/'private/worker-input/initial.pt').read_bytes(),b'frozen weights')

    def test_missed_native_intervals_do_not_catch_up_or_renew(self):
        from oeis_learn.fresh.engineering import Control
        from oeis_learn.pipeline_validation.soak_contract import lease
        h='sha256:'+'a'*64;clock=[100.]
        cfg=seal(dict(version='pipeline-engineering-worker/v1',source=h,work=h,choice=h,mode='window',
            batch_size=8,window=0,resume=False,lease=lease(h,h,h,now=100),token_target=1,seconds=3600,
            instrumented=False,cadence_tokens=10))
        c=Control({'engineering':cfg},{},Path('/unused'),clock=lambda:clock[0])
        self.assertTrue(c.pulse_due());self.assertEqual(c.active_pulse,0);c.pulsed()
        self.assertFalse(c.pulse_due())
        clock[0]=1500
        self.assertTrue(c.pulse_due());self.assertEqual(c.active_pulse,2);c.pulsed()
        self.assertEqual(c.state['pulse_slots'][1]['state'],'UNSTARTED')
        with self.assertRaises(ValueError):c.pulsed()
        clock[0]=3690;self.assertFalse(c.pulse_due())

    def test_unstarted_diagnostic_roster_cannot_complete_a_window(self):
        from oeis_learn.fresh.engineering import Control
        from oeis_learn.pipeline_validation.soak_contract import lease
        h='sha256:'+'a'*64;clock=[100.]
        cfg=seal(dict(version='pipeline-engineering-worker/v1',source=h,work=h,choice=h,mode='window',
            batch_size=8,window=0,resume=False,lease=lease(h,h,h,now=100),token_target=1,seconds=3600,
            instrumented=False,cadence_tokens=10))
        c=Control({'engineering':cfg},{},Path('/unused'),clock=lambda:clock[0],minimum_examples=2)
        c.update([{'tokens':[1]}],.1);c.cycled(complete=False);clock[0]=3700
        self.assertFalse(c.finish()['complete'])
        c.cycled();self.assertFalse(c.finish()['complete'])
        c.update([{'tokens':[1]}],.1);self.assertTrue(c.finish()['complete'])

    def test_native_peer_drains_and_gpu_probe_stays_on_owner_thread(self):
        import tempfile,threading,time,types
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.soak_native import pulse
        from oeis_learn.fresh.pipeline_jobs import JobLedger
        owner=threading.get_ident();release=threading.Event();peer=[];completed=[]
        boundary=JobLedger().boundary({})
        class NonQuiescentWorker(RuntimeError):pass
        # Substitute only the container-owned client import, not pulse's supervision.
        module=types.SimpleNamespace(NonQuiescentWorker=NonQuiescentWorker)
        with tempfile.TemporaryDirectory() as td:
            control=types.SimpleNamespace(output=Path(td),active_pulse=0,
                component=lambda *a,**kw:None,pulsed=lambda:completed.append(True))
            client=types.SimpleNamespace(worker=types.SimpleNamespace(cancel_all=release.set))
            def native(*args):
                peer.append(threading.get_ident())
                if not release.wait(2):raise RuntimeError('owner did not release native peer')
                return dict(ordinal=0,boundary=boundary,seconds=.01,results=[])
            def gpu():
                self.assertEqual(threading.get_ident(),owner);release.set()
            with patch.dict('sys.modules',{'oeis_learn.synthesis.worker_pool':module}),patch('oeis_learn.pipeline_validation.soak_native._native_pulse',side_effect=native):
                selection={'pipeline_boundary':boundary}
                pulse(client,control,selection,{},time.monotonic()+2,gpu_probe=gpu)
            self.assertNotEqual(peer,[owner]);self.assertEqual(completed,[True])
            self.assertEqual(selection['pipeline_boundary']['active'],0)

    def test_oom_event_remains_fatal_after_usage_drops(self):
        import tempfile
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.memory import MemoryWatchdog
        with tempfile.TemporaryDirectory() as td:
            killed=[];watch=MemoryWatchdog(td,ceiling=1000,reader=lambda:dict(physical=10000,available=9990,current=10,limit=1000),killer=lambda:killed.append(True),audit_samples=True)
            read=Path.read_text
            def fixture(path,*args,**kwargs):
                if path.name=='memory.events':return 'oom 1\noom_kill 1\n'
                if path.name=='memory.stat':return 'anon 10\nfile 0\n'
                if path.name=='cpu.stat':return 'usage_usec 10\n'
                return read(path,*args,**kwargs)
            with patch.object(Path,'read_text',fixture):watch.sample()
            self.assertEqual(killed,[True]);self.assertFalse(watch.report()['qualified'])
            self.assertIn('OOM',watch.breach['reason'])

    def test_roles_reject_unregistered_options_and_resume_outside_windows(self):
        from oeis_learn.pipeline_validation.soak_runtime import role_options
        opt=dict(kind='batch',batch_size=8,seconds=300,memory_ceiling=1000,render_node='/dev/dri/renderD128',
            instrumented=False,choice='sha256:'+'a'*64,resume=False,window=0,lease=None)
        role_options({'options':opt})
        for changed in (dict(opt,extra=True),dict(opt,resume=True),dict(opt,window=1),dict(opt,batch_size=True)):
            with self.assertRaises(ValueError):role_options({'options':changed})

    def test_foreign_uid_preserves_rootless_security_arguments(self):
        from oeis_learn.pipeline_validation.soak_operations import foreign_user_args,fixture_code
        before=['--user','0:0','--userns=keep-id','--security-opt=label=disable']
        after=foreign_user_args(before)
        self.assertEqual(after,['--user','12345:12345','--userns=keep-id:uid=12345,gid=12345','--security-opt=label=disable'])
        self.assertEqual(before[1],'0:0')
        self.assertEqual(foreign_user_args(['--user','1000:100','--group-add=keep-groups']),
            ['--user','12345:12345','--group-add=keep-groups'])
        compile(fixture_code(12345),'uid-fixture','exec')
        compile(fixture_code(1000,wait=True),'interruption-fixture','exec')

    def test_phase_spans_split_without_fractional_completed_work(self):
        from oeis_learn.fresh.engineering import Control
        h='sha256:'+'a'*64;clock=[100.]
        cfg=seal(dict(version='pipeline-engineering-worker/v1',source=h,work=h,choice=h,mode='batch',batch_size=8,
            window=0,resume=False,lease=None,token_target=10,seconds=600,instrumented=False,cadence_tokens=100))
        c=Control({'engineering':cfg},{},Path('/unused'),clock=lambda:clock[0])
        clock[0]=410;c.component('training',20,tokens=16,examples=8,updates=1)
        self.assertEqual(c.windows[0]['components']['training']['seconds'],10)
        self.assertNotIn('tokens',c.windows[0]['components']['training'])
        self.assertEqual(c.windows[1]['components']['training']['tokens'],16)
        self.assertEqual(c.windows[1]['components']['training']['seconds'],10)

    def test_failed_child_evidence_is_exported_without_tensors(self):
        import tempfile
        from oeis_learn.pipeline_validation.soak_runtime import export_worker_evidence
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'private/trainer';p.mkdir(parents=True)
            (p/'engineering-attempts.jsonl').write_text('{"event":"STARTED"}\n')
            (p/'initial.pt').write_bytes(b'private')
            (p/'cycle-plan-0.json').write_text('{"planned":32}')
            export_worker_evidence(root)
            self.assertTrue((root/'fresh-evidence/engineering-attempts.jsonl').exists())
            self.assertTrue((root/'fresh-evidence/cycle-plan-0.json').exists())
            self.assertFalse((root/'fresh-evidence/initial.pt').exists())

    def test_partial_soak_does_not_promote_learning_or_defaults(self):
        from oeis_learn.pipeline_validation.soak_report import report
        r=report({})
        self.assertEqual(r['sustained_operation'],'INCONCLUSIVE')
        self.assertEqual(r['supported_default'],'RETAIN_BASELINE')
        self.assertEqual(r['learning'],'NOT_ADMITTED')
        self.assertEqual(len(r['confirmation']['slots']),20)
        self.assertFalse(r['costs']['within_eighty_percent'])

    def test_drift_uses_all_declared_steady_windows(self):
        from oeis_learn.pipeline_validation.soak_report import window_review
        windows=[dict(leg=leg,ordinal=i,start=i*300,stop=(i+1)*300,complete=True,
            components={'training':dict(tokens=1000,examples=100,seconds=10 if i<8 else 12)})
            for leg in (0,1) for i in range(1,11)]
        r=window_review(windows)
        self.assertEqual([x['verdict'] for x in r],['FAIL','FAIL'])
        self.assertEqual(window_review(windows[:-1])[1]['verdict'],'INCONCLUSIVE')

    def test_long_stress_program_is_valid_but_not_corpus(self):
        from oeis_learn.decoder.bigint_codec import encode_body
        source='(program (local $b0 bigint) '+('(set $b0 (int 1)) '*100)+'(return (get $b0)))'
        self.assertEqual(len(encode_body(source)),1016)

    def test_complete_host_orchestration_shares_state_and_exports_one_archive(self):
        import tempfile,types
        from contextlib import nullcontext
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.soak import SoakStages,prepare_inputs,peak
        from oeis_learn.pipeline_validation.soak_contract import manifest
        from oeis_learn.pipeline_validation.registry import soak_registry,resolve
        from oeis_learn.pipeline_validation.runner import execute
        from oeis_learn.pipeline_validation.identity import write,digest
        from oeis_learn.pipeline_validation.model_sizes import EXECUTION,PROFILES,parameters
        from oeis_learn.pipeline_validation.bundle import verify
        from oeis_learn.fresh.bigint_config import defaults
        root=Path(__file__).resolve().parents[2]
        examples={k:[dict(id=k+str(i),values=['1']*20,indices=list(map(str,range(20))),tokens=[1]*20) for i in range(n)]
            for k,n in [('training',512),('parameter_development',128),('structural_development',128)]}
        facts=dict(physical=128<<30,available=120<<30,limits=[],cpus=16,quota=16,cpuset=16)
        calls=[];leases=[];clock=[1000.]
        class Adapter:
            def __init__(self):
                self.args=types.SimpleNamespace(engine='podman',pipeline_engines=['podman'],run='unused',inputs='unused')
                self.root=root;self.contract=manifest();self.receipts={}
                self.images=types.SimpleNamespace(images={('podman','gpu'):'image'})
                self.fail_operations=False
                self.s=SoakStages(self)
            def host(self,name):return types.SimpleNamespace(policy=dict(memory_bytes=112<<30,cpus=16,render_node='/dev/dri/renderD128'),select_gpu=lambda:None,gpu_lock=nullcontext)
            def __call__(self,stage,output):
                if stage.adapter=='soak-operations' and self.fail_operations:raise PermissionError('fixture UID cannot write')
                if stage.adapter in {'soak-operations','soak-scorer'}:write(output/'fixture.json',{'fixture':True});return {'fixture':True}
                if stage.adapter.startswith('soak-'):return self.s(stage,output)
                write(output/'fixture.json',{'fixture':True});return {'fixture':True}
        def prepare(run,inputs,output,**kw):
            return prepare_inputs(defaults(),examples,output,source=kw['source'],image='image',parent='parent',parent_checkpoint={},data='data')
        def container(host,image,inputs,output,**kw):
            output.mkdir(parents=True,exist_ok=True);cfg=json.loads((inputs/'expected.json').read_text());opt=cfg['options']
            work=json.loads((inputs/'work.json').read_text())['manifest']
            memory=dict(qualified=True,ceiling_bytes=112<<30,breach=None)
            if cfg['mode']=='soak-reference':
                p=output/'private';p.mkdir();(p/'initial.pt').write_bytes(b'initial');(p/'oracle.pt').write_bytes(b'oracle')
                write(output/'reference.json',{'fixture':True});r=dict(status='COMPLETE',execution=EXECUTION)
            else:
                calls.append(opt['kind']);(output/'private').mkdir(exist_ok=True)
                eng=dict(complete=True,examples=512,window=opt['window'],active_seconds=3600 if opt['kind']=='window' else 10,
                    state=dict(components={'training':dict(tokens=2*work['data']['tokens']['training'],seconds=800/opt['batch_size'])},windows=[]))
                if opt['kind']=='window':
                    leases.append(opt['lease'])
                    start=opt['lease']['started']
                    if opt['window']==1:self.assertTrue((output/'private/state-marker').exists())
                    (output/'private/state-marker').write_text('same state')
                    windows=[dict(leg=l,ordinal=i,start=start+l*3600+i*300,planned_stop=start+l*3600+(i+1)*300,
                        stop=start+l*3600+(i+1)*300,complete=True,components={'training':dict(seconds=100,tokens=10000,examples=500)})
                        for l in range(opt['window']+1) for i in range(12)]
                    eng['state']['windows']=windows
                    eng['state']['pulse_slots']=[dict(leg=l,slot=i,state='PASS') for l in range(opt['window']+1) for i in range(6)]
                    samples=[dict(time=start+opt['window']*3600+i,anon=100,file=100,current=100,driver=100,
                        cpu_usage_usec=i*1000000,gpu_busy_percent=80) for i in range(3600)]
                    (output/'resource-samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
                    clock[0]+=3600
                r=dict(status='COMPLETE',execution=EXECUTION,profile=PROFILES['m100'],source=cfg['source'],work=work['identity'],
                    device={'hip':'fixture'},resources=memory,trainer=dict(engineering=eng,counts={'actual':parameters(PROFILES['m100'])},
                        final_parameter_digest='state',initial_parameter_digest='state'))
                if opt['kind']=='parity':r['numerical_parity']=True
            write(output/'summary.json',seal(r));write(output/'host-timing.json',{'seconds_through_cleanup':10})
        with tempfile.TemporaryDirectory() as td,patch('oeis_learn.pipeline_validation.benchmarks.legacy',return_value=types.SimpleNamespace(prepare=prepare,container_stage=container)),patch('oeis_learn.pipeline_validation.resources.observe',return_value=facts),patch('time.monotonic',side_effect=lambda:clock[0]):
            a=Adapter();out=Path(td)/'run'
            code=execute(resolve(soak_registry(('podman',),'podman'),'all-ready'),out,adapter=a,binding={'source':'fixture'},cap=86400,reserve=1200,contract=a.contract)
            self.assertEqual(code,0)
            self.assertEqual(calls.count('batch'),12);self.assertEqual(calls.count('window'),2)
            self.assertEqual(leases[0],leases[1]);verify(out.with_suffix('.zip'))
            calls.clear();failed=Adapter();failed.fail_operations=True;out=Path(td)/'failed'
            code=execute(resolve(soak_registry(('podman',),'podman'),'all-ready'),out,adapter=failed,binding={'source':'fixture'},cap=86400,reserve=1200,contract=failed.contract)
            self.assertNotEqual(code,0);self.assertEqual(calls,[])
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['stages']['soak.prepare']['state'],'BLOCKED')
            self.assertEqual(summary['stages']['soak.report']['state'],'INCONCLUSIVE')
            verify(out.with_suffix('.zip'))
    def test_continuation_cannot_reset_counters_or_lose_resume_state(self):
        from oeis_learn.fresh.engineering import Control
        from oeis_learn.pipeline_validation.soak_contract import lease
        h='sha256:'+'a'*64
        clock=[100.]
        l=lease(h,h,h,now=100)
        cfg=seal(dict(version='pipeline-engineering-worker/v1',source=h,work=h,choice=h,mode='window',
            batch_size=8,window=0,resume=False,lease=l,token_target=1,seconds=3600,
            instrumented=False,cadence_tokens=10))
        c=Control({'engineering':cfg},{},Path('/unused'),clock=lambda:clock[0])
        c.update([{'tokens':[1]*10}],.1);c.cycled();clock[0]=3700
        first=c.finish();self.assertTrue(first['complete'])
        second=seal(dict(cfg,window=1,resume=True))
        resumed=Control({'engineering':second},{'engineering':copy.deepcopy(first['state'])},Path('/unused'),clock=lambda:clock[0])
        self.assertEqual(resumed.state['tokens'],10)
        self.assertEqual(resumed.deadline,7300)
        with self.assertRaises(ValueError):Control({'engineering':second},{},Path('/unused'),clock=lambda:clock[0])

    def test_ordinary_timer_does_not_synchronize_gpu(self):
        import ast
        path=Path(__file__).resolve().parents[2]/'src/oeis_learn/tracking/foundation_metrics.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PhaseTimer')
        namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        # Execute the actual class without importing its container-owned dependencies.
        values=iter([1,4]);timer=namespace['PhaseTimer']('cuda',clock=lambda:next(values),synchronized=False)
        with timer.phase('ordinary'):pass
        self.assertEqual(timer.values,{'ordinary':3})

    def test_registered_scope_keeps_learning_denied(self):
        from oeis_learn.pipeline_validation.contracts import parse
        from oeis_learn.pipeline_validation.soak_contract import manifest
        value=manifest()
        self.assertEqual(parse(value)['scope'],'engineering-soak')
        for field in ('training','resume','final'):
            bad=copy.deepcopy(value);bad['authority'][field]=True
            with self.assertRaises(ValueError):parse(seal(bad))

    def test_roster_and_phase_reservations_fit(self):
        from oeis_learn.pipeline_validation.registry import soak_registry,resolve
        from oeis_learn.pipeline_validation.soak_contract import SPEC
        stages=resolve(soak_registry(('podman','docker'),'podman'),'all-ready')
        self.assertEqual(len({s.id for s in stages}),len(stages))
        self.assertEqual(len([s for s in stages if s.adapter=='soak-batch']),12)
        self.assertEqual(SPEC['work_windows_seconds'],[3600,3600])
        for phase,cap in [(1,3600),(2,10800),(3,7200),(4,18000),(5,14400)]:
            self.assertLessEqual(sum(s.seconds for s in stages if s.phase==phase and s.adapter!='image'),cap)

    def test_selection_requires_all_trials_and_prefers_small_within_band(self):
        from oeis_learn.pipeline_validation.soak_contract import select_batch
        rows={str(b):[dict(complete=True,tokens=100,seconds=1/rate) for _ in range(3)] for b,rate in [(8,1),(16,1.03),(32,1.2)]}
        self.assertEqual(select_batch(rows),32)
        rows['32'][1]['complete']=False
        self.assertEqual(select_batch(rows),8)
        rows['8'][0]['seconds']=float('nan')
        self.assertEqual(select_batch(rows),16)

    def test_lease_does_not_renew_across_restart(self):
        from oeis_learn.pipeline_validation.soak_contract import lease,remaining
        l=lease(*['sha256:'+'a'*64]*3,now=100)
        self.assertEqual(remaining(l,now=4000),3900)
        self.assertEqual(remaining(l,now=7900),0)
        bad=dict(l,expires=8000)
        with self.assertRaises(ValueError):remaining(bad,now=4000)

    def test_entire_partition_and_token_cadence_are_frozen(self):
        from oeis_learn.pipeline_validation.soak_contract import data_contract
        examples={k:[dict(id=k+str(i),values=['1']*20,indices=list(map(str,range(20))),tokens=[1]*((i%3)+1)) for i in range(n)]
                  for k,n in [('training',513),('parameter_development',128),('structural_development',128)]}
        d=data_contract(examples)
        self.assertEqual(d['counts']['training'],513)
        self.assertEqual(d['cadence_tokens'],8*sum(len(r['tokens']) for r in examples['training']))
        bad=copy.deepcopy(examples);bad['training']=bad['training'][:128]
        with self.assertRaises(ValueError):data_contract(bad)


if __name__=='__main__':unittest.main()

class FailedInspection(unittest.TestCase):
    def run_case(self,*,launch_failure,remove_failure=None):
        import tempfile,types,subprocess
        from unittest.mock import Mock,patch
        from oeis_learn.pipeline_validation.benchmarks import legacy
        from oeis_learn.pipeline_validation.runner import ResourceBreach
        module=legacy(Path(__file__).resolve().parents[2])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inputs=root/'input';inputs.mkdir();out=root/'out';out.mkdir()
            engine=Mock();engine.command.return_value=['fixture-engine']
            inspection_error=subprocess.CalledProcessError(125,['fixture-engine','inspect','missing'],stderr=b'container unavailable',output=b'')
            engine.call.side_effect=inspection_error
            if remove_failure:engine.remove.side_effect=remove_failure
            host=types.SimpleNamespace(engine=engine,policy=dict(memory_bytes=1<<30,cpus=2))
            with patch('oeis_learn.pipeline_validation.supervision.launch',side_effect=launch_failure):
                expected=ResourceBreach if remove_failure else type(launch_failure) if launch_failure else subprocess.CalledProcessError
                with self.assertRaises(expected) as raised:
                    module.container_stage(host,'image',inputs,out,mode='fixture',seconds=1,gpu=False,argv=['python','-c','fixture'])
            audit=json.loads((out/'isolation-audit.json').read_text());detail=json.loads((out/'inspection-error.json').read_text())
            cleanup=json.loads((out/'cleanup.json').read_text())
            self.assertEqual(audit['status'],'INCOMPLETE');self.assertEqual(detail['returncode'],125)
            self.assertEqual(detail['stderr'],'container unavailable')
            self.assertEqual(cleanup['acknowledged'],not bool(remove_failure));engine.remove.assert_called_once()
            self.assertFalse((out/'container-inspect.json').exists())
            if remove_failure:self.assertFalse(cleanup['within_bound'])
            elif launch_failure:self.assertIs(raised.exception,launch_failure)
            else:self.assertIs(raised.exception,inspection_error)
            return detail

    def test_inspection_failure_retains_primary_timeout_and_fatal_resource_error(self):
        from oeis_learn.pipeline_validation.runner import ResourceBreach
        for error in (TimeoutError('original launch deadline'),ResourceBreach('original allocation failure')):
            with self.subTest(error=type(error).__name__):
                detail=self.run_case(launch_failure=error)
                self.assertIn(str(error),detail['launcher_error'])

    def test_successful_launch_without_inspection_remains_failure(self):
        detail=self.run_case(launch_failure=None)
        self.assertIsNone(detail['launcher_error'])

    def test_failed_reclamation_remains_fatal_after_missing_inspection(self):
        self.run_case(launch_failure=TimeoutError('original launch deadline'),remove_failure=RuntimeError('no acknowledgement'))
