"""Host-only soak authority, work, selection and cumulative recovery contracts."""
import copy
import json
from pathlib import Path
import unittest

from oeis_learn.pipeline_validation.identity import seal


class SoakContracts(unittest.TestCase):
    def test_foreign_uid_preserves_rootless_security_arguments(self):
        from oeis_learn.pipeline_validation.soak_operations import foreign_user_args,fixture_code
        before=['--user','0:0','--userns=keep-id','--security-opt=label=disable']
        after=foreign_user_args(before)
        self.assertEqual(after,['--user','12345:12345','--userns=keep-id','--security-opt=label=disable'])
        self.assertEqual(before[1],'0:0')
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
                self.s=SoakStages(self)
            def host(self,name):return types.SimpleNamespace(policy=dict(memory_bytes=112<<30,cpus=16,render_node='/dev/dri/renderD128'),select_gpu=lambda:None,gpu_lock=nullcontext)
            def __call__(self,stage,output):
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
