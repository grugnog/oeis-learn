"""Combined finite hardware hypotheses, mode/capacity selection and no promotion."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.pipeline_validation.replay import TRIALS
from oeis_learn.pipeline_validation.hardware_ladder import (
    ladder_plan, mode_report, freeze_child, validate_child, branch_report, slots, BASIS,
)

ROOT=Path(__file__).resolve().parents[2]


def fixture():
    ref=seal(dict(source='source',contract='validation',workload='work',weights='weights'))
    def result(seconds,measurement,batch=16):
        return seal(dict(status='COMPLETE',reference=ref['identity'],workload='work',
            verified_seconds=seconds,decode_seconds=seconds-1,batch_size=batch,boundaries=640,
            bookkeeping='hoisted/v1',measurement_mode=measurement,
            torch_threads={'intraop':2,'interop':1},device={'hip':'test','name':'gpu'},
            resources={'qualified':True,'ceiling_bytes':32<<30,'role_peak_upper_bytes':2<<30}))
    rows={key:dict(state='PASS',identity=key,observations={'result':result(100,'aggregate1Hz/v1')}) for key in TRIALS}
    rows.update({key:dict(state='PASS',identity=key,observations={'result':result(seconds,mode)}) for key,seconds,mode in [
        ('hardware.overhead.ordinary',100,'ordinary/v1'),('hardware.overhead.telemetry',101,'aggregate1Hz/v1')]})
    return ref,rows,result


class Ladder(unittest.TestCase):
    def run_host(self,*,durations,telemetry=101,fail_at=None):
        import io
        import tempfile
        from contextlib import nullcontext,redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from oeis_learn.pipeline_validation.hardware_ladder import HardwareStages
        from oeis_learn.pipeline_validation.registry import hardware_registry,resolve
        from oeis_learn.pipeline_validation.runner import execute,ResourceBreach
        from oeis_learn.pipeline_validation.identity import write
        from oeis_learn.pipeline_validation.bundle import verify
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);contract=json.loads((ROOT/'configs/foundation/pipeline_hardware_ladder.json').read_text())
            host=SimpleNamespace(policy={'render_node':'/dev/dri/renderD128','memory_bytes':32<<30,'cpus':16},select_gpu=Mock(),gpu_lock=lambda:nullcontext())
            a=SimpleNamespace(root=root,args=SimpleNamespace(run='parent',inputs='data',engine='podman'),contract=contract,
                images=SimpleNamespace(images={('podman','gpu'):'image'}),host=lambda _:host,receipts={})
            def prepare(run,data,output,**kw):
                self.assertEqual(kw['replay'],'ladder/v1')
                inputs=output/'input';inputs.mkdir();(inputs/'weights.pt').write_bytes(b'frozen')
                m=seal(dict(binding={'source':'source'},weights='weights',plan=ladder_plan(1)))
                write(inputs/'work.json',{'manifest':m});return m
            def container(h,image,inputs,output,**kw):
                cfg=json.loads((inputs/'expected.json').read_text());options=cfg['options'];output.mkdir()
                slot=json.loads((inputs/'comparison-slot.json').read_text()) if (inputs/'comparison-slot.json').exists() else None
                key=slot['id'] if slot else str(output.parent.name);calls.append((key,options['batch_size'],options['measurement']))
                if slot and fail_at==slot['id']:raise RuntimeError('injected candidate failure')
                if slot and fail_at=='mutate-child' and slot['id']==slots(32)[0]['id']:
                    (output.parents[2]/'comparison-contract.json').write_text('{"identity":"changed"}')
                if not kw['gpu']:
                    self.assertEqual(kw['mode'],'hardware-reference')
                    (output/'private').mkdir();(output/'private/reference.pt').write_bytes(b'reference')
                    ref=seal(dict(workload='work',source='source',contract=contract['identity'],weights='weights'))
                    write(output/'reference.json',ref)
                else:
                    ref=json.loads((inputs/'reference.json').read_text())
                seconds=telemetry if key==OVERHEAD_KEY else durations[options['batch_size']]
                result=dict(status='COMPLETE',reference=ref['identity'],workload='work',verified_seconds=seconds,decode_seconds=seconds-1,
                    batch_size=options['batch_size'],boundaries=640,bookkeeping='hoisted/v1',measurement_mode=options['measurement'],
                    resources={'qualified':True,'ceiling_bytes':32<<30,'role_peak_upper_bytes':2<<30},
                    device={'hip':'test','name':'gpu'},torch_threads={'intraop':2,'interop':1})
                if slot:
                    child=json.loads((inputs/'comparison-contract.json').read_text())
                    self.assertEqual(options['comparison'],child['identity']);self.assertEqual(options['slot'],slot['id'])
                    self.assertEqual(kw['seconds'],child['limits']['external_seconds'])
                    self.assertEqual(options['seconds'],child['limits']['internal_seconds'])
                    result.update(comparison=child['identity'],slot=slot['id'])
                write(output/'summary.json',seal(result))
            OVERHEAD_KEY='hardware.overhead.telemetry'
            module=SimpleNamespace(prepare=prepare,container_stage=container)
            observations=dict(physical=64<<30,available=48<<30,cpus=16,limits=[],quota=None,cpuset=None)
            hardware=HardwareStages(a)
            class Harness:
                receipts={}
                def __call__(self,stage,output):
                    a.receipts=self.receipts
                    return hardware(stage,output) if stage.adapter.startswith('hardware-') else {'fixture':'prerequisite'}
            with patch('oeis_learn.pipeline_validation.benchmarks.legacy',return_value=module),patch('oeis_learn.pipeline_validation.resources.observe',return_value=observations),patch('oeis_learn.experiments.source_snapshot.collect_snapshot',return_value={}),redirect_stdout(io.StringIO()):
                code=execute(resolve(hardware_registry(('podman',),'podman'),'all-ready'),root/'out',adapter=Harness(),
                    binding={'source':'source'},cap=86400,reserve=1200,contract=contract)
            verify(root/'out.zip')
            summary=json.loads((root/'out/summary.json').read_text())
            report=json.loads((root/'out/stages/hardware.report/hardware-report.json').read_text())
            nested={}
            for batch in (32,64,128):
                for slot in slots(batch):
                    path=root/'out/stages'/f'hardware.b{batch}'/'roles'/slot['id']/'receipt.json'
                    if path.exists():nested[slot['id']]=json.loads(path.read_text())['state']
            return code,summary,report,nested,calls

    def test_combined_host_runs_all_branches_in_one_verified_archive(self):
        code,summary,report,nested,calls=self.run_host(durations={16:100,32:50,64:25,128:20})
        self.assertEqual(code,0);self.assertEqual(summary['scope'],'engineering-hardware')
        self.assertEqual(report['experimental_batch'],128)
        self.assertEqual(len(calls),23)  # CPU reference,2 observation,5 baseline,15 paired roles.
        self.assertEqual(set(nested.values()),{'PASS'})
        self.assertEqual(len(nested),15)

    def test_mode_fallback_and_stopping_leave_unexecuted_roles_visible(self):
        code,summary,report,nested,calls=self.run_host(durations={16:100,32:90,64:88,128:80},telemetry=106)
        self.assertEqual(code,0);self.assertEqual(report['measurement']['mode'],'ordinary/v1')
        self.assertEqual(report['experimental_batch'],32)  # Within5% of64, prefer smaller.
        self.assertEqual(report['branches']['128']['status'],'NOT_RUN')
        self.assertTrue(all(nested[s['id']]=='NOT_SELECTED' for s in slots(128)))
        self.assertFalse(any(key.startswith('hardware.b128.') for key,_,_ in calls))
        self.assertTrue(all(mode=='ordinary/v1' for key,_,mode in calls if key not in {'hardware.reference','hardware.overhead.telemetry'}))

    def test_failed_nested_role_blocks_remaining_pairs_and_larger_branches(self):
        code,summary,report,nested,calls=self.run_host(durations={16:100,32:50,64:25,128:20},fail_at=slots(32)[2]['id'])
        self.assertEqual(code,1)
        self.assertEqual(summary['stages']['hardware.b64']['state'],'BLOCKED')
        self.assertEqual(nested[slots(32)[2]['id']],'FAIL')
        self.assertTrue(all(nested[s['id']]=='BLOCKED' for s in slots(32)[3:]))
        self.assertIsNone(report['branches']['32']['paired_gain'])

    def test_changed_child_stops_before_any_scored_pair(self):
        code,summary,report,nested,calls=self.run_host(durations={16:100,32:50,64:25,128:20},fail_at='mutate-child')
        self.assertEqual(code,1)
        self.assertEqual(nested[slots(32)[0]['id']],'PASS')
        self.assertTrue(all(nested[s['id']]=='BLOCKED' for s in slots(32)[1:]))
        self.assertFalse(any(key==slots(32)[1]['id'] for key,_,_ in calls))

    def test_role_authority_and_weights_are_checked_before_numeric_dispatch(self):
        import tempfile
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.hardware_ladder import hardware_role
        ref,rows,_=fixture();measurement=mode_report(rows,ref['identity'],'work')
        child=freeze_child(rows,ref,measurement,32,weight_bytes=1024);slot=slots(32)[2]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'weights.pt').write_bytes(b'x'*1024)
            for name,value in [('reference.json',ref),('comparison-contract.json',child),('comparison-slot.json',slot)]:
                (root/name).write_text(json.dumps(value))
            cfg=dict(source='source',contract='validation',options=dict(seconds=child['limits']['internal_seconds'],memory_ceiling=32<<30,
                render_node=None,batch_size=32,measurement=measurement['mode'],comparison=child['identity'],slot=slot['id']))
            value=dict(device=child['resource_device']['device'],torch_threads=child['threads'])
            with patch('oeis_learn.pipeline_validation.hardware_ladder.numeric_role',return_value=value) as run:
                self.assertEqual(hardware_role(root,root,cfg)['comparison'],child['identity'])
                for field,value in [('batch_size',128),('comparison','other'),('measurement','ordinary/v1'),('seconds',570)]:
                    bad=copy.deepcopy(cfg);bad['options'][field]=value
                    with self.assertRaises(ValueError):hardware_role(root,root,bad)
                (root/'weights.pt').write_bytes(b'x'*1025)
                with self.assertRaises(ValueError):hardware_role(root,root,cfg)
                self.assertEqual(run.call_count,1)

    def test_visible_basis_is32_rows_and_no_private_truth_is_read(self):
        import ast
        import importlib.util
        import tempfile
        source=ROOT/'tests/unit/test_pipeline_profile.py'
        fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parent_fixture')
        namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        spec=importlib.util.spec_from_file_location('hardware_prepare_test',ROOT/'scripts/profile_bigint_pipeline.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            run,inputs,out=namespace['parent_fixture'](Path(directory));(inputs/'private/synthetic-truth.json').unlink()
            module.prepare(run,inputs,out,image='image',source='source',render_node=None,replay='ladder/v1')
            work=json.loads((out/'input/work.json').read_text())
            self.assertEqual(len(work['prompts']),32)
            self.assertEqual(len(work['manifest']['plan']['tokens']),128)
            self.assertFalse((out/'private').exists())

    def test_fixed_work_modes_and_closed_public_graph(self):
        from oeis_learn.pipeline_validation.contracts import parse
        plan=ladder_plan(1)
        self.assertEqual(plan['visible_rows'],32)
        self.assertEqual(plan['row_mapping'],[r%32 for r in range(128)])
        self.assertEqual(sum(map(len,plan['tokens'])),131072)
        path='configs/foundation/pipeline_hardware_ladder.json';m=json.loads((ROOT/path).read_text());parse(m)
        for field,value in [('comparisons',[]),('workloads',[]),('scope','engineering')]:
            with self.assertRaises(ValueError):parse(seal(dict(m,**{field:value})))
        r=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/foundation_run.py'),'--engine','podman','validate','--suite','pipeline','--manifest',path,'--engines','selected','--scope-report'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);graph=json.loads(r.stdout)
        self.assertEqual(graph['scope'],'engineering-hardware')
        self.assertLessEqual(sum(s['seconds'] for s in graph['selected'] if s['phase']==2 and s['adapter']!='image'),10800)
        self.assertEqual([s['id'] for s in graph['selected'] if s['adapter']=='hardware-branch'],['hardware.b32','hardware.b64','hardware.b128'])

    def test_mode_selection_requires_complete_matching_work(self):
        ref,rows,result=fixture();report=mode_report(rows,ref['identity'],'work')
        self.assertEqual(report['mode'],'aggregate1Hz/v1')
        rows['hardware.overhead.telemetry']['observations']['result']=result(105,'aggregate1Hz/v1')
        self.assertEqual(mode_report(rows,ref['identity'],'work')['mode'],'aggregate1Hz/v1')
        rows['hardware.overhead.telemetry']['observations']['result']=result(106,'aggregate1Hz/v1')
        self.assertEqual(mode_report(rows,ref['identity'],'work')['mode'],'ordinary/v1')
        rows.pop('hardware.overhead.ordinary')
        self.assertEqual(mode_report(rows,ref['identity'],'work')['status'],'INCONCLUSIVE')

    def test_child_freezes_capacity_roster_mode_and_prior_basis(self):
        ref,rows,_=fixture();mode=mode_report(rows,ref['identity'],'work')
        for batch in (32,64,128):
            child=freeze_child(rows,ref,mode,batch,weight_bytes=1024)
            validate_child(child)
            self.assertEqual(child['slots'],slots(batch))
            self.assertEqual(child['basis'],BASIS)
            self.assertLessEqual(child['limits']['external_seconds'],340)
            for field,value in [('candidate_batch',256),('measurement',dict(mode,mode='ordinary/v1'))]:
                with self.assertRaises(ValueError):validate_child(seal(dict(child,**{field:value})))
        changed=copy.deepcopy(rows)
        for key in TRIALS:
            r=changed[key]['observations']['result'];r['resources']['role_peak_upper_bytes']=8<<30
            changed[key]['observations']['result']=seal(r)
        with self.assertRaisesRegex(ValueError,'capacity'):freeze_child(changed,ref,mode,128,weight_bytes=1024)

    def test_expansion_stops_without_favorable_subset_or_promotion(self):
        ref,baseline,result=fixture();mode=mode_report(baseline,ref['identity'],'work')
        child=freeze_child(baseline,ref,mode,32,weight_bytes=1024);rows={}
        for slot in slots(32):
            value=result(50 if slot['arm']=='B' else 100,mode['mode'],slot['batch_size'])
            value=seal(dict(value,comparison=child['identity'],slot=slot['id']))
            rows[slot['id']]=dict(state='PASS',identity=slot['id'],observations={'result':value})
        good=branch_report(child,rows,previous_seconds=100)
        self.assertTrue(good['advance']);self.assertFalse(good['optimization_adoption'])
        self.assertFalse(good['research_cost_admission'])
        self.assertFalse(branch_report(child,rows,previous_seconds=51)['advance'])
        rows.pop(slots(32)[2]['id']);bad=branch_report(child,rows,previous_seconds=100)
        self.assertEqual(bad['status'],'INCONCLUSIVE');self.assertIsNone(bad['paired_gain'])

