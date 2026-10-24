"""Fixed-work hardware batching, closed capacity/authority and complete reports."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.pipeline_validation.replay import replay_report
from oeis_learn.pipeline_validation.replay_batching import (
    batch_plan, BATCH_SLOTS, freeze_batching, validate_batching, batching_report,
)
from tests.unit.test_pipeline_replay_comparison import baseline_fixture

ROOT=Path(__file__).resolve().parents[2]


def fixture():
    ref,rows,_=baseline_fixture()
    for row in rows.values():
        value=row['observations']['result']
        value.update(batch_size=8,bookkeeping='hoisted/v1',boundaries=160)
        value['resources']['role_peak_upper_bytes']=2<<30
        row['observations']['result']=seal(value)
    return ref,rows,replay_report(rows,ref['identity'],ref['workload'])


class Batching(unittest.TestCase):
    def test_preparation_reads_all32_visible_rows_without_hidden_payloads(self):
        import ast
        import importlib.util
        import tempfile
        source=ROOT/'tests/unit/test_pipeline_profile.py'
        function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parent_fixture')
        namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        spec=importlib.util.spec_from_file_location('batching_prepare_test',ROOT/'scripts/profile_bigint_pipeline.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            run,inputs,output=namespace['parent_fixture'](Path(directory))
            (inputs/'private/synthetic-truth.json').unlink()
            module.prepare(run,inputs,output,image='image',source='source',render_node=None,replay='batching/v1')
            work=json.loads((output/'input/work.json').read_text())
            self.assertEqual(len(work['prompts']),32)
            self.assertTrue(all(set(p)=={'id','values'} and len(p['values'])==20 for p in work['prompts']))
            self.assertFalse((output/'private').exists())

    def test_role_parser_binds_batch_and_child_before_gpu_dispatch(self):
        import tempfile
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.replay_batching import comparison_role
        ref,baseline,report=fixture();child=freeze_batching(report,baseline,ref);slot=BATCH_SLOTS[2]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,value in [('reference.json',ref),('comparison-contract.json',child),('comparison-slot.json',slot)]:
                (root/name).write_text(json.dumps(value))
            cfg=dict(source=child['source'],contract=child['validation'],options=dict(
                seconds=child['limits']['internal_seconds'],memory_ceiling=16<<30,render_node=None,
                comparison=child['identity'],slot=slot['id'],batch_size=16))
            value=dict(device=child['resource_device']['device'],torch_threads=child['threads'])
            with patch('oeis_learn.pipeline_validation.replay_batching.batch_role',return_value=value) as run:
                result=comparison_role(root,root,cfg)
                self.assertEqual(result['comparison'],child['identity'])
                self.assertEqual(run.call_args.args[2]['options']['batch_size'],16)
                for field,value in [('batch_size',8),('comparison','changed'),('seconds',570)]:
                    changed=copy.deepcopy(cfg);changed['options'][field]=value
                    with self.assertRaises(ValueError):comparison_role(root,root,changed)
                self.assertEqual(run.call_count,1)

    def test_fixed_rows_tokens_and_bounded_manifest_graph(self):
        from oeis_learn.pipeline_validation.contracts import parse
        plan=batch_plan(1)
        self.assertEqual(len(plan['tokens']),32)
        self.assertEqual(sum(map(len,plan['tokens'])),32768)
        self.assertEqual(plan['positions'],[1,32,128,513,1024])
        self.assertEqual(plan['bookkeeping'],'hoisted/v1')
        path='configs/foundation/pipeline_replay_batching.json'
        m=json.loads((ROOT/path).read_text());parse(m)
        for field,value in [('comparisons',[]),('workloads',['cache-replay/v1']),('scope','engineering')]:
            with self.assertRaises(ValueError):parse(seal(dict(m,**{field:value})))
        result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/foundation_run.py'),'--engine','podman','validate','--suite','pipeline','--manifest',path,'--phase','all-ready','--engines','selected','--scope-report'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        graph=json.loads(result.stdout)
        self.assertEqual(graph['scope'],'engineering-replay-batching')
        self.assertEqual(len([s for s in graph['selected'] if s['adapter']=='replay-batching-trial']),10)
        self.assertLessEqual(sum(s['seconds'] for s in graph['selected'] if s['phase']==2 and s['adapter']!='image'),10800)

    def test_capacity_and_complete_baseline_required_before_candidate(self):
        ref,rows,report=fixture();child=freeze_batching(report,rows,ref)
        validate_batching(child)
        self.assertEqual(child['slots'],BATCH_SLOTS)
        self.assertEqual([s['batch_size'] for s in BATCH_SLOTS],[16,8,16,16,8])
        self.assertEqual(child['capacity']['candidate_peak_bytes'],8<<30)
        self.assertEqual(child['limits']['external_seconds'],120)
        for field,value in [('batch_size',16),('boundaries',159),('bookkeeping','baseline')]:
            changed=copy.deepcopy(rows)
            r=next(iter(changed.values()))['observations']['result'];r[field]=value
            next(iter(changed.values()))['observations']['result']=seal(r)
            with self.assertRaises(ValueError):freeze_batching(replay_report(changed,ref['identity'],ref['workload']),changed,ref)
        changed=copy.deepcopy(rows)
        for r in changed.values():
            v=r['observations']['result'];v['resources']['role_peak_upper_bytes']=8<<30;r['observations']['result']=seal(v)
        with self.assertRaisesRegex(ValueError,'capacity'):freeze_batching(replay_report(changed,ref['identity'],ref['workload']),changed,ref)

    def test_partial_unequal_or_unqualified_work_cannot_produce_throughput(self):
        ref,baseline,report=fixture();child=freeze_batching(report,baseline,ref)
        rows={}
        for slot in BATCH_SLOTS:
            v=copy.deepcopy(next(iter(baseline.values()))['observations']['result'])
            v.update(batch_size=slot['batch_size'],comparison=child['identity'],slot=slot['id'],verified_seconds=20 if slot['arm']=='B' else 30,decode_seconds=19)
            rows[slot['id']]=dict(state='PASS',identity=slot['id'],observations={'result':seal(v)})
        good=batching_report(child,rows)
        self.assertEqual(good['status'],'COMPLETE')
        self.assertEqual(good['supplied_tokens'],32768)
        self.assertFalse(good['optimization_adoption'])
        self.assertFalse(good['research_cost_admission'])
        for slot in BATCH_SLOTS:
            changed=copy.deepcopy(rows);changed.pop(slot['id'])
            bad=batching_report(child,changed)
            self.assertEqual(bad['status'],'INCONCLUSIVE')
            self.assertIsNone(bad['paired_gain'])
        changed=copy.deepcopy(rows);r=changed[BATCH_SLOTS[2]['id']]['observations']['result'];r['batch_size']=8;changed[BATCH_SLOTS[2]['id']]['observations']['result']=seal(r)
        self.assertEqual(batching_report(child,changed)['status'],'INCONCLUSIVE')

