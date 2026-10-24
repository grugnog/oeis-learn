"""Prospective component comparison contracts; no hardware claims from fixtures."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.pipeline_validation.replay import TRIALS, replay_report
from oeis_learn.pipeline_validation.replay_comparison import (
    SLOTS, freeze_comparison, validate_comparison, comparison_report,
)

ROOT=Path(__file__).resolve().parents[2]


def baseline_fixture():
    ref=seal(dict(source='source',contract='validation',workload='work',weights='weights'))
    rows={key:dict(state='PASS',identity=key,observations={'result':seal(dict(
        status='COMPLETE',reference=ref['identity'],workload='work',verified_seconds=30,
        resources={'qualified':True,'ceiling_bytes':16<<30},device={'hip':'test','name':'gpu'},
        torch_threads={'intraop':2,'interop':1}))}) for key in TRIALS}
    return ref,rows,replay_report(rows,ref['identity'],'work')


class ReplayComparison(unittest.TestCase):
    def test_isolated_parser_requires_frozen_slot_reference_and_setup_deadline(self):
        import tempfile
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.replay_comparison import comparison_role
        ref,rows,report=baseline_fixture();child=freeze_comparison(report,rows,ref)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'reference.json').write_text(json.dumps(ref))
            (root/'comparison-contract.json').write_text(json.dumps(child))
            slot=SLOTS[2];(root/'comparison-slot.json').write_text(json.dumps(slot))
            cfg=dict(source=child['source'],contract=child['validation'],options=dict(
                seconds=child['limits']['internal_seconds'],memory_ceiling=16<<30))
            value=dict(device=child['resource_device']['device'],torch_threads=child['threads'])
            with patch('oeis_learn.pipeline_validation.replay.replay_role',return_value=value) as run:
                result=comparison_role(root,root,cfg)
                self.assertEqual(result['comparison'],child['identity'])
                self.assertEqual(run.call_args.kwargs,{'bookkeeping':'hoisted/v1'})
                for bad in [dict(cfg,source='changed'),dict(cfg,options=dict(cfg['options'],seconds=570))]:
                    with self.assertRaises(ValueError):comparison_role(root,root,bad)
                (root/'comparison-slot.json').write_text(json.dumps(dict(slot,bookkeeping='baseline')))
                with self.assertRaises(ValueError):comparison_role(root,root,cfg)
                self.assertEqual(run.call_count,1)

    def test_host_child_is_immutable_and_no_candidate_runs_before_freeze(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import Mock
        from oeis_learn.pipeline_validation.replay_comparison import ComparisonStages
        from oeis_learn.pipeline_validation.registry import Stage
        ref,rows,_=baseline_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reference.json').write_text(json.dumps(ref))
            replay=Mock();replay.a=SimpleNamespace(receipts=rows);replay.reference=root
            stages=ComparisonStages(replay)
            trial=Stage(SLOTS[0]['id'],2,seconds=600,adapter='replay-comparison-trial')
            with self.assertRaisesRegex(ValueError,'freeze'):stages(trial,root)
            frozen=root/'frozen';frozen.mkdir()
            child=stages(Stage('replay.compare.freeze',2,adapter='replay-comparison-freeze'),frozen)
            stages(trial,root)
            self.assertEqual(replay.call_args.kwargs['comparison'],(child,SLOTS[0]))
            changed=dict(child,slots=list(reversed(SLOTS)))
            (frozen/'comparison-contract.json').write_text(json.dumps(seal(changed)))
            with self.assertRaisesRegex(ValueError,'changed before dispatch'):stages(trial,root)
            self.assertEqual(replay.call_count,1)

    def test_closed_manifest_and_public_graph(self):
        from oeis_learn.pipeline_validation.contracts import parse
        from oeis_learn.pipeline_validation.registry import replay_comparison_registry, resolve
        path='configs/foundation/pipeline_replay_comparison.json'
        manifest=json.loads((ROOT/path).read_text());parse(manifest)
        for field,value in [('comparisons',[]),('workloads',[]),('scope','engineering-replay')]:
            bad=copy.deepcopy(manifest);bad[field]=value
            with self.assertRaises(ValueError):parse(seal(bad))
        graph=resolve(replay_comparison_registry(('podman',),'podman'),'all-ready')
        ids=[s.id for s in graph]
        self.assertEqual([s.id for s in graph if s.adapter=='replay-comparison-trial'],[s['id'] for s in SLOTS])
        self.assertLess(ids.index('replay.report'),ids.index('replay.compare.freeze'))
        self.assertFalse({s.adapter for s in graph}&{'inference','native','joint','study','cost'})
        self.assertLessEqual(sum(s.seconds for s in graph if s.phase==2 and s.adapter!='image'),10800)
        result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/foundation_run.py'),'--engine','podman','validate','--suite','pipeline','--manifest',path,'--phase','all-ready','--engines','selected','--scope-report'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['scope'],'engineering-replay-comparison')

    def test_freeze_requires_complete_baselines_and_separates_setup_allowances(self):
        ref,rows,report=baseline_fixture()
        child=freeze_comparison(report,rows,ref)
        validate_comparison(child)
        self.assertEqual(child['limits'],dict(component_seconds=60,internal_seconds=90,external_seconds=120))
        self.assertEqual(child['slots'],SLOTS)
        self.assertEqual([s['arm'] for s in SLOTS],['B','A','B','B','A'])
        for key in TRIALS:
            changed=copy.deepcopy(rows);changed[key]['state']='PARTIAL'
            with self.assertRaises(ValueError):freeze_comparison(report,changed,ref)
        changed=copy.deepcopy(rows)
        changed[TRIALS[-1]]['observations']['result']['torch_threads']['intraop']=8
        with self.assertRaises(ValueError):freeze_comparison(report,changed,ref)
        changed=copy.deepcopy(report);changed['calibration']['reservation_seconds']=600
        with self.assertRaises(ValueError):freeze_comparison(seal(changed),rows,ref)

    def test_child_policy_order_and_deadlines_are_closed(self):
        ref,rows,report=baseline_fixture();child=freeze_comparison(report,rows,ref)
        for field,value in [('slots',list(reversed(SLOTS))),('measurement','ordinary'),('limits',dict(component_seconds=60,internal_seconds=570,external_seconds=600))]:
            bad=copy.deepcopy(child);bad[field]=value
            with self.assertRaises(ValueError):validate_comparison(seal(bad))
        bad=dict(child,training=True)
        with self.assertRaises(ValueError):validate_comparison(seal(bad))

    def test_report_keeps_unstarted_pairs_and_denies_adoption_even_when_faster(self):
        ref,baseline,base_report=baseline_fixture();child=freeze_comparison(base_report,baseline,ref)
        rows={}
        for slot in SLOTS:
            result=seal(dict(status='COMPLETE',comparison=child['identity'],slot=slot['id'],
                bookkeeping=slot['bookkeeping'],reference=child['reference'],workload=child['workload'],
                verified_seconds=30 if slot['arm']=='A' else 20,decode_seconds=18,boundaries=12,
                resources={'qualified':True,'ceiling_bytes':16<<30},device=child['resource_device']['device'],
                torch_threads=child['threads']))
            rows[slot['id']]=dict(state='PASS',identity=slot['id'],observations={'result':result})
        report=comparison_report(child,rows)
        self.assertEqual(report['status'],'COMPLETE')
        self.assertEqual(len(report['pairs']),2)
        self.assertAlmostEqual(report['paired_gain']['median'],1/3)
        self.assertFalse(report['optimization_adoption']);self.assertFalse(report['research_cost_admission'])
        for slot in SLOTS:
            changed=copy.deepcopy(rows);changed.pop(slot['id'])
            partial=comparison_report(child,changed)
            self.assertEqual(partial['status'],'INCONCLUSIVE')
            self.assertEqual(len(partial['slots']),5)
            self.assertIsNone(partial['paired_gain'])
        for field,value in [('comparison','other'),('bookkeeping','baseline'),('boundaries',11),('verified_seconds',float('nan'))]:
            changed=copy.deepcopy(rows);changed[SLOTS[2]['id']]['observations']['result'][field]=value
            self.assertEqual(comparison_report(child,changed)['status'],'INCONCLUSIVE')

