"""Read-only diagnostic authority, populations and artifact collection (stdlib)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from oeis_learn.pipeline_validation.identity import seal,digest,write


class Diagnostics(unittest.TestCase):
    def test_plan_is_closed_finite_and_nonlearning(self):
        from oeis_learn.pipeline_validation.diagnostics_contract import plan,parse
        p=plan(operation='models',device='hip',snapshots=[0,2,5],training_rows=128,spectra=False)
        self.assertEqual(parse(p),p)
        self.assertFalse(p['training']);self.assertFalse(p['resume']);self.assertFalse(p['final_access'])
        self.assertEqual(p['budgets']['global_seconds'],28800)
        for change in ({'training':True},{'hidden_truth':True},{'snapshots':[0,0]},{'training_rows':32769}):
            with self.subTest(change=change),self.assertRaises(ValueError):parse(seal(dict(p,**change)))
        for bad in ([2,0],[-1],[6],[]):
            with self.assertRaises(ValueError):plan(operation='models',snapshots=bad)

    def test_population_and_shuffle_are_frozen_and_independent_of_scores(self):
        from oeis_learn.pipeline_validation.diagnostics_contract import populations
        examples={split:[dict(id=split+str(i),family='f'+str(i%2),stage='A',tokens=[1],values=['0']*20) for i in range(n)]
            for split,n in [('training',40),('parameter_development',4),('structural_development',4)]}
        roster={'training_panel':[r['id'] for r in examples['training'][:32]]}
        a=populations(examples,roster,training_rows=32)
        b=populations({k:list(reversed(v)) for k,v in examples.items()},roster,training_rows=32)
        self.assertEqual(a,b)
        self.assertEqual(len(a['correct']),40)
        self.assertEqual(len(a['matched']),40)
        self.assertTrue(all(k!=v for k,v in a['shuffled'].items()))
        split_by_id={r['id']:k for k,rows in examples.items() for r in rows}
        self.assertTrue(all(split_by_id[k]==split_by_id[v] for k,v in a['shuffled'].items()))
        self.assertEqual(len(a['activation']),40)

    def test_snapshot_binds_actual_weight_record_pointer_and_public_audit(self):
        from tests.unit.test_pipeline_pilot_calibration import SnapshotBindings
        from oeis_learn.pipeline_validation.diagnostics_import import snapshot
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trainer,audit,settings=SnapshotBindings().fixture(root)
            audit['contract']=settings['contract_sha256'];trainer['final_parameter_digest']=digest('parameters')
            got=snapshot(root,5,trainer,audit)
            self.assertEqual(got['ordinal'],5)
            (root/'weights.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):snapshot(root,5,trainer,audit)

    def test_computational_changes_invalidate_import(self):
        from oeis_learn.pipeline_validation.diagnostics_import import compatible_source
        old={'src/oeis_learn/decoder/sampler.py':digest('a')}
        with self.assertRaises(ValueError):compatible_source(old,{**old,'unknown.py':digest('new')})
        with self.assertRaises(ValueError):compatible_source(old,{k:digest('changed') for k in old})
        self.assertEqual(compatible_source(old,old),{})
        self.assertEqual(set(compatible_source(old,{**old,'src/oeis_learn/pipeline_validation/diagnostics.py':digest('new')})),{'src/oeis_learn/pipeline_validation/diagnostics.py'})

    def test_main_cli_parses_both_diagnostic_operations(self):
        import importlib.util
        root=Path(__file__).resolve().parents[2]
        spec=importlib.util.spec_from_file_location('foundation_diagnostics',root/'scripts/foundation_local.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        collect=m.parser().parse_args(['pilot-diagnose','collect','--run','old.zip','--output','fresh'])
        self.assertEqual(collect.operation,'collect')
        models=m.parser().parse_args(['pilot-diagnose','models','--run','old','--admission','admit','--inputs','producer','--output','fresh'])
        self.assertEqual(models.snapshots,[0,1,2,3,4,5]);self.assertEqual(models.device,'hip')


class Failures(unittest.TestCase):
    def test_snapshot_refuses_unbound_record_or_nonregular_weights(self):
        from tests.unit.test_pipeline_pilot_calibration import SnapshotBindings
        from oeis_learn.pipeline_validation.diagnostics_import import snapshot
        for kind in ('pointer','record','contract','symlink'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);trainer,audit,settings=SnapshotBindings().fixture(root)
                audit['contract']=settings['contract_sha256'];trainer['final_parameter_digest']=digest('parameters')
                if kind=='pointer':trainer['checkpoint']=dict(trainer['checkpoint'],sha256=digest('other'))
                elif kind=='contract':audit['contract']=digest('other')
                elif kind=='record':
                    p=root/'checkpoint.json';d=json.loads(p.read_text());d['record']['binding']['weights']=digest('other');p.write_text(json.dumps(d))
                else:
                    p=root/'weights.pt';p.rename(root/'outside.pt');p.symlink_to(root/'outside.pt')
                with self.assertRaises(ValueError):snapshot(root,0,trainer,audit)

    def test_models_failure_exports_and_never_initializes_engine(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.diagnostics import main
        from oeis_learn.pipeline_validation.bundle import verify
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('.specify/memory/constitution.md','specs/007-experiment-foundation/validation/adoption.md','specs/007-experiment-foundation/validation/review-inputs.json','specs/007-experiment-foundation/tasks.md'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            args=SimpleNamespace(operation='models',device='hip',snapshots=[0,2,5],training_rows=128,spectra=False,
                engine='podman',output='output',run='missing',inputs='missing',admission='missing',max_seconds=None,scope_report=False)
            def fake(self,stage,out):
                if stage.adapter!='host':raise AssertionError('engine reached despite missing import')
                return {}
            with patch('oeis_learn.pipeline_validation.adapters.Adapters.__call__',fake):
                result=main(args,root,lambda *_:self.fail('engine initialized'))
            self.assertEqual(result,1)
            summary=json.loads((root/'output/summary.json').read_text())
            self.assertEqual(summary['stages']['engine.podman']['state'],'BLOCKED')
            self.assertEqual(summary['stages']['study.report']['state'],'PASS')
            self.assertFalse(verify(root/'output.zip')['omissions'])

    def test_registry_device_bounds_and_no_training_adapters(self):
        from oeis_learn.pipeline_validation.diagnostics import registry
        from oeis_learn.pipeline_validation.diagnostics_contract import plan
        from oeis_learn.pipeline_validation.registry import resolve
        for device in ('cpu','hip'):
            p=plan(operation='models',device=device);stages=registry(p,'podman')
            self.assertEqual(len(resolve(stages,'all-ready')),len(stages))
            self.assertLessEqual(sum(s.seconds for s in stages)+1200,p['budgets']['global_seconds'])
            self.assertFalse(any(s.adapter in ('pilot-train','pilot-prepare','calibration-session') for s in stages))
            self.assertEqual(any(s.adapter=='hip' for s in stages),device=='hip')
        self.assertEqual([s.id for s in registry(plan(operation='collect'),'podman')],['diagnostics.collect'])

    def test_row_summary_preserves_split_family_and_token_weighting(self):
        from oeis_learn.pipeline_validation.diagnostics import summarize_rows
        rows=[dict(id='a',split='training',family='f',stage='A',tokens=1,loss_sum=1,correct_tokens=1,exact_program=True,entropy_sum=0),
            dict(id='b',split='parameter_development',family='f',stage='A',tokens=3,loss_sum=9,correct_tokens=0,exact_program=False,entropy_sum=3)]
        result=summarize_rows(rows)
        self.assertEqual(result['all']['loss'],2.5)
        self.assertEqual(result['split:training|family:f']['loss'],1)
        self.assertEqual(result['split:parameter_development|family:f']['loss'],3)

    def test_unsafe_archive_rejected_before_any_materialization(self):
        import zipfile
        from oeis_learn.pipeline_validation.diagnostics_collect import Evidence
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'unsafe.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','payload')
            with self.assertRaisesRegex(ValueError,'unsafe'):Evidence(p)

if __name__=='__main__':unittest.main()
