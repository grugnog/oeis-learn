"""Dependency-free regressions for original, bounded qualification diagnostics."""
import json
from pathlib import Path
import tempfile
import unittest

from oeis_learn.pipeline_validation.identity import digest,seal,write,file_hash
from oeis_learn.pipeline_validation.generalization_failures import stage_failures


class QualificationFailures(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.pipeline=self.root/'execution/qualification/pipeline/stages'

    def references(self):
        return [dict(path=str(p.relative_to(self.root)),bytes=p.stat().st_size,sha256=file_hash(p))
                for p in self.root.rglob('*') if p.is_file()]

    def fixture(self):
        for stage,state in [('image.qualification','PASS'),('cpu.semantic','FAIL'),('image.gpu','PASS'),('hip.preflight','BLOCKED')]:
            write(self.pipeline/stage/'receipt.json',seal(dict(version='pipeline-stage/v1',stage=stage,
                state=state,reason='private /home/operator/teacher-source',files={})))
        path=self.pipeline/'cpu.semantic/bigint-cpu-regressions/results.xml'
        path.parent.mkdir(parents=True)
        path.write_text('''<testsuites><testsuite tests="3" failures="2" errors="0" skipped="0">
          <testcase classname="tests.integration.test_generalization_rehearsal" name="test_rehearsal_claims_only_fresh_private_work[private-value]">
            <failure message="ImportError: cannot import name 'claim_output' from 'oeis_learn.pipeline_validation.generalization_pac_rehearsal'">private traceback /home/operator/teacher-source</failure>
          </testcase>
          <testcase classname="private.teacher" name="secret"><error message="ValueError: protected values">private traceback</error></testcase>
          <testcase classname="private.teacher" name="another-secret"/>
        </testsuite></testsuites>''')
        write(self.root/'execution/result.json',seal(dict(version='generalization-command-result/v2',status='FAIL',
            result=dict(status='FAIL',reason='QUALIFICATION_FAILED',exit_code=1))))
        return path

    def test_original_tests_and_stage_failures_survive_without_private_text(self):
        self.fixture();before={r['path']:r['sha256'] for r in self.references()}
        result=stage_failures(self.root,refs=self.references())
        self.assertEqual(result['failures'][0]['detail']['code'],'qualification_tests_failed')
        tests=result['qualification']['tests'][0]
        self.assertEqual(tests['counts'],dict(tests=3,failures=1,errors=1,skipped=0))
        self.assertEqual(tests['reported_counts'],dict(tests=3,failures=2,errors=0,skipped=0))
        self.assertEqual(tests['failures'][0]['code'],'missing_r0_output_claim')
        self.assertEqual(tests['failures'][0]['test'],'tests/integration/test_generalization_rehearsal.py::test_rehearsal_claims_only_fresh_private_work')
        self.assertIsNone(tests['failures'][1]['test'])
        self.assertEqual({r['stage']:r['state'] for r in result['qualification']['stages']},
            {'image.qualification':'PASS','cpu.semantic':'FAIL','image.gpu':'PASS','hip.preflight':'BLOCKED'})
        rendered=json.dumps(result)
        for private in ('/home/operator','teacher-source','private-value','private.teacher','another-secret','private traceback'):
            self.assertNotIn(private,rendered)
        self.assertEqual(before,{r['path']:r['sha256'] for r in self.references()})

    def test_unbound_and_tampered_xml_are_omitted(self):
        path=self.fixture();refs=self.references();path.write_text('<testsuites/>')
        result=stage_failures(self.root,refs=refs)
        self.assertFalse(result['qualification']['tests'])
        self.assertTrue(result['omissions'])
        self.assertFalse(stage_failures(self.root,refs=[])['qualification']['stages'])

    def test_unsealed_pipeline_receipt_is_not_projected(self):
        self.fixture();path=self.pipeline/'cpu.semantic/receipt.json'
        value=json.loads(path.read_text());value['state']='PASS';path.write_text(json.dumps(value))
        result=stage_failures(self.root,refs=self.references())
        self.assertNotIn('cpu.semantic',[r['stage'] for r in result['qualification']['stages']])
        self.assertTrue(result['omissions'])

    def test_xml_entities_and_discovery_overflow_are_explicit(self):
        path=self.fixture();path.write_text('<!DOCTYPE testsuite [<!ENTITY secret "private">]><testsuite/>')
        result=stage_failures(self.root,refs=self.references())
        self.assertFalse(result['qualification']['tests'])
        self.assertTrue(result['omissions'])
        self.assertTrue(stage_failures(self.root,max_entries=1)['omissions'])

    def test_generic_qualification_error_is_not_a_private_empty_exception(self):
        from oeis_learn.pipeline_validation.generalization_failures import public_failure
        result=public_failure(kind='QUALIFICATION_FAILED')
        self.assertEqual(result['code'],'qualification_failed')

    def test_log_projection_keeps_stage_outcome_without_reason_or_path(self):
        from oeis_learn.pipeline_validation.generalization_qualification_failures import progress_lines
        line='QUALIFICATION '+json.dumps(dict(stage='cpu.semantic',status='FAIL',reason='/home/operator/teacher-source'))
        self.assertEqual(progress_lines(line),['qualification: cpu.semantic FAIL'])
        self.assertEqual(progress_lines('QUALIFICATION '+json.dumps(dict(stage='private.teacher',status='FAIL'))),[])

    def test_failure_tail_is_bounded_and_counts_remain_complete(self):
        path=self.fixture()
        path.write_text('<testsuite>'+''.join('<testcase name="secret"><failure message="AssertionError: secret"/></testcase>' for _ in range(100))+'</testsuite>')
        tests=stage_failures(self.root,refs=self.references())['qualification']['tests'][0]
        self.assertEqual(tests['counts']['failures'],100)
        self.assertEqual(len(tests['failures']),64)
        self.assertEqual(tests['omitted_failures'],36)

    def test_canonical_wrapper_binds_outer_supervision_evidence(self):
        import importlib.util
        from unittest.mock import patch
        path=Path(__file__).resolve().parents[2]/'specs/012-parameterized-adaptive-curriculum/validation/run.py'
        spec=importlib.util.spec_from_file_location('pac_diagnostic_wrapper',path)
        wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
        target=self.root/'qualification';target.mkdir()
        value=seal(dict(status='FAIL',identity=digest('producer'),inventory=[]))
        write(target/'execution/result.json',value)
        supervision=target/'execution-supervision';supervision.mkdir()
        write(supervision/'process.json',dict(outcome='FAIL',returncode=1))
        (supervision/'stdout.log').write_text('QUALIFICATION '+json.dumps(dict(stage='cpu.semantic',status='FAIL',reason='private')))
        with patch.object(wrapper,'invoke',return_value=value):
            result=wrapper._canonical_command_stage(dict(engine='podman'),
                dict(operation='qualify',manifest='unused',extra=[],seconds=60),target,self.root,{})
        paths={r['path'] for r in result['artifacts']}
        self.assertIn('execution-supervision/process.json',paths)
        self.assertIn('execution-supervision/stdout.log',paths)

    def test_rich_and_minimal_archive_recovery_keep_original_test_causes(self):
        import shutil,zipfile
        from unittest.mock import patch
        from oeis_learn.pipeline_validation import generalization_pac_export as exporter
        from oeis_learn.pipeline_validation.generalization_archive import verify_archive
        self.fixture();original=self.root/'campaign';stage=original/'qualification'
        shutil.copytree(self.root/'execution',stage/'execution')
        (original/'admission').mkdir()
        order=['admission','qualification','calibration','acceptance']
        plan=seal(dict(version='pac-serial-plan/v1',engine='podman',research=False,arms=['B','C'],source=digest('source'),
            images=[digest('cpu'),digest('gpu')],helper=digest('helper'),scientific={},seconds=18000,finalization_seconds=3600,
            stages=[dict(name=n,extra=[]) for n in order]))
        write(original/'plan.json',plan)
        refs=[dict(path=str(p.relative_to(stage)),bytes=p.stat().st_size,sha256=file_hash(p)) for p in stage.rglob('*') if p.is_file()]
        write(original/'workflow.json',seal(dict(plan=plan['identity'],status='FAIL',errors=[],receipts=[
            dict(stage=n,directory=n,status='PASS' if n=='admission' else 'FAIL',producer=digest(n),binding=digest(n),
                artifacts=[] if n=='admission' else refs,process=dict(outcome='COMPLETE' if n=='admission' else 'FAIL',returncode=0 if n=='admission' else 1),
                cleanup=dict(acknowledged=True,within_bound=True)) for n in order[:2]])))
        before=(original/'workflow.json').read_bytes()
        archives=[exporter.export(original,self.root/'rich')]
        with patch.object(exporter,'collect',side_effect=ValueError('private exporter failure')):
            archives.append(exporter.export(original,self.root/'minimal'))
        for archive in archives:
            self.assertEqual(verify_archive(archive)['software_status'],'FAIL')
            with zipfile.ZipFile(archive) as z:
                details=json.loads(z.read('failures/qualification.json'))
                self.assertEqual(details['qualification']['tests'][0]['counts']['tests'],3)
                self.assertEqual(details['failures'][0]['detail']['code'],'qualification_tests_failed')
                for name in z.namelist():
                    self.assertNotIn(b'teacher-source',z.read(name))
                    self.assertNotIn(b'private traceback',z.read(name))
        self.assertEqual((original/'workflow.json').read_bytes(),before)


class RehearsalOutput(unittest.TestCase):
    def test_fresh_mount_and_exclusive_private_claim(self):
        from oeis_learn.pipeline_validation.generalization_pac_rehearsal import claim_output
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'role';root.mkdir();(root/'.runtime').mkdir();(root/'supervision').mkdir()
            marker=root/'.runtime/launcher';marker.write_text('preserve')
            self.assertEqual(claim_output(root),root)
            with self.assertRaises(FileExistsError):claim_output(root)
            self.assertEqual(marker.read_text(),'preserve')

    def test_absent_and_stale_private_or_terminal_paths(self):
        from oeis_learn.pipeline_validation.generalization_pac_rehearsal import claim_output
        for existing in ('absent','directory','file','symlink','terminal'):
            with self.subTest(existing=existing),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'role'
                if existing=='absent':
                    self.assertEqual(claim_output(root),root);continue
                root.mkdir();private=root/'private'
                if existing=='directory':private.mkdir()
                elif existing=='file':private.write_text('preserve')
                elif existing=='symlink':private.symlink_to(Path(folder),target_is_directory=True)
                else:(root/'result.json').write_text('preserve')
                before=sorted(root.iterdir())
                with self.assertRaises(FileExistsError):claim_output(root)
                self.assertEqual(sorted(root.iterdir()),before)


if __name__=='__main__':unittest.main()
