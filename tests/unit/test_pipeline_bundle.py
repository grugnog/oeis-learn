import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import errno
from unittest.mock import patch
from oeis_learn.pipeline_validation.bundle import pack, verify
from oeis_learn.pipeline_validation.identity import write


class Bundle(unittest.TestCase):
    def test_pilot_member_cap_preserves_mandatory_core(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out';root.mkdir();self.fixture(root)
            (root/'detail.txt').write_text('public')
            with self.assertRaisesRegex(ValueError,'member cap'):pack(root,mandatory=['summary.json','contract.json'],member_cap=2)
            manifest=verify(pack(root,mandatory=['summary.json','contract.json'],member_cap=3))
            self.assertEqual(len(manifest['members']),2)
            self.assertEqual(manifest['omissions'][0]['path'],'detail.txt')

    def fixture(self, root):
        write(root/'summary.json',dict(selected_scope='FAIL'))
        write(root/'contract.json',dict(identity='contract'))

    def test_member_integrity_and_private_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'out'; root.mkdir(); self.fixture(root)
            (root/'weights.pt').write_text('secret')
            archive=pack(root, mandatory=['summary.json','contract.json'])
            verify(archive)
            with zipfile.ZipFile(archive) as z:
                self.assertNotIn('weights.pt',z.namelist())
                self.assertIn('manifest.json',z.namelist())

    def test_symlink_and_missing_mandatory_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out'; root.mkdir(); self.fixture(root)
            (root/'link.json').symlink_to(root/'summary.json')
            with self.assertRaises(ValueError):
                pack(root,mandatory=['link.json'])
            with self.assertRaises((ValueError,OSError)):
                pack(root,mandatory=['missing.json'])

    def test_excluded_tree_links_are_private_not_public_omissions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out';root.mkdir();self.fixture(root)
            for name in ('private','pytest'):
                (root/name).mkdir();(root/name/'alias.json').symlink_to(root/'summary.json')
            (root/'public-alias.json').symlink_to(root/'summary.json')
            manifest=verify(pack(root,mandatory=['summary.json']))
            self.assertEqual(manifest['omissions'],[{'path':'public-alias.json','reason':'symlink'}])

    def test_unreadable_mandatory_rejected_even_for_privileged_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out';root.mkdir();self.fixture(root)
            (root/'contract.json').chmod(0)
            with self.assertRaisesRegex(ValueError,'unreadable'):
                pack(root,mandatory=['summary.json','contract.json'])
            self.assertFalse(root.with_suffix('.zip').exists())

    def test_mandatory_traversal_and_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out';root.mkdir();self.fixture(root)
            write(Path(folder)/'outside.json',{'private':True})
            (root/'link.json').symlink_to(Path(folder)/'outside.json')
            for name in ('../outside.json','link.json'):
                with self.subTest(name=name),self.assertRaises(ValueError):
                    pack(root,mandatory=['summary.json',name])

    def test_optional_cap_preserves_core_and_explicit_omissions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out';root.mkdir();self.fixture(root)
            (root/'detail.txt').write_text('x'*32)
            manifest=verify(pack(root,mandatory=['summary.json','contract.json'],optional_cap=8))
            self.assertEqual(manifest['omissions'],[{'path':'detail.txt','reason':'optional byte cap'}])
            self.assertEqual({x['path'] for x in manifest['members']},{'summary.json','contract.json'})

    def test_contradictory_stage_summary_cannot_export_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out';root.mkdir();self.fixture(root)
            write(root/'summary.json',{'selected_scope':'PASS','stages':{'failed':{'state':'FAIL'}}},exclusive=False)
            with self.assertRaisesRegex(ValueError,'contradict'):
                pack(root,mandatory=['summary.json'])
            self.assertFalse(root.with_suffix('.zip').exists())

    def test_enospc_deletes_partial_zip_and_minimal_failure_survives(self):
        from oeis_learn.pipeline_validation.recovery import minimal_archive
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out';root.mkdir();self.fixture(root)
            write(root/'invocation.json',{'fixture':True})
            original=zipfile.ZipFile.write
            def full(stream,*args,**kw):
                original(stream,*args,**kw)
                raise OSError(errno.ENOSPC,'owned export fixture')
            with patch.object(zipfile.ZipFile,'write',full),self.assertRaises(OSError):
                pack(root,mandatory=['summary.json'])
            self.assertFalse(root.with_suffix('.zip').exists())
            archive,summary=minimal_archive(root,contract={},binding={},receipts={},error='ENOSPC',cap=16<<20)
            self.assertEqual(verify(archive)['summary_state'],'FAIL')
            self.assertEqual(json.loads(summary.read_text())['selected_scope'],'FAIL')

    def test_build_or_startup_interruption_has_terminal_ledger_and_archive(self):
        from oeis_learn.pipeline_validation.runner import execute
        from oeis_learn.pipeline_validation.registry import Stage
        for failure,expected in ((KeyboardInterrupt(),130),(FileNotFoundError('startup'),1)):
            with self.subTest(failure=type(failure).__name__),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'out'
                def fail(stage,out):
                    write(out/'attempt.json',{'state':'STARTED'})
                    raise failure
                code=execute([Stage('build',1),Stage('child',2,('build',))],root,
                    adapter=fail,binding={},cap=60,reserve=1)
                self.assertEqual(code,expected)
                self.assertNotEqual(verify(root.with_suffix('.zip'))['summary_state'],'PASS')
                summary=json.loads((root/'summary.json').read_text())
                self.assertEqual(summary['stages']['child']['state'],'BLOCKED')
                self.assertIn('attempt.json',summary['stages']['build']['files'])
