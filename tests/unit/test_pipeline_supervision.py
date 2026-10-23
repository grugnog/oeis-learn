from pathlib import Path
import sys
import tempfile
import unittest
from oeis_learn.pipeline_validation.supervision import launch


class Supervision(unittest.TestCase):
    def test_log_streams_are_drained_and_capped(self):
        with tempfile.TemporaryDirectory() as folder:
            result=launch([sys.executable,'-c',"import sys;sys.stdout.write('x'*100000);sys.stderr.write('y'*100000)"],Path(folder)/'logs',seconds=3,stream_cap=4096)
            self.assertEqual((Path(folder)/'logs/stdout.log').stat().st_size,4096)
            self.assertEqual(result['omitted_bytes']['stdout'],100000-4096)

    def test_startup_failure_and_deadline_return(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(OSError):launch(['/no-such-executable'],Path(folder)/'a',seconds=1)
            with self.assertRaises(TimeoutError):launch([sys.executable,'-c','import time;time.sleep(30)'],Path(folder)/'b',seconds=.05)

    def test_failure_records_command_and_preserves_bounded_tail(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(RuntimeError):
                launch([sys.executable,'-c',"import sys;sys.stderr.write('x'*100000+'LAST');sys.exit(1)"],root,seconds=3,stream_cap=4096)
            self.assertEqual((root/'stderr.log').stat().st_size,4096)
            self.assertTrue((root/'stderr.log').read_text().endswith('LAST'))
            result=json.loads((root/'process.json').read_text())
            self.assertEqual(result['omitted_bytes']['stderr'],100004-4096)
            self.assertEqual(result['outcome'],'FAIL')
