from pathlib import Path
import sys
import tempfile
import unittest
from oeis_learn.pipeline_validation.supervision import launch


class Supervision(unittest.TestCase):
    def test_unresponsive_client_leaves_time_for_container_removal(self):
        import json,time
        from unittest.mock import Mock
        from oeis_learn.pipeline_validation.supervision import reclaim
        with tempfile.TemporaryDirectory() as folder:
            state={};root=Path(folder)
            with self.assertRaises(TimeoutError):
                launch([sys.executable,'-u','-c',
                        'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready");time.sleep(30)'],
                       root/'logs',seconds=.3,cleanup_state=state)
            assert (root/'logs/stdout.log').read_text().strip()=='ready'
            assert state['deadline']-time.monotonic()>3.5
            engine=Mock()
            reclaim(engine,'fixture',root/'cleanup',state)
            assert engine.remove.call_args.kwargs['timeout']>3.5
            assert json.loads((root/'cleanup/cleanup.json').read_text())['acknowledged']

    def test_extended_cleanup_uses_one_shared_deadline_and_still_fails_closed(self):
        import json,time,subprocess
        from unittest.mock import Mock,patch
        from oeis_learn.pipeline_validation.supervision import reclaim
        from oeis_learn.pipeline_validation.runner import ResourceBreach
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state={}
            launch([sys.executable,'-c','pass'],root/'launch',seconds=5,
                   cleanup_seconds=30,cleanup_state=state)
            assert state['allowance_seconds']==30
            clock=[state['deadline']-29]
            engine=Mock()
            def slow_remove(name,timeout):
                assert timeout==29
                clock[0]+=12
            engine.remove.side_effect=slow_remove
            with patch('oeis_learn.pipeline_validation.supervision.time.monotonic',side_effect=lambda:clock[0]):
                reclaim(engine,'fixture',root/'cleanup',state)
            receipt=json.loads((root/'cleanup/cleanup.json').read_text())
            assert receipt['acknowledged'] and receipt['seconds']==12 and receipt['allowance_seconds']==30
            engine.remove.side_effect=subprocess.TimeoutExpired('rm',17)
            with patch('oeis_learn.pipeline_validation.supervision.time.monotonic',return_value=clock[0]):
                with self.assertRaises(ResourceBreach):reclaim(engine,'fixture',root/'failure',state)
            assert not json.loads((root/'failure/cleanup.json').read_text())['acknowledged']

    def test_container_reclamation_uses_remaining_allowance_and_failure_is_global(self):
        import json
        import subprocess
        from unittest.mock import Mock, patch
        from oeis_learn.pipeline_validation.supervision import reclaim
        from oeis_learn.pipeline_validation.runner import ResourceBreach
        engine = Mock()
        with tempfile.TemporaryDirectory() as folder, patch('oeis_learn.pipeline_validation.supervision.time.monotonic',return_value=103):
            reclaim(engine,'role',Path(folder),{'deadline':105})
            engine.remove.assert_called_once_with('role',timeout=2)
        engine.remove.side_effect = subprocess.TimeoutExpired('rm',2)
        with tempfile.TemporaryDirectory() as folder, patch('oeis_learn.pipeline_validation.supervision.time.monotonic',return_value=103):
            with self.assertRaises(ResourceBreach):
                reclaim(engine,'role',Path(folder),{'deadline':105})
            receipt=json.loads((Path(folder)/'cleanup.json').read_text())
            self.assertFalse(receipt['acknowledged'])
            self.assertFalse(receipt['within_bound'])

    def test_launch_publishes_one_cleanup_deadline_for_container_reclamation(self):
        import json
        import time
        with tempfile.TemporaryDirectory() as folder:
            state = {}
            with self.assertRaises(TimeoutError):
                launch([sys.executable,'-c','import time;time.sleep(30)'],Path(folder),
                    seconds=.03,cleanup_state=state,grace_seconds=.1)
            self.assertLessEqual(state['deadline']-time.monotonic(), 5)
            self.assertGreater(state['deadline']-time.monotonic(), 4)
            self.assertEqual(json.loads((Path(folder)/'process.json').read_text())['outcome'],'FAIL')

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
