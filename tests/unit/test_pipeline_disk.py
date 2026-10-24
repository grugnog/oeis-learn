import tempfile
from pathlib import Path
import unittest
from oeis_learn.pipeline_validation.disk import DiskWatchdog
from oeis_learn.pipeline_validation.runner import ResourceBreach


class DiskBounds(unittest.TestCase):
    def test_low_free_space_observation_survives_recovery_during_cleanup(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            samples=[SimpleNamespace(f_bavail=n,f_frsize=1) for n in (64,7,64)]
            with patch('oeis_learn.pipeline_validation.disk.os.statvfs',side_effect=samples),patch('oeis_learn.pipeline_validation.disk.threading.Thread'):
                with self.assertRaisesRegex(ResourceBreach,'free-space reserve exhausted: 7 < 16'):
                    with DiskWatchdog(root,root/'stage',cap=1024,trace_cap=512,free_reserve=16) as guard:
                        guard.check()
            report=json.loads((root/'stage/disk.json').read_text())
            self.assertEqual(report['minimum_free_bytes'],7)
            self.assertEqual(report['free_bytes'],64)
            self.assertEqual(report['breach_observation'],dict(artifact_bytes=0,trace_bytes=0,free_bytes=7))
            self.assertEqual(report['cap_bytes'],1024)
            self.assertEqual(report['free_reserve_bytes'],16)

    def test_artifact_and_trace_breaches_are_distinct_from_free_space(self):
        for traced,reason in [(False,'transient artifact cap'),(True,'trace cap')]:
            with self.subTest(traced=traced),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);directory=root/'traces' if traced else root
                directory.mkdir(exist_ok=True);(directory/'data').write_bytes(b'x'*100)
                guard=DiskWatchdog(root,root/'stage',cap=1000 if traced else 50,trace_cap=50,free_reserve=0)
                with self.assertRaisesRegex(ResourceBreach,reason):guard.check()
                self.assertEqual(guard.breach_observation['artifact_bytes'],100)
                self.assertEqual(guard.breach_observation['trace_bytes'],100 if traced else 0)

    def test_hard_links_count_once_and_breach_stops_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'a').write_bytes(b'x'*100)
            (root/'b').hardlink_to(root/'a')
            guard=DiskWatchdog(root,root/'stage',cap=150,free_reserve=0)
            self.assertEqual(guard.usage(),100)
            (root/'c').write_bytes(b'x'*100)
            with self.assertRaises(ResourceBreach):guard.check()

    def test_continuous_watchdog_reports_without_touching_readonly_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with DiskWatchdog(root,root/'stage',cap=1<<20,free_reserve=0):
                (root/'public.json').write_text('{}')
            self.assertTrue((root/'stage/disk.json').is_file())
