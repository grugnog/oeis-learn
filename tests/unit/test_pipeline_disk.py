import tempfile
from pathlib import Path
import unittest
from oeis_learn.pipeline_validation.disk import DiskWatchdog
from oeis_learn.pipeline_validation.runner import ResourceBreach


class DiskBounds(unittest.TestCase):
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
