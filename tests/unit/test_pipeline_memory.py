import tempfile
from pathlib import Path
import unittest
from oeis_learn.pipeline_validation.memory import MemoryWatchdog


class MemoryAccounting(unittest.TestCase):
    def test_shared_pages_use_conservative_driver_bound_without_allocator(self):
        calls=[]
        sample=dict(physical=1000,available=500,current=100,limit=900,driver=50)
        with tempfile.TemporaryDirectory() as folder:
            guard=MemoryWatchdog(Path(folder),ceiling=900,gpu=True,reader=lambda:sample,killer=lambda:calls.append('kill'))
            guard.sample()
            self.assertTrue(guard.report()['qualified'])
            self.assertEqual(guard.report()['machine_peak_upper_bytes'],550)
            self.assertEqual(guard.report()['role_peak_upper_bytes'],150)
            sample['driver']=500
            guard.sample()
            self.assertEqual(calls,['kill'])
            self.assertFalse(guard.report()['qualified'])

    def test_unknown_gpu_accounting_never_qualifies(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=MemoryWatchdog(Path(folder),ceiling=900,gpu=True,reader=lambda:dict(physical=1000,available=500,current=100,limit=900,driver=None),killer=lambda:None)
            guard.sample()
            self.assertFalse(guard.report()['qualified'])
            self.assertIn('unavailable',guard.report()['accounting'])
