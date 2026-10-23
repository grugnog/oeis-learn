import unittest
from oeis_learn.pipeline_validation.resources import resolve, reserve


class Resources(unittest.TestCase):
    def test_workstation_ancestor_and_available(self):
        result = resolve(physical=128 << 30, available=100 << 30, limits=[120 << 30, 96 << 30], cpus=32, quota=12, cpuset=8)
        self.assertEqual(result['ceiling_bytes'], 96 << 30)
        self.assertEqual(result['cpus'], 8)
        reserve(result, 60 << 30)
        with self.assertRaises(ValueError):
            reserve(result, 95 << 30)
        self.assertGreater(result['ceiling_bytes'], 16 << 30)
        self.assertNotEqual(result['identity'], resolve(physical=128 << 30, available=100 << 30, limits=[96 << 30], cpus=32, profile='historical16')['identity'])

    def test_unknown_not_zero(self):
        with self.assertRaises(ValueError):
            resolve(physical=None, available=None)
        policy = resolve(physical=8 << 30, available=None)
        with self.assertRaises(ValueError):
            reserve(policy, 1 << 30)
        self.assertIsNone(policy['available_bytes'])
        with self.assertRaises(ValueError):
            resolve(physical=8 << 30, available=8 << 30, profile='capacity64')
