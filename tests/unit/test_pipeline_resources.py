import unittest
from oeis_learn.pipeline_validation.resources import resolve, reserve


class Resources(unittest.TestCase):
    def test_docker_shim_or_missing_memory_is_an_explicit_capability_gap(self):
        from oeis_learn.container_runtime import Engine
        engine=Engine.__new__(Engine);engine.name='docker'
        engine.details={'host':{'memTotal':128<<30},'store':{},'version':{'Version':'5.8'}}
        with self.assertRaisesRegex(RuntimeError,'Podman'):
            engine.facts()
        engine.details={'NCPU':16,'ServerVersion':'test'}
        with self.assertRaisesRegex(RuntimeError,'MemTotal'):
            engine.facts()
        engine.details={'MemTotal':128<<30,'NCPU':16}
        self.assertEqual(engine.facts()['memory_bytes'],128<<30)

    def test_study_resource_identity_binds_envelope_but_not_free_memory(self):
        from oeis_learn.pipeline_validation.resources import execution_identity
        policy=resolve(physical=128<<30,available=100<<30,cpus=16)
        self.assertEqual(execution_identity(policy),execution_identity(dict(policy,available_bytes=90<<30)))
        self.assertNotEqual(execution_identity(policy),execution_identity(dict(policy,ceiling_bytes=16<<30)))
        self.assertNotEqual(execution_identity(policy),execution_identity(dict(policy,cpus=8)))

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

    def test_resolved_envelope_reaches_host_before_legacy_default(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.adapters import Adapters
        class Engine:
            def __init__(self,name):self.name=name
            def facts(self):return {'memory_bytes':128<<30}
        class Host:
            def __init__(self,options):
                self.options=options
                self.policy={'memory_bytes':int((options.memory_gib or 16)*(1<<30)),'cpus':options.cpus}
        options=SimpleNamespace(engine='docker',memory_gib=None,cpus=None)
        measured=dict(physical=128<<30,available=100<<30,limits=[],cpus=32,quota=None,cpuset=None)
        with patch('oeis_learn.container_runtime.Engine',Engine),patch('oeis_learn.pipeline_validation.adapters.observe',lambda:measured):
            adapters=Adapters(options,'.',Host,{'budgets':{'build_per_engine_seconds':14400},'resource_policy':'workstation90'})
            host=adapters.host('docker')
        self.assertGreater(host.policy['memory_bytes'],16<<30)
        self.assertEqual(host.policy['memory_bytes'],host.policy['cpu_memory_bytes'])
        self.assertEqual(host.options.cpus,16)
