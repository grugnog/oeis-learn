"""Role-wide cache ownership and independent current-deadline admission."""
from dataclasses import replace
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
from oeis_learn.synthesis.client import SynthesisClient
from oeis_learn.synthesis.service_policy import ServiceLimits
from oeis_learn.synthesis.worker_pool import RoleCaches


class PipelineCache(unittest.TestCase):
    def test_exclusive_disk_owner_and_aggregate_eviction(self):
        limits=replace(ServiceLimits(),result_cache_max_entries=2,result_cache_max_bytes=512,
            disk_cache_max_entries=2,disk_cache_max_bytes=512)
        with tempfile.TemporaryDirectory() as folder:
            with RoleCaches(limits,Path(folder)) as cache:
                with self.assertRaises(BlockingIOError):
                    RoleCaches(limits,Path(folder))
                for i in range(4):cache.put(source_identity(str(i)),{'record':i})
                for name in ('result','disk'):
                    self.assertLessEqual(cache.metrics()[name]['entries'],2)
                    self.assertLessEqual(cache.metrics()[name]['bytes'],512)
                self.assertIsNone(cache.get(source_identity('0')))
                self.assertEqual(cache.get(source_identity('3')),{'record':3})
            with RoleCaches(limits,Path(folder)) as reopened:
                self.assertEqual(reopened.get(source_identity('3')),{'record':3})

    def test_only_complete_shortlists_are_cached_and_hits_are_readmitted(self):
        prompt=VisiblePrompt(tuple(map(str,range(20))),tuple(map(str,range(20))))
        source='(program (return $index))'
        item=dict(original_source_sha256=source_identity(source),source_sha256=source_identity(source))
        native=dict(outcome='assignment_found',reason=None,uniqueness='unique_in_scope',
            assignments=[item],searches=[{'outcome':'assignment_found','reason':None}])
        for complete in (False,True):
            client=SynthesisClient(source_identity('contract'),worker=Mock())
            client.runtime_identity=lambda profile:source_identity('runtime')
            client.request=Mock(return_value={'complete':complete,'results':[native]})
            def admission(item,prompt,**kwargs):
                return SimpleNamespace(outcome='verified_solution' if kwargs['deadline']>time.monotonic() else 'resource_exhausted',
                    verification={'independent':True},reason=None)
            try:
                with patch('oeis_learn.synthesis.client.admit',side_effect=admission) as verify:
                    for _ in range(2):client.fit(source,prompt,deadline=time.monotonic()+3)
                    self.assertEqual(client.request.call_count,1 if complete else 2)
                    self.assertEqual(verify.call_count,2)
                    # Changing the fitter policy cannot hit the previous key.
                    client.fit(source,prompt,policy='eligible-literals/v1',deadline=time.monotonic()+3)
                    self.assertEqual(client.request.call_count,2 if complete else 3)
                    expired=client.fit(source,prompt,deadline=time.monotonic()-1)
                    self.assertFalse(expired['admitted'])
                    self.assertFalse(expired['complete'])
            finally:client.close()

    def test_separate_arm_cache_namespaces_do_not_share_prewarm(self):
        key=source_identity('same semantic job')
        with tempfile.TemporaryDirectory() as folder:
            with RoleCaches(ServiceLimits(),Path(folder)/'baseline') as a, RoleCaches(ServiceLimits(),Path(folder)/'candidate') as b:
                a.put(key,{'complete':True})
                self.assertIsNone(b.get(key))
                b.put(key,{'complete':True})
                self.assertEqual(a.get(key),b.get(key))
