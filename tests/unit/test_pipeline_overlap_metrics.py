import tempfile
from pathlib import Path
import unittest
from oeis_learn.tracking.pipeline_profile import interval_union, exclusive_seconds
from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.fresh.bigint_state import CheckpointStore
from oeis_learn.fresh.pipeline_jobs import JobLedger, logical_job


class OverlapMetrics(unittest.TestCase):
    def test_overlapped_intervals_are_union_not_sum(self):
        self.assertEqual(interval_union([(0,4),(2,6),(6,7),(1,3)]),7)
        rows=[dict(name='parent',start=0,end=10,depth=0),
            dict(name='a',start=1,end=5,depth=1),dict(name='b',start=3,end=7,depth=1)]
        self.assertEqual(exclusive_seconds(rows)['parent'],4)

    def test_checkpoint_rejects_unaccounted_or_changed_policy_boundary(self):
        policies={'workers':2}
        ledger=JobLedger()
        job=logical_job(contract='c',snapshot='s',mode='m',arm='a',prompt_id='p',attempt_ordinal=0,method='x',seed=1)
        ledger.reserve(job,deadline=10,work=4)
        ledger.cancel(job['job_id'])
        boundary=ledger.boundary(policies)
        self.assertEqual(boundary['consumed'],4)
        with tempfile.TemporaryDirectory() as folder:
            store=CheckpointStore(Path(folder),None,policies=policies)
            store._boundary({'pipeline_boundary':boundary})
            for altered in (dict(boundary,active=1),dict(boundary,reserved=1),dict(boundary,policies={'workers':4})):
                with self.assertRaises(ValueError):store._boundary({'pipeline_boundary':seal(altered)})
            with self.assertRaises(ValueError):store._boundary({})
