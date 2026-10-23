import time
import unittest
from oeis_learn.fresh.pipeline_jobs import NativeRequest,logical_job
from oeis_learn.fresh.pipeline_scheduler import drive


class Overlap(unittest.TestCase):
    def test_owner_publication_and_one_active_per_prompt(self):
        calls=[]
        def program(i):
            for attempt in range(2):
                job=logical_job(contract='c',snapshot='s',mode='m',arm='a',prompt_id=str(i),attempt_ordinal=attempt,method='x',seed=attempt)
                value=yield NativeRequest(job,i,time.monotonic()+10,lambda i=i,attempt=attempt: (i,attempt))
                calls.append(value)
            return i
        results,boundary=drive(None,[program(0),program(1)],workers=2,lookahead=1,policies={})
        self.assertEqual(results,[0,1])
        self.assertEqual(set(calls),{(0,0),(0,1),(1,0),(1,1)})
        self.assertEqual(boundary['active'],0)
        self.assertEqual(len(boundary['frontier']),4)
