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

    def test_checkpoint_merge_never_refunds_or_repeats_work(self):
        from oeis_learn.fresh.pipeline_jobs import JobLedger, merge_boundary
        def boundary(i):
            ledger=JobLedger()
            job=logical_job(contract='c',snapshot=str(i),mode='m',arm='a',prompt_id='p',attempt_ordinal=0,method='x',seed=0)
            ledger.reserve(job,deadline=10,work=3)
            ledger.cancel(job['job_id'])
            return ledger.boundary({'workers':1})
        a,b=boundary(0),boundary(1)
        merged=merge_boundary(a,b)
        self.assertEqual(merged['consumed'],6)
        self.assertEqual(len(merged['frontier']),2)
        with self.assertRaises(ValueError):merge_boundary(merged,a)
        altered=dict(b, policies={'workers':2})
        with self.assertRaises(ValueError):merge_boundary(a,altered)

    def test_cancellation_signalled_before_join(self):
        import threading
        stopped=threading.Event()
        def program():
            job=logical_job(contract='c',snapshot='s',mode='m',arm='a',prompt_id='p',attempt_ordinal=0,method='x',seed=0)
            yield NativeRequest(job,0,time.monotonic()+.03,lambda:stopped.wait(2),cancel=stopped.set)
        start=time.monotonic()
        with self.assertRaises(TimeoutError):drive(None,[program()],workers=1)
        self.assertTrue(stopped.is_set())
        self.assertLess(time.monotonic()-start,1)
