import time
import unittest
from oeis_learn.fresh.pipeline_jobs import NativeRequest,logical_job
from oeis_learn.fresh.pipeline_scheduler import drive


class Overlap(unittest.TestCase):
    def test_sequential_local_expiry_drops_late_data_and_keeps_other_prompts(self):
        from unittest.mock import patch, Mock
        from oeis_learn.fresh.bigint_diagnostics import _drive_portfolios
        from oeis_learn.fresh.pipeline_jobs import JobLedger
        for queued in (False,True):
            clock=[2.0 if queued else 0.0];operation=Mock()
            def work():
                operation();clock[0]=2.0
                return {'selected':'ineligible late candidate'}
            def program(i):
                job=logical_job(contract='c',snapshot='s',mode='m',arm='a',prompt_id=str(i),attempt_ordinal=0,method='x',seed=1)
                result=yield NativeRequest(job,i,1.0 if i==0 else 10.0,work if i==0 else lambda:{'complete':True},
                    on_expired=lambda:{'complete':False,'reason':'local deadline'})
                return result
            ledger=JobLedger()
            with patch('time.monotonic',side_effect=lambda:clock[0]):
                results=_drive_portfolios(None,[program(0),program(1)],ledger=ledger)
            self.assertEqual(results,[{'complete':False,'reason':'local deadline'},{'complete':True}])
            self.assertEqual(operation.call_count,0 if queued else 1)
            expired=next(r for r in ledger.records.values() if r['job']['prompt_id']=='0')
            self.assertEqual(expired['state'],'INCOMPLETE' if not queued else 'UNSTARTED')
            self.assertEqual(expired['consumed'],0 if queued else 1)

    def test_overlap_completed_late_job_is_not_published_as_eligible(self):
        from unittest.mock import patch
        from oeis_learn.fresh.pipeline_jobs import JobLedger
        clock=[0.0]
        class Thread:
            def __init__(self,*,target,args,daemon):self.target=target;self.args=args
            def start(self):self.target(*self.args)
            def is_alive(self):return False
            def join(self,timeout):pass
        def program():
            job=logical_job(contract='c',snapshot='s',mode='m',arm='a',prompt_id='p',attempt_ordinal=0,method='x',seed=1)
            def work():clock[0]=2.0;return {'selected':'late'}
            return (yield NativeRequest(job,0,1.0,work,on_expired=lambda:{'complete':False}))
        ledger=JobLedger()
        with patch('time.monotonic',side_effect=lambda:clock[0]),patch('oeis_learn.fresh.pipeline_scheduler.threading.Thread',Thread):
            results,boundary=drive(None,[program()],ledger=ledger)
        self.assertEqual(results,[{'complete':False}])
        self.assertEqual(boundary['consumed'],1)

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
