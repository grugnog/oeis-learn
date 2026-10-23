import unittest
from oeis_learn.fresh.pipeline_jobs import logical_job, JobLedger
from oeis_learn.synthesis.service_policy import PipelineCapacity


class NativePool(unittest.TestCase):
    def test_affinity_aggregate_and_result_reservation(self):
        cap=PipelineCapacity(workers=2,pending_bytes=100,active_bytes=80,result_bytes=100)
        self.assertTrue(cap.enqueue('a',40,50,prompt=0,deadline=10,now=0))
        self.assertTrue(cap.enqueue('b',40,50,prompt=2,deadline=10,now=0))
        self.assertIsNone(cap.dispatch(1,now=1))
        self.assertEqual(cap.dispatch(0,now=1),'a')
        self.assertIsNone(cap.dispatch(0,now=1))
        cap.finish('a',acknowledged=True)
        self.assertEqual(cap.dispatch(0,now=2),'b')
        self.assertFalse(cap.enqueue('c',1,1,prompt=1,deadline=10,now=0))
        cap.commit('a')
        self.assertTrue(cap.enqueue('c',1,1,prompt=1,deadline=10,now=0))

    def test_stable_id_and_exactly_once_strict_deadline(self):
        job=logical_job(contract='c',snapshot='s',mode='correct',arm='fit',prompt_id='p',attempt_ordinal=0,method='greedy',seed=1)
        self.assertEqual(job,logical_job(contract='c',snapshot='s',mode='correct',arm='fit',prompt_id='p',attempt_ordinal=0,method='greedy',seed=1))
        ledger=JobLedger();ledger.reserve(job,deadline=10,work=4)
        self.assertFalse(ledger.publish(job['job_id'],{'ok':True},observed=10))
        self.assertEqual(ledger.records[job['job_id']]['state'],'INCOMPLETE')
        self.assertFalse(ledger.publish(job['job_id'],{'ok':True},observed=9))
        with self.assertRaises(ValueError):
            ledger.reserve(job,deadline=20,work=4)


class ProcessSlots(unittest.TestCase):
    def test_unacknowledged_termination_keeps_handle_and_charge(self):
        from unittest.mock import Mock, patch
        import subprocess
        from oeis_learn.synthesis.worker_pool import WorkerPool
        from oeis_learn.synthesis.service_policy import ServiceLimits
        pool = WorkerPool(contract_sha256='sha256:'+'a'*64,
            limits=ServiceLimits(max_inflight_batches=2))
        process = Mock(pid=1234)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired('worker', 5)
        pool.slots[0].process = process
        request = dict(request_id='job', candidates=[dict(candidate_id='a')])
        def fail(*args, **kwargs):
            pool.slots[0]._retire()
        with patch('oeis_learn.synthesis.worker_pool.os.killpg'), patch.object(pool.slots[0], 'call', side_effect=fail):
            with self.assertRaises(RuntimeError):
                pool.call(request, deadline=10**20)
        self.assertIs(pool.slots[0].process, process)
        self.assertIn('job', pool.capacity.active)
        self.assertTrue(pool.closed)
        process.stdin.close.assert_not_called()

    def test_cleanup_has_one_deadline_for_all_slots(self):
        from unittest.mock import Mock, patch
        from oeis_learn.synthesis.worker_pool import WorkerPool
        from oeis_learn.synthesis.service_policy import ServiceLimits
        pool = WorkerPool(contract_sha256='sha256:'+'a'*64,
            limits=ServiceLimits(max_inflight_batches=2))
        clock = [100.0]
        waits = []
        def wait(timeout):
            waits.append(timeout)
            clock[0] += min(3, timeout)
        for slot in pool.slots:
            slot.process = Mock(pid=1234)
            slot.process.poll.return_value = None
            slot.process.wait.side_effect = wait
        with patch('oeis_learn.synthesis.worker_pool.time.monotonic', side_effect=lambda: clock[0]), patch('oeis_learn.synthesis.worker_pool.os.killpg'):
            pool.close()
        self.assertEqual(waits, [5, 2])

    def test_two_process_generations_and_recovery(self):
        import json
        from pathlib import Path
        import sys
        import tempfile
        import threading
        import time
        from oeis_learn.synthesis.worker_pool import WorkerPool
        from oeis_learn.synthesis.service_policy import ServiceLimits
        worker = '''import json,struct,sys,os
while True:
    head=sys.stdin.buffer.read(4)
    if not head: break
    n=struct.unpack('<I',head)[0]
    r=json.loads(sys.stdin.buffer.read(n))
    out={k:r[k] for k in ['protocol','request_id','worker_generation','contract_sha256']}
    if r['request_id']=='bad':out['worker_generation']+=1
    out.update(results=[dict(candidate_id=c['candidate_id'],outcome='ok',values=[os.environ.get('OMP_NUM_THREADS')]) for c in r['candidates']],unstarted=[],complete=True)
    raw=json.dumps(out).encode();sys.stdout.buffer.write(struct.pack('<I',len(raw))+raw);sys.stdout.buffer.flush()
'''
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'worker.py';path.write_text(worker)
            pool=WorkerPool(command=[sys.executable,str(path)],contract_sha256='sha256:'+'a'*64,limits=ServiceLimits(max_inflight_batches=2))
            rows=[]
            def call(i):
                request=dict(protocol='synthesis-worker/v1',request_id=str(i),contract_sha256=pool.contract,
                    candidates=[{'candidate_id':str(i)}],remaining_budget={'wall_time_ms':1000},worker_generation=0)
                with pool.affinity(i):
                    rows.append(pool.call(request,deadline=time.monotonic()+3))
            try:
                threads=[threading.Thread(target=call,args=(i,)) for i in range(2)]
                for t in threads:t.start()
                for t in threads:t.join(timeout=5)
                self.assertEqual(len(rows),2)
                self.assertTrue(all(r['complete'] for r in rows),rows)
                self.assertTrue(all(r['results'][0]['values']==['1'] for r in rows),rows)
                self.assertEqual(len({s.generation for s in pool.slots}),2)
                self.assertFalse(pool.capacity.active)
                self.assertFalse(pool.capacity.pending)
                request=dict(protocol='synthesis-worker/v1',request_id='bad',contract_sha256=pool.contract,
                    candidates=[{'candidate_id':'bad'}],remaining_budget={'wall_time_ms':1000},worker_generation=0)
                failed=pool.call(request,deadline=time.monotonic()+3)
                self.assertEqual(failed['outcome'],'verification_failure')
                self.assertFalse(pool.capacity.active)
                recovered=pool.call(dict(request,request_id='healthy'),deadline=time.monotonic()+3)
                self.assertTrue(recovered['complete'])
                self.assertGreater(recovered['generation'],failed['generation'])
            finally:
                pool.close()
