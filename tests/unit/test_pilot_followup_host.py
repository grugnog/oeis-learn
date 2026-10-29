"""Standard-library regression checks; no engine/GPU needed."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from oeis_learn.decoder.bigint_codec import encode_body,decode_body,fitting_policy
from oeis_learn.fresh.template_baseline import template_bank
from oeis_learn.pipeline_validation.diagnostics_contract import plan,parse
from oeis_learn.pipeline_validation.followup_training import profile,validate,sample
from oeis_learn.pipeline_validation.pilot_clocks import MATCHED,DRAIN,attempts,clock_record,FullClockEvidence


class FollowupTests(unittest.TestCase):
    def test_codec_dispatch(self):
        for source in template_bank()['sources']:
            self.assertEqual(fitting_policy(decode_body(encode_body(source))),'none')
        self.assertEqual(fitting_policy('(program (return (int 3)))'),'eligible-literals/v1')

    def test_plans_are_closed_and_legacy_unchanged(self):
        legacy=plan(operation='models')
        self.assertEqual(legacy['version'],'pipeline-pilot-diagnostics/v1')
        self.assertEqual(parse(legacy),legacy)
        for policy in (MATCHED,DRAIN):
            p=plan(operation='evaluate',snapshots=[2,5],include_selected=True,evaluation_policy=policy)
            self.assertEqual(p['snapshots'],[2,5,6]);self.assertEqual(parse(p),p)
            p['training']=True
            with self.assertRaises(ValueError):parse(p)
        p=plan(operation='models',include_selected=True)
        self.assertEqual(parse(p),p)
        with self.assertRaises(ValueError):plan(operation='models',snapshots=[6])

    def test_balancing_and_reproducibility(self):
        rows=[dict(id=i,family='large') for i in range(99)]+[dict(id=99,family='small')]
        uniform=sample(rows,random.Random(12),16,'uniform')
        balanced=sample(rows,random.Random(12),1000,'family_balanced')
        self.assertEqual(balanced,sample(rows,random.Random(12),1000,'family_balanced'))
        self.assertTrue(400<sum(r['family']=='small' for r in balanced)<600)
        self.assertEqual(len(uniform),16)
        self.assertEqual(validate(profile()),profile())
        with self.assertRaises(ValueError):profile(tokens=10**12)

    def test_matched_clocks_keep_full_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            evidence=FullClockEvidence(d,[dict(id='a')],source='s',work='w',policy=DRAIN)
            self.assertEqual(len(evidence.jobs),32)
            for mode in attempts(DRAIN):
                for i in range(8):
                    evidence('a',mode,i,'STARTED',{})
                    evidence('a',mode,i,'GENERATED',{})
                    evidence('a',mode,i,'INCOMPLETE',{})
                evidence.portfolio('a',mode,clock_record(prompt='a',mode=mode,snapshot='s',start=0,
                    scope_deadline=100,finish=10,expected=8,started=8,terminal=8,policy=DRAIN))
            evidence.finish()
            report=json.loads((Path(d)/'portfolio-clocks.json').read_text())
            self.assertEqual(report['expected_portfolios'],4)
            self.assertTrue(report['cost_complete'])
            self.assertEqual(report['policy'],DRAIN)

    def test_selected_checkpoint_hashes_and_contract(self):
        from oeis_learn.pipeline_validation.diagnostics_import import selected_snapshot
        from oeis_learn.pipeline_validation.identity import file_hash
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'stages/pilot.train-5/train-5/private/trainer/checkpoints'
            state=folder/'state-test';state.mkdir(parents=True)
            weight=state/'weights.pt';weight.write_bytes(b'bound weights only')
            record=dict(files=dict(weights=dict(path='state-test/weights.pt',sha256=file_hash(weight))),
                binding=dict(weights=file_hash(weight),contract='contract'))
            checkpoint=state/'checkpoint.json';checkpoint.write_text(json.dumps(record))
            trainer=dict(selected=dict(path='state-test/checkpoint.json',sha256=file_hash(checkpoint)))
            self.assertEqual(selected_snapshot(root,trainer,dict(contract='contract'))['ordinal'],6)
            with self.assertRaises(ValueError):selected_snapshot(root,trainer,dict(contract='other'))
            weight.write_bytes(b'tampered')
            with self.assertRaises(ValueError):selected_snapshot(root,trainer,dict(contract='contract'))

    def test_native_priority_drains_before_more_decoding(self):
        import time
        from types import SimpleNamespace
        from oeis_learn.fresh.pipeline_jobs import logical_job,NativeRequest
        from oeis_learn.fresh.pipeline_scheduler import drive
        completed=[];calls=[]
        def worker(i):
            yield SimpleNamespace(temperature=0,memory=SimpleNamespace(shape=(1,1,1)))
            job=logical_job(contract='c',snapshot='s',mode='correct',arm='fit',prompt_id=str(i),attempt_ordinal=0,method='test',seed=i)
            def operation():
                time.sleep(.01);completed.append(i);return i
            value=yield NativeRequest(job,i,time.monotonic()+5,operation)
            self.assertEqual(value,i)
            yield SimpleNamespace(temperature=0,memory=SimpleNamespace(shape=(1,1,1)))
            return i
        def decode(decoder,requests):
            calls.append(len(completed))
            if len(calls)>1:self.assertEqual(len(completed),2)
            return [None]*len(requests)
        result,boundary=drive(None,[worker(0),worker(1)],workers=1,batch_size=8,sample=decode,native_priority=True)
        self.assertEqual(result,[0,1]);self.assertEqual(boundary['active'],0)

if __name__=='__main__':unittest.main()
