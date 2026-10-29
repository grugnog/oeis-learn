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
    def test_evaluation_handoff_and_training_stage_write_each_record_once(self):
        from types import SimpleNamespace
        from oeis_learn.pipeline_validation.identity import write,file_hash
        from oeis_learn.pipeline_validation.followup_evaluation import evaluate,report
        from oeis_learn.pipeline_validation.followup_study import Study
        from oeis_learn.pipeline_validation.registry import Stage
        arms=('raw','rescue','equal_budget','shuffled','absent','template')
        for training in (False,True):
            with self.subTest(training=training),tempfile.TemporaryDirectory() as d:
                root=Path(d);prepared=root/'prepared'
                write(prepared/'recipe.json',{})
                source=root/'weights.pt';source.write_bytes(b'original weights')
                snapshots={2:dict(path=source,weights_sha256=file_hash(source),pointer={})}
                calls=[]
                def invoke(output,label,mode,files,**kwargs):
                    target=output/label;target.mkdir()
                    cfg=json.loads(files['study.json' if mode=='pilot-followup-train' else 'settings.json'].read_text())
                    calls.append((mode,files,cfg))
                    if mode=='pilot-followup-train':
                        weight=target/'private/best-weights.pt';weight.parent.mkdir();weight.write_bytes(b'new weights')
                        write(target/'weights.json',dict(best_weights=file_hash(weight)))
                        return dict(status='COMPLETE'),target
                    if mode=='pilot-generate':
                        self.assertNotIn('truth.json',files)
                        self.assertNotIn('pilot_selections_sha256',cfg)
                        write(target/'selections.json',dict(a=dict(strata=dict(split='parameter_development'))))
                        return dict(clock_complete=True),target
                    self.assertEqual(mode,'pilot-score')
                    self.assertNotIn('weights.pt',files)
                    self.assertEqual(cfg['pilot_selections_sha256'],file_hash(files['selections.json']))
                    self.assertEqual(cfg['pilot_truth_sha256'],file_hash(files['truth.json']))
                    write(target/'outcomes.json',dict(a={arm:dict(success=False,complete=True,reason='visible_mismatch') for arm in arms}))
                    return dict(complete=True),target
                write(prepared/'private/truth.json',{})
                pilot=SimpleNamespace(invoke=invoke)
                args=SimpleNamespace(tokens=25000000,seed=1)
                owner=Study(args,root,dict(evaluation_policy=MATCHED,snapshots=[2]),SimpleNamespace(pilot=pilot))
                owner.imported=dict(prepared=prepared,manifest=dict(identity='work',files={
                    'prompts.json':dict(sha256='prompts'),'private/truth.json':dict(sha256=file_hash(prepared/'private/truth.json'))}),
                    contract='contract',snapshots=snapshots)
                output=root/'stages/diagnostics.snapshot-2'
                if training:
                    result=owner(Stage('followup.uniform',4,adapter='followup-train'),output)
                    self.assertEqual(result['stage_state'],'PASS')
                else:
                    result=evaluate(owner,output,2)
                    self.assertTrue(result['scientific_complete'])
                    self.assertEqual(report(owner,root/'report',root,{})['evaluation'],'COMPLETE')
                generation=json.loads((output/'private/generation-settings.json').read_text())
                scoring=json.loads((output/'private/scoring-settings.json').read_text())
                self.assertNotIn('pilot_selections_sha256',generation)
                self.assertIn('pilot_selections_sha256',scoring)
                self.assertEqual(json.loads((output/'evaluation.json').read_text())['groups']['parameter_development|raw']['denominator'],1)
                self.assertEqual(len(calls),3 if training else 2)

    def test_serial_helper_surfaces_child_failure(self):
        import contextlib
        import importlib.util
        import io
        import sys
        path=Path(__file__).resolve().parents[2]/'scripts/pilot_hip_serial.py'
        spec=importlib.util.spec_from_file_location('pilot_hip_serial',path)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);log=root/'1.log';stderr=io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(stderr):
                code=helper.run_command([sys.executable,'-c',"print('actual qualification failure');raise SystemExit(7)"],root,log)
            self.assertEqual(code,7)
            self.assertIn('actual qualification failure',stderr.getvalue())
            self.assertIn(str(log),stderr.getvalue())
            self.assertEqual(log.read_text().strip(),'actual qualification failure')

    def test_actual_portfolios_accept_every_registered_policy(self):
        import time
        from oeis_learn.fresh.bigint_diagnostics import _portfolio
        from oeis_learn.decoder.bigint_generation import GenerationRequest
        from oeis_learn.fresh.pipeline_jobs import NativeRequest
        from oeis_learn.pipeline_validation.pilot_clocks import POLICY
        from oeis_learn.synthesis.contracts import ResourceProfile
        class Client:
            limits=ResourceProfile()
            def execute(self,source,indices,**kwargs):
                return dict(outcome='ok',values=list(indices))
            def fit(self,source,prompt,**kwargs):
                return dict(outcome='unsat_in_scope',admitted=[],complete=True)
        row=dict(id='a',indices=list(map(str,range(20))),values=['7']*20)
        recipe=dict(model=dict(seed=1),search=dict(beam=None),fitting=dict(disabled=False),pipeline=dict(workers=1))
        source=decode_body(encode_body('(program (return $index))'))
        for policy in (POLICY,MATCHED,DRAIN):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as d:
                evidence=FullClockEvidence(d,[row],source='source',work='work',policy=policy)
                for mode,count in attempts(policy).items():
                    gen=_portfolio(None,None,row,recipe,Client(),deadline=time.monotonic()+60,
                        attempt_count=count,template=mode=='template',mode=mode,snapshot='snapshot',
                        evidence=evidence,evaluation_policy=policy)
                    value=None;native=0
                    while True:
                        try:request=gen.send(value)
                        except StopIteration as stop:
                            self.assertTrue(stop.value['complete']);break
                        if isinstance(request,GenerationRequest):
                            value=dict(source=source,complete=True,tokens=encode_body(source),generation_seconds=0.,generation_batch_size=1)
                        else:
                            self.assertIsInstance(request,NativeRequest);native+=1;value=request.operation()
                    self.assertEqual(native,count+(policy!=POLICY and mode!='template'))
                evidence.finish()
                self.assertTrue(evidence.clock_summary['cost_complete'])
        for policy,mode,count,seconds in [(MATCHED,'shuffled',1,60),(DRAIN,'correct',8,180),('unknown','correct',8,60)]:
            gen=_portfolio(None,None,row,recipe,Client(),deadline=time.monotonic()+60,mode=mode,
                attempt_count=count,portfolio_seconds=seconds,evaluation_policy=policy)
            with self.assertRaises(ValueError):next(gen)

    def test_isolated_raw_evidence_survives_expired_fit(self):
        import time
        from oeis_learn.fresh.bigint_diagnostics import _portfolio
        from oeis_learn.decoder.bigint_generation import GenerationRequest
        from oeis_learn.fresh.pipeline_jobs import NativeRequest
        from oeis_learn.synthesis.contracts import ResourceProfile
        class Client:
            limits=ResourceProfile()
            def execute(self,source,indices,**kwargs):return dict(outcome='ok',values=list(indices))
            def runtime_identity(self,profile):return 'sha256:'+'1'*64
        row=dict(id='a',indices=list(map(str,range(20))),values=list(map(str,range(20))))
        recipe=dict(model=dict(seed=1),search=dict(beam=None),fitting=dict(disabled=False),pipeline=dict(workers=1))
        source=decode_body(encode_body('(program (return $index))'))
        for policy in (MATCHED,DRAIN):
            with tempfile.TemporaryDirectory() as d:
                evidence=FullClockEvidence(d,[row],source='s',work='w',policy=policy)
                gen=_portfolio(None,None,row,recipe,Client(),deadline=time.monotonic()+60,mode='correct',
                    attempt_count=8,snapshot='s',evidence=evidence,evaluation_policy=policy)
                value=None
                while True:
                    try:request=gen.send(value)
                    except StopIteration as stop:result=stop.value;break
                    if isinstance(request,GenerationRequest):
                        value=dict(source=source,complete=True,tokens=encode_body(source),generation_seconds=0.,generation_batch_size=1)
                    else:
                        self.assertIsInstance(request,NativeRequest)
                        value=request.operation() if request.job['arm']=='raw' else request.expired()
                self.assertFalse(result['complete'])
                self.assertTrue(result['arm_evidence']['raw']['complete'])
                self.assertIsNotNone(result['raw'])
                self.assertFalse(result['arm_evidence']['equal_budget']['complete'])

    def test_serial_start_step_uses_new_output_and_keeps_original_numbers(self):
        import contextlib,importlib.util,io
        from unittest.mock import patch
        path=Path(__file__).resolve().parents[2]/'scripts/pilot_hip_serial.py'
        spec=importlib.util.spec_from_file_location('pilot_hip_serial',path)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);helper.__file__=str(root/'scripts/pilot_hip_serial.py');seen=[]
            argv=['--run','r','--admission','a','--inputs','p','--output','out','--start-step','4']
            with patch.object(helper,'run_command',side_effect=lambda cmd,cwd,log:(seen.append(log.name) or 0)),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(helper.main(argv),0)
                with self.assertRaises(FileExistsError):helper.main(argv)
            self.assertEqual(seen,['4.log','5.log','6.log'])
            self.assertEqual(json.loads((root/'out/execution-plan.json').read_text())['skipped_steps'],[1,2,3])

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
