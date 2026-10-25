import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from oeis_learn.pipeline_validation.identity import seal


class NativeCapacityTests(unittest.TestCase):
    def test_closed_manifest_graph_cpu_only_and_budget(self):
        from oeis_learn.pipeline_validation.contracts import parse
        from oeis_learn.pipeline_validation.registry import native_capacity_registry,resolve
        root=Path(__file__).resolve().parents[2]
        manifest=json.loads((root/'configs/foundation/pipeline_native_capacity.json').read_text())
        parse(manifest)
        stages=resolve(native_capacity_registry(('podman',),'podman'),'all-ready')
        self.assertFalse(any(s.adapter=='hip' or s.id.endswith('.gpu') for s in stages))
        self.assertLessEqual(sum(s.seconds for s in stages if s.phase==3),7200)
        self.assertEqual([s.id for s in stages if s.phase==3],['native.capacity','native.capacity.report'])
        for key,value in [('workloads',[]),('phases',[1,2]),('comparisons',[])]:
            with self.subTest(key=key),self.assertRaises(ValueError):parse(seal(dict(manifest,**{key:value})))

    def trial(self,output,*,workers,corpus,contract,cap,admission='queued8/v1'):
        rows=[dict(id=r['id'],stratum=r['stratum'],complete=True,outcome='FIT',admitted=[dict(assignment={'p0':'7'},verification='exact')]) for r in corpus['rows']]
        from oeis_learn.fresh.pipeline_jobs import logical_job,JobLedger
        ledger=JobLedger();timing={}
        for r in rows:
            job=logical_job(contract=contract,snapshot='fixed-native-corpus',mode='native',arm='fit',prompt_id=r['id'],attempt_ordinal=0,method='fixture',seed=20261016)
            ledger.reserve(job,deadline=4,work=1);ledger.publish(job['job_id'],r,observed=.3)
            timing[r['id']]=dict(admitted=0,dispatched=.1,native_completed=.2,deadline=4)
        return seal(dict(version='pipeline-native-wave-trial/v1',workers=workers,corpus=corpus['identity'],seconds={1:2,2:1,4:1.02}[workers],primary_seconds={1:2,2:1,4:1.02}[workers],results=rows,complete=True,boundary=ledger.boundary({}),metrics={},failure=None,admission=admission,jobs=ledger.records,timing=timing,started=0,parent_deadline=cap))


    def run_screen(self,trial=None,**kwargs):
        from oeis_learn.pipeline_validation.native_capacity import investigate
        with tempfile.TemporaryDirectory() as f:
            p=Path(f)
            with patch('oeis_learn.pipeline_validation.role.native_trial',trial or self.trial),patch('oeis_learn.pipeline_validation.native_capacity.observed_trial',side_effect=lambda fn,*a,**k:fn(*a,**k)):
                result=investigate(None,p,dict(source='source',contract='parent',options=dict(seconds=4470,memory_ceiling=16<<30)),**kwargs)
            receipts={n.parent.name:json.loads(n.read_text()) for n in p.glob('slots/*/receipt.json')}
            return result,receipts

    def test_all_slots_are_planned_and_ordinary_pairs_select_smaller(self):
        result,receipts=self.run_screen()
        self.assertEqual(len(receipts),18)
        self.assertTrue(all(r['state']=='PASS' for r in receipts.values()))
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['selected_workers'],2)
        self.assertFalse(result['optimization_adoption'])
        self.assertFalse(result['research_cost_admission'])
        self.assertEqual(len(result['comparisons']),2)
        for c in result['comparisons']:
            self.assertEqual([p['order'] for p in c['pairs']],['AB','BA'])
            self.assertTrue(c['supported'])

    def test_incomplete_baseline_blocks_both_no_subset_or_favorable_ratios(self):
        def incomplete(output,**kw):
            r=self.trial(output,**kw)
            if Path(output).name=='calibration-1':r=seal(dict(r,complete=False,results=[dict(x,complete=False) for x in r['results']]))
            return r
        result,receipts=self.run_screen(incomplete)
        self.assertEqual(result['status'],'INCONCLUSIVE')
        self.assertEqual(result['selected_workers'],1)
        self.assertEqual(result['comparisons'],[])
        self.assertEqual(len(receipts),18)
        self.assertTrue(all(r['state']=='BLOCKED' for k,r in receipts.items() if k.startswith('w2') or k.startswith('w4') or k.startswith('observed')))

    def test_incomplete_candidate_retains_other_named_hypothesis(self):
        def incomplete(output,**kw):
            r=self.trial(output,**kw)
            return seal(dict(r,complete=False)) if Path(output).name=='w2-pair0-B' else r
        result,receipts=self.run_screen(incomplete)
        self.assertEqual(result['comparisons'][0]['status'],'INCONCLUSIVE')
        self.assertEqual(result['comparisons'][0]['pairs'],[])
        self.assertTrue(result['comparisons'][1]['supported'])
        self.assertEqual(result['selected_workers'],4)
        self.assertNotEqual(result['status'],'COMPLETE')

    def test_semantic_mismatch_cannot_produce_gain(self):
        def changed(output,**kw):
            r=self.trial(output,**kw)
            if Path(output).name=='w4-pair1-B':
                rows=[dict(x) for x in r['results']];rows[0]['admitted']=[];r=seal(dict(r,results=rows))
            return r
        result,_=self.run_screen(changed)
        self.assertEqual(result['comparisons'][1]['pairs'],[])
        self.assertFalse(result['comparisons'][1]['supported'])

    def test_resource_failure_and_interrupt_preserve_terminal_rosters_and_propagate(self):
        from oeis_learn.pipeline_validation.native_capacity import investigate
        for error in (MemoryError('stop'),KeyboardInterrupt()):
            with self.subTest(type=type(error).__name__),tempfile.TemporaryDirectory() as f:
                def failed(output,**kw):raise error
                with patch('oeis_learn.pipeline_validation.role.native_trial',failed),self.assertRaises(type(error)):
                    investigate(None,Path(f),dict(source='source',contract='parent',options=dict(seconds=4470,memory_ceiling=16<<30)))
                rows=[json.loads(p.read_text()) for p in Path(f).glob('slots/*/receipt.json')]
                self.assertEqual(len(rows),18)
                self.assertFalse(any(r['state'] in ('PENDING','RUNNING','READY') for r in rows))
                self.assertTrue((Path(f)/'native-capacity.json').is_file())

    def test_changed_prospective_child_stops_before_next_candidate(self):
        def changed(output,**kw):
            r=self.trial(output,**kw)
            if Path(output).name=='w2-warmup':
                p=Path(output).parents[1]/'comparison-contract.json';p.write_text('{}')
            return r
        result,receipts=self.run_screen(changed)
        self.assertEqual(result['status'],'INCONCLUSIVE')
        self.assertEqual(result['selected_workers'],1)
        self.assertEqual(receipts['w2-pair0-A']['state'],'BLOCKED')
        self.assertTrue(all(not c['pairs'] for c in result['comparisons']))

    def test_wave_admission_never_constructs_more_than_slot_count_or_renews(self):
        from oeis_learn.pipeline_validation.role import native_trial,native_corpus
        from types import SimpleNamespace
        import sys
        active=[];created=[]
        class Client:
            def __init__(self,*a,**k):self.worker=SimpleNamespace(stats={},cancel_all=lambda:None)
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def fit(self,source,prompt,deadline):
                active.append(deadline);return dict(complete=True,outcome='FIT',admitted=[])
        def drive(decoder,generators,*,workers,policies,ledger):
            requests=[next(g) for g in generators];created.append(len(requests));self.assertLessEqual(len(requests),workers)
            values=[]
            for g,r in zip(generators,requests):
                ledger.reserve(r.job,deadline=r.deadline,work=1);v=r.operation();self.assertLessEqual(active[-1],r.deadline)
                ledger.publish(r.job['job_id'],v,observed=r.deadline-.001)
                try:g.send(v)
                except StopIteration as s:values.append(s.value)
            return values,ledger.boundary(policies)
        mods={'oeis_learn.synthesis.client':SimpleNamespace(SynthesisClient=Client),'oeis_learn.synthesis.admission':SimpleNamespace(VisiblePrompt=lambda *a:a)}
        for workers in (1,2,4):
            created.clear()
            with tempfile.TemporaryDirectory() as f,patch.dict(sys.modules,mods),patch('oeis_learn.fresh.pipeline_scheduler.drive',drive):
                r=native_trial(Path(f),workers=workers,corpus=native_corpus(),contract='parent',cap=30,admission='slot-waves/v1')
            self.assertTrue(r['complete']);self.assertEqual(sum(created),18)
            self.assertEqual(len(r['jobs']),18)
            self.assertEqual(r['admission'],'slot-waves/v1')

    def test_clock_mutation_missing_debits_and_wrong_workers_cannot_score(self):
        from copy import deepcopy
        for mutation in ('deadline','observed','debit','frontier','workers'):
            def changed(output,**kw):
                r=self.trial(output,**kw)
                if Path(output).name=='w2-pair0-B':
                    r=deepcopy(r);job=next(iter(r['jobs'].values()))
                    if mutation=='deadline':r['timing'][job['job']['prompt_id']]['deadline']+=1
                    if mutation=='observed':job['observed']=job['deadline']
                    if mutation=='debit':job['consumed']=0
                    if mutation=='frontier':r['boundary']=seal(dict(r['boundary'],frontier=[]))
                    if mutation=='workers':r['workers']=4
                    r=seal(r)
                return r
            with self.subTest(mutation=mutation):
                result,_=self.run_screen(changed)
                self.assertEqual(result['comparisons'][0]['pairs'],[])
                self.assertEqual(result['comparisons'][0]['status'],'INCONCLUSIVE')

    def test_primary_wall_overrun_and_unsupported_gain_retain_serial(self):
        def slow(output,**kw):
            r=self.trial(output,**kw)
            if Path(output).name.startswith(('w2','w4')) and not Path(output).name.endswith('-A'):
                r=seal(dict(r,primary_seconds=kw['cap']+1))
            return r
        result,_=self.run_screen(slow)
        self.assertEqual(result['selected_workers'],1)
        self.assertTrue(all(c['pairs']==[] for c in result['comparisons']))
        def negative(output,**kw):
            r=self.trial(output,**kw)
            return seal(dict(r,primary_seconds=2))
        result,_=self.run_screen(negative)
        self.assertEqual(result['status'],'COMPLETE');self.assertEqual(result['selected_workers'],1)
        self.assertTrue(all(not c['supported'] for c in result['comparisons']))

    def test_full_simulated_cpu_graph_exports_one_verified_archive(self):
        from oeis_learn.pipeline_validation.runner import execute
        from oeis_learn.pipeline_validation.registry import native_capacity_registry
        from oeis_learn.pipeline_validation.native_capacity import investigate,NativeCapacityStages
        from oeis_learn.pipeline_validation.bundle import verify
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out';a=SimpleNamespace(receipts={});report=NativeCapacityStages(a)
            def adapter(stage,output):
                if stage.id=='native.capacity':
                    return investigate(None,output,dict(source='source',contract='parent',options=dict(seconds=4470,memory_ceiling=16<<30)))
                if stage.id=='native.capacity.report':return report(stage,output)
                return {'fixture':True}
            adapter.receipts=a.receipts
            with patch('oeis_learn.pipeline_validation.role.native_trial',self.trial),patch('oeis_learn.pipeline_validation.native_capacity.observed_trial',side_effect=lambda fn,*args,**kw:fn(*args,**kw)):
                code=execute(native_capacity_registry(('podman',),'podman'),root,adapter=adapter,binding={'source':'source'},cap=10000,reserve=1200,contract={'scope':'engineering-native-capacity'})
            self.assertEqual(code,0)
            manifest=verify(root.with_suffix('.zip'));self.assertEqual(manifest['summary_state'],'PASS')
            d=json.loads((root/'summary.json').read_text());self.assertEqual(d['scope'],'engineering-native-capacity');self.assertFalse(d['feature_complete'])
            self.assertEqual(len(list((root/'stages/native.capacity/slots').glob('*/receipt.json'))),18)
