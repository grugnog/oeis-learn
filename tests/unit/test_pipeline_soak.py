"""Host-only soak authority, work, selection and cumulative recovery contracts."""
import copy
import json
from pathlib import Path
import unittest

from oeis_learn.pipeline_validation.identity import seal


class SoakContracts(unittest.TestCase):
    def test_continuation_cannot_reset_counters_or_lose_resume_state(self):
        from oeis_learn.fresh.engineering import Control
        from oeis_learn.pipeline_validation.soak_contract import lease
        h='sha256:'+'a'*64
        clock=[100.]
        l=lease(h,h,h,now=100)
        cfg=seal(dict(version='pipeline-engineering-worker/v1',source=h,work=h,choice=h,mode='window',
            batch_size=8,window=0,resume=False,lease=l,token_target=1,seconds=3600,
            instrumented=False,cadence_tokens=10))
        c=Control({'engineering':cfg},{},Path('/unused'),clock=lambda:clock[0])
        c.update([{'tokens':[1]*10}],.1);c.cycled();clock[0]=3700
        first=c.finish();self.assertTrue(first['complete'])
        second=seal(dict(cfg,window=1,resume=True))
        resumed=Control({'engineering':second},{'engineering':copy.deepcopy(first['state'])},Path('/unused'),clock=lambda:clock[0])
        self.assertEqual(resumed.state['tokens'],10)
        self.assertEqual(resumed.deadline,7300)
        with self.assertRaises(ValueError):Control({'engineering':second},{},Path('/unused'),clock=lambda:clock[0])

    def test_ordinary_timer_does_not_synchronize_gpu(self):
        import ast
        path=Path(__file__).resolve().parents[2]/'src/oeis_learn/tracking/foundation_metrics.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PhaseTimer')
        namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        # Execute the actual class without importing its container-owned dependencies.
        values=iter([1,4]);timer=namespace['PhaseTimer']('cuda',clock=lambda:next(values),synchronized=False)
        with timer.phase('ordinary'):pass
        self.assertEqual(timer.values,{'ordinary':3})

    def test_registered_scope_keeps_learning_denied(self):
        from oeis_learn.pipeline_validation.contracts import parse
        from oeis_learn.pipeline_validation.soak_contract import manifest
        value=manifest()
        self.assertEqual(parse(value)['scope'],'engineering-soak')
        for field in ('training','resume','final'):
            bad=copy.deepcopy(value);bad['authority'][field]=True
            with self.assertRaises(ValueError):parse(seal(bad))

    def test_roster_and_phase_reservations_fit(self):
        from oeis_learn.pipeline_validation.registry import soak_registry,resolve
        from oeis_learn.pipeline_validation.soak_contract import SPEC
        stages=resolve(soak_registry(('podman','docker'),'podman'),'all-ready')
        self.assertEqual(len({s.id for s in stages}),len(stages))
        self.assertEqual(len([s for s in stages if s.adapter=='soak-batch']),12)
        self.assertEqual(SPEC['work_windows_seconds'],[3600,3600])
        for phase,cap in [(1,3600),(2,10800),(3,7200),(4,18000),(5,14400)]:
            self.assertLessEqual(sum(s.seconds for s in stages if s.phase==phase and s.adapter!='image'),cap)

    def test_selection_requires_all_trials_and_prefers_small_within_band(self):
        from oeis_learn.pipeline_validation.soak_contract import select_batch
        rows={str(b):[dict(complete=True,tokens=100,seconds=1/rate) for _ in range(3)] for b,rate in [(8,1),(16,1.03),(32,1.2)]}
        self.assertEqual(select_batch(rows),32)
        rows['32'][1]['complete']=False
        self.assertEqual(select_batch(rows),8)
        rows['8'][0]['seconds']=float('nan')
        self.assertEqual(select_batch(rows),16)

    def test_lease_does_not_renew_across_restart(self):
        from oeis_learn.pipeline_validation.soak_contract import lease,remaining
        l=lease('source','work','choice',now=100)
        self.assertEqual(remaining(l,now=4000),3900)
        self.assertEqual(remaining(l,now=7900),0)
        bad=dict(l,expires=8000)
        with self.assertRaises(ValueError):remaining(bad,now=4000)

    def test_entire_partition_and_token_cadence_are_frozen(self):
        from oeis_learn.pipeline_validation.soak_contract import data_contract
        examples={k:[dict(id=k+str(i),values=['1']*20,indices=list(map(str,range(20))),tokens=[1]*((i%3)+1)) for i in range(n)]
                  for k,n in [('training',513),('parameter_development',128),('structural_development',128)]}
        d=data_contract(examples)
        self.assertEqual(d['counts']['training'],513)
        self.assertEqual(d['cadence_tokens'],8*sum(len(r['tokens']) for r in examples['training']))
        bad=copy.deepcopy(examples);bad['training']=bad['training'][:128]
        with self.assertRaises(ValueError):data_contract(bad)


if __name__=='__main__':unittest.main()
