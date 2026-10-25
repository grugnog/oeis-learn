"""Stdlib sizing authority, fixed work, accounting and complete-only reports."""
import json
from pathlib import Path
import unittest
from oeis_learn.pipeline_validation.identity import seal

ROOT=Path(__file__).resolve().parents[2]


class Sizing(unittest.TestCase):
    def test_exact_profiles_counts_and_closed_authority(self):
        from oeis_learn.pipeline_validation.model_sizes import PROFILES,parameters,validate_profile
        self.assertEqual([p['width'] for p in PROFILES.values()],[256,512,768])
        self.assertEqual(parameters(PROFILES['m8'])['total'],8118294)
        for p in PROFILES.values():
            self.assertEqual(validate_profile(p),p)
            with self.assertRaises(ValueError):validate_profile(seal(dict(p,width=1024)))
            with self.assertRaises(ValueError):validate_profile(seal(dict(p,precision='bfloat16')))

    def test_closed_manifest_and_fifteen_slots_fit_phases(self):
        from oeis_learn.pipeline_validation.contracts import parse
        from oeis_learn.pipeline_validation.registry import model_sizing_registry,resolve
        from oeis_learn.pipeline_validation.model_sizes import SLOTS
        m=json.loads((ROOT/'configs/foundation/pipeline_model_sizing.json').read_text());parse(m)
        stages=resolve(model_sizing_registry(('podman',),'podman'),'all-ready')
        self.assertEqual(len(SLOTS),15)
        self.assertEqual([s.id for s in stages if s.adapter=='sizing-trial'],SLOTS)
        self.assertLessEqual(sum(s.seconds for s in stages if s.phase==4),18000)
        for key,v in [('workloads',[]),('phases',[1,4]),('authority',dict(m['authority'],training=True))]:
            with self.assertRaises(ValueError):parse(seal(dict(m,**{key:v})))

    def test_resource_reservation_grows_with_model_and_is_not_weight_storage(self):
        from oeis_learn.pipeline_validation.model_sizes import PROFILES,parameters,peak_estimate
        peaks=[peak_estimate(p) for p in PROFILES.values()]
        self.assertEqual(sorted(peaks),peaks)
        self.assertTrue(all(peak_estimate(p)>parameters(p)['total']*16 for p in PROFILES.values()))

    def test_all_rows_planned_and_partial_rows_not_program_success(self):
        from oeis_learn.pipeline_validation.model_sizing import attempt_plan,generation_counts
        prompts=[dict(id=str(i),values=['0']*20,indices=list(map(str,range(20)))) for i in range(32)]
        plan=attempt_plan(prompts)
        self.assertEqual(len(plan),64);self.assertEqual(len({r['id'] for r in plan}),64)
        rows=[dict(r,state='PARTIAL',tokens=[1,2],decoded=False,native=None) for r in plan]
        c=generation_counts(rows,10)
        self.assertEqual(c['decoded'],0);self.assertEqual(c['verified'],0);self.assertEqual(c['scheduled'],64)
        self.assertEqual(c['emitted_tokens'],128)

    def test_partial_calibration_cannot_produce_favorable_aggregate(self):
        from oeis_learn.pipeline_validation.model_sizing import summarize_profiles
        from oeis_learn.pipeline_validation.model_sizes import SLOTS,PROFILES
        receipts={k:seal(dict(state='PASS',observations={'result':self.result(PROFILES[k.split('.')[1]])})) for k in SLOTS}
        receipts['sizing.m100.calibration-1']=seal(dict(receipts['sizing.m100.calibration-1'],state='PARTIAL'))
        report=summarize_profiles(receipts)
        self.assertIsNone(report['profiles']['m100']['training_tokens_per_second_median'])
        self.assertNotIn('m100',report['hardware_eligible_profiles'])
        self.assertFalse(report['optimization_adoption']);self.assertFalse(report['research_cost_admission'])
        self.assertEqual(report['model_selection'],'NOT_DECIDED')

    def result(self,p):
        from oeis_learn.pipeline_validation.model_sizes import parameters
        return seal(dict(status='COMPLETE',profile=p,counts=parameters(p),training_tokens=100,
            training_updates=32,validation_rows=32,training_tokens_per_second=10,training_updates_per_second=1,
            whole_seconds=100,resources={'qualified':True,'ceiling_bytes':32<<30,'breach':None},
            numerical_parity=True,changed_parameters=True,torch_threads={'intraop':2,'interop':1},device={'hip':'fixture'},
            source='source',contract='contract',workload='work',reference='ref',initial_parameters='initial',trained_parameters='trained',
            generation={'scheduled':64,'started':64,'decoded':0,'verified':0}))
