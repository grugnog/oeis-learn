"""Finite trial roster tests use synthetic observations, never device evidence."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from oeis_learn.pipeline_validation.benchmarks import Benchmarks
from oeis_learn.pipeline_validation.identity import digest
from oeis_learn.pipeline_validation.runner import CapabilityError


class FiniteBenchmarks(unittest.TestCase):
    def fixture(self):
        bench=Benchmarks(SimpleNamespace(contract={'identity':digest('contract')}))
        bench.calibration={'reservation_seconds':1,'identity':digest('calibration')}
        bench.subset=['p'];calls=[]
        def trial(output,**options):
            calls.append((str(output),options))
            # Independent fake measurements. These stand for no real qualification.
            candidate=options['policy'].get('overlap') is True
            return dict(complete=True,seconds=1 if candidate else 2,work='same-work',
                gpu=dict(semantic='same-selections',tokens='same-tokens',
                    resources={'qualified':True},timings=dict(updates=1,validation=1,generation_native=1,checkpoint=1)),
                score=dict(resources={'qualified':True},result={'splits':{'equal_budget':{'success':3,'complete':32}}}))
        bench.trial=trial
        return bench,calls

    def test_exact_fresh_confirmation_roster_and_warmups(self):
        bench,calls=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            result=bench.joint(Path(folder),cap=100)
        self.assertEqual(len(calls),71)
        self.assertEqual(result['adoption'],'ADOPT')
        for scope in ('cold','warm'):
            fresh=[p for p,_ in calls if '/confirm-'+scope+'-' in p and 'warmup' not in p]
            self.assertEqual(len(fresh),10)
            budget=[p for p,_ in calls if '/budget-'+scope+'-' in p]
            self.assertEqual(len(budget),10)
        self.assertEqual(sum('candidate' in p and 'warmup' in p for p,_ in calls),5)

    def test_no_candidate_when_all_reservations_cannot_fit(self):
        bench,calls=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(CapabilityError):bench.joint(Path(folder),cap=75)
        self.assertEqual(calls,[])

    def test_batch32_only_after_positive_batch16_and_capacity_proof(self):
        bench,calls=self.fixture()
        original=bench.trial
        def measured(output,**options):
            row=original(output,**options)
            row['identity']=digest(str(output))
            row['gpu']['resources'].update(role_peak_upper_bytes=10<<30,ceiling_bytes=64<<30)
            return row
        bench.trial=measured
        with tempfile.TemporaryDirectory() as folder:
            bench.joint(Path(folder),cap=100)
        expanded=[options['policy'] for _,options in calls if options['policy']['batch_size']==32]
        self.assertTrue(expanded)
        proof=expanded[0]['capacity_qualification']
        self.assertTrue(proof['batch16_positive'])
        self.assertEqual(proof['predicted_peak_bytes'],24<<30)
        self.assertEqual(len(calls),76)
