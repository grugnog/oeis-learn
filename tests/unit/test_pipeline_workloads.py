import unittest
from oeis_learn.pipeline_validation.workloads import material_strata


class WorkloadStrata(unittest.TestCase):
    def test_partial_cases_never_invent_long_or_large_latency(self):
        rows={'p':{'values':['1'],'indices':['0']}}
        attempts=[dict(prompt='p',mode='correct',tokens=30,batch_id='one',batch_size=1,
            generation_seconds=2,generation={'complete':True},native_seconds=1,fitting={'complete':True},native_kind='other')]
        result=material_strata(attempts,rows)
        self.assertEqual(result['decode.short'],2)
        self.assertNotIn('decode.long',result)
        self.assertNotIn('native.large',result)
        self.assertNotIn('decode.mixed',result)

    def test_actual_mixed_long_large_nonlinear_cases_have_observed_bounds(self):
        rows={'p':{'values':[str(1<<600)],'indices':['0']},'q':{'values':['1'],'indices':['0']}}
        def item(prompt,tokens):
            return dict(prompt=prompt,mode='correct',tokens=tokens,batch_id='same',batch_size=2,
                generation_seconds=4,generation={'complete':True},native_seconds=2,
                fitting={'complete':True},native_kind='nonlinear')
        result=material_strata([item('p',600),item('q',40)],rows)
        self.assertEqual(result['decode.long'],4)
        self.assertEqual(result['decode.mixed'],4)
        self.assertEqual(result['native.large'],2)
        self.assertEqual(result['native.nonlinear'],2)
