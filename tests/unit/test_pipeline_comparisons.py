import copy
import unittest
from oeis_learn.pipeline_validation.comparisons import calibrate, decide


class Comparisons(unittest.TestCase):
    def pairs(self):
        return [dict(order=order,baseline=100,candidate=85,complete=True,work='same',candidate_work='same',strata={'long':1.03},correct=True,resources=True,baseline_success=20,candidate_success=20,baseline_complete=32,candidate_complete=32,reset=True,scope='warm',kind='confirmation') for order in ['AB','BA','AB','BA','AB']]

    def test_exact_five_fresh_complete(self):
        pairs=self.pairs()
        self.assertEqual(decide(pairs)['adoption'],'ADOPT')
        for field,value in [('complete',False),('correct',False),('candidate_work','other'),('reset',False),('kind','exploratory')]:
            bad=copy.deepcopy(pairs);bad[0][field]=value
            self.assertNotEqual(decide(bad)['adoption'],'ADOPT')
        self.assertNotEqual(decide(pairs[:3])['adoption'],'ADOPT')
        bad=copy.deepcopy(pairs);bad[0]['strata']['long']=1.06
        self.assertNotEqual(decide(bad)['adoption'],'ADOPT')

    def test_no_adaptive_calibration(self):
        rows=[dict(complete=True,seconds=n,identity=str(n)) for n in (2,3,4)]
        result=calibrate(rows,minimum=5,cap=10,parent='parent',workload='workload')
        self.assertEqual(result['reservation_seconds'],8)
        with self.assertRaises(ValueError):
            calibrate(rows,minimum=5,cap=6,parent='p',workload='w')
        with self.assertRaises(ValueError):
            calibrate(rows[:2],minimum=1,cap=10,parent='p',workload='w')
