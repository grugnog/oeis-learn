import unittest
from oeis_learn.pipeline_validation.qualification import qualification
from oeis_learn.fresh.cost_policy import pipeline_forecast


class Qualification(unittest.TestCase):
    def test_history_and_gates_do_not_promote_learning(self):
        result=qualification({'cpu':{'state':'PASS'},'hip':{'state':'BLOCKED'}},source='source')
        self.assertEqual(result['engineering'],'BLOCKED')
        self.assertEqual(result['historical']['008:T096']['consumed_seconds'],7730)
        self.assertEqual(result['historical']['008:T093']['disposition'],'DEFERRED')
        self.assertFalse(result['long_run_authorized'])
        self.assertEqual(len(result['carryovers']),13)

    def test_unknown_forecast_components_block_scope(self):
        result=pipeline_forecast({},'diverse32',updates=100,cadence=100,budget=7200)
        self.assertFalse(result['within_budget'])
        self.assertIsNone(result['projected_seconds'])
        self.assertIn('startup',result['missing'])
