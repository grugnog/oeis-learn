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

    def test_full_scope_cost_observations_cannot_relabel_reduced_work(self):
        from oeis_learn.pipeline_validation.costs import forecast_trials
        components={name:.0001 for name in ['training_p95','teacher_forcing_p95','generation_p95','fit_p95','reference_p95','hidden_scoring_p95','checkpoint_p95','startup','export','cleanup']}
        def row(i):
            return dict(identity=str(i),complete=True,work='same',options={'assessment':'cost-full','scope':'diverse32'},
                gpu={'source':'source','weights':'weights','prompt_ids':list(range(32)), 'scheduled_attempts':1024,
                     'resources':{'qualified':True},'cost_components':components},
                score={'resources':{'qualified':True},'result':{'splits':{name:{'complete':32,'denominator':32} for name in ['raw','rescue','equal_budget','shuffled','absent','template']}}},
                cost_components=components)
        rows=[row(i) for i in range(3)]
        receipt=forecast_trials(rows,'diverse32',updates=3000,cadence=500,budget=7200)
        self.assertTrue(receipt['forecast']['within_budget'])
        self.assertEqual(receipt['forecast']['scope'],'diverse32')
        for changed in ([dict(rows[0],complete=False),*rows[1:]],rows[:2]):
            self.assertFalse(forecast_trials(changed,'diverse32',updates=3000,cadence=500,budget=7200)['forecast']['within_budget'])
        self.assertFalse(forecast_trials(rows,'transfer-screen',updates=3000,cadence=500,budget=7200)['forecast']['within_budget'])
