import unittest
from oeis_learn.pipeline_validation.qualification import qualification
from oeis_learn.fresh.cost_policy import pipeline_forecast


class Qualification(unittest.TestCase):
    def test_cost_profile_binds_opt_in_policy_and_preserves_legacy_identity(self):
        from copy import deepcopy
        from oeis_learn.fresh.bigint_config import defaults
        from oeis_learn.fresh.common import digest
        from oeis_learn.fresh.cost_policy import profile_identity, schedule_identity, validate_preflight
        recipe = defaults()
        original_profile = digest({k:recipe[k] for k in ('execution','inputs','model','fitting','search')})
        original_schedule = digest(dict(scope='diverse32',data=recipe['data'],evaluation=recipe['evaluation'],
            budgets=recipe['budgets'],learning={k:v for k,v in recipe['learning'].items() if k!='cost_preflight'}))
        self.assertEqual(profile_identity(recipe), original_profile)
        self.assertEqual(schedule_identity(recipe, 'diverse32'), original_schedule)
        recipe['pipeline'] = {'workers':1, 'cache':'baseline'}
        bound = profile_identity(recipe)
        schedule = schedule_identity(recipe, 'diverse32')
        self.assertNotEqual(bound, original_profile)
        self.assertNotEqual(schedule, original_schedule)
        for changed in ({'workers':2,'cache':'baseline'}, {'workers':1,'cache':'larger'}):
            other = deepcopy(recipe);other['pipeline'] = changed
            self.assertNotEqual(profile_identity(other), bound)
            self.assertNotEqual(schedule_identity(other, 'diverse32'), schedule)
        # A previously admitted legacy receipt cannot authorize opt-in execution.
        recipe['learning']['cost_preflight'] = dict(scope='diverse32',image='i',source_sha256='s',
            shape_sha256='h',schedule_sha256=original_schedule,profile_sha256=original_profile,scoring_data=False)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            validate_preflight(recipe, scope='diverse32',image='i',source_sha256='s',shape_sha256='h')

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
        from oeis_learn.pipeline_validation.workloads import MANDATORY_STRATA
        def row(i):
            return dict(identity=str(i),complete=True,work='same',strata={k:1 for k in MANDATORY_STRATA},options={'assessment':'cost-full','scope':'diverse32','policy':{'workers':1}},
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
        changed=[dict(rows[0],strata={}),*rows[1:]]
        self.assertFalse(forecast_trials(changed,'diverse32',updates=3000,cadence=500,budget=7200)['forecast']['within_budget'])
        changed=[dict(rows[0],options=dict(rows[0]['options'],policy={'workers':4})),*rows[1:]]
        self.assertFalse(forecast_trials(changed,'diverse32',updates=3000,cadence=500,budget=7200)['forecast']['within_budget'])

    def test_missing_or_changed_source_gate_cannot_qualify(self):
        from oeis_learn.pipeline_validation.identity import seal
        cpu=seal(dict(state='PASS',binding={'source':'current'},files={'test.xml':'hash'}))
        good=qualification({'cpu':cpu},source='current',required={'cpu'})
        self.assertEqual(good['engineering'],'PASS')
        self.assertEqual(qualification({'cpu':cpu},source='new',required={'cpu'})['engineering'],'BLOCKED')
        self.assertEqual(qualification({'cpu':cpu},source='current',required={'cpu','hip'})['engineering'],'BLOCKED')
        broken=dict(cpu,identity='changed')
        self.assertEqual(qualification({'cpu':broken},source='current',required={'cpu'})['engineering'],'BLOCKED')
