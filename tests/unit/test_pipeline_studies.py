import copy
import unittest
from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.pipeline_validation.studies import parse_study, gate_requirements, lifecycle


class Studies(unittest.TestCase):
    def test_non_circular_learnability_and_final_authority(self):
        self.assertNotIn('learning',gate_requirements('learnability'))
        self.assertIn('learning',gate_requirements('representation'))
        self.assertIn('final_plan',gate_requirements('final'))
        with self.assertRaises(ValueError):gate_requirements('arbitrary')

    def test_immutable_study_lifecycle(self):
        self.assertEqual(lifecycle('RUNNING','NEGATIVE'),'NEGATIVE')
        with self.assertRaises(ValueError):lifecycle('NEGATIVE','RUNNING')
        with self.assertRaises(ValueError):lifecycle('DRAFT','RUNNING')

    def test_missing_authority_never_mounts_inputs(self):
        row=seal(dict(version='pipeline-study/v1',kind='learnability',decision_id='new-attempt',
            hypothesis='prospective',state='FROZEN',arms=[],seeds=[20261016],track='generic_only',
            prior_decisions=[],gates={},cost=None,budgets={'wall_seconds':100,'work':100,'disk_bytes':100,'cleanup_seconds':5,'archive_seconds':5},
            denominators={'visible':20,'horizon':100},rules={'stop':'negative-or-inconclusive','promote':'review'},
            authority=None,final_plan=None))
        result=parse_study(row)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertIn('authority',result['missing'])
        with self.assertRaises(ValueError):parse_study(seal(dict(row,unknown=True)))
