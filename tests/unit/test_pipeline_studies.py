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

    def test_durable_study_debits_survive_new_output_and_interruption(self):
        import tempfile
        from pathlib import Path
        from oeis_learn.pipeline_validation.studies import StudyLedger
        with tempfile.TemporaryDirectory() as folder:
            row={'decision_id':'decision', 'identity':'sha256:'+'a'*64,
                 'arms':[{'id':'A','wall_seconds':10,'work':2,'disk_bytes':100}]}
            first=StudyLedger(Path(folder), row)
            first.reserve(row['arms'][0])
            with self.assertRaises(ValueError):StudyLedger(Path(folder),row).reserve(row['arms'][0])
            with self.assertRaises(ValueError):StudyLedger(Path(folder),dict(row,identity='sha256:'+'b'*64))
            first.finish('A','INTERRUPTED')
            with self.assertRaises(ValueError):first.finish('A','COMPLETE')

    def test_decision_payload_has_no_reference_hash_cycle(self):
        from oeis_learn.pipeline_validation.studies import decision_payload
        row=dict(identity='self',state='FROZEN',authority=None,gates={},cost=None,hypothesis='same')
        admitted=dict(row,identity='other',state='ADMITTED',authority={'path':'review'},gates={'runtime':'receipt'},cost={'path':'cost'})
        self.assertEqual(decision_payload(row),decision_payload(admitted))
        self.assertNotEqual(decision_payload(row),decision_payload(dict(row,hypothesis='changed')))

    def test_same_final_plan_retry_spends_remaining_budget_without_refund(self):
        import tempfile
        from pathlib import Path
        from oeis_learn.pipeline_validation.studies import StudyLedger
        arm=dict(id='A',wall_seconds=100,work=10,disk_bytes=1000,
                 attempt={'wall_seconds':40,'work':4,'disk_bytes':100})
        contract=dict(decision_id='final-decision',identity='sha256:'+'a'*64,kind='final',arms=[arm],
            budgets={'wall_seconds':120,'work':12,'disk_bytes':2000,'cleanup_seconds':5,'archive_seconds':5})
        with tempfile.TemporaryDirectory() as folder:
            ledger=StudyLedger(Path(folder),contract)
            first=ledger.reserve(arm)
            self.assertEqual(first['wall_seconds'],40)
            ledger.finish('A','INTERRUPTED')
            second=ledger.reserve(arm)
            self.assertEqual(second['attempt'],1)
            ledger.finish('A','INTERRUPTED')
            with self.assertRaises(ValueError):ledger.reserve(arm)
            changed=dict(contract,identity='sha256:'+'b'*64)
            with self.assertRaises(ValueError):StudyLedger(Path(folder),changed)

    def test_existing_final_exposure_persists_debit_and_same_plan_remaining_budget(self):
        import tempfile
        from pathlib import Path
        from oeis_learn.fresh.bigint_state import expose_final,finish_final
        from oeis_learn.synthesis.contracts import identity
        from oeis_learn.synthesis.state import FinalPlan
        from oeis_learn.fresh.common import read
        h='sha256:'+'a'*64
        decisions={name:{'status':'not_planned','reason':'explicit fixture-only deferral'} for name in FinalPlan.DECISIONS}
        plan=dict(version='final-evaluation/v1',cohort=h,
            decisions=[[name,identity(row)] for name,row in decisions.items()],decision_manifest=decisions,
            arms=[['A',17,h,h,h,h,100]],denominator=1,
            controls=['correct','shuffled','absent','template'],budget=100)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            kwargs=dict(cohort=h,checkpoint=h,recipe=h,portfolio=h,runtime=h,seed=17,debit=40)
            arm,state=expose_final(root,plan,**kwargs)
            persisted=read(root/'state.json')
            self.assertEqual(persisted['state'],'exposed')
            self.assertEqual(persisted['arm_spent'],{'A':40})
            self.assertEqual(persisted,state)
            finish_final(root,plan,arm,'interrupted')
            arm,state=expose_final(root,plan,**kwargs)
            self.assertEqual(state['arm_spent'],{'A':80})
            finish_final(root,plan,arm,'interrupted')
            with self.assertRaises(ValueError):expose_final(root,plan,**kwargs)
            with self.assertRaises(ValueError):expose_final(root,dict(plan,budget=200),**kwargs)
            self.assertEqual(read(root/'state.json')['state'],'exposed')

    def test_authority_inputs_do_not_form_a_compiled_source_hash_cycle(self):
        import tempfile
        from pathlib import Path
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ['.specify/memory/constitution.md','specs/007-experiment-foundation/validation/adoption.md',
                         'specs/007-experiment-foundation/validation/review-inputs.json','specs/007-experiment-foundation/tasks.md',
                         'configs/foundation/pipeline_validation.json']:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
            before=collect_snapshot(root)
            path=root/'configs/foundation/pipeline_studies/study.json'
            path.parent.mkdir();path.write_text('{"gates":{}}')
            self.assertEqual(collect_snapshot(root),before)
            path.write_text('{"gates":{"runtime":"content-bound receipt"}}')
            self.assertEqual(collect_snapshot(root),before)
            (root/'configs/foundation/pipeline_validation.json').write_text('changed runtime policy')
            self.assertNotEqual(collect_snapshot(root),before)

    def test_incomplete_matched_batch_preserves_missing_arms_without_promotion(self):
        import tempfile
        from pathlib import Path
        from oeis_learn.pipeline_validation.study_reports import study_report
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'report'
            contract=dict(kind='representation',identity='study',arms=[dict(id='A',operation='train',run=None,seed=20261016)])
            report=study_report(contract,root,out,{'A':{'state':'FAIL','reason':'injected interruption'}},workspace=root)
            self.assertEqual(report['research'],'INCONCLUSIVE')
            self.assertFalse(report['automatic_run_authorized'])
            self.assertEqual(report['arms'][0]['state'],'FAIL')
            self.assertTrue((out/'report.json').is_file())

    def test_recipe_work_and_intervention_cannot_exceed_reviewed_arm(self):
        from oeis_learn.pipeline_validation.studies import validate_recipe
        recipe={'model':{'seed':17},'track':{'kind':'generic_only'},
                'learning':{'max_updates':1000,'validation_every':500,'target_view':'literal','curriculum':True,'replay':False},
                'search':{'attempts':8,'tokens':1024,'seconds':60,'beam':None}}
        contract={'kind':'curriculum','track':'generic_only'}
        arm={'seed':17,'operation':'train','work':100000}
        required=validate_recipe(contract,arm,recipe)
        self.assertGreater(required,1000)
        with self.assertRaises(ValueError):validate_recipe(contract,dict(arm,work=1),recipe)
        recipe['learning']['replay']=True
        with self.assertRaises(ValueError):validate_recipe(contract,arm,recipe)
