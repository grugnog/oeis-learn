import tempfile
from pathlib import Path
import unittest
from oeis_learn.pipeline_validation.identity import seal,write,file_hash
from oeis_learn.pipeline_validation.study_batch import parse_batch


def study(name):
    return seal(dict(version='pipeline-study/v1',kind='learnability',decision_id=name,
        hypothesis='prospective fixture',state='FROZEN',arms=[],seeds=[20261016],track='generic_only',
        prior_decisions=[],gates={},cost=None,budgets=dict(wall_seconds=100,work=100,disk_bytes=1000,cleanup_seconds=5,archive_seconds=5),
        denominators={'visible':20,'horizon':100},rules={'stop':'negative-or-inconclusive','promote':'review'},authority=None,final_plan=None))


class StudyBatches(unittest.TestCase):
    def test_batch_authority_cannot_supply_missing_individual_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);refs=[]
            for name in ['one','two']:
                write(root/(name+'.json'),study(name))
                refs.append(dict(path=name+'.json',sha256=file_hash(root/(name+'.json'))))
            batch=seal(dict(version='pipeline-study-batch/v1',state='FROZEN',studies=refs,authority=None,
                budgets=dict(wall_seconds=250,work=250,disk_bytes=2500,cleanup_seconds=5,archive_seconds=5)))
            result=parse_batch(batch,root=root,source='source')
            self.assertEqual(result['status'],'BLOCKED')
            self.assertEqual([r['status'] for r in result['studies']],['BLOCKED','BLOCKED'])
            self.assertTrue(all('authority' in r['missing'] for r in result['studies']))

    def test_duplicate_decision_and_shared_oversubscription_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write(root/'one.json',study('same'))
            ref=dict(path='one.json',sha256=file_hash(root/'one.json'))
            batch=dict(version='pipeline-study-batch/v1',state='FROZEN',studies=[ref,ref],authority=None,
                budgets=dict(wall_seconds=250,work=250,disk_bytes=2500,cleanup_seconds=5,archive_seconds=5))
            with self.assertRaises(ValueError):parse_batch(seal(batch),root=root,source='source')
            batch['studies']=[ref];batch['budgets']['wall_seconds']=105
            with self.assertRaises(ValueError):parse_batch(seal(batch),root=root,source='source')

    def test_approved_batch_with_blocked_studies_exports_once_without_engine_access(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.study_batch import main
        from oeis_learn.pipeline_validation.identity import digest
        from oeis_learn.pipeline_validation.bundle import verify
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);refs=[]
            for name in ['one','two']:
                write(root/(name+'.json'),study(name))
                refs.append(dict(path=name+'.json',sha256=file_hash(root/(name+'.json'))))
            payload=dict(version='pipeline-study-batch/v1',studies=refs,
                budgets=dict(wall_seconds=250,work=250,disk_bytes=2500,cleanup_seconds=5,archive_seconds=5))
            write(root/'authority.json',seal(dict(version='pipeline-study-authority/v1',decision_payload=digest(payload),approved=True,reviewed_by='fixture')))
            batch=seal(dict(payload,state='ADMITTED',authority={'path':'authority.json','sha256':file_hash(root/'authority.json')}))
            args=SimpleNamespace(scope_report=False,output='out',max_seconds=None,manifest=None,final_plan=None)
            def forbidden(*args):raise AssertionError('blocked study touched an engine')
            with patch('oeis_learn.experiments.source_snapshot.collect_snapshot',lambda root:{}):
                code=main(args,root,forbidden,batch,digest({}),None)
            self.assertEqual(code,3)
            self.assertEqual(len(list(root.rglob('*.zip'))),1)
            verify(root/'out.zip')
            result=json.loads((root/'out/summary.json').read_text())
            self.assertEqual(set(result['stages']),{'s00.study.admission','s01.study.admission','study.report'})
