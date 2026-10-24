from pathlib import Path
import tempfile
import unittest
from oeis_learn.pipeline_validation.identity import seal, write, file_hash
from oeis_learn.pipeline_validation.reuse import compatible


class Reuse(unittest.TestCase):
    def test_only_complete_pass_bound_to_artifacts_and_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'a').write_text('a')
            binding={k:k+'-original' for k in ('source','image','lock','adapter','workload','hardware','resources','policy','closure')}
            row=seal(dict(state='PASS',binding=binding,prerequisites={'p':'hash'},files={'a':file_hash(root/'a')}))
            self.assertTrue(compatible(row,root,binding,{'p':'hash'}))
            changes=[seal(dict(row,binding=dict(binding,**{k:'changed'}))) for k in binding]
            for bad in [seal(dict(row,state='FAIL')),*changes,seal(dict(row,prerequisites={})),dict(row,identity='broken')]:
                self.assertFalse(compatible(bad,root,binding,{'p':'hash'}))
            (root/'a').write_text('b')
            self.assertFalse(compatible(row,root,binding,{'p':'hash'}))

    def test_dependency_closure_exclusions_and_unknown_changes(self):
        from oeis_learn.pipeline_validation.closures import dependency_closure, compatible_closure
        old={'src/a.py':'old','uv.lock':'lock','README.md':'old-doc'}
        a=dependency_closure(old,prerequisites={'p':'producer'})
        b=dependency_closure(dict(old,**{'README.md':'new-doc'}),prerequisites={'p':'producer'})
        self.assertTrue(compatible_closure(a,b))
        for current in [dict(old,**{'src/a.py':'new'}),dict(old,**{'uv.lock':'new'}),dict(old,**{'unmapped/new.data':'new'})]:
            self.assertFalse(compatible_closure(a,dependency_closure(current,prerequisites={'p':'producer'})))
        self.assertFalse(compatible_closure(a,dependency_closure(old,prerequisites={'p':'different'})))

    def test_current_image_inspection_can_support_prior_pure_test_evidence(self):
        import json
        from oeis_learn.pipeline_validation.runner import execute
        from oeis_learn.pipeline_validation.registry import Stage
        from oeis_learn.pipeline_validation.identity import digest
        calls=[]
        class Adapter:
            def stage_binding(self,stage,base):
                return dict(base,image='immutable-image',hardware='same-host',resources='same-limits') if stage.adapter=='cpu' else base
            def __call__(self,stage,output):
                calls.append(stage.id)
                if stage.adapter=='image':
                    write(output/'image.json',{'image':'immutable-image'})
                    return {'image':'immutable-image'}
                (output/'results.xml').write_text('<testsuite><testcase name="semantic"/></testsuite>')
                return {'tests':1}
        stages=[Stage('image.docker.qualification',1,adapter='image'),Stage('cpu.docker.semantic',1,('image.docker.qualification',),adapter='cpu')]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);binding={'source':digest('source')}
            self.assertEqual(execute(stages,root/'old',adapter=Adapter(),binding=binding,cap=60,reserve=1),0)
            calls.clear()
            self.assertEqual(execute(stages,root/'new',adapter=Adapter(),binding=binding,cap=60,reserve=1,reuse_roots=[root/'old']),0)
            summary=json.loads((root/'new/summary.json').read_text())
            self.assertEqual(calls,['image.docker.qualification'])
            cpu=summary['stages']['cpu.docker.semantic']
            self.assertEqual(cpu['state'],'REUSED')
            self.assertEqual(cpu['observations']['result']['tests'],1)
            self.assertIn('imported/producer-receipt.json',cpu['files'])
