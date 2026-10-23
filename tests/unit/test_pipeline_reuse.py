from pathlib import Path
import tempfile
import unittest
from oeis_learn.pipeline_validation.identity import seal, write, file_hash
from oeis_learn.pipeline_validation.reuse import compatible


class Reuse(unittest.TestCase):
    def test_only_complete_pass_bound_to_artifacts_and_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'a').write_text('a')
            binding=dict(policy='a',hardware='hw',closure='c')
            row=seal(dict(state='PASS',binding=binding,prerequisites={'p':'hash'},files={'a':file_hash(root/'a')}))
            self.assertTrue(compatible(row,root,binding,{'p':'hash'}))
            for bad in [seal(dict(row,state='FAIL')),seal(dict(row,binding=dict(binding,policy='b'))),seal(dict(row,prerequisites={})),dict(row,identity='broken')]:
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
