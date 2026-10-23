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
