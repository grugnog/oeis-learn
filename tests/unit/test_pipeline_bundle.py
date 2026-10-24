import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from oeis_learn.pipeline_validation.bundle import pack, verify
from oeis_learn.pipeline_validation.identity import write


class Bundle(unittest.TestCase):
    def fixture(self, root):
        write(root/'summary.json',dict(selected_scope='FAIL'))
        write(root/'contract.json',dict(identity='contract'))

    def test_member_integrity_and_private_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'out'; root.mkdir(); self.fixture(root)
            (root/'weights.pt').write_text('secret')
            archive=pack(root, mandatory=['summary.json','contract.json'])
            verify(archive)
            with zipfile.ZipFile(archive) as z:
                self.assertNotIn('weights.pt',z.namelist())
                self.assertIn('manifest.json',z.namelist())

    def test_symlink_and_missing_mandatory_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'out'; root.mkdir(); self.fixture(root)
            (root/'link.json').symlink_to(root/'summary.json')
            with self.assertRaises(ValueError):
                pack(root,mandatory=['link.json'])
            with self.assertRaises((ValueError,OSError)):
                pack(root,mandatory=['missing.json'])
