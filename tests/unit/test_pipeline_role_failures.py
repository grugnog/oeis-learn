import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from oeis_learn.pipeline_validation.identity import digest,write
from oeis_learn.pipeline_validation.role import main


class RoleFailures(unittest.TestCase):
    def test_resource_breach_survives_role_summary(self):
        class Guard:
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):raise MemoryError('injected aggregate limit')
            def __exit__(self,*args):pass
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inputs=root/'input';inputs.mkdir()
            write(inputs/'expected.json',dict(version='pipeline-role/v1',source=digest({}),contract='fixture',mode='neural',options={'memory_ceiling':1<<30}))
            with patch('oeis_learn.experiments.source_snapshot.source_snapshot',lambda:{}),patch('oeis_learn.tracking.pipeline_profile.device_view',lambda:{}),patch('oeis_learn.tracking.pipeline_profile.audit_device_view',lambda *args:None),patch('oeis_learn.pipeline_validation.memory.MemoryWatchdog',Guard),patch.dict(os.environ,{'OEIS_EXPECTED_PIPELINE_DEVICES':'{}'}):
                code=main(['neural','--input',str(inputs),'--output',str(root/'out')])
            self.assertEqual(code,1)
            summary=json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['fatal'],'RESOURCE_BREACH')
            self.assertIn('MemoryError',summary['error'])
