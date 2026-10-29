import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from oeis_learn.pipeline_validation.role import native_compare
from oeis_learn.pipeline_validation.identity import seal


class NativeMeasurementRoster(unittest.TestCase):
    def test_subset_frozen_before_candidates_and_fewer_worker_tie(self):
        calls=[]
        def trial(output,*,workers,corpus,contract,cap):
            calls.append((Path(output).name,workers,corpus['identity']))
            cold=Path(output).name=='cold'
            rows=[dict(id=row['id'],complete=not cold or i==0,admitted=[]) for i,row in enumerate(corpus['rows'])]
            return seal(dict(results=rows,complete=all(r['complete'] for r in rows),seconds={1:2,2:1,4:1.02}[workers]))
        with tempfile.TemporaryDirectory() as folder,patch('oeis_learn.pipeline_validation.role.native_trial',trial):
            result=native_compare(None,Path(folder),{'contract':'parent','options':{'seconds':3000}})
        self.assertEqual(len(calls),15)
        self.assertEqual(result['selected_workers'],2)
        self.assertEqual(result['adoption'],'EXPERIMENTAL_ONLY')
        self.assertNotEqual(calls[0][2],calls[1][2])
        self.assertEqual(len({c[2] for c in calls[1:]}),1)
        self.assertEqual([c[0] for c in calls[2:5]],['calibration-0','calibration-1','calibration-2'])
