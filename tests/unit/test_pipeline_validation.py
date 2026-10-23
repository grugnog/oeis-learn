import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from oeis_learn.pipeline_validation.registry import Stage, resolve
from oeis_learn.pipeline_validation.runner import execute
from oeis_learn.pipeline_validation.identity import digest

ROOT = Path(__file__).resolve().parents[2]


class Validation(unittest.TestCase):
    def test_graph_rejections_and_selection(self):
        stages = [Stage('a', 1), Stage('b', 2, ('a',))]
        self.assertEqual([s.id for s in resolve(stages, '2')], ['a','b'])
        for bad in [[Stage('a', 1, ('b',)), Stage('b', 1, ('a',))], [Stage('a',1), Stage('a',2)], [Stage('a',1,('missing',))]]:
            with self.assertRaises(ValueError):
                resolve(bad, 'all-ready')

    def test_failure_blocks_descendants_continues_independent_and_archives(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            def run(stage, output):
                calls.append(stage.id)
                if stage.id == 'bad':
                    raise RuntimeError('intentional')
                return {}
            out = Path(folder) / 'out'
            code = execute([Stage('bad',1),Stage('child',1,('bad',)),Stage('safe',1)], out,
                adapter=run, binding={'source':digest('source')}, cap=60, reserve=1)
            self.assertEqual(code,1)
            self.assertEqual(calls, ['bad','safe'])
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['stages']['child']['state'],'BLOCKED')
            self.assertTrue(out.with_suffix('.zip').is_file())

    def test_scope_report_requires_only_stdlib(self):
        result = subprocess.run([sys.executable,'-S',str(ROOT/'scripts/foundation_run.py'),
            'validate','--suite','pipeline','--scope-report'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('selected',result.stdout)
