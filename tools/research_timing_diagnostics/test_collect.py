import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import collect


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.timing = self.root / 'prepare-research/evaluation-timing'
        self.write('request.json', dict(version='progressive-research-evaluation-request/v1',
            scope='training-only prospective throughput; no panel exposure', source='original-source',
            cases=[dict(identity=str(i)) for i in range(14)]))
        for i in range(14):
            self.write(f'cases/{i}/generation/candidates.json', dict(version='discovery-candidates/v1',
                identity='original', records={'prompt': dict(selections={'equal_budget': {'source': 'SECRET'}},
                    arm_evidence={'equal_budget': dict(complete=False, reason='token limit')},
                    costs={'correct': dict(complete=False, elapsed_seconds=3,
                        details=[dict(tokens=[999, 888], source='SECRET', native_seconds=1,
                            generation=dict(complete=False, reason='token limit'),
                            fitting=dict(complete=False, admission=['SECRET']))])})}))
            self.write(f'cases/{i}/scoring/scores.json', dict(version='discovery-scoring/v1',
                outcomes={'prompt': {'equal_budget': dict(complete=False, success=False,
                    scoring=dict(complete=False, outcome='no_selected_candidate'))}}))
        self.write('private/training-samples.json', ['SECRET'])

    def write(self, relative, value):
        path = self.timing / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def original_bytes(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for p in self.timing.rglob('*') if p.is_file()}

    def test_complete_private_exclusion_unchanged_originals_and_hashes(self):
        before = self.original_bytes()
        output, report = collect.collect(self.root)
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(len(report['files']), 28)
        self.assertEqual(before, self.original_bytes())
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(len(archive.namelist()), 29)
            for entry in report['files']:
                original = entry['original']
                self.assertEqual(original['sha256'], 'sha256:' + hashlib.sha256(before[original['path']]).hexdigest())
                raw = archive.read(entry['projection'])
                self.assertNotIn(b'SECRET', raw)
                self.assertNotIn(b'999', raw)
                self.assertEqual(entry['sha256'], 'sha256:' + hashlib.sha256(raw).hexdigest())
            projected = json.loads(archive.read('cases/0/generation/candidates.json'))
            detail = projected['records']['prompt']['costs']['correct']['details'][0]
            self.assertEqual(detail['token_count'], 2)
            self.assertEqual(detail['generation']['reason'], 'token limit')
        other, _ = collect.collect(self.root)
        self.assertNotEqual(output, other)
        self.assertTrue(output.is_file())

    def test_missing_malformed_symlink_and_oversize_continue(self):
        (self.timing / 'cases/0/generation/candidates.json').unlink()
        (self.timing / 'cases/1/scoring/scores.json').write_text('{bad')
        path = self.timing / 'cases/2/scoring/scores.json'
        path.unlink()
        path.symlink_to(self.timing / 'private/training-samples.json')
        (self.timing / 'cases/3/scoring/scores.json').write_text(' ' * 4097)
        self.write('cases/4/scoring/scores.json', [])
        with patch.object(collect, 'MAX_FILE', 4096):
            output, report = collect.collect(self.root)
        self.assertEqual(report['status'], 'PARTIAL')
        self.assertEqual(len(report['errors']), 5)
        self.assertEqual(len(report['files']), 23)
        with zipfile.ZipFile(output) as archive:
            self.assertIn('cases/13/scoring/scores.json', archive.namelist())
            self.assertEqual(json.loads(archive.read('manifest.json'))['status'], 'PARTIAL')

    def test_refuses_wrong_scope(self):
        self.write('request.json', dict(version='assessment', cases=[]))
        with self.assertRaises(ValueError):
            collect.collect(self.root)
        self.assertFalse((self.root / 'share').exists())

    def test_refuses_symlink_output(self):
        (self.root / 'share').symlink_to(self.timing, target_is_directory=True)
        with self.assertRaises(ValueError):
            collect.collect(self.root)
        self.assertFalse(list(self.timing.glob('*.zip')))

    def test_stdlib_only_cli(self):
        result = subprocess.run([sys.executable, '-S', str(Path(collect.__file__).resolve()),
            '--session', str(self.root)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('COMPLETE: 28/28', result.stdout)


if __name__ == '__main__':
    unittest.main()
