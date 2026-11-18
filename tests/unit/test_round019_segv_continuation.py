"""Exercise the host-only retry with real supervision, journal and budget files."""
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from oeis_learn.pipeline_validation.identity import digest, seal
from oeis_learn.pipeline_validation.progressive_workstation import Workflow
from oeis_learn.pipeline_validation.progressive_workstation_state import Session, read, write
from oeis_learn.generalization.progressive.contracts import reference

SCRIPT = (Path(__file__).resolve().parents[2] / 'specs/013-progressive-mathematical-discovery'
          / 'validation/continue-round019-segv.py')
spec = importlib.util.spec_from_file_location('segv_helper', SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

# Stand-in for expensive producer operations. Supervision runs the normal
# child argv with -S, and the canonical workflow consumes its exit code.
PRODUCER = """
import hashlib, json, sys
from pathlib import Path
op = sys.argv[sys.argv.index('discovery') + 1]
output = Path(sys.argv[sys.argv.index('--output') + 1])
output.mkdir(parents=True)
def publish(name, **value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    value['identity'] = 'sha256:' + hashlib.sha256(encoded).hexdigest()
    (output/name).write_text(json.dumps(value))
with Path('calls').open('a') as log:
    print(op, file=log)
if Path('block-' + op).exists():
    publish('result.json', operation=op, mechanical_status='BLOCKED', reason='test', missing_evidence=['test'])
    sys.exit(2)
publish('result.json', operation=op, mechanical_status='COMPLETE')
if op == 'freeze':
    publish('campaign.json', campaign='test')
"""


class CrashRecovery(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        original_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original_cwd)
        self.s = Session(self.root, helper.SESSION, {'source': 'test'}, helper.PHASE)
        self.base = self.s.folder / helper.PHASE
        self.old = self.base / 'storage-continuation'
        self.output = self.base / 'frozen'
        self.args = SimpleNamespace(engine='podman', phase=helper.PHASE, render_node=None, cpus=8, memory_gib=64)
        self.script = self.root / 'run-state/tools/continue.py'
        self.script.parent.mkdir()
        self.script.write_bytes(SCRIPT.read_bytes())
        arguments = {k: str((self.base / v).relative_to(self.root)) for k, v in
                     [('blueprint','calibration-reuse/blueprint.json'), ('qualification','qualification'),
                      ('prepared','prepared'), ('output','frozen')]}
        owner = seal(dict(version='discovery-owner/v1', invocation='3'*32, output=arguments['output']))
        write(self.output / 'owner.json', owner)
        steps = self.s.journal['steps']
        bound = seal(dict(continuation=helper.STORAGE, arguments=arguments))
        self.crash = dict(status='FAILED', binding=bound, reason='exit -11', history=[{}, {}])
        steps[helper.PREFIX + 'freeze'] = self.crash
        for name in ('registered-data', 'native-census', 'sensitivity', 'materials'):
            steps[helper.PREFIX + name] = dict(status='COMPLETE', value={})
        steps[helper.PREFIX + 'execution-preflight'] = dict(status='COMPLETE', value={'setup_seconds':{'acceptance':300}})
        for name in ('acceptance','readiness'):
            steps[helper.PREFIX + name] = dict(status='BLOCKED')
        debit = seal(dict(binding=bound, spent_seconds=150))
        write(self.old / 'continuation-allowance.json', debit)
        parent = self.base / 'prepare-allowance.json'
        write(parent, seal(dict(seconds=1800, spent_seconds=90)))
        self.previous = dict(original_allowance=reference(self.root, parent), previous_seconds=400,
                             retry_allowance=reference(self.root, self.old / 'continuation-allowance.json'))
        logs = {}
        for name in helper.LOGS:
            path = self.old / 'logs/0000' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('reviewed ' + name)
            logs[name] = reference(self.root, path)['sha256'][7:]
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/foundation_run.py').write_text(PRODUCER)
        storage = SimpleNamespace(collect_trace=lambda *a: None)
        self.report = seal(dict(scope='full', status='READY'))
        for p in [patch.object(helper, 'LOGS', logs), patch.object(helper, 'DEBIT', debit['identity']),
                  patch.object(helper, 'CRASH_ROW', digest(self.crash)),
                  patch.object(helper, 'storage_helper', return_value=(storage,self.previous)),
                  patch.object(helper, 'check_capacity', side_effect=lambda *a: self.report),
                  patch('oeis_learn.pipeline_validation.progressive_research_launch.readiness', return_value=({'status':'READY'}, []))]:
            p.start()
            self.addCleanup(p.stop)
        def record_research(workflow):
            workflow.s.journal['steps']['run-research/invocation'] = dict(status='COMPLETE')
        p = patch.object(Workflow, 'research', autospec=True, side_effect=record_research)
        self.research = p.start()
        self.addCleanup(p.stop)

    def execute(self, proceed=True):
        return helper.execute(self.s, self.args, self.script, proceed)

    def test_check_then_continue(self):
        self.assertEqual(self.execute(False), 0)
        self.assertTrue((self.output / 'owner.json').exists())
        self.assertFalse((self.root / 'calls').exists())
        self.assertEqual(self.execute(), 0)
        self.assertEqual((self.root / 'calls').read_text().splitlines(), ['freeze','acceptance'])
        self.research.assert_called_once()
        folder = self.base / 'segv-continuation'
        receipt = read(folder / 'continuation.json')
        self.assertEqual(receipt['remaining_seconds'], 1250)
        self.assertEqual(receipt['previous_seconds'], 550)
        self.assertTrue((folder / 'original-crashed-freeze/owner.json').is_file())
        argv = json.loads((folder / 'logs/0000/command.json').read_text())['argv']
        self.assertEqual(argv[1:3], ['-S', str(self.root / 'scripts/foundation_run.py')])
        self.assertEqual(read(folder / 'continuation-allowance.json')['attempts'][0]['outcome'], 'COMPLETE')
        history = self.s.journal['steps'][helper.PREFIX + 'freeze']['history']
        self.assertEqual([*self.crash['history'], {k:v for k,v in self.crash.items() if k!='history'}], history)
        self.s.phase = helper.PHASE
        self.assertEqual(self.execute(), 0)
        self.assertEqual((self.root / 'calls').read_text().splitlines(), ['freeze','acceptance'])

    def test_new_failure_retained(self):
        (self.root / 'block-freeze').touch()
        self.assertEqual(self.execute(), 2)
        with self.assertRaisesRegex(ValueError, 'reviewed and charged'):
            self.execute()
        self.assertEqual((self.root / 'calls').read_text().splitlines(), ['freeze'])
        self.research.assert_not_called()

    def test_acceptance_stops_research(self):
        (self.root / 'block-acceptance').touch()
        self.assertEqual(self.execute(), 2)
        self.research.assert_not_called()

    def test_storage_block_preserves_owner_and_journal(self):
        self.report = seal(dict(scope='lower_bound', status='BLOCKED'))
        self.assertEqual(self.execute(), 2)
        self.assertTrue((self.output / 'owner.json').is_file())
        self.assertFalse((self.base / 'segv-continuation').exists())
        self.assertIs(self.s.journal['steps'][helper.PREFIX + 'freeze'], self.crash)

    def test_partial_publication_rejected(self):
        (self.output / 'tasks').mkdir()
        with self.assertRaisesRegex(ValueError, 'owner-only'):
            self.execute()

    def test_changed_log_rejected(self):
        (self.old / 'logs/0000/stderr.log').write_text('other crash')
        with self.assertRaisesRegex(ValueError, 'crash evidence'):
            self.execute()

    def test_changed_accounting_rejected(self):
        write(self.old / 'continuation-allowance.json', seal(dict(binding=self.crash['binding'], spent_seconds=0)))
        with self.assertRaisesRegex(ValueError, 'reviewed and charged'):
            self.execute()

    def test_wrong_owner_rejected(self):
        write(self.output / 'owner.json', seal(dict(version='discovery-owner/v1', invocation='3'*32, output='elsewhere')))
        with self.assertRaisesRegex(ValueError, 'owner differs'):
            self.execute()

    def test_changed_crash_binding_rejected(self):
        self.crash['reason'] = 'different failure'
        with self.assertRaisesRegex(ValueError, 'reviewed and charged'):
            self.execute()

    def test_existing_recovery_retained(self):
        folder = self.base / 'segv-continuation'
        folder.mkdir()
        marker = folder / 'marker'
        marker.write_text('retained')
        with self.assertRaisesRegex(ValueError, 'existing diagnostic-crash recovery'):
            self.execute()
        self.assertEqual(marker.read_text(), 'retained')
        self.assertFalse((self.root / 'calls').exists())
