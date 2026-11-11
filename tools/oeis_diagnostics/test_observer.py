"""Independent observer tests. Reference-backed transport doubles are not native evidence."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
if not (ROOT / 'src/oeis_learn').is_dir():
    import oeis_learn
    ROOT = Path(oeis_learn.__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis
import run
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.generalization.progressive.admission import admit


class ReferenceTransportDouble:
    def execute(self, source, indices, deadline):
        r = evaluate(source, indices, deadline=deadline)
        return dict(outcome=r.outcome, values=list(r.values))
    def runtime_identity(self, *args):
        return 'sha256:' + 'a' * 64


def member(n=1):
    return dict(id=f'OEIS:A{n:06d}', group=f'g{n}', role='training',
                source_snapshot=analysis.digest('fixture'),
                indices=list(map(str, range(100))), values=list(map(str, range(100))))


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.client = ReferenceTransportDouble()
    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, *, short=False):
        pool = [member(1), member(2)]
        run.write(self.root / 'pool.json', pool)
        cycle = analysis.seal(dict(version='progressive-oeis-cycle/v1',
                                   pool=run.ref(self.root, self.root / 'pool.json')))
        run.write(self.root / 'cycle.json', cycle)
        corpus = self.root / 'corpus'
        corpus.mkdir()
        with sqlite3.connect(corpus / 'inventory.sqlite') as db:
            db.execute('CREATE TABLE processed(aid TEXT, result TEXT)')
            for i, m in enumerate(pool):
                raw = corpus / f'b{i}.txt'
                count = 100 if short and i == 0 else 200
                raw.write_text(''.join(f'{j} {j}\n' for j in range(count)))
                db.execute('INSERT INTO processed VALUES (?,?)', (m['id'][5:], json.dumps(dict(
                    eligible100=True, metadata=dict(offset=0), raw_path=raw.name,
                    bfile=dict(oid=run.ref(self.root, raw)['sha256'][7:])))))
        run.write(corpus / 'snapshot.json', dict(commit='pinned-fixture'))
        run.write(corpus / 'diagnostics/result.json', dict(status='COMPLETE', inventory=run.ref(corpus, corpus / 'inventory.sqlite')))
        args = SimpleNamespace(cycle_plan='cycle.json', cycle_output='run-state/cycle',
                               corpus='corpus', sample=2, seed=20261111, output='run-state/observer-plan')
        plan = run.freeze(self.root, args, 'sha256:' + 'b' * 64)
        return pool, args, plan

    def produce(self, pool, source='(program (return $index))'):
        receipts = [admit(m, source, client=self.client) for m in pool]
        root = self.root / 'run-state/cycle'
        for i, receipt in enumerate(receipts):
            run.write(root / f'phase-{i:05d}-admit-training/role/admission.json', receipt)
        run.write(root / 'host/yield-audit.json', analysis.seal(dict(version='discovery-oeis-yield/v1',
            payload=dict(pool=analysis.digest(pool), admissions=sorted(r['identity'] for r in receipts)))))
        return receipts

    def test_constant_digits_concrete_and_parameterised_sizes(self):
        small = analysis.metrics('(program (return (int 1)))', '(program (return (param.get p0)))')
        big = analysis.metrics('(program (return (int 12345678901234567890)))', '(program (return (param.get p0)))')
        self.assertEqual(small['target_tokens'], big['target_tokens'])
        self.assertGreater(big['concrete_tokens'], small['concrete_tokens'])
        self.assertEqual(big['literal_decimal_digits'], 20)
        self.assertGreater(big['largest_literal_bits'], small['largest_literal_bits'])

    def test_exception_flags_preserve_legitimate_base_case(self):
        source = '(program (return (select (eq (int 0) $index) (int 0) $index)))'
        stats = analysis.metrics(source, source)
        self.assertEqual(stats['direct_index_equalities'], 1)
        self.assertEqual(stats['branches'], 1)
        r = analysis.compare(source, list(map(str, range(100,200))), list(map(str, range(100,200))),
                             self.client, deadline=time.monotonic()+4)
        self.assertEqual(r['status'], 'MATCH')
        self.assertEqual(sum(v['false'] for v in r['branches'].values()), 100)
        self.assertFalse(stats['selection_changed'])

    def test_existing_admission_archive_regressions(self):
        report = analysis.regressions(self.client, self.root)
        self.assertEqual(len(report['cases']), 3)
        self.assertTrue(all(c['status'] == 'PASS' for c in report['cases']))

    def test_plan_is_prospective_and_unavailable_members_are_not_replaced(self):
        pool, args, plan = self.fixture(short=True)
        self.assertEqual(len(plan['sample']), 2)
        self.assertEqual(sum(r['status'] == 'UNAVAILABLE' for r in plan['sample']), 1)
        (self.root / 'run-state/cycle').mkdir()
        with self.assertRaisesRegex(ValueError, 'before'):
            run.freeze(self.root, args, plan['image'])

    def test_extra_terms_preserve_nonzero_source_offset(self):
        m = member()
        m['indices'] = list(map(str, range(-2,98)))
        path = self.root / 'shifted.txt'
        path.write_text(''.join(f'{i} {i+2}\n' for i in range(-2,198)))
        truth = run.extra_terms(path, m)
        self.assertEqual(truth['indices'][0], '98')
        self.assertEqual(truth['indices'][-1], '197')
        result = analysis.compare('(program (return (add $index (int 2))))',
            truth['indices'], truth['values'], self.client, deadline=time.monotonic()+4)
        self.assertEqual(result['status'], 'MATCH')

    def test_training_membership_and_offsets_required(self):
        m = member()
        with self.assertRaises(ValueError):
            run.validate_pool([dict(m, role='assessment')])
        with self.assertRaises(ValueError):
            run.validate_pool([dict(m, indices=['1']*100)])
        with self.assertRaises(ValueError):
            run.validate_pool([m,m])

    def test_extended_mismatch_reported_without_changing_admission(self):
        pool, _, plan = self.fixture()
        source = '(program (return (select (lt $index (int 100)) $index (int 0))))'
        self.produce(pool, source)
        req = run.request(self.root, self.root / 'run-state/observer-plan/plan.json')
        originals = [run.ref(self.root, p) for p in (self.root / 'run-state/cycle').rglob('*.json')]
        report = analysis.analyse(req, self.client, self.root / 'analysis')
        self.assertEqual(report['continuation_counts'], {'MISMATCH': 2})
        self.assertTrue(all(r['baseline']['status'] == 'MATCH' for r in report['rows']))
        self.assertTrue(all(r['continuation']['first_mismatch'] == '100' for r in report['rows']))
        self.assertFalse(report['admission_changed'])
        for ref in originals:
            run.checked(self.root, ref)

    def test_missing_and_changed_admission_fail_closed(self):
        pool, _, _ = self.fixture()
        self.produce(pool)
        path = self.root / 'run-state/cycle/phase-00000-admit-training/role/admission.json'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing admission'):
            run.request(self.root, self.root / 'run-state/observer-plan/plan.json')

    def test_changed_extended_truth_rejected(self):
        pool, _, _ = self.fixture()
        self.produce(pool)
        path = self.root / 'run-state/observer-plan/private/extra.json'
        data = run.load(path)
        data[pool[0]['id']]['values'][0] = '0'
        run.write(path, data)
        with self.assertRaisesRegex(ValueError, 'truth changed'):
            run.request(self.root, self.root / 'run-state/observer-plan/plan.json')

    def test_failures_do_not_discard_other_cases_or_unsolved_denominator(self):
        pool, _, _ = self.fixture()
        self.produce(pool[:1])
        path = self.root / 'run-state/cycle/host/yield-audit.json'
        y = run.load(path); y['payload']['pool'] = analysis.digest(pool); run.write(path, analysis.seal(y))
        req = run.request(self.root, self.root / 'run-state/observer-plan/plan.json')
        class Broken:
            def execute(self, *args, **kwargs):
                raise RuntimeError('independent case failure')
        report = analysis.analyse(req, Broken(), self.root / 'analysis')
        self.assertEqual({r['status'] for r in report['rows']}, {'INCOMPLETE','NO_ADMITTED_PROGRAM'})
        self.assertEqual(report['sampled_groups'], 2)
        self.assertTrue(list((self.root / 'analysis/cases').glob('*.json')))

    def test_executor_disagreement_never_becomes_match(self):
        class Disagrees:
            def execute(self, *args, **kw):
                return dict(outcome='ok', values=['0']*100)
        result = analysis.compare('(program (return $index))', list(map(str, range(100))),
            list(map(str, range(100))), Disagrees(), deadline=time.monotonic()+4)
        self.assertEqual(result['status'], 'EXECUTOR_DISAGREEMENT')

    def test_symlink_rejected_and_zip_excludes_private_truth(self):
        (self.root / 'alias').symlink_to('/tmp')
        with self.assertRaises(ValueError):
            run.confined(self.root, 'alias/file')
        run.write(self.root / 'private/extra.json', dict(secret='PRIVATE'))
        run.write(self.root / 'report.json', dict(status='MEASURED'))
        run.package(self.root)
        with zipfile.ZipFile(self.root / 'diagnostics.zip') as z:
            self.assertNotIn('private/extra.json', z.namelist())
            self.assertEqual(len(json.loads(z.read('manifest.json'))['files']), 1)

    def test_fixed_scope_cannot_be_enlarged_by_resealed_plan(self):
        pool, _, _ = self.fixture(); self.produce(pool)
        req = run.request(self.root, self.root / 'run-state/observer-plan/plan.json')
        req['plan']['seconds'] = 1000000
        req['plan'] = analysis.seal(req['plan']); req = analysis.seal(req)
        with self.assertRaisesRegex(ValueError, 'bounded'):
            analysis.analyse(req, self.client, self.root / 'analysis')

    def test_one_failed_case_still_runs_other_selected_case(self):
        pool, _, _ = self.fixture(); self.produce(pool)
        req = run.request(self.root, self.root / 'run-state/observer-plan/plan.json')
        class Once(ReferenceTransportDouble):
            first = True
            def execute(self, *args, **kwargs):
                if self.first:
                    self.first = False
                    raise RuntimeError('first case failed')
                return super().execute(*args, **kwargs)
        report = analysis.analyse(req, Once(), self.root / 'analysis')
        self.assertEqual([r['status'] for r in report['rows']].count('INCOMPLETE'), 1)
        self.assertEqual(report['continuation_counts']['MATCH'], 1)

    def test_host_uses_no_build_device_network_or_experiment_mount_and_cleans_on_failure(self):
        import oeis_learn.container_runtime as runtime
        import oeis_learn.pipeline_validation.supervision as supervision
        seen = []
        class Engine:
            def __init__(self, name):
                seen.append(name)
            def command(self, **kw):
                self_kw = kw
                if self_kw['gpu'] or self_kw['network']:
                    raise AssertionError('observer got device/network')
                if len(self_kw['mounts']) != 2 or self_kw['mounts'][0][2] is not False:
                    raise AssertionError('unexpected input mounts')
                seen.append('command')
                return ['observer-container']
            def call(self, *a, **k):
                seen.append('inspect')
                return SimpleNamespace(stdout=b'[{"State":{"OOMKilled":true}}]')
            def remove(self, name):
                seen.append('remove')
        def fail(*a, **k):
            raise RuntimeError('first launch failure')
        args = SimpleNamespace(engine='podman', operation='self-test')
        with patch.object(runtime, 'Engine', Engine), patch.object(supervision, 'launch', fail):
            with self.assertRaisesRegex(RuntimeError, 'first launch failure'):
                run.container(ROOT, args, 'sha256:'+'b'*64, self.root / 'observer')
        self.assertEqual(seen, ['podman','command','inspect','remove'])
        self.assertEqual(run.load(self.root / 'observer/cleanup.json')['status'], 'COMPLETE')

    def test_tool_files_are_outside_production_qualification_snapshot(self):
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        # No exclusion rule was changed to achieve this: the independent tool
        # has no production imports/callers and cannot confer qualification.
        snapshot = collect_snapshot(ROOT)
        self.assertFalse(any(name.startswith('tools/oeis_diagnostics/') for name in snapshot))

    def test_collection_needs_no_engine_and_preserves_original_failure(self):
        destination = self.root / 'run-state/observer'
        run.write(destination / 'failure.json', {'error':'first failure'})
        with patch.object(run, 'ROOT', self.root):
            self.assertEqual(run.main(['--engine','podman','collect','--output','run-state/observer']), 0)
        with zipfile.ZipFile(destination / 'diagnostics.zip') as z:
            self.assertEqual(json.loads(z.read('failure.json'))['error'], 'first failure')


if __name__ == '__main__':
    unittest.main()
