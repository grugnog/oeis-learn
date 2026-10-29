"""Independent host contract counterexamples; runnable without project dependencies."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from oeis_learn.pipeline_validation.identity import digest, seal, validate, reference, number
from oeis_learn.pipeline_validation.contracts import parse
from oeis_learn.pipeline_validation.receipts import Ledger

ROOT = Path(__file__).resolve().parents[2]


class Contracts(unittest.TestCase):
    def test_canonical_identity(self):
        self.assertEqual(digest({'a': 1}), 'sha256:015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862')
        row = seal({'version': 'test/v1', 'policy': 'a'})
        validate(row, {'version', 'policy'}, 'test/v1')
        for change in ({'policy': 'b'}, {'extra': 1}):
            with self.assertRaises(ValueError):
                validate(dict(row, **change), {'version', 'policy'}, 'test/v1')

    def test_seal_detaches_nested_payload_and_matches_existing_hashes(self):
        from oeis_learn.fresh.common import digest as existing_digest
        payload = {'version': 'test/v1', 'roster': [{'source': 'λ'}]}
        sealed = seal(payload)
        self.assertEqual(sealed['identity'], existing_digest(payload))
        payload['roster'][0]['source'] = 'changed'
        self.assertEqual(sealed['roster'][0]['source'], 'λ')
        validate(sealed, {'version', 'roster'}, 'test/v1')

    def test_observations(self):
        for value in [-1, float('inf'), float('nan'), True]:
            with self.assertRaises(ValueError):
                number(value)

    def test_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'a').write_text('a')
            good = {'path': 'a', 'sha256': 'sha256:ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb'}
            self.assertEqual(reference(root, good), root / 'a')
            for bad in ['a', dict(good, path='../a'), dict(good, sha256=digest('wrong'))]:
                with self.assertRaises((ValueError, OSError)):
                    reference(root, bad)

    def test_manifest_fails_before_engine(self):
        cfg = json.loads((ROOT / 'configs/foundation/pipeline_validation.json').read_text())
        parse(cfg)
        for change in [{'extra': True}, {'authority': {'training': True}}, {'scope': 'research'}]:
            with self.assertRaises(ValueError):
                parse(seal(dict(cfg, **change)))

    def test_terminal_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder), 'a', {'contract': digest('a')})
            ledger.transition('READY')
            ledger.transition('RUNNING')
            ledger.transition('PASS', files={})
            with self.assertRaises(ValueError):
                ledger.transition('FAIL')
            self.assertEqual(ledger.read()['state'], 'PASS')

    def test_manifest_rejects_ignored_overrides_and_raised_resource_caps(self):
        cfg=json.loads((ROOT/'configs/foundation/pipeline_validation.json').read_text())
        for field in ['workloads','comparisons']:
            with self.assertRaises(ValueError):parse(seal(dict(cfg,**{field:[{'unregistered':'override'}]})))
        for name,value in [('stream_bytes',9<<20),('transient_bytes',9<<30),('trace_bytes',129<<20),('zip_bytes',513<<20)]:
            changed=copy.deepcopy(cfg);changed['budgets'][name]=value
            with self.assertRaises(ValueError):parse(seal(changed))
        with self.assertRaises(ValueError):parse(seal(dict(cfg,source={'ignored':'source'})))

    def test_current_dispositions_use_the_package_implementation_serializer(self):
        import ast
        import hashlib
        # Execute the actual small identity functions, avoiding the runtime's
        # non-stdlib dependency imports on the portable host.
        namespace=dict(json=json,hashlib=hashlib,Path=Path,Any=object)
        def definitions(path,names):
            functions=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
            self.assertEqual({n.name for n in functions},set(names))
            exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
        definitions(ROOT/'src/oeis_learn/experiments/artifacts.py',['canonical_bytes','compute_canonical_digest'])
        namespace['digest']=namespace['compute_canonical_digest']
        source=ROOT/'src/oeis_learn/evaluation/experiment_runner.py'
        namespace['__file__']=str(source)
        definitions(source,['implementation_identity'])
        self.assertEqual(namespace['canonical_bytes']({'text':'λ'}),'{"text":"λ"}'.encode())
        record=json.loads((ROOT/'specs/007-experiment-foundation/validation/current-dispositions.json').read_text())
        self.assertEqual(record['implementation_sha256'],namespace['implementation_identity']())
        self.assertEqual(record['original_inputs'],68)
        self.assertEqual(len(record['items']),68)
        self.assertIs(record['full_007_complete'],False)
