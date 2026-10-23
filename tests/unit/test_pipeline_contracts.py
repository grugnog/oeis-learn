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
