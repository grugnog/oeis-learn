"""Append-only state transitions and exclusive immutable terminal receipts."""
import json
import uuid
from pathlib import Path
from .identity import seal, write

TERMINAL = {'PASS','FAIL','PARTIAL','INCONCLUSIVE','BLOCKED','REUSED','UNAVAILABLE','NOT_SELECTED'}
EDGES = {'PENDING': {'READY','BLOCKED','REUSED','UNAVAILABLE','NOT_SELECTED'},
         'READY': {'RUNNING','BLOCKED'}, 'RUNNING': {'PASS','FAIL','PARTIAL','INCONCLUSIVE'}}


class Ledger:
    def __init__(self, root, stage, binding):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.payload = dict(version='pipeline-stage/v1', stage=stage, attempt=uuid.uuid4().hex,
                            binding=binding, state='PENDING', reason='', files={})
        if (self.root / 'receipt.json').exists() or (self.root / 'transitions.jsonl').exists():
            raise ValueError('fresh stage output required')
        self._append()

    def _append(self):
        row = seal(self.payload)
        with (self.root / 'transitions.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        if self.payload['state'] in TERMINAL:
            write(self.root / 'receipt.json', row)

    def transition(self, state, **details):
        if state not in EDGES.get(self.payload['state'], set()):
            raise ValueError('illegal or immutable terminal transition')
        if set(details) - {'reason','files','observations','reused_from','prerequisites','command','counts','cleanup'}:
            raise ValueError('unknown receipt details')
        self.payload.update(state=state, **details)
        self._append()
        return self.read()

    def read(self):
        return seal(self.payload)


def read_terminal(path):
    path = Path(path)
    if not (path / 'receipt.json').is_file():
        return dict(state='PARTIAL', reason='interrupted: terminal record missing')
    value = json.loads((path / 'receipt.json').read_text())
    if seal(value) != value or value['state'] not in TERMINAL:
        raise ValueError('invalid terminal receipt')
    return value
