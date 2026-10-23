"""Canonical, immutable public records and confined content-addressed references."""
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return 'sha256:' + h.hexdigest()


def seal(payload):
    payload = json.loads(json.dumps({k: v for k, v in payload.items() if k != 'identity'}, allow_nan=False))
    return dict(payload, identity=digest(payload))


def validate(record, fields, version):
    if not isinstance(record, dict) or set(record) != set(fields) | {'identity'}:
        raise ValueError('unknown or missing fields: ' + version)
    if record['version'] != version or seal(record) != record:
        raise ValueError('version/identity mismatch: ' + version)
    return record


def number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError('finite nonnegative number required')
    return value


def confined(root, value, *, exists=True):
    root = Path(root).resolve(strict=True)
    relative = Path(value)
    if relative.is_absolute():
        try:
            relative = relative.relative_to(root)
        except ValueError:
            raise ValueError('path outside workspace: ' + str(value)) from None
    if '..' in relative.parts:
        raise ValueError('path traversal: ' + str(value))
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('symlink reference: ' + str(path))
    path.resolve(strict=exists).relative_to(root)
    return path


def reference(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('content identity required; a path alone is not identity')
    path = confined(root, ref['path'])
    if not path.is_file() or file_hash(path) != ref['sha256']:
        raise ValueError('reference hash mismatch: ' + str(path))
    return path


def write(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
