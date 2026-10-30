"""Offline authority validation. A seal binds bytes; it does not certify an origin."""
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from oeis_learn.pipeline_validation.identity import seal, digest, confined


@lru_cache(maxsize=1)
def _schemas():
    schemas = {p.name.removesuffix('.schema.json'): json.loads(p.read_text())
               for p in files('oeis_learn.generalization').joinpath('schemas').iterdir()
               if p.name.endswith('.schema.json')}
    registry = Registry().with_resources((s['$id'], Resource.from_contents(s)) for s in schemas.values())
    return schemas, registry


def shape(kind, record):
    schemas, registry = _schemas()
    if kind not in schemas:
        raise ValueError('unknown authority kind')
    errors = list(Draft202012Validator(schemas[kind], registry=registry).iter_errors(record))
    if errors:
        raise ValueError(str(errors[0].message))


@dataclass(frozen=True)
class FrozenAuthority:
    kind: str
    payload: bytes

    @property
    def identity(self):
        return self.record()['identity']

    def record(self):
        return json.loads(self.payload)


def freeze(kind, record):
    return validate(kind, seal(record))


def validate(kind, record):
    shape(kind, record)
    if 'identity' not in record or seal(record) != record:
        raise ValueError('canonical identity mismatch')
    return FrozenAuthority(kind, json.dumps(record, sort_keys=True, allow_nan=False).encode())


def verify_artifact(root, ref, *, max_bytes=64 * 1024**3):
    """Read a bounded regular file, rejecting every symlink component and tampering.

    Roots must be immutable while verification runs (container read-only mounts in C).
    """
    if set(ref) != {'path', 'sha256', 'bytes'}:
        raise ValueError('artifact fields')
    value = ref['path']
    if not isinstance(value, str) or not value or PurePosixPath(value).is_absolute() or '\\' in value or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('noncanonical relative path')
    if type(ref['bytes']) is not int or not 0 <= ref['bytes'] <= max_bytes:
        raise ValueError('artifact size bound')
    try:
        path = confined(root, value)
        if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size != ref['bytes']:
            raise ValueError('artifact size/type mismatch')
        with path.open('rb') as stream:
            data = stream.read(ref['bytes'] + 1)
    except OSError as exc:
        raise ValueError('missing artifact') from exc
    if len(data) != ref['bytes'] or 'sha256:' + hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError('artifact hash/size mismatch')
    return data


def training_ancestry(origins):
    """Require a complete supplied lineage; callers bind it to verified inventories."""
    if not origins or 'mechanical' not in origins or any(x not in {'mechanical', 'training', 'random_initialization', 'verified_training_replay'} for x in origins):
        raise ValueError('forbidden or unknown training ancestry')


def artifact_refs(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256', 'bytes'}:
            yield value
        else:
            for child in value.values():
                yield from artifact_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from artifact_refs(child)


def resolve_study(preparation, proposed, manifests, *, shortfalls=(), root=None):
    """Commit only resolved authority, preserving all preregistered settings."""
    prep = validate('preparation', preparation.record()).record()
    if shortfalls or not manifests:
        raise ValueError('preparation shortfall: no trainable study')
    if root is None:
        raise ValueError('resolved artifact root required')
    candidate = json.loads(json.dumps(proposed))
    for key, value in prep['study_settings'].items():
        if candidate.get(key) != value:
            raise ValueError('preparation settings changed: ' + key)
    for key in ('seeds', 'budgets'):
        if candidate.get(key) != prep[key]:
            raise ValueError('preparation policy changed: ' + key)
    for key in ('generator', 'descriptors', 'exclusions'):
        if candidate['data'][key] != prep[key]:
            raise ValueError('preparation data policy changed')
    records = []
    for ref in manifests:
        record = json.loads(verify_artifact(root, ref))
        records.append(validate('cohort', record).record())
    for record in records:
        if record['exclusions'] != prep['exclusions']:
            raise ValueError('changed exclusion authority')
        if any(record['coverage'].get(k, 0) < v for k, v in record['quotas'].items()):
            raise ValueError('cohort coverage shortfall')
        for ref in artifact_refs(record):
            verify_artifact(root, ref)
        for ref in record['public_payloads']:
            payload = json.loads(verify_artifact(root, ref))
            for row in payload:
                shape('visible-prompt', row)
    resolved_quotas = {}
    for record in records:
        for key, value in record['quotas'].items():
            resolved_quotas[key] = resolved_quotas.get(key, 0) + value
    if resolved_quotas != prep['quotas']:
        raise ValueError('preparation quota commitments changed')
    if {r['role'] for r in records} != set(prep['roles']):
        raise ValueError('incomplete role manifests')
    candidate['data']['manifests'] = list(manifests)
    for ref in artifact_refs(candidate):
        verify_artifact(root, ref)
    return freeze('study', candidate)
