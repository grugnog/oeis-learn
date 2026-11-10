"""Research input registration from original, hash-bound workstation artifacts.

No tensor deserialization, model scoring, inferred offsets, or authority upgrades.
The disk-backed exclusion index stays private; public records contain identities
and counts only. Registration does not authorize legacy optimizer continuation.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import stat
import time
import traceback

from .identity import confined, seal
from .progressive_workstation_state import write
from oeis_learn.generalization.progressive.contracts import digest, reference

MANIFEST_VERSION = 'research-input-resolution-review/v1'


def validate_manifest(value):
    if (not isinstance(value, dict) or value.get('version') != MANIFEST_VERSION
        or seal(value) != value or value.get('executable') is not False
        or value.get('registration_applied') is not False
        or value.get('research_ready') is not False
        or value.get('retained', {}).get('selected_arm') != 'D'
        or set(value['retained']['checkpoints']) != {'B', 'D'}):
        raise ValueError('original non-executable D/B input resolution manifest required')
    return value


def checked_file(root, ref, *, maximum=32 << 30):
    """Hash in bounded blocks; never read a model-sized tensor into host memory."""
    if (not isinstance(ref, dict) or set(ref) != {'path', 'bytes', 'sha256'}
        or type(ref['bytes']) is not int or not 0 <= ref['bytes'] <= maximum):
        raise ValueError('bounded original artifact reference required')
    path = confined(root, ref['path'])
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != ref['bytes']:
        raise ValueError('original artifact identity/size differs: ' + ref['path'])
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    after = path.stat()
    fingerprint = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fingerprint(before) != fingerprint(after) or 'sha256:' + h.hexdigest() != ref['sha256']:
        raise ValueError('original artifact identity changed: ' + ref['path'])
    return path


def _json(path, *, maximum=256 << 20):
    if path.stat().st_size > maximum:
        raise ValueError('bounded original JSON required')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON member')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


class ExposureIndex:
    """Exact finite-window and structural exclusion index, never a fresh panel.

    Keys are hashes of authoritative values; no SQLite dependency on the host
    beyond Python's standard library. Originals are processed one cell at a time.
    """
    def __init__(self, path, *, readonly=False):
        path = Path(path)
        self.readonly = readonly
        if readonly:
            if path.is_symlink() or not path.is_file():
                raise ValueError('original exposure index required')
            self.db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists():
            raise ValueError('fresh exposure index required')
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA cache_size=-32768')
        page_size = self.db.execute('PRAGMA page_size').fetchone()[0]
        self.db.execute('PRAGMA max_page_count=' + str((8 << 30) // page_size))
        self.db.execute('CREATE TABLE keys(kind TEXT, value TEXT, PRIMARY KEY(kind,value)) WITHOUT ROWID')
        self.db.execute('CREATE TABLE origins(name TEXT PRIMARY KEY, rows INTEGER NOT NULL)')
        path.chmod(0o600)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.db.commit()
        self.db.close()

    @staticmethod
    def keys(row):
        from oeis_learn.generalization.progressive.generator import structure_id
        from oeis_learn.synthesis.contracts import canonical_integer
        values = row.get('values')
        indices = row.get('indices')
        if not isinstance(values, list) or not isinstance(indices, list) or len(values) != 100 or len(indices) != 100:
            raise ValueError('complete original indexed exposure required; no inferred offsets')
        numeric = [canonical_integer(x) for x in indices]
        if numeric != list(range(numeric[0], numeric[0] + 100)):
            raise ValueError('consecutive original source indices required')
        for value in values:
            canonical_integer(value)
        result = {('id', digest(row['id'])), ('behavior', digest(values))}
        result.update(('window', digest(values[i:i + 20])) for i in range(81))
        if row.get('source'):
            result.add(('structure', structure_id(row['source'])))
            result.add(('source', digest(row['source'])))
        for field in ('group', 'family'):
            if row.get(field):
                result.add((field, digest(row[field])))
        return sorted(result)

    def add(self, row, *, origin, role):
        if self.readonly:
            raise ValueError('original exposure index is immutable')
        if role not in {'training', 'development', 'confirmation', 'assessment'}:
            raise ValueError('explicit original exposure role required')
        self.db.executemany('INSERT OR IGNORE INTO keys VALUES (?,?)', self.keys(row))
        self.db.execute('INSERT INTO origins VALUES (?,1) ON CONFLICT(name) DO UPDATE SET rows=rows+1', (origin,))

    def matches(self, row):
        return sorted({kind for kind, value in self.keys(row)
                       if self.db.execute('SELECT 1 FROM keys WHERE kind=? AND value=?', (kind, value)).fetchone()})

    def summary(self):
        origins = dict(self.db.execute('SELECT name,rows FROM origins ORDER BY name'))
        return dict(rows=sum(origins.values()), origins=origins,
                    keys=dict(self.db.execute('SELECT kind,count(*) FROM keys GROUP BY kind')),
                    detector_limits='exact 100-term behavior, all 20-term windows, canonical structure and original identities; no all-domain equivalence claim')


def import_exposure(root, manifest, registration, output, *, seconds=7200):
    """All supplied original cells and later observed tasks, never scored anew."""
    root, output = Path(root), Path(output)
    if registration['status'] != 'COMPLETE' or registration['manifest'] != manifest['identity']:
        raise ValueError('complete matching original registration required')
    deadline = time.monotonic() + seconds
    by_name = {row['name']: row['workspace_artifact'] for row in registration['checks']
               if row.get('status') == 'PASS'}
    sources = [('original.' + item['name'], item['role'], item['count'])
               for item in manifest['prior_inputs']['originals']]
    sources += [('prior.probes', 'training', None), ('prior.observed', 'selection', None),
                ('fresh-002.targets', 'selection', manifest['fresh_002']['task_count'])]
    reports, errors = [], []
    index_path = output / 'private/exposure.sqlite'
    with ExposureIndex(index_path) as index:
        for name, role, expected in sources:
            before = time.monotonic()
            try:
                rows = _json(checked_file(root, by_name[name]))
                if not isinstance(rows, list) or not rows or len(rows) > 65536:
                    raise ValueError('bounded nonempty original exposure roster required')
                if expected is not None and len(rows) != expected:
                    raise ValueError('original exposure count differs')
                if len({r['id'] for r in rows}) != len(rows):
                    raise ValueError('original exposure identities conflict')
                if name == 'fresh-002.targets':
                    roster = _json(checked_file(root, by_name['fresh-002.roster']))
                    if isinstance(roster, dict) and roster.get('version') == 'pac-evaluation-roster/v2':
                        roster = roster['tasks']
                    if not isinstance(roster, dict) or set(roster) != {r['id'] for r in rows}:
                        raise ValueError('fresh-002 private truth/roster differs')
                for ordinal, row in enumerate(rows):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('original exposure import allowance exhausted')
                    index.add(row, origin=name, role='development' if role == 'selection' else role)
                    if ordinal % 256 == 0:
                        index.db.commit()
                        if index_path.stat().st_size > 8 << 30:
                            raise ValueError('private exposure index exceeds reserved eight GiB')
                        print('research-exposure: ' + name + ' ' + str(ordinal) + '/' + str(len(rows)), flush=True)
                checked_file(root, by_name[name])
                reports.append(dict(name=name, status='COMPLETE', rows=len(rows),
                                    artifact=by_name[name], elapsed_seconds=time.monotonic() - before))
            except Exception as exc:
                errors.append(dict(name=name, type=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc()))
            finally:
                write(output / 'exposure-progress.json', seal(dict(version='progressive-exposure-import-progress/v1',
                    reports=reports, failures=errors)))
        summary = index.summary()
    result = seal(dict(version='progressive-exposure-import/v1',
        status='BLOCKED' if errors else 'COMPLETE', registration=registration['identity'],
        reports=reports, failures=errors, summary=summary,
        index=reference(output, index_path),
        assessment='prior assessment used for exclusion only; no new assessment performed',
        fresh_002='previously observed; never a fresh panel'))
    write(output / 'exposure-import.json', result)
    return result


def source_candidates(root, *, maximum_files=20000):
    """Bounded breadth-first metadata inventory; no raw corpus traversal."""
    import os
    from collections import deque
    root = Path(root)
    if type(maximum_files) is not int or not 1 <= maximum_files <= 1000000:
        raise ValueError('finite source inventory bound required')
    found, caches, scans, errors = [], [], [], []
    excluded = {'checkpoints', '.git', 'private', 'share', 'rows', 'terms',
                'metadata', 'raw', 'logs', 'oeisdata'}
    for base in ('data', 'downloads', 'tmp-oeis-bulk', 'run-state'):
        folder = root / base
        if not folder.is_dir() or folder.is_symlink():
            continue
        queue = deque([folder])
        scanned = directories = 0
        began = time.monotonic()
        truncated = False
        while queue:
            directory = queue.popleft()
            directories += 1
            # Inspect the metadata before applying the bulk directory entry cap.
            # A manifest beside 20,000 terms must not disappear behind that cap.
            for name in ('manifest.json', 'fetch-plan.json', 'fetch-result.json'):
                path = directory / name
                if not path.is_file() or path.is_symlink():
                    continue
                try:
                    if path.stat().st_size > 128 << 20:
                        raise ValueError('source metadata exceeds inventory bound')
                    value = _json(path)
                    if (name == 'manifest.json' and isinstance(value, dict)
                        and value.get('version') == 'indexed-source/v1'
                        and isinstance(value.get('records'), list)):
                        found.append(dict(artifact=reference(root, path), release=value.get('release'),
                                          records=len(value['records'])))
                    elif name != 'manifest.json':
                        caches.append(dict(artifact=reference(root, path),
                            kind=name.removesuffix('.json'), scope='inventory only; offline import not yet validated'))
                except (ValueError, OSError, UnicodeError) as exc:
                    errors.append(dict(path=str(path.relative_to(root)), reason=str(exc)))
            children = []
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        scanned += 1
                        if entry.is_dir(follow_symlinks=False) and entry.name not in excluded:
                            children.append(Path(entry.path))
                        if scanned >= maximum_files:
                            truncated = True
                            break
            except OSError as exc:
                errors.append(dict(path=str(directory.relative_to(root)), reason=str(exc)))
            queue.extend(sorted(children))
            if truncated or time.monotonic() - began >= 60:
                truncated = truncated or bool(queue)
                break
        scans.append(dict(root=base, entries=scanned, directories=directories, truncated=truncated))
    return seal(dict(version='progressive-source-candidates/v1', candidates=found,
        bulk_caches=caches, scans=scans, scanned_files=sum(s['entries'] for s in scans),
        scanned_directories=sum(s['directories'] for s in scans),
        truncated=any(s['truncated'] for s in scans), errors=errors, selected=None,
        note='Inventory only; private values omitted; exact source selection still required. Each root has a separate entry/time cap.'))


def audit(root, manifest, output):
    """Check independent originals even after failures, preserving every cause."""
    from oeis_learn.generalization.contracts import artifact_refs
    root, output = Path(root), Path(output)
    manifest = validate_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    checks, valid, deficits = [], {}, []

    def check(name, ref, namespace=root, validate=None):
        started = time.monotonic()
        row = dict(name=name, artifact=dict(ref), namespace=str(Path(namespace).relative_to(root)))
        try:
            path = checked_file(namespace, ref)
            if validate:
                validate(path)
            workspace_ref = reference(root, path)
            if any(workspace_ref[key] != ref[key] for key in ('bytes', 'sha256')):
                raise ValueError('original changed during semantic verification')
            valid[name] = path
            row.update(status='PASS', workspace_artifact=workspace_ref)
        except Exception as exc:
            row.update(status='BLOCKED' if isinstance(exc, FileNotFoundError) else 'INVALID',
                       error=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc())
            deficits.append(name)
        row['elapsed_seconds'] = time.monotonic() - started
        checks.append(row)
        write(output / 'registration-progress.json', seal(dict(version='progressive-registration-progress/v1', checks=checks)))
        print('research-input: ' + name + ' ' + row['status'], flush=True)
        return valid.get(name)

    retained = manifest['retained']
    def recipe_check(path):
        from oeis_learn.fresh.bigint_config import resolve
        value = _json(path)
        if resolve(value) != value or value['model'] != retained['model']:
            raise ValueError('original resolved retained recipe/model differs')
    check('recipe', retained['recipe'], validate=recipe_check)
    for arm in ('D', 'B'):
        entry = retained['checkpoints'][arm]
        descriptor = entry['descriptor']
        if seal(descriptor) != descriptor or descriptor['kind'] != 'full_state':
            raise ValueError('original full-state descriptor seal required')
        def roster_check(path, descriptor=descriptor):
            roster = _json(path)
            if seal(roster) != roster or descriptor not in roster['checkpoints']:
                raise ValueError('selected original checkpoint absent from sealed roster')
        check(arm + '.roster', entry['roster'], validate=roster_check)
        check(arm + '.producer', entry['producer'])
        namespace = confined(root, entry['root'], exists=False)
        check(arm + '.weights', descriptor['weights'], namespace)
        path = check(arm + '.state', descriptor['state'], namespace)
        if path is not None:
            try:
                state = _json(path)
                from oeis_learn.generalization.state import _complete
                from oeis_learn.generalization.budgets import Ledger
                if (seal(state) != state or state['version'] != 'generalization-learner-state/v1'
                    or state['kind'] != 'full_state' or state['contract'] != descriptor['compatibility']
                    or state['boundary'] != 'before_or_after_complete_update'
                    or state['states']['weights'] != descriptor['weights']):
                    raise ValueError('original descriptor/state binding differs')
                _complete(state['states'])
                ledger = Ledger.restore(state['ledger'], cap=state['ledger']['cap'])
                committed = ledger.consumed - (ledger.pending[1] if ledger.debited else 0)
                if descriptor['tokens'] != committed or descriptor['updates'] != ledger.updates:
                    raise ValueError('original descriptor/ledger exposure differs')
                for i, ref in enumerate(artifact_refs(state['states'])):
                    check(arm + '.sidecar.' + str(i), ref, namespace)
            except Exception as exc:
                deficits.append(arm + '.state-contract')
                checks.append(dict(name=arm + '.state-contract', status='INVALID',
                                   reason=str(exc), traceback=traceback.format_exc()))
    for name in ('original_plan', 'workflow', 'inventory', 'probes', 'observed'):
        check('prior.' + name, manifest['prior_inputs'][name])
    for item in manifest['prior_inputs']['originals']:
        check('original.' + item['name'], item['ref'])
    for name in ('plan', 'result', 'publication', 'targets', 'roster'):
        check('fresh-002.' + name, manifest['fresh_002'][name])
    # Register the originals, never mutate their legacy namespace or descriptors.
    refs = [dict(name=name, role='custodian', kind='recipe' if name == 'recipe' else
                 'checkpoint' if name.endswith('.weights') else 'source',
                 artifact=reference(root, path)) for name, path in sorted(valid.items())]
    write(output / 'retained-inputs.json', seal(dict(version='discovery-retained-inputs/v1', references=refs)))
    result = seal(dict(version='progressive-research-registration/v1', manifest=manifest['identity'],
        selected_arm='D', status='BLOCKED' if deficits else 'COMPLETE', checks=checks,
        deficits=sorted(set(deficits)), research_ready=False, resume_authorized=False,
        initialization=dict(capability='original_D_weights', diversity='paired_fresh_random',
                            legacy_optimizer='retained for provenance; never a progressive resume state')))
    write(output / 'registration.json', result)
    return result
