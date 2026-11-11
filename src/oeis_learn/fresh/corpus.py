"""Full pinned OEIS inventory; raw data is never a training-membership grant.

One native LFS transfer queue per attempt, a bare Git store, and transactional
per-entry processing. No per-sequence HTTP, smudge loops or inferred offsets.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
from collections import Counter
from contextlib import closing

from . import bulk
from .common import confined, digest, file_hash, lock, read, write

BFILES = 'files/**/b*.txt'
VERSION = 'oeis-corpus/v1'
PROCESSOR = 'indexed-corpus-entry/v1'
VERIFICATION = 'sha256-content-bounded-actual-size/v2'


class SourceDisagreement(ValueError):
    """Hash-verified source records disagree; exclude, never choose a winner."""


def configuration(value):
    keys = {'version', 'cache', 'stripped', 'revision', 'max_source_bytes',
            'max_projection_bytes', 'free_disk_floor_bytes', 'seconds', 'attempts',
            'retry_seconds', 'concurrent_transfers'}
    if not isinstance(value, dict) or set(value) != keys or value['version'] != VERSION:
        raise ValueError('closed oeis-corpus/v1 configuration required')
    for key in ('cache', 'stripped'):
        if key == 'stripped' and value[key] is None:
            continue
        confined(Path.cwd(), value[key])
    if value['revision'] is not None and not re.fullmatch('[0-9a-f]{40}', value['revision']):
        raise ValueError('explicit revision must be a Git commit SHA')
    caps = {'max_source_bytes': 2 << 40, 'max_projection_bytes': 64 << 30,
            'free_disk_floor_bytes': 2 << 40, 'seconds': 86400, 'attempts': 5,
            'retry_seconds': 60, 'concurrent_transfers': 16}
    for key, cap in caps.items():
        if type(value[key]) is not int or not 1 <= value[key] <= cap:
            raise ValueError('positive bounded corpus setting required: ' + key)
    return value


def stamp(path):
    st = path.stat()
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular corpus object required')
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


class Runner:
    def __init__(self, cache, config):
        self.cache, self.config = cache, config
        self.started = time.monotonic()
        self.deadline = self.started + config['seconds']
        self.logs = cache / 'diagnostics' / ('attempt-' + str(time.time_ns()))
        self.logs.mkdir(parents=True)
        self.ordinal = 0

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('corpus invocation allowance exhausted; resume retained work')
        return remaining

    def run(self, argv, *, retry=False):
        from oeis_learn.pipeline_validation.supervision import launch
        last = None
        for attempt in range(self.config['attempts'] if retry else 1):
            self.ordinal += 1
            output = self.logs / str(self.ordinal).zfill(4)
            try:
                launch(argv, output, seconds=self.remaining(), stream_cap=4 << 20)
                return
            except (RuntimeError, OSError, TimeoutError) as exc:
                last = exc
                print(f'Native command attempt {attempt + 1} failed: {exc}', flush=True)
                if attempt + 1 < self.config['attempts'] and retry:
                    time.sleep(min(self.config['retry_seconds'], self.remaining()))
        raise last


def pinned_repository(cache, config, runner):
    repo = cache / 'oeisdata.git'
    if not repo.exists():
        # init/fetch remains resumable even if network disappears during the
        # first fetch; clone's partially-created directory is not reusable.
        runner.run(['git', 'init', '--bare', str(repo)])
    if not (repo / 'config').is_file():
        raise ValueError('bare corpus repository required')
    remotes = bulk.git(repo, 'remote', capture=True).splitlines()
    if 'origin' not in remotes:
        bulk.git(repo, 'remote', 'add', 'origin', bulk.REPOSITORY)
    if bulk.git(repo, 'remote', 'get-url', 'origin', capture=True).strip() != bulk.REPOSITORY:
        raise ValueError('unexpected corpus remote')
    pin_path = cache / 'snapshot.json'
    if pin_path.exists():
        pin = read(pin_path)
        if pin['repository'] != bulk.REPOSITORY or config['revision'] not in (None, pin['commit']):
            raise ValueError('immutable corpus revision changed')
    else:
        runner.run(['git', '-C', str(repo), 'fetch', '--depth=1', 'origin',
                    config['revision'] or 'HEAD'], retry=True)
        commit = bulk.git(repo, 'rev-parse', 'FETCH_HEAD^{commit}', capture=True).strip()
        pin = dict(repository=bulk.REPOSITORY, commit=commit,
                   exported_at=bulk.git(repo, 'show', commit + ':time.txt', capture=True).strip())
        # A stable ref retains the pinned tree and lets LFS discover .lfsconfig
        # in a bare repository without materializing hundreds of thousands files.
        bulk.git(repo, 'update-ref', 'refs/heads/corpus', commit)
        bulk.git(repo, 'symbolic-ref', 'HEAD', 'refs/heads/corpus')
        write(pin_path, pin, immutable=True)
    if bulk.git(repo, 'rev-parse', 'HEAD', capture=True).strip() != pin['commit']:
        raise ValueError('corpus repository moved from pinned commit')
    return repo, pin


def blobs(repo, entries):
    """One persistent native reader, not one git-show subprocess per entry."""
    proc = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for name, oid in entries:
            proc.stdin.write((oid + '\n').encode()); proc.stdin.flush()
            header = proc.stdout.readline().decode().split()
            if len(header) != 3 or header[:2] != [oid, 'blob']:
                raise ValueError('missing pinned Git blob: ' + name)
            size = int(header[2])
            if size > 4 << 20:
                # Consume without allocating an unbounded object; retain its
                # exclusion in the inventory rather than dropping the member.
                remaining = size
                while remaining:
                    part = proc.stdout.read(min(remaining, 1 << 20))
                    if not part:
                        raise ValueError('truncated Git object')
                    remaining -= len(part)
                payload = None
            else:
                payload = proc.stdout.read(size)
                if len(payload) != size:
                    raise ValueError('truncated Git object')
            if proc.stdout.read(1) != b'\n':
                raise ValueError('malformed Git batch boundary')
            yield name, oid, payload
    finally:
        proc.stdin.close(); proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)


def main_terms(payload, aid):
    fields = {}
    for line in payload.splitlines():
        match = re.fullmatch(r'%([STUVWX]) ' + aid + r' (.*)', line)
        if match:
            if match[1] in fields:
                raise ValueError('duplicate main-entry term field')
            fields[match[1]] = match[2]
    keys = 'VWX' if any(k in fields for k in 'VWX') else 'STU'
    text = ','.join(fields[k].strip(',') for k in keys if k in fields)
    if not text:
        return []
    values = text.split(',')
    if any(not bulk.INTEGER.fullmatch(v) or len(v) > 4096 for v in values):
        raise ValueError('non-integer or oversized main-entry term')
    return [str(int(v)) for v in values]


def inventory(repo, pin, cache, runner):
    path = cache / 'inventory.sqlite'
    if path.exists():
        return path
    pending = cache / 'inventory.pending.sqlite'
    pending.unlink(missing_ok=True)
    tree = []
    for entry in bulk.git(repo, 'ls-tree', '-rz', pin['commit'], 'seq', 'files', capture=True).split('\0'):
        if not entry:
            continue
        fields, name = entry.split('\t', 1)
        mode, kind, oid = fields.split()
        if (re.fullmatch(r'seq/A\d{3}/A\d{6}\.seq', name)
                or re.fullmatch(r'files/.*/b[^/]*\.txt', name)):
            if mode != '100644' or kind != 'blob':
                raise ValueError('nonregular corpus source: ' + name)
            tree.append((name, oid))
    with closing(sqlite3.connect(pending)) as db, db:
        db.execute('CREATE TABLE entries(aid TEXT PRIMARY KEY, entry_oid TEXT, metadata TEXT, main_terms TEXT, error TEXT)')
        db.execute('CREATE TABLE objects(path TEXT PRIMARY KEY, oid TEXT, bytes INTEGER, error TEXT, storage TEXT)')
        db.execute('CREATE TABLE identity(value TEXT)')
        db.execute('INSERT INTO identity VALUES (?)', (digest(pin),))
        for ordinal, (name, oid, payload) in enumerate(blobs(repo, tree)):
            runner.remaining()
            error = None
            if name.startswith('files/'):
                match = re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n', payload or b'')
                if match:
                    object_oid, size, storage = match[1].decode(), int(match[2]), 'lfs'
                elif payload is not None and not payload.startswith(b'version https://git-lfs.'):
                    object_oid, size, storage = hashlib.sha256(payload).hexdigest(), len(payload), 'git'
                    write(cache / 'git-bfiles' / object_oid, payload, immutable=True)
                else:
                    object_oid, size, storage = None, 0, 'invalid'
                    error = 'oversized_inline_object_or_malformed_LFS_pointer'
                db.execute('INSERT INTO objects VALUES (?,?,?,?,?)',
                           (name, object_oid, size, error, storage))
            else:
                aid = Path(name).stem
                meta, values = {}, []
                try:
                    if payload is None:
                        raise ValueError('oversized entry')
                    text = payload.decode()
                    meta = bulk.metadata(text, aid, f"{bulk.REPOSITORY}@{pin['commit']}:{name}")
                    values = main_terms(text, aid)
                except (ValueError, UnicodeError) as exc:
                    error = str(exc)
                db.execute('INSERT INTO entries VALUES (?,?,?,?,?)',
                           (aid, oid, json.dumps(meta), json.dumps(values), error))
            if ordinal % 10000 == 0:
                print(f'Corpus inventory: {ordinal}/{len(tree)} blobs', flush=True)
        if not db.execute('SELECT count(*) FROM entries').fetchone()[0]:
            raise ValueError('empty corpus inventory')
    os.replace(pending, path)
    return path


def cache_terms(db, source, runner):
    """Cache values are used only after agreement with the pinned main entry."""
    db.execute('CREATE TABLE IF NOT EXISTS cached(aid TEXT PRIMARY KEY, terms TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS cache_identity(hash TEXT)')
    if source is None or not source.exists():
        return dict(status='ABSENT', path=str(source) if source else None)
    before = stamp(source)
    identity = file_hash(source)
    previous = db.execute('SELECT hash FROM cache_identity').fetchone()
    if previous:
        if previous[0] != identity:
            raise ValueError('stripped cache changed; retain corpus binding')
        return dict(status='REUSED', sha256=identity)
    with db:
        db.execute('DELETE FROM cached')
        with gzip.open(source, 'rt') as stream:
            while True:
                line = stream.readline(1 << 20)
                if not line:
                    break
                runner.remaining()
                if len(line) >= 1 << 20:
                    raise ValueError('oversized stripped-cache row')
                if line.startswith('#') or not line.strip():
                    continue
                match = re.fullmatch(r'(A\d{6}) ,(.+),\s*', line)
                if not match:
                    raise ValueError('malformed stripped-cache row')
                values = match[2].split(',')
                if any(not bulk.INTEGER.fullmatch(v) or len(v) > 4096 for v in values):
                    raise ValueError('malformed stripped-cache terms')
                db.execute('INSERT INTO cached VALUES (?,?)',
                           (match[1], json.dumps([str(int(v)) for v in values])))
        if before != stamp(source):
            raise ValueError('stripped cache changed while reading')
        db.execute('INSERT INTO cache_identity VALUES (?)', (identity,))
    return dict(status='IMPORTED', sha256=identity)


def object_path(media, oid):
    return media / oid[:2] / oid[2:4] / oid


def inspect_bfile(path, row, offset, runner):
    """Hash and validate all terms in one streaming pass; retain 120 indexed terms."""
    h, values, previous, count = hashlib.sha256(), {}, None, 0
    before = stamp(path)
    if before[2] > runner.config['max_source_bytes']:
        raise ValueError('b-file exceeds reserved actual source byte allowance')
    malformed = None
    consumed = 0
    with path.open('rb') as stream:
        for chunk, raw in enumerate(iter(lambda: stream.readline(16385), b'')):
            if chunk % 4096 == 0:
                runner.remaining()
            consumed += len(raw)
            if consumed > before[2]:
                raise ValueError('b-file grew while reading')
            h.update(raw)
            if len(raw) > 16384:
                malformed = 'oversized b-file line'
            if malformed:
                # Still hash the complete original, without numeric parsing.
                continue
            try:
                line = raw.decode().split('#', 1)[0].strip()
                if not line:
                    continue
                fields = line.split()
                if (len(fields) != 2 or any(not bulk.INTEGER.fullmatch(v) for v in fields)
                        or len(fields[1]) > 4096 or len(fields[0]) > 32):
                    raise ValueError('malformed indexed b-file')
                index, value = map(int, fields)
                if previous is not None and index <= previous:
                    raise ValueError('duplicate or unordered b-file index')
                previous = index
                count += 1
                if offset is not None and offset <= index < offset + 120:
                    values[index] = str(value)
            except (ValueError, UnicodeError) as exc:
                malformed = str(exc)
            if count % 4096 == 0:
                runner.remaining()
    if before != stamp(path) or h.hexdigest() != row['oid']:
        raise ValueError('b-file hash changed or differs from pinned LFS OID')
    return values, count, malformed, before


def process(db, media, cache, runner):
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS processed(aid TEXT PRIMARY KEY, signature TEXT, result TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS verified_objects(oid TEXT PRIMARY KEY, fingerprint TEXT)')
    projection_bytes = sum(json.loads(r[0]).get('projection_bytes', 0)
                           for r in db.execute('SELECT result FROM processed'))
    reused = 0
    for ordinal, row in enumerate(db.execute('SELECT * FROM entries ORDER BY aid')):
        runner.remaining()
        aid, meta, main = row['aid'], json.loads(row['metadata']), json.loads(row['main_terms'])
        if ordinal % 1000 == 0:
            print(f'Corpus scan: {ordinal + 1}; reused {reused}; latest {aid}', flush=True)
            resource_floor(cache, runner.config)
        source = db.execute('SELECT * FROM objects WHERE path=?',
                            (f'files/{aid[:4]}/b{aid[1:]}.txt',)).fetchone()
        raw = ((cache / 'git-bfiles' / source['oid']) if source['storage'] == 'git'
               else object_path(media, source['oid'])) if source and source['oid'] else None
        current_stamp = stamp(raw) if raw and raw.exists() else None
        signature = digest([PROCESSOR, row['entry_oid'], dict(source) if source else None, current_stamp])
        old = db.execute('SELECT signature,result FROM processed WHERE aid=?', (aid,)).fetchone()
        if old:
            prior = json.loads(old['result'])
            if prior['status'] == 'BLOCKED':
                # Keep the original cause after the current summary is repaired.
                with (runner.logs / 'prior-blocked-members.jsonl').open('a') as stream:
                    stream.write(json.dumps({k: v for k, v in prior.items()
                                             if k not in {'values', 'indices'}}) + '\n')
        if old and old['signature'] == signature:
            if prior['status'] != 'BLOCKED':
                # Published projections are cheap to hash; raw multi-GB b-files
                # need no repeated parsing when their exact stat identity holds.
                for ref in prior.get('projections', {}).values():
                    if file_hash(confined(cache, ref['path'])) != ref['sha256']:
                        raise ValueError('published corpus projection changed: ' + aid)
                reused += 1
                continue
        result = dict(aid=aid, status='PROCESSED', reasons=[], metadata=meta,
                      entry_oid=row['entry_oid'], bfile=dict(source) if source else None,
                      raw_path=str(raw.relative_to(cache)) if raw else None,
                      actual_bytes=current_stamp[2] if current_stamp else None,
                      main_terms=len(main), available_terms=0, eligible100=False)
        values = {meta['offset'] + i: v for i, v in enumerate(main[:120])} if meta else {}
        cached = db.execute('SELECT terms FROM cached WHERE aid=?', (aid,)).fetchone()
        if cached:
            cached_values = json.loads(cached[0])
            compatible = len(cached_values) >= len(main) and cached_values[:len(main)] == main and bool(main)
            result['cache'] = 'VERIFIED_MAIN_TERMS' if compatible else 'DIFFERENT_RELEASE_OR_TERMS'
            if compatible and meta:
                values = {meta['offset'] + i: v for i, v in enumerate(cached_values[:min(120, len(main))])}
        try:
            if row['error']:
                result['reasons'].append(row['error'])
            if source:
                if source['error'] or not current_stamp:
                    raise ValueError(source['error'] or 'missing pinned b-file object')
                bvalues, count, malformed, verified_stamp = inspect_bfile(raw, source, meta.get('offset'), runner)
                if verified_stamp[2] != source['bytes']:
                    result['warnings'] = ['declared_size_mismatch_hash_verified']
                with db:
                    db.execute('INSERT OR REPLACE INTO verified_objects VALUES (?,?)',
                               (source['oid'], json.dumps(verified_stamp)))
                result['available_terms'] = count
                if malformed:
                    result['reasons'].append(malformed)
                    values = {}
                else:
                    conflicts = [i for i, v in bvalues.items() if i in values and values[i] != v]
                    if conflicts:
                        result['conflict'] = dict(count=len(conflicts), first_index=str(min(conflicts)))
                        raise SourceDisagreement('b-file/main-entry value disagreement')
                    # Do not bridge gaps in a supplied b-file with main-entry terms.
                    values = bvalues
            else:
                result['available_terms'] = len(main)
                result['reasons'].append('no_bfile_main_terms_only')
            if meta.get('finite'):
                result['reasons'].append('finite')
            if meta.get('table'):
                result['reasons'].append('table')
            complete = bool(meta) and all(meta['offset'] + i in values for i in range(100))
            if not complete:
                result['reasons'].append('incomplete100')
            result['eligible100'] = bool(complete and not row['error'] and not meta['finite'] and not meta['table'])
            # Keep bounded exact terms for every processable entry, including
            # finite/short/table members; full originals remain in Git/LFS.
            result['indices'] = [str(i) for i in sorted(values)]
            result['values'] = [values[i] for i in sorted(values)]
            if result['eligible100']:
                terms = f'indexed/terms/{aid[:4]}/{aid}.txt'
                metadata = f'indexed/metadata/{aid[:4]}/{aid}.json'
                payload = ''.join(f'{i} {v}\n' for i, v in sorted(values.items())).encode()
                required = len(payload) + len(json.dumps(meta).encode()) + 512
                if projection_bytes + required + (cache / 'inventory.sqlite').stat().st_size > runner.config['max_projection_bytes']:
                    raise OSError('corpus projection byte allowance exhausted; raise reservation to resume')
                write(cache / terms, payload, immutable=True)
                write(cache / metadata, meta, immutable=True)
                result['projection_bytes'] = required
                projection_bytes += required
                result['projections'] = dict(terms=dict(path=terms, sha256=file_hash(cache / terms)),
                                            metadata=dict(path=metadata, sha256=file_hash(cache / metadata)))
        except SourceDisagreement as exc:
            result.update(status='EXCLUDED', eligible100=False)
            result['reasons'].append(str(exc))
        except (ValueError, OSError, UnicodeError) as exc:
            result.update(status='BLOCKED', eligible100=False, error=str(exc))
        with db:
            db.execute('INSERT OR REPLACE INTO processed VALUES (?,?,?)',
                       (aid, signature, json.dumps(result)))



def verify_objects(db, media, cache, runner):
    """Account for every selected object, including auxiliary/orphan b-files."""
    failures = 0
    warnings = Counter()
    path = cache / 'diagnostics/object-outcomes.jsonl'
    if path.exists():
        with path.open() as previous, (runner.logs / 'prior-object-failures.jsonl').open('w') as saved:
            for ordinal, line in enumerate(previous):
                runner.remaining()
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError:
                    # A killed writer can leave an unfinished diagnostic line.
                    # This is not source evidence; verify the actual object anew.
                    saved.write(json.dumps(dict(status='INCOMPLETE_DIAGNOSTIC_LINE', line=ordinal + 1)) + '\n')
                    continue
                if prior.get('status') == 'BLOCKED':
                    saved.write(line)
    with path.open('w') as out:
        for ordinal, row in enumerate(db.execute('SELECT * FROM objects ORDER BY path')):
            runner.remaining()
            if ordinal % 10000 == 0:
                print(f'Corpus object verification: {ordinal + 1}; blocked {failures}', flush=True)
            outcome = dict(row)
            try:
                if row['error']:
                    raise ValueError(row['error'])
                raw = cache / 'git-bfiles' / row['oid'] if row['storage'] == 'git' else object_path(media, row['oid'])
                before = stamp(raw)
                outcome['actual_bytes'] = before[2]
                if before[2] > runner.config['max_source_bytes']:
                    raise ValueError('object exceeds reserved actual source byte allowance')
                verified = db.execute('SELECT fingerprint FROM verified_objects WHERE oid=?', (row['oid'],)).fetchone()
                if not verified or json.loads(verified[0]) != before:
                    h = hashlib.sha256()
                    consumed = 0
                    with raw.open('rb') as stream:
                        for block in iter(lambda: stream.read(1 << 20), b''):
                            runner.remaining()
                            consumed += len(block)
                            if consumed > before[2]:
                                raise ValueError('object grew while reading')
                            h.update(block)
                    outcome['sha256'] = 'sha256:' + h.hexdigest()
                    if h.hexdigest() != row['oid'] or stamp(raw) != before:
                        raise ValueError('object differs from pinned hash')
                    with db:
                        db.execute('INSERT OR REPLACE INTO verified_objects VALUES (?,?)',
                                   (row['oid'], json.dumps(before)))
                outcome.update(status='VERIFIED', sha256='sha256:' + row['oid'])
                if before[2] != row['bytes']:
                    outcome['warnings'] = ['declared_size_mismatch_hash_verified']
                    warnings.update(outcome['warnings'])
            except (ValueError, OSError) as exc:
                failures += 1
                outcome.update(status='BLOCKED', reason=str(exc))
            out.write(json.dumps(outcome, sort_keys=True) + '\n')
    return dict(failures=failures, warnings=dict(warnings), policy=VERIFICATION)


def source_budget(db, media, cache, runner):
    """Reserve actual retained bytes too: a pointer's size is not authoritative.

    Count per path, conservatively including shared objects more than once,
    matching the original declared-byte reservation. No content reads here.
    """
    actual = declared = missing = 0
    for ordinal, row in enumerate(db.execute('SELECT * FROM objects')):
        runner.remaining()
        if ordinal % 10000 == 0:
            print(f'Corpus size preflight: {ordinal + 1}; actual bytes {actual}', flush=True)
        declared += row['bytes']
        if not row['oid']:
            continue
        raw = cache / 'git-bfiles' / row['oid'] if row['storage'] == 'git' else object_path(media, row['oid'])
        if raw.exists():
            actual += stamp(raw)[2]
        else:
            missing += 1
    result = dict(actual_bytes=actual, declared_bytes=declared, missing_objects=missing,
                  limit=runner.config['max_source_bytes'], policy=VERIFICATION)
    write(cache / 'diagnostics/source-budget.json', result)
    write(runner.logs / 'source-budget.json', result)
    if actual > runner.config['max_source_bytes']:
        raise ValueError('actual source bytes exceed reservation; increase reserved source allowance')
    return result

def resource_floor(cache, config, *, required=0, inodes=0):
    if shutil.disk_usage(cache).free < config['free_disk_floor_bytes'] + required:
        raise OSError('corpus disk reserve insufficient; cache retained')
    if os.statvfs(cache).f_favail < inodes + 10000:
        raise OSError('corpus inode reserve insufficient; cache retained')


def summarize(db, cache, pin, *, fetch_error=None, objects=None):
    counts, reasons, records = Counter(), Counter(), []
    db.row_factory = sqlite3.Row
    target = cache / 'diagnostics/member-outcomes.jsonl'
    with target.open('w') as stream:
        for row in db.execute('SELECT e.aid,p.result FROM entries e LEFT JOIN processed p USING(aid) ORDER BY aid'):
            result = json.loads(row['result']) if row['result'] else dict(aid=row['aid'], status='PENDING')
            counts[result['status']] += 1
            counts['eligible100'] += bool(result.get('eligible100'))
            reasons.update(result.get('reasons', []))
            public = {k: v for k, v in result.items() if k not in {'values', 'indices'}}
            stream.write(json.dumps(public, sort_keys=True) + '\n')
            if result.get('eligible100'):
                refs = {key: dict(ref, path=str(Path(ref['path']).relative_to('indexed')))
                        for key, ref in result['projections'].items()}
                records.append(dict(record_id='OEIS:' + result['aid'], **refs))
    complete = objects is not None and not objects['failures'] and not fetch_error and not counts['PENDING'] and not counts['BLOCKED']
    report = dict(version='oeis-corpus-result/v1', snapshot=pin,
                  status='COMPLETE' if complete else 'PARTIAL', counts=dict(counts),
                  reasons=dict(reasons), fetch_error=fetch_error,
                  verification_policy=VERIFICATION,
                  object_warnings=objects['warnings'] if objects else {},
                  objects_verified=objects is not None and not objects['failures'],
                  completion_scope='full inventory accounted for; EXCLUDED entries never enter indexed training sources',
                  membership='UNASSIGNED; no training or evaluation authority',
                  license='CC BY-SA 4.0; OEIS Foundation', qualified=False,
                  inventory=dict(path='inventory.sqlite', bytes=(cache / 'inventory.sqlite').stat().st_size,
                                 sha256=file_hash(cache / 'inventory.sqlite')))
    if complete:
        write(cache / 'indexed/manifest.json', dict(version='indexed-source/v1',
              release=f"{pin['repository']}@{pin['commit']}", records=records), immutable=True)
    write(cache / 'diagnostics/result.json', report)
    return report


def prepare(root, config):
    config = configuration(config)
    cache = confined(root, config['cache'])
    cache.mkdir(parents=True, exist_ok=True)
    runner = Runner(cache, config)
    with lock(cache / 'corpus.lock'):
        # Resource reservations can be raised prospectively on a later attempt;
        # the pinned source, cache input and processor contract remain bound.
        write(cache / 'binding.json', dict(version=VERSION, processor=PROCESSOR,
              stripped=config['stripped']), immutable=True)
        write(runner.logs / 'request.json', config)
        resource_floor(cache, config)
        repo, pin = pinned_repository(cache, config, runner)
        dbpath = inventory(repo, pin, cache, runner)
        with closing(sqlite3.connect(dbpath)) as db:
            if db.execute('SELECT value FROM identity').fetchone()[0] != digest(pin):
                raise ValueError('corpus inventory belongs to another snapshot')
            source = confined(root, config['stripped']) if config['stripped'] else None
            write(runner.logs / 'cache-reuse.json', cache_terms(db, source, runner))
            total, bad = db.execute('SELECT coalesce(sum(bytes),0),sum(error IS NOT NULL) FROM objects').fetchone()
            entries = db.execute('SELECT count(*) FROM entries').fetchone()[0]
            write(cache / 'diagnostics/corpus-plan.json', dict(snapshot=pin, entries=entries,
                  bfiles=db.execute('SELECT count(*) FROM objects').fetchone()[0],
                  declared_source_bytes=total, malformed_pointers=bad or 0, limits=config))
            bulk.git(repo, 'lfs', 'install', '--local', '--skip-repo')
            media = next(Path(line.split('=', 1)[1]) for line in
                         bulk.git(repo, 'lfs', 'env', capture=True).splitlines()
                         if line.startswith('LocalMediaDir='))
            media = confined(cache, str(media.relative_to(cache)))
            fetch_error = None
            try:
                if total > config['max_source_bytes']:
                    raise ValueError(f'full b-file source needs {total} bytes; increase reserved source allowance')
                missing = sum(size for oid, size in db.execute("SELECT oid,max(bytes) FROM objects WHERE storage='lfs' GROUP BY oid")
                              if not object_path(media, oid).is_file())
                resource_floor(cache, config, required=missing + config['max_projection_bytes'], inodes=3 * entries)
                if missing:
                    runner.run(['git', '-C', str(repo), '-c', 'lfs.fetchrecentalways=false',
                                '-c', f"lfs.concurrenttransfers={config['concurrent_transfers']}",
                                '-c', 'lfs.transfer.maxretries=8', '-c', 'lfs.transfer.maxretrydelay=30',
                                '-c', 'lfs.skipdownloaderrors=false', 'lfs', 'fetch',
                                '--include=' + BFILES, '--exclude=', 'origin', pin['commit']], retry=True)
            except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
                fetch_error = f'{type(exc).__name__}: {exc}'
            objects = None
            try:
                db.row_factory = sqlite3.Row
                try:
                    source_budget(db, media, cache, runner)
                except (ValueError, OSError) as exc:
                    fetch_error = fetch_error or f'{type(exc).__name__}: {exc}'
                else:
                    process(db, media, cache, runner)
                    objects = verify_objects(db, media, cache, runner)
                    if objects['failures']:
                        fetch_error = fetch_error or f"{objects['failures']} pinned b-file objects unavailable or invalid"
            finally:
                report = summarize(db, cache, pin, fetch_error=fetch_error, objects=objects)
                write(runner.logs / 'result.json', dict(report, elapsed_seconds=time.monotonic() - runner.started))
            return report
