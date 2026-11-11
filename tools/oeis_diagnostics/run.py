#!/usr/bin/env python3
"""Independent stdlib host observer; uses a retained CPU image, never builds one."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import sys
import time
import traceback
import uuid
import zipfile

from analysis import digest, seal

ROOT = Path(__file__).resolve().parents[2]
TOOL = Path(__file__).resolve().parent


def confined(root, name):
    root = Path(root).resolve()
    name = Path(name)
    if name.is_absolute():
        name = name.relative_to(root)
    if '..' in name.parts:
        raise ValueError('path traversal')
    path = root
    for part in name.parts:
        path /= part
        if path.is_symlink():
            raise ValueError('symlink input/output')
    return path


def load(path, maximum=32 << 20):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError('bounded regular JSON required')
    def pairs(items):
        value = {}
        for k, v in items:
            if k in value:
                raise ValueError('duplicate JSON key')
            value[k] = v
        return value
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def write(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    if path.is_symlink():
        raise ValueError('symlink output')
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise ValueError('immutable diagnostic output changed')
        return
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('wb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def ref(root, path, maximum=1 << 30, *, deadline=None):
    path = confined(root, path)
    before = path.stat()
    if not path.is_file() or before.st_size > maximum:
        raise ValueError('bounded regular original required')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('diagnostic source hashing deadline')
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('source changed while hashing')
    return dict(path=str(path.relative_to(Path(root).resolve())), bytes=after.st_size, sha256='sha256:' + h.hexdigest())


def checked(root, original, *, maximum=1 << 30, deadline=None):
    path = confined(root, original['path'])
    if ref(root, path, maximum=maximum, deadline=deadline) != original:
        raise ValueError('original artifact changed: ' + original['path'])
    return path


def tool_identity():
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TOOL.glob('*.py'))})


def validate_pool(pool):
    if not isinstance(pool, list) or not 1 <= len(pool) <= 512:
        raise ValueError('bounded frozen training pool required')
    seen = set()
    for m in pool:
        if (set(m) != {'id','group','role','source_snapshot','indices','values'}
                or m['role'] != 'training' or not isinstance(m['group'], str) or not m['group']
                or not re.fullmatch(r'OEIS:A\d{6}', m['id']) or m['id'] in seen
                or not re.fullmatch(r'sha256:[0-9a-f]{64}', m['source_snapshot'])
                or len(m['indices']) != 100 or len(m['values']) != 100):
            raise ValueError('unique canonical OEIS training members required')
        for n in m['indices'] + m['values']:
            if not isinstance(n, str) or len(n) > 1235 or not re.fullmatch(r'0|-?[1-9]\d*', n):
                raise ValueError('canonical bounded integer required')
        start = int(m['indices'][0])
        if m['indices'] != list(map(str, range(start, start + 100))):
            raise ValueError('exact consecutive source indices required')
        seen.add(m['id'])
    return pool


def extra_terms(path, member, *, deadline=None):
    """Parse the full original; use only the prespecified next hundred terms."""
    offset = int(member['indices'][0])
    values, previous = {}, None
    with path.open() as f:
        ordinal = 0
        while True:
            ordinal += 1
            if ordinal % 4096 == 0 and deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('diagnostic source parsing deadline')
            line = f.readline(16385)
            if not line:
                break
            if len(line) > 16384:
                raise ValueError('oversized b-file line')
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2 or any(len(x) > 1235 or not re.fullmatch(r'[+-]?\d+', x) for x in parts):
                raise ValueError('malformed indexed original')
            index, value = map(int, parts)
            if previous is not None and index <= previous:
                raise ValueError('duplicate/unordered source indices')
            previous = index
            if offset <= index < offset + 200:
                values[index] = str(value)
    if [values.get(i) for i in range(offset, offset + 100)] != member['values']:
        raise ValueError('corpus and training member disagree')
    if any(i not in values for i in range(offset + 100, offset + 200)):
        return None
    return dict(indices=list(map(str, range(offset + 100, offset + 200))),
                values=[values[i] for i in range(offset + 100, offset + 200)])


def freeze(root, args, image):
    deadline = time.monotonic() + 600
    output = confined(root, args.output)
    cycle_path = confined(root, args.cycle_plan)
    cycle = load(cycle_path)
    if cycle.get('version') != 'progressive-oeis-cycle/v1' or seal(cycle) != cycle:
        raise ValueError('original sealed OEIS cycle plan required')
    future_run = confined(root, args.cycle_output)
    if future_run.exists():
        raise ValueError('freeze diagnostic sample before the cycle output exists')
    pool_path = checked(root, cycle['pool'])
    pool = validate_pool(load(pool_path, maximum=128 << 20))
    corpus = confined(root, args.corpus)
    receipt = load(corpus / 'diagnostics/result.json')
    if receipt['status'] != 'COMPLETE':
        raise ValueError('complete original corpus required')
    checked(corpus, receipt['inventory'], maximum=64 << 30, deadline=deadline)
    if not 1 <= args.sample <= 32 or not 0 <= args.seed < 2**32:
        raise ValueError('sample must contain one to 32 training groups')
    groups = {}
    for m in pool:
        groups.setdefault(m['group'], []).append(m)
    selected = []
    for group in sorted(groups, key=lambda g: digest([args.seed, g]))[:args.sample]:
        selected.append(min(groups[group], key=lambda m: digest([args.seed, m['id']])))
    rows, extra, originals, failures = [], {}, [], []
    consumed = 0
    with closing(sqlite3.connect((corpus / 'inventory.sqlite').as_uri() + '?mode=ro', uri=True, timeout=1)) as db:
        for member in selected:
            row = dict(task=member['id'], group=member['group'], status='UNAVAILABLE', reason='no_extended_bfile')
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError('diagnostic preparation deadline')
                result = db.execute('SELECT result FROM processed WHERE aid=?', (member['id'][5:],)).fetchone()
                if result is None:
                    raise ValueError('training member absent from pinned corpus')
                result = json.loads(result[0])
                if not result['eligible100'] or result['metadata']['offset'] != int(member['indices'][0]):
                    raise ValueError('training member offset/profile differs from corpus')
                if result.get('raw_path'):
                    path = confined(corpus, result['raw_path'])
                    consumed += path.stat().st_size * 3
                    if consumed > 1 << 30:
                        raise ValueError('diagnostic source-read allowance exhausted')
                    original = ref(root, path, maximum=256 << 20, deadline=deadline)
                    if original['sha256'] != 'sha256:' + result['bfile']['oid']:
                        raise ValueError('original b-file differs from pinned corpus')
                    truth = extra_terms(path, member, deadline=deadline)
                    checked(root, original, deadline=deadline)
                    originals.append(original)
                    if truth:
                        extra[member['id']] = truth
                        row.update(status='READY', reason=None, truth_identity=digest(truth))
                    else:
                        row['reason'] = 'incomplete200'
            except Exception as exc:
                row.update(status='BLOCKED', reason=str(exc))
                failures.append(dict(task=member['id'], error=type(exc).__name__, reason=str(exc)))
            rows.append(row)
    write(output / 'freeze-outcomes.json', dict(rows=rows, failures=failures))
    if failures:
        raise ValueError('diagnostic source integrity/resource deficits; inspect freeze-outcomes.json')
    plan = seal(dict(version='oeis-observer-plan/v1', tool=tool_identity(), image=image,
        cycle=ref(root, cycle_path), cycle_output=str(future_run.relative_to(root)),
        pool=ref(root, pool_path), pool_identity=digest(pool),
        corpus=ref(root, corpus / 'snapshot.json'), originals=originals,
        seed=args.seed, requested_groups=args.sample, sample=rows, seconds=600,
        per_case_seconds=8, visible=20, admitted=100, diagnostic_total=200,
        prepared_unix_ns=time.time_ns(), assessment_exposed=False,
        authority='read-only observer; no candidate selection, training, qualification or promotion authority'))
    write(output / 'private/extra.json', extra, immutable=True)
    write(output / 'plan.json', plan, immutable=True)
    return plan


def request(root, plan_path):
    plan = load(plan_path)
    if seal(plan) != plan or plan['tool'] != tool_identity():
        raise ValueError('original plan/tool identity differs')
    checked(root, plan['cycle']); checked(root, plan['corpus'])
    pool = validate_pool(load(checked(root, plan['pool']), maximum=128 << 20))
    if digest(pool) != plan['pool_identity']:
        raise ValueError('original pool differs')
    run = confined(root, plan['cycle_output'])
    yield_path = run / 'host/yield-audit.json'
    yielded = load(yield_path)
    if (seal(yielded) != yielded or yielded.get('version') != 'discovery-oeis-yield/v1'
            or yielded['payload']['pool'] != plan['pool_identity']):
        raise ValueError('original completed yield audit required')
    expected = yielded['payload']['admissions']
    if len(expected) > 512 or len(set(expected)) != len(expected):
        raise ValueError('bounded unique yield admission inventory required')
    found, refs = {}, []
    paths = sorted(run.glob('phase-*-admit-training*/role/admission.json'))
    if len(paths) > 4096:
        raise ValueError('admission scan allowance exceeded')
    for path in paths:
        path = confined(root, path)
        data = load(path, maximum=1 << 20)
        if data.get('identity') in expected:
            if seal(data) != data:
                raise ValueError('admission identity differs')
            found[data['identity']] = data
            refs.append(ref(root, path))
    if set(found) != set(expected):
        raise ValueError('yield audit has missing admission originals')
    extra = load(Path(plan_path).parent / 'private/extra.json')
    if set(extra) != {r['task'] for r in plan['sample'] if r['status'] == 'READY'}:
        raise ValueError('exact frozen extended-truth roster required')
    for row in plan['sample']:
        if row['status'] == 'READY' and digest(extra[row['task']]) != row['truth_identity']:
            raise ValueError('extended truth changed')
    return seal(dict(plan=plan, pool=pool, extra=extra,
                     admissions=list(found.values()), accepted_identities=expected,
                     originals=[ref(root, yield_path), *refs]))


def package(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files, total = [], 0
    with zipfile.ZipFile(output / 'diagnostics.pending.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if not path.is_file() or path.is_symlink() or 'private' in path.relative_to(output).parts:
                continue
            if path.suffix not in {'.json', '.log'}:
                continue
            # Requests include training programs and truth; never export them.
            if path.name == 'request.json':
                continue
            item = ref(output, path)
            total += item['bytes']
            if total > 128 << 20:
                raise ValueError('diagnostic ZIP allowance exceeded; originals retained')
            z.write(path, item['path']); files.append(item)
        z.writestr('manifest.json', json.dumps(seal(dict(version='oeis-observer-support/v1', files=files,
                                                       qualified=False, training_changed=False))))
    os.replace(output / 'diagnostics.pending.zip', output / 'diagnostics.zip')
    print('Diagnostics: ' + str(output / 'diagnostics.zip'), flush=True)


def container(root, args, image, output, payload=None):
    sys.path.insert(0, str(root / 'src'))
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation.supervision import launch
    engine = Engine(args.engine)
    output.mkdir(parents=True, exist_ok=True)
    name = 'oeis-observer-' + uuid.uuid4().hex
    mounts = [(TOOL, '/work/tool', False), (output, '/work/output', True)]
    if payload:
        (output / 'private').mkdir(mode=0o700, exist_ok=True)
        write(output / 'private/request.json', payload, immutable=True)
        write(output / 'plan.json', payload['plan'], immutable=True)
        write(output / 'input-inventory.json', dict(originals=payload['originals'],
              request=payload['identity'], plan=payload['plan']['identity']))
    argv = ['python', '/work/tool/worker.py', args.operation]
    command = engine.command(image=image, name=name, mounts=mounts, argv=argv,
        memory_bytes=2 << 30, cpus=2, tmp_bytes=256 << 20, gpu=False, network=False,
        auto_remove=False, workdir='/opt/oeis-build',
        environment={'PYTHONPATH': '/opt/oeis-build/src:/work/tool',
                     'HOME': '/work/output/private/home', 'OMP_NUM_THREADS': '2'})
    write(output / 'producer.json', dict(image=image, tool=tool_identity(), gpu=False, network=False))
    failure = None
    try:
        launch(command, output / 'logs', seconds=660, cleanup_seconds=45,
               grace_seconds=30, stream_cap=4 << 20)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            result = engine.call(['inspect', name], capture_output=True, timeout=30)
            write(output / 'container-inspect.json', json.loads(result.stdout))
        except Exception as exc:
            write(output / 'inspection-error.json', dict(error=str(exc)))
        try:
            engine.remove(name)
            write(output / 'cleanup.json', dict(status='COMPLETE'))
        except Exception as exc:
            write(output / 'cleanup.json', dict(status='FAILED', error=str(exc)))
            if failure is None:
                raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=['podman','docker'], required=True)
    p.add_argument('--image', default='oeis-learn:corpus', help='existing CPU image; never built here')
    sub = p.add_subparsers(dest='operation', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--cycle-plan', required=True); f.add_argument('--cycle-output', required=True)
    f.add_argument('--corpus', default='data/oeis-corpus-001')
    f.add_argument('--sample', type=int, default=32); f.add_argument('--seed', type=int, default=20261111)
    f.add_argument('--output', required=True)
    a = sub.add_parser('audit'); a.add_argument('--plan', required=True); a.add_argument('--output', required=True)
    s = sub.add_parser('self-test'); s.add_argument('--output', required=True)
    c = sub.add_parser('collect'); c.add_argument('--output', required=True)
    args = p.parse_args(argv)
    output = confined(ROOT, args.output)
    if args.operation == 'collect':
        if not output.is_dir() or not output.is_relative_to(ROOT / 'run-state'):
            p.error('existing diagnostic output under run-state required')
        package(output)
        return 0
    if output.exists():
        p.error('fresh diagnostic output required; keep previous reports and costs')
    if not output.is_relative_to(ROOT / 'run-state'):
        p.error('diagnostic output must live under run-state')
    output.mkdir(parents=True)
    def interrupt(*_):
        raise KeyboardInterrupt()
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        sys.path.insert(0, str(ROOT / 'src'))
        from oeis_learn.container_runtime import Engine
        if args.operation == 'audit':
            payload = request(ROOT, confined(ROOT, args.plan))
            image = payload['plan']['image']
            container(ROOT, args, image, output, payload)
        else:
            image = Engine(args.engine).image(args.image)
            if args.operation == 'freeze':
                freeze(ROOT, args, image)
            else:
                container(ROOT, args, image, output)
        return 0
    except BaseException as exc:
        write(output / 'failure.json', dict(error=type(exc).__name__, traceback=traceback.format_exc()))
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        package(output)


if __name__ == '__main__':
    raise SystemExit(main())
