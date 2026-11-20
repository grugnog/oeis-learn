#!/usr/bin/env python3
"""Spec 013 inventory. Python stdlib only; never builds, trains or uses the GPU.
Run from the repository root. Writes only a new diagnostic directory under run-state.
Default: stat every registered file and hash files <=16 MiB. --hash-all runs the
canonical read-only input audit, writing its receipts only under this diagnostic.
--probe-images starts CPU-only containers from existing images (no pulls).
No result from this script is qualification or permission to resume a campaign.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import zipfile


def run(argv, cwd, timeout=120):
    # Temporary files avoid unbounded PIPE buffering; only bounded tails retained.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        p = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                             stdout=out, stderr=err, start_new_session=True)
        expired = False
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired = True
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=10)
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=10)
            raise
        def tail(f):
            size = f.tell(); f.seek(max(0, size - (1 << 20)))
            return f.read().decode(errors='replace'), max(0, size - (1 << 20))
        stdout, omitted_out = tail(out); stderr, omitted_err = tail(err)
        result = dict(argv=argv, returncode=p.returncode, stdout=stdout, stderr=stderr,
                      omitted_bytes=omitted_out + omitted_err, timed_out=expired)
        if p.returncode or expired:
            raise RuntimeError(json.dumps(result))
        return result


def main(argv=None, *, default_root=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=str(default_root or Path.cwd()))
    ap.add_argument('--engine', choices=['podman', 'docker'], default='podman')
    ap.add_argument('--registration-session', default='run-state/discovery/013-research-inputs-003')
    ap.add_argument('--session', default='run-state/discovery/013-research-round-001')
    ap.add_argument('--render-node')
    ap.add_argument('--memory-gib', type=int, default=64)
    ap.add_argument('--cpus', type=int, default=8)
    ap.add_argument('--hash-all', action='store_true')
    ap.add_argument('--probe-images', action='store_true')
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    sys.dont_write_bytecode = True
    out = root / 'run-state' / ('environment-preflight-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output_error = None
    try:
        out.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        output_error = str(exc)
        out = Path(tempfile.mkdtemp(prefix='oeis-preflight-'))
    rows = []
    report = dict(version='discovery-environment-preflight/v1', root=str(root),
                  python=sys.version, platform=platform.platform(), checks=rows,
                  limitations=['No HIP execution or measured model capacity',
                               'No registry/network build test', 'No research support or scientific feasibility test',
                               'No authority to bypass source mismatch or recovery budgets'])
    def check(name, action):
        start = time.monotonic()
        try:
            value = action()
            status = value.pop('_status', 'PASS') if isinstance(value, dict) else 'PASS'
            row = dict(name=name, status=status, value=value)
        except Exception as exc:
            row = dict(name=name, status='FAIL', error=type(exc).__name__,
                       detail=str(exc), traceback=traceback.format_exc())
        row['seconds'] = round(time.monotonic() - start, 3)
        rows.append(row)
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(row['status'] + ' ' + name, flush=True)
        return row.get('value') if row['status'] != 'FAIL' else None
    def require(condition, message, value=None):
        if not condition: raise ValueError(message)
        return value if value is not None else dict(detail=message)
    def file_check(ref):
        from .identity import confined
        from .progressive_research_inputs import checked_file
        path = confined(root, ref['path'])
        info = path.stat()
        require(path.is_file() and info.st_size == ref.get('bytes', info.st_size), 'Missing or wrong-size original: ' + ref['path'])
        with path.open('rb') as f: f.read(1)
        if info.st_size <= 16 << 20:
            if 'bytes' in ref: checked_file(root, ref)
            else:
                require('sha256:' + hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'], 'Hash mismatch: ' + ref['path'])
            mode = 'hash_verified'
        else: mode = 'readable_size_only; full hash deferred to --hash-all/canonical preparation'
        return dict(path=ref['path'], bytes=info.st_size, verification=mode)
    originals = {}
    interrupted = False
    previous = {}
    def interrupt(*_):
        raise KeyboardInterrupt()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[sig] = signal.signal(sig, interrupt)
    try:
        if output_error:
            check('repository output directory', lambda: require(False, output_error))
        check('repository layout', lambda: require((root / 'scripts/discovery_hip_handoff.py').is_file(),
              'Expected discovery checkout missing; select the repository with --root'))
        for module in ('identity', 'progressive_workstation_state', 'progressive_research_inputs', 'progressive_research_bridge'):
            check('checkout module ' + module, lambda module=module: repository_module(root, module))
        check('checkout', lambda: run(['git', 'status', '--short', '--branch'], root))
        check('revision', lambda: run(['git', 'rev-parse', 'HEAD'], root))
        check('host', lambda: host_telemetry(root))
        def resources():
            free = shutil.disk_usage(root).free; nodes = os.statvfs(root).f_favail
            mem = {a.split(':')[0]: int(a.split()[1]) * 1024 for a in Path('/proc/meminfo').read_text().splitlines() if ':' in a}
            require(free >= 10 << 30 and nodes > 0, 'Below pipeline 10 GiB free-disk floor or no free inodes')
            return dict(_status='WARN' if mem['MemAvailable'] < 24 << 30 else 'PASS',
                        free_disk_gib=round(free / (1 << 30), 1), free_inodes=nodes,
                        available_memory_gib=round(mem['MemAvailable'] / (1 << 30), 1),
                        note='24 GiB host headroom; checkpoint storage needs measured preparation. This is not a storage sufficiency claim.')
        check('host resource floors', resources)
        check('diagnostic directory write/rename', lambda: write_probe(out))
        for name in ('benchmarks/symbolic_definitions_v1.json', 'benchmarks/trustworthy_synthesis_v1.json',
                     'sample_oeis/names', 'sample_oeis/stripped'):
            check('tracked fixture ' + name, lambda name=name: require((root / 'data' / name).is_file(), 'Fixture exists: ' + name))
        def engine_info():
            raw = run([args.engine, 'info', '--format', '{{json .}}'], root)
            from oeis_learn.container_runtime import Engine
            details = json.loads(raw['stdout'])
            engine = Engine(args.engine, details=details)
            facts = engine.facts()
            require(facts['memory_bytes'] >= args.memory_gib << 30 and facts['cpus'] >= args.cpus,
                    'Engine cannot supply requested memory/CPUs: ' + json.dumps(facts))
            # Report only normalized facts; no complete engine/config/environment dump.
            storage = details.get('store', {}).get('graphRoot') if args.engine == 'podman' else details.get('DockerRootDir')
            if storage and Path(storage).is_dir():
                facts['image_storage_free_gib'] = round(shutil.disk_usage(storage).free / (1 << 30), 1)
            else:
                facts['image_storage_free_gib'] = 'unavailable; engine storage may be remote/inaccessible'
            report['engine'] = facts
            return engine, facts
        holder = {}
        def inspect_engine():
            holder['engine'], facts = engine_info()
            return facts
        check('container engine and requested limits', inspect_engine)
        devices = [Path(args.render_node)] if args.render_node else sorted(Path('/dev/dri').glob('renderD*'))
        check('render-node selection', lambda: require(len(devices) == 1, 'Exactly one render node required; supply --render-node when ambiguous', dict(nodes=list(map(str, devices)))))
        for device in [Path('/dev/kfd'), *devices]:
            check('device permission ' + str(device), lambda device=device: require(device.is_char_device() and os.access(device, os.R_OK | os.W_OK), 'Readable/writable character device: ' + str(device)))
        def session_inventory(session, exposure_only=False):
            from .identity import confined
            from .progressive_workstation_state import read
            from .progressive_research_bridge import REGISTRATION_STEPS
            folder = confined(root, session)
            binding = read(folder / 'session.json')['binding']
            journal = read(folder / 'journal.json')
            required = ('register-originals', 'import-exposure') if exposure_only else REGISTRATION_STEPS
            for name in required:
                step = journal['steps'].get('register-inputs/' + name, {})
                check(session + ' / ' + name, lambda step=step: require(step.get('status') == 'COMPLETE' and step.get('outputs'), 'Original registration step must be COMPLETE'))
                for ref in step.get('outputs', []):
                    if ref['path'] in originals and originals[ref['path']] != ref:
                        raise ValueError('Conflicting original references: ' + ref['path'])
                    originals[ref['path']] = ref
            manifest = binding.get('input_manifest')
            if manifest: originals.setdefault(manifest['path'], manifest)
            return binding
        binding = check('registration journal', lambda: session_inventory(args.registration_session))
        if binding:
            donor = binding.get('reuse_exposure_session')
            if donor and donor != args.registration_session:
                check('original exposure donor', lambda: session_inventory(donor, exposure_only=True))
            # Audit records bind checkpoints, optimizer/RNG sidecars and prior inputs.
            def retained():
                from .identity import confined
                from .progressive_workstation_state import read
                registration = read(confined(root, args.registration_session) / 'register-inputs/registration/registration.json')
                require(registration['status'] == 'COMPLETE', 'Original retained registration is incomplete')
                for entry in registration['checks']:
                    ref = entry.get('workspace_artifact')
                    if ref: originals.setdefault(ref['path'], ref)
                return dict(status=registration['status'], checks=len(registration['checks']))
            check('retained checkpoint/sidecar inventory', retained)
        for ref in originals.values(): check('original ' + ref['path'], lambda ref=ref: file_check(ref))
        def existing_session():
            from .identity import confined, digest
            from .progressive_workstation_state import read
            path = confined(root, args.session, exists=False)
            if not path.exists(): return dict(state='new session; parents are created by pipeline')
            from oeis_learn.experiments.source_snapshot import collect_snapshot
            old = read(path / 'session.json')['binding']
            expected = dict(source=digest(collect_snapshot(root)), engine=args.engine,
                            memory_gib=args.memory_gib, cpus=args.cpus, render_node=args.render_node,
                            registration_session=args.registration_session)
            changed = [k for k, v in expected.items() if old.get(k) != v]
            require(not changed, 'Existing session binding differs: ' + ', '.join(changed) + '; do not delete/reset evidence')
            return dict(state='binding matches', note='Budgets, active processes and recovery remain canonical pipeline checks')
        check('research session compatibility', existing_session)
        if args.hash_all:
            code = 'import sys;sys.dont_write_bytecode=True;sys.path.insert(0,sys.argv[1]);from oeis_learn.pipeline_validation.progressive_research_bridge import adopt;adopt(*sys.argv[2:])'
            check('canonical full original-input audit', lambda: run([sys.executable, '-S', '-c', code, str(root/'src'), str(root), args.registration_session, str(out/'private-audit')], root, 3600))
        else:
            check('full data validation', lambda: dict(_status='WARN', detail='Large-file hashes and semantic original-input validation deferred. Use --hash-all when convenient.'))
        if args.probe_images and 'engine' in holder:
            engine = holder['engine']
            for target in ('qualification', 'gpu'):
                tag = 'oeis-learn:discovery-' + target
                check('CPU-only image probe ' + tag, lambda tag=tag: image_probe(engine, tag, root, out))
        else:
            check('installed image contents', lambda: dict(_status='WARN', detail='Not inspected. --probe-images uses existing tags only, without GPU access, pulls or builds.'))
    except KeyboardInterrupt:
        interrupted = True
        rows.append(dict(name='interruption', status='WARN', detail='Interrupted; partial results retained'))
    except Exception:
        rows.append(dict(name='collector', status='FAIL', traceback=traceback.format_exc()))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report['counts'] = {s: sum(r['status'] == s for r in rows) for s in ('PASS', 'WARN', 'FAIL')}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        archive = out / 'preflight.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(out/'report.json', 'report.json')
            z.write(Path(__file__).resolve(), 'environment_preflight.py')
            launcher = root / 'scripts/check_discovery_environment.py'
            if launcher.is_file(): z.write(launcher, 'check_discovery_environment.py')
        print(json.dumps(report['counts']), flush=True)
        print('Share: ' + str(archive), flush=True)
    return 130 if interrupted else 1 if report['counts']['FAIL'] else 0


def repository_module(root, name):
    import importlib
    module = importlib.import_module('oeis_learn.pipeline_validation.' + name)
    expected = root / 'src/oeis_learn/pipeline_validation' / (name + '.py')
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError('Imported module is outside selected checkout: ' + str(module.__file__))
    return dict(path=str(expected))


def host_telemetry(root):
    from .progressive_workstation_state import telemetry
    return telemetry(root)


def write_probe(folder):
    with tempfile.TemporaryDirectory(prefix='permissions-', dir=folder) as t:
        p = Path(t)/'a'; p.write_bytes(b'preflight'); p.rename(Path(t)/'b')
    return dict(create_write_rename_remove=True)


def image_probe(engine, tag, root, out):
    info = json.loads(run([engine.executable, 'image', 'inspect', tag], root)['stdout'])[0]
    image = info.get('Id') or info.get('ID')
    name = 'oeis-preflight-' + uuid.uuid4().hex
    code = """import json,os,sys
from pathlib import Path
import torch,oeis_learn
from oeis_learn.experiments.source_snapshot import source_snapshot
from oeis_learn.pipeline_validation.identity import digest
base=Path('/opt/oeis-build')
paths=['data/benchmarks/symbolic_definitions_v1.json','data/benchmarks/trustworthy_synthesis_v1.json',
       'data/sample_oeis/names','data/sample_oeis/stripped']
assert all((base/p).is_file() for p in paths), 'required benchmark fixture absent'
assert Path('/opt/oeis-build-info/synthesis-worker').is_file(), 'installed native worker absent'
assert os.access('/opt/oeis-build-info/synthesis-worker',os.X_OK), 'native worker not executable'
Path('/work/output/mount-probe').write_text('ok')
assert not Path('/dev/kfd').exists(), 'CPU-only probe unexpectedly sees GPU'
print(json.dumps(dict(python=sys.version,torch=torch.__version__,hip_build=torch.version.hip,source=digest(source_snapshot()),mount_write=True,fixtures=paths)))
"""
    with tempfile.TemporaryDirectory(prefix='image-probe-', dir=out) as t:
        argv = engine.command(image=image, name=name, mounts=[(t, '/work/output', True)],
                              argv=['python', '-c', code], gpu=False,
                              environment={'HOME':'/tmp'}, tmp_bytes=256 << 20)
        try:
            result = run(argv, root, 300)
            observed = json.loads(result['stdout'].strip().splitlines()[-1])
            from oeis_learn.experiments.source_snapshot import collect_snapshot
            from oeis_learn.pipeline_validation.identity import digest
            if observed['source'] != digest(collect_snapshot(root)):
                result.update(_status='WARN', note='Existing image source differs from checkout; pipeline must build current-source image before qualification')
            return result
        finally: engine.remove(name, timeout=120)


if __name__ == '__main__':
    raise SystemExit(main())
