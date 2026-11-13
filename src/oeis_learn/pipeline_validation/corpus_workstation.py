"""CPU-only full-corpus host operation with durable failure diagnostics."""
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid
import zipfile

from oeis_learn.fresh.common import confined, file_hash, read, write
from oeis_learn.fresh.corpus import configuration
from .supervision import launch


def package(cache):
    cache = Path(cache)
    share = cache / 'share'
    share.mkdir(exist_ok=True)
    files = sorted((cache / 'diagnostics').rglob('*'))
    files += [cache / name for name in ('snapshot.json', 'binding.json')]
    inventory, omissions, total = [], [], 0
    with zipfile.ZipFile(share / 'corpus.pending.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix not in {'.json', '.jsonl', '.log'}:
                continue
            size = path.stat().st_size
            ref = dict(path=str(path.relative_to(cache)), bytes=size, sha256=file_hash(path))
            if total + size > (512 << 20) - (1 << 20):
                omissions.append(dict(ref, reason='support ZIP cap; original retained locally'))
                continue
            archive.write(path, ref['path']); inventory.append(ref); total += size
        archive.writestr('manifest.json', json.dumps(dict(version='oeis-corpus-support/v1',
            files=inventory, omissions=omissions, qualified=False,
            note='Data preparation diagnostics. Raw corpus and private exposure index stay local.')))
    (share / 'corpus.pending.zip').replace(share / 'corpus.zip')
    print('Share diagnostics: ' + str(share / 'corpus.zip'), flush=True)


def run(root, args):
    from oeis_learn.container_runtime import Engine
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.fresh.common import digest, lock
    root = Path(root)
    config_path = confined(root, args.config)
    config = configuration(read(config_path))
    cache = confined(root, config['cache'])
    cache.mkdir(parents=True, exist_ok=True)
    with lock(cache / 'host.lock'):
        return _run(root, args, config, cache, Engine, collect_snapshot, digest)


def _run(root, args, config, cache, Engine, collect_snapshot, digest):
    logs = cache / 'diagnostics' / ('host-' + str(time.time_ns()))
    logs.mkdir(parents=True)
    started = time.monotonic()
    engine, name = None, 'oeis-corpus-' + uuid.uuid4().hex
    code, failure = 1, None
    container_attempted = False
    previous = {}
    def interrupt(*_):
        raise KeyboardInterrupt()
    try:
        package(cache)  # Also reconstruct diagnostics after a prior SIGKILL.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, interrupt)
        write(logs / 'request.json', dict(config=config, engine=args.engine,
              source=digest(collect_snapshot(root)), argv=sys.argv, container_name=name,
              gpu=False, training_started=False))
        engine = Engine(args.engine)
        write(logs / 'environment.json', engine.facts())
        # A killed host client may leave an engine-owned worker. Reclaim it
        # before any new download/processing owner can acquire the corpus lock.
        for request in sorted((cache / 'diagnostics').glob('host-*/request.json')):
            if request.parent == logs or (request.parent / 'outcome.json').exists():
                continue
            old = read(request)
            old_name = old.get('container_name')
            if old_name:
                try:
                    state = engine.call(['inspect', old_name], capture_output=True, timeout=30)
                    write(request.parent / 'container-inspect.json', json.loads(state.stdout))
                except Exception as exc:
                    write(request.parent / 'inspection-error.json', dict(error=str(exc)))
                engine.remove(old_name)
                write(request.parent / 'outcome.json', dict(status='INTERRUPTED',
                      recovered_by=name, prior_cost='unknown interrupted tail retained; never zeroed'))
        from .progressive_workstation_state import telemetry
        write(logs / 'telemetry.json', telemetry(root))
        # Build through the root Dockerfile/locks. The build cache is reusable;
        # do not silently reuse an old installed producer after pulling code.
        launch([engine.executable, 'build', '--target', 'qualification', '--tag',
                'oeis-learn:corpus', str(root)], logs / 'build', seconds=3600, stream_cap=4 << 20)
        image = engine.image('oeis-learn:corpus')
        write(logs / 'image.json', dict(image=image))
        worker = dict(config, cache='corpus', stripped='stripped.gz' if config['stripped'] else None)
        mounts = [(cache, '/work/corpus', True)]
        if config['stripped']:
            stripped = confined(root, config['stripped'])
            if stripped.exists():
                mounts.append((stripped, '/work/stripped.gz', False))
        write(logs / 'worker-config.json', worker)
        mounts.append((logs / 'worker-config.json', '/work/config.json', False))
        command = engine.command(image=image, name=name, mounts=mounts,
            argv=['python', '-m', 'oeis_learn.fresh.corpus_worker', '/work/config.json'],
            memory_bytes=int((args.memory_gib or 12) * (1 << 30)), cpus=args.cpus or 4,
            tmp_bytes=1 << 30, network=True, gpu=False, auto_remove=False, workdir='/work',
            environment={'PYTHONPATH': '/opt/oeis-build/src', 'HOME': '/work/corpus/home',
                         'GIT_TERMINAL_PROMPT': '0', 'GIT_LFS_SKIP_SMUDGE': '1'})
        container_attempted = True
        launch(command, logs / 'container', seconds=config['seconds'] + 120,
               cleanup_seconds=45, grace_seconds=30, stream_cap=8 << 20,
               allowed_returncodes=(0, 1, 2, 130))
        code = read(logs / 'container/process.json')['returncode']
    except KeyboardInterrupt:
        code = 130; failure = traceback.format_exc()
    except Exception:
        code = 1; failure = traceback.format_exc()
    finally:
        if engine and container_attempted:
            try:
                inspection = engine.call(['inspect', name], capture_output=True, timeout=30)
                write(logs / 'container-inspect.json', json.loads(inspection.stdout))
            except Exception as exc:
                write(logs / 'inspection-error.json', dict(error=str(exc)))
            try:
                engine.remove(name)
                write(logs / 'cleanup.json', dict(status='COMPLETE'))
            except Exception as exc:
                write(logs / 'cleanup.json', dict(status='FAILED', error=str(exc)))
                code = code or 1
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        write(logs / 'outcome.json', dict(returncode=code, traceback=failure,
              elapsed_seconds=time.monotonic() - started, qualified=False))
        package(cache)
    return code
