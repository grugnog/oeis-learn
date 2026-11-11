"""External subprocess watchdog, bounded draining logs and process-group cleanup."""
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


def diagnostic_program(body, label):
    """Wrap installed code with progress only; never walk another thread's frames.

    The host supervisor still owns deadlines, including when a blocked GIL stops
    the heartbeat. A heartbeat indicates Python liveness, not work completion.
    """
    import textwrap
    prefix="""import sys,threading,time
_diagnostic_label=LABEL
print(f'{_diagnostic_label}: python-start unix={time.time():.6f}',file=sys.stderr,flush=True)
_diagnostic_stop=threading.Event()
def _diagnostic_heartbeat():
    while not _diagnostic_stop.wait(30):
        try:
            print(f'{_diagnostic_label}: heartbeat unix={time.time():.6f}',file=sys.stderr,flush=True)
        except OSError:
            return
threading.Thread(target=_diagnostic_heartbeat,daemon=True,name='diagnostic-heartbeat').start()
try:
""".replace('LABEL',repr(label))
    return prefix+textwrap.indent(body,'    ')+'\nfinally:\n    _diagnostic_stop.set()\n'


def launch(argv, output, *, seconds, stream_cap=8<<20, cleanup_seconds=5, heartbeat=30,
           cleanup_state=None, grace_seconds=None,allowed_returncodes=(0,)):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();next_beat=start+heartbeat
    process=None;omitted={'stdout':0,'stderr':0}
    streams={};tails={'stdout':b'','stderr':b''};received={'stdout':0,'stderr':0}
    tail_cap=min(8192,stream_cap//4);prefix_cap=stream_cap-tail_cap
    outcome='COMPLETE';failure=None
    from .identity import write
    write(output/'command.json',dict(argv=list(argv),seconds=seconds,stream_cap=stream_cap,cleanup_seconds=cleanup_seconds))
    interrupted=False
    try:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        with selectors.DefaultSelector() as selector:
            for name, pipe in [('stdout',process.stdout),('stderr',process.stderr)]:
                streams[name]=(output/(name+'.log')).open('xb')
                os.set_blocking(pipe.fileno(),False);selector.register(pipe,selectors.EVENT_READ,name)
            while selector.get_map() or process.poll() is None:
                now=time.monotonic()
                if now-start>=seconds:
                    raise TimeoutError('external deadline: '+str(argv))
                if now>=next_beat:
                    print(f'  running {now-start:.1f}s; remaining {seconds-(now-start):.1f}s; logs {output}',flush=True)
                    next_beat=now+heartbeat
                for key,_ in selector.select(timeout=min(0.2,max(0,seconds-(now-start)))):
                    block=os.read(key.fileobj.fileno(),65536)
                    if not block:
                        selector.unregister(key.fileobj);continue
                    stream=streams[key.data]
                    keep=max(0,prefix_cap-stream.tell())
                    stream.write(block[:keep])
                    received[key.data]+=len(block)
                    rest=block[keep:]
                    if rest and tail_cap:tails[key.data]=(tails[key.data]+rest)[-tail_cap:]
                    omitted[key.data]=received[key.data]-stream.tell()-len(tails[key.data])
            if process.returncode not in allowed_returncodes:
                raise RuntimeError(f'exit {process.returncode}: {output}/stderr.log')
    except KeyboardInterrupt:
        interrupted=True;outcome='INTERRUPTED';failure='KeyboardInterrupt'
        raise
    except BaseException as exc:
        outcome='FAIL';failure=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        cleanup_end=time.monotonic()+cleanup_seconds
        if cleanup_state is not None:
            cleanup_state.setdefault('allowance_seconds',cleanup_seconds)
            cleanup_state.setdefault('deadline',cleanup_end)
            cleanup_end=cleanup_state['deadline']
        cleanup_failure=None
        if process is not None:
            # Give the controller one shared grace envelope to persist debits and
            # remove its containers before escalating to the entire host group.
            if process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGINT if interrupted else signal.SIGTERM)
                    grace=max(0,cleanup_end-time.monotonic())
                    # A container client can wait for its engine indefinitely.
                    # Preserve the shared deadline for engine.remove(), rather
                    # than exhausting it waiting for that client's exit.
                    if cleanup_state is not None and grace_seconds is None:
                        grace=min(grace,.25)
                    process.wait(timeout=grace if grace_seconds is None else min(grace,grace_seconds))
                except (ProcessLookupError,subprocess.TimeoutExpired):
                    pass
            # Descendants may retain pipes even after the controller exits.
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                # A client stuck in kernel I/O may not acknowledge SIGKILL. Keep
                # an explicit share for engine removal even on that path.
                reap=max(0,cleanup_end-time.monotonic())
                if cleanup_state is not None:reap=min(1,reap/4)
                process.wait(timeout=reap)
            except subprocess.TimeoutExpired:
                cleanup_failure='process-group termination unacknowledged within cleanup deadline'
                outcome='FAIL' if not interrupted else 'INTERRUPTED'
            process.stdout.close();process.stderr.close()
        for name,stream in streams.items():
            try:stream.write(tails[name])
            finally:stream.close()
        try:
            write(output/'process.json',dict(outcome=outcome,failure=failure,
                elapsed_seconds=time.monotonic()-start,omitted_bytes=omitted,
                returncode=process.returncode if process is not None else None,
                cleanup_failure=cleanup_failure,
                logging='opening prefix plus final tail within one per-stream cap'))
        except OSError:
            if outcome=='COMPLETE':raise
        if cleanup_failure and not interrupted:
            from .runner import ResourceBreach
            raise ResourceBreach(cleanup_failure)
    return dict(elapsed_seconds=time.monotonic()-start,omitted_bytes=omitted,command=list(argv))


def reclaim(engine, name, output, cleanup_state):
    """Finish engine-owned work within the same deadline as its client process."""
    from .identity import write
    from .runner import ResourceBreach
    started=time.monotonic()
    allowance=cleanup_state.get('allowance_seconds',5)
    deadline=cleanup_state.setdefault('deadline',started+allowance)
    failure=None
    try:
        left=deadline-time.monotonic()
        if left<=0:
            raise TimeoutError('cleanup deadline exhausted before container reclamation')
        engine.remove(name,timeout=left)
        if time.monotonic()>deadline:
            raise TimeoutError('container reclamation exceeded cleanup deadline')
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
        raise ResourceBreach('container termination unacknowledged: '+failure) from exc
    finally:
        write(Path(output)/'cleanup.json',dict(seconds=time.monotonic()-started,
            allowance_seconds=allowance,within_bound=failure is None and time.monotonic()<=deadline,
            acknowledged=failure is None,failure=failure))


def pending_gpu(host, name):
    """Durable owner marker; only engine acknowledgment permits its removal."""
    import re
    from .identity import write
    state = host.policy.get('state_root')
    if state is None:  # Explicit engine-boundary test doubles have no workspace.
        return None
    if not re.fullmatch(r'(?:oeis-phase5|generalization-inspect)-[0-9a-f]{32}', name):
        raise ValueError('owned preparation container required')
    path = Path(state) / 'gpu-pending.json'
    write(path, dict(version='preparation-gpu-owner/v1', engine=host.engine.name, name=name))
    return path


def reconcile_gpu(state, engine):
    """Called under the global GPU lock, before any new engine-owned work."""
    import json,re
    from .runner import ResourceBreach
    path = Path(state) / 'gpu-pending.json'
    if not path.exists():
        return
    if path.is_symlink() or path.stat().st_size > 4096:
        raise ResourceBreach('invalid pending GPU owner; no new GPU work')
    value = json.loads(path.read_bytes())
    if (set(value) != {'version','engine','name'} or value['version'] != 'preparation-gpu-owner/v1'
            or value['engine'] != engine.name
            or not re.fullmatch(r'(?:oeis-phase5|generalization-inspect)-[0-9a-f]{32}', value['name'])):
        raise ResourceBreach('pending GPU owner scope differs; no new GPU work')
    # Retain the marker if the engine cannot acknowledge termination. Original
    # producer/cleanup errors remain in their original stage directory.
    from uuid import uuid4
    reclaim(engine, value['name'], Path(state) / 'gpu-recovery' / uuid4().hex, {'allowance_seconds':120})
    path.unlink()
