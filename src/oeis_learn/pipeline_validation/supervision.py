"""External subprocess watchdog, bounded draining logs and process-group cleanup."""
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


def launch(argv, output, *, seconds, stream_cap=8<<20, cleanup_seconds=5, heartbeat=30):
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
            if process.returncode:
                raise RuntimeError(f'exit {process.returncode}: {output}/stderr.log')
    except KeyboardInterrupt:
        interrupted=True;outcome='INTERRUPTED';failure='KeyboardInterrupt'
        raise
    except BaseException as exc:
        outcome='FAIL';failure=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if process is not None:
            # Give the controller one shared grace envelope to persist debits and
            # remove its containers before escalating to the entire host group.
            cleanup_end=time.monotonic()+cleanup_seconds
            if process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGINT if interrupted else signal.SIGTERM)
                    process.wait(timeout=max(0,cleanup_end-time.monotonic()))
                except (ProcessLookupError,subprocess.TimeoutExpired):
                    pass
            # Descendants may retain pipes even after the controller exits.
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=max(.01,cleanup_end-time.monotonic()))
            process.stdout.close();process.stderr.close()
        for name,stream in streams.items():
            try:stream.write(tails[name])
            finally:stream.close()
        try:
            write(output/'process.json',dict(outcome=outcome,failure=failure,
                elapsed_seconds=time.monotonic()-start,omitted_bytes=omitted,
                returncode=process.returncode if process is not None else None,
                logging='opening prefix plus final tail within one per-stream cap'))
        except OSError:
            if outcome=='COMPLETE':raise
    return dict(elapsed_seconds=time.monotonic()-start,omitted_bytes=omitted,command=list(argv))
