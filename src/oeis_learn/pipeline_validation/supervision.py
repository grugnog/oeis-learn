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
    streams={}
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
                    keep=max(0,stream_cap-stream.tell())
                    stream.write(block[:keep]);omitted[key.data]+=max(0,len(block)-keep)
            if process.returncode:
                raise RuntimeError(f'exit {process.returncode}: {output}/stderr.log')
    finally:
        if process is not None:
            # Children may retain pipes after the immediate engine process exits.
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=cleanup_seconds)
            process.stdout.close();process.stderr.close()
        for stream in streams.values():
            stream.close()
    return dict(elapsed_seconds=time.monotonic()-start,omitted_bytes=omitted,command=list(argv))
