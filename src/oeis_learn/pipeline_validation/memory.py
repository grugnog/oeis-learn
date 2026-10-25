"""Observed aggregate upper bounds; no Torch allocator/cgroup double-count claim."""
import os
from pathlib import Path
import signal
import threading
import json
import time
from .identity import write


def observe(render_node=None):
    def integer(path):
        try:
            return int(Path(path).read_text().strip())
        except (OSError,ValueError):
            return None
    try:
        memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    except (OSError,ValueError):
        memory={}
    driver=None
    if render_node:
        root=Path('/sys/class/drm')/Path(render_node).name/'device'
        values=[integer(root/name) for name in ('mem_info_vram_used','mem_info_gtt_used')]
        if all(v is not None for v in values):
            driver=sum(values)
    return dict(physical=memory.get('MemTotal'),available=memory.get('MemAvailable'),
        current=integer('/sys/fs/cgroup/memory.current'),limit=integer('/sys/fs/cgroup/memory.max'),driver=driver)


class MemoryWatchdog:
    def __init__(self, output, *, ceiling, gpu=False, render_node=None, reader=None, killer=None,audit_samples=False):
        self.output=Path(output);self.ceiling=ceiling;self.gpu=gpu
        self.reader=reader or (lambda:observe(render_node))
        self.killer=killer or (lambda:os.kill(os.getpid(),signal.SIGTERM))
        self.stop=threading.Event();self.samples=0;self.known=True
        self.machine_peak=0;self.role_peak=0;self.breach=None;self.last=None
        self.audit_samples=audit_samples;self.render_node=render_node

    def sample(self, *, terminate=True):
        try:
            row=self.reader()
        except Exception:
            row={}
        self.samples+=1;self.last=row
        required=['physical','available','current','limit']+(['driver'] if self.gpu else [])
        known=all(type(row.get(k)) is int and row[k]>=0 for k in required)
        self.known &= known
        if self.audit_samples:
            def table(name):
                try:return {x.split()[0]:int(x.split()[1]) for x in Path('/sys/fs/cgroup',name).read_text().splitlines()}
                except (OSError,ValueError,IndexError):return {}
            memory=table('memory.stat');cpu=table('cpu.stat')
            def sensor(name):
                if not self.render_node:return None
                try:return int((Path('/sys/class/drm')/Path(self.render_node).name/'device'/name).read_text().strip())
                except (OSError,ValueError):return None
            sample=dict(time=time.monotonic(),**row,anon=memory.get('anon'),file=memory.get('file'),
                kernel=memory.get('kernel'),shmem=memory.get('shmem'),cpu_usage_usec=cpu.get('usage_usec'),
                gpu_busy_percent=sensor('gpu_busy_percent'))
            payload=json.dumps(sample,sort_keys=True,allow_nan=False)+'\n'
            if self.samples>9000 or len(payload.encode())>4096:
                self.breach=dict(reason='mandatory memory sample bound exceeded',observation=row)
                if terminate:self.killer()
                return
            try:
                with (self.output/'resource-samples.jsonl').open('a') as stream:stream.write(payload)
            except OSError:
                self.breach=dict(reason='mandatory resource audit cannot be written',observation=row)
                if terminate:self.killer()
                return
        if not known:
            return
        # Summing driver residency gives a conservative upper bound even on UMA
        # where some pages may already occur in MemAvailable/cgroup accounting.
        # Never add allocator-reserved or allocator-active bytes to these counters.
        driver=row['driver'] if self.gpu else 0
        machine=row['physical']-row['available']+driver
        role=row['current']+driver
        self.machine_peak=max(self.machine_peak,machine);self.role_peak=max(self.role_peak,role)
        if machine>row['physical']*9//10 or role>self.ceiling or row['limit']>self.ceiling:
            if self.breach is None:
                self.breach=dict(reason='aggregate memory upper bound or enforcement exceeds authorized envelope',observation=row)
                try:
                    write(self.output/'memory-breach.json',self.breach)
                finally:
                    if terminate:self.killer()

    def report(self):
        return dict(version='pipeline-memory-observation/v1',qualified=bool(self.samples and self.known and not self.breach),
            samples=self.samples,ceiling_bytes=self.ceiling,machine_peak_upper_bytes=self.machine_peak,
            role_peak_upper_bytes=self.role_peak,last=self.last,breach=self.breach,
            accounting='conservative cgroup plus driver residency upper bound; allocator excluded' if self.known else 'unavailable aggregate memory accounting; cannot qualify')

    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=True)
        self.previous=signal.getsignal(signal.SIGTERM)
        def terminate(signum,frame):
            if self.breach:raise MemoryError(self.breach['reason'])
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,terminate)
        self.sample(terminate=False)
        if self.breach:
            signal.signal(signal.SIGTERM,self.previous)
            write(self.output/'memory.json',self.report())
            raise MemoryError(self.breach['reason'])
        def monitor():
            while not self.stop.wait(1):self.sample()
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
        return self

    def __exit__(self,*exc):
        self.stop.set();self.thread.join(timeout=2)
        signal.signal(signal.SIGTERM,self.previous)
        self.sample(terminate=False)
        write(self.output/'memory.json',self.report())
        if self.breach and exc[0] is not KeyboardInterrupt:raise MemoryError(self.breach['reason']) from exc[1]
