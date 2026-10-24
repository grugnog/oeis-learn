"""Invocation-wide transient/trace bounds during builds, startup and role work."""
import os
from pathlib import Path
import signal
import threading
from .identity import write
from .runner import ResourceBreach


class DiskWatchdog:
    def __init__(self, root, output, *, cap=8<<30, trace_cap=128<<20, free_reserve=16<<20, extra_roots=()):
        self.root=Path(root);self.extra_roots=tuple(Path(p) for p in extra_roots);self.output=Path(output);self.cap=cap;self.trace_cap=trace_cap
        self.free_reserve=free_reserve;self.stop=threading.Event();self.breach=None
        self.peak=0;self.trace_peak=0
        self.free=None;self.minimum_free=None;self.breach_observation=None

    def usage(self):
        seen=set();size=traces=0
        from itertools import chain
        for path in chain.from_iterable(root.rglob('*') for root in (self.root,*self.extra_roots) if root.exists()):
            try:
                if path.is_symlink() or not path.is_file():continue
                info=path.stat();key=(info.st_dev,info.st_ino)
                if key in seen:continue
                seen.add(key);size+=info.st_size
                if 'traces' in path.parts:traces+=info.st_size
            except FileNotFoundError:
                continue  # Atomic role progress/cache publication.
        self.peak=max(self.peak,size);self.trace_peak=max(self.trace_peak,traces)
        return size

    def check(self):
        try:
            size=self.usage()
            fs=os.statvfs(self.root);free=fs.f_bavail*fs.f_frsize
            self.free=free
            self.minimum_free=free if self.minimum_free is None else min(self.minimum_free,free)
            reason=None
            if size>self.cap:
                reason=f'invocation transient artifact cap exceeded: {size} > {self.cap} bytes'
            elif self.trace_peak>self.trace_cap:
                reason=f'invocation trace cap exceeded: {self.trace_peak} > {self.trace_cap} bytes'
            elif free<self.free_reserve:
                reason=f'invocation free-space reserve exhausted: {free} < {self.free_reserve} bytes at {self.root}'
            if reason:
                if self.breach is None:
                    self.breach=reason
                    self.breach_observation=dict(artifact_bytes=size,trace_bytes=self.trace_peak,free_bytes=free)
                raise ResourceBreach(self.breach)
        except OSError as exc:
            raise ResourceBreach('invocation disk accounting unavailable: '+str(exc)) from exc

    def __enter__(self):
        self.check()
        self.previous=signal.getsignal(signal.SIGUSR1)
        def breach(signum,frame):
            raise ResourceBreach(self.breach or 'disk accounting failure')
        signal.signal(signal.SIGUSR1,breach)
        def monitor():
            while not self.stop.wait(1):
                try:self.check()
                except ResourceBreach as exc:
                    self.breach=str(exc)
                    os.kill(os.getpid(),signal.SIGUSR1)
                    break
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
        return self

    def __exit__(self,*exc):
        self.stop.set();self.thread.join(timeout=2)
        signal.signal(signal.SIGUSR1,self.previous)
        try:
            self.check()
        except ResourceBreach as failure:
            self.breach=str(failure)
        try:
            write(self.output/'disk.json',dict(version='pipeline-disk-observation/v1',
                peak_bytes=self.peak,trace_peak_bytes=self.trace_peak,cap_bytes=self.cap,
                trace_cap_bytes=self.trace_cap,free_reserve_bytes=self.free_reserve,breach=self.breach,
                filesystem_path=str(self.root),free_bytes=self.free,minimum_free_bytes=self.minimum_free,
                breach_observation=self.breach_observation))
        except OSError:
            # Earlier ledgers remain usable; resource exhaustion must still propagate.
            if exc[0] is None:raise ResourceBreach('disk report could not be persisted')
        if self.breach and exc[0] is not KeyboardInterrupt:raise ResourceBreach(self.breach) from exc[1]
