"""Opt-in, bounded observational timing; never an input to learning decisions."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
import time

_ACTIVE = ContextVar('learner_timing', default=None)


@contextmanager
def span(name):
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
        return
    start = time.perf_counter_ns()
    frame = [0]
    recorder.stack.append(frame)
    try:
        yield
    finally:
        elapsed = time.perf_counter_ns() - start
        recorder.stack.pop()
        if recorder.stack:
            recorder.stack[-1][0] += elapsed
        value = recorder.components.setdefault(name, dict(calls=0, inclusive_us=0, self_us=0))
        value['calls'] += 1
        value['inclusive_us'] += elapsed / 1000
        value['self_us'] += max(0, elapsed - frame[0]) / 1000


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if _ACTIVE.get() is None:
                return fn(*args, **kwargs)
            with span(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


class Recorder:
    def __init__(self):
        self.stack = []
        self.components = {}
        self.windows = []
        self.updates = self.tokens = self.loop_us = 0
        self.window = dict(updates=0, tokens=0, elapsed_us=0, maximum_shard=0)
        self.maximum_shard = 0

    def update(self, start, tokens, shard):
        elapsed = (time.perf_counter_ns() - start) / 1000
        self.updates += 1
        self.tokens += tokens
        self.loop_us += elapsed
        self.maximum_shard = max(self.maximum_shard, shard)
        self.window['updates'] += 1
        self.window['tokens'] += tokens
        self.window['elapsed_us'] += elapsed
        self.window['maximum_shard'] = max(self.window['maximum_shard'], shard)
        # First eight updates are cold; subsequent windows include allocation
        # probes, packing, accounting, quarantine and persistence in the loop.
        if self.updates == 8 or self.window['updates'] == 32:
            self.flush()

    def flush(self):
        if self.window['updates']:
            self.windows.append(dict(self.window, end_update=self.updates, cold=self.updates <= 8))
            self.windows = self.windows[:1] + self.windows[1:][-63:]
            self.window = dict(updates=0, tokens=0, elapsed_us=0, maximum_shard=0)


def completed_update(start, tokens, shard):
    recorder = _ACTIVE.get()
    if recorder is not None:
        recorder.update(start, tokens, shard)


@contextmanager
def recording(output):
    if os.environ.get('OEIS_PROGRESSIVE_TIMING') != '1':
        yield
        return
    from pathlib import Path
    from oeis_learn.generalization.progressive.contracts import atomic_write, seal
    recorder = Recorder()
    token = _ACTIVE.set(recorder)
    began = time.perf_counter_ns()
    status = 'FAILED'
    try:
        with span('invocation'):
            yield
        status = 'RETURNED'
    finally:
        _ACTIVE.reset(token)
        recorder.flush()
        device_memory = None
        try:
            import sys
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                free, total = torch.cuda.mem_get_info()
                device_memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes=free,total_bytes=total)
        except Exception as exc:
            device_memory = dict(collector_error=type(exc).__name__+': '+str(exc))
        atomic_write(Path(output)/'timing.json', seal(dict(
            version='progressive-learner-timing/v1', status=status,
            wall_us=(time.perf_counter_ns()-began)/1000,
            components=recorder.components, windows=recorder.windows,device_memory=device_memory,
            updates=recorder.updates, tokens=recorder.tokens,
            loop_us=recorder.loop_us, maximum_shard=recorder.maximum_shard,
            scope='observational wall time; no per-operation GPU synchronization; not qualification')))
