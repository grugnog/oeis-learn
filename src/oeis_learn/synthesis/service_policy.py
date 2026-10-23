"""Aggregate owning-role FIFO reservations and deterministic LRU publication.

Storage and worker supervision call one shared instance per role in Phase 3.
No operation allocates request/cache payloads or issues a fresh time allowance.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math


def positive(n, *, zero=False):
    if type(n) is not int or n < (0 if zero else 1):
        raise ValueError("invalid bounded count")


@dataclass(frozen=True)
class ServiceLimits:
    max_inflight_batches: int = 1
    max_pending_batches: int = 8
    max_pending_bytes: int = 32 * 1024**2
    result_cache_max_entries: int = 4096
    result_cache_max_bytes: int = 64 * 1024**2
    compiled_cache_max_entries: int = 32
    compiled_cache_max_bytes: int = 64 * 1024**2
    disk_cache_max_entries: int = 256
    disk_cache_max_bytes: int = 512 * 1024**2

    def __post_init__(self):
        for key, value in vars(self).items():
            positive(value, zero=key.startswith("disk_"))
        if bool(self.disk_cache_max_entries) != bool(self.disk_cache_max_bytes):
            raise ValueError("disable both disk limits together")


class RoleCapacity:
    def __init__(self, max_pending=8, max_pending_bytes=32*1024**2, max_active=1):
        for n in (max_pending, max_pending_bytes, max_active): positive(n)
        self.max_pending, self.max_pending_bytes, self.max_active = max_pending, max_pending_bytes, max_active
        self.pending, self.active = OrderedDict(), {}
        self.expired = 0

    @property
    def pending_bytes(self): return sum(item[0] for item in self.pending.values())

    def expire(self, now):
        for key, (_, deadline, _) in list(self.pending.items()):
            if now >= deadline:
                del self.pending[key]
                self.expired += 1

    def enqueue(self, key, size, *, deadline, now):
        positive(size)
        if not isinstance(key, str) or not key or key in self.pending or key in self.active:
            raise ValueError("duplicate or invalid request")
        if not math.isfinite(now) or not math.isfinite(deadline) or now >= deadline:
            raise ValueError("queue:backpressure_timeout")
        self.expire(now)
        if len(self.pending) >= self.max_pending or self.pending_bytes + size > self.max_pending_bytes:
            return False  # Caller may retry only within this same original deadline.
        self.pending[key] = (size, deadline, now)
        return True

    def dispatch(self, *, now):
        self.expire(now)
        if len(self.active) >= self.max_active or not self.pending: return None
        key, item = self.pending.popitem(last=False)
        self.active[key] = item  # Original deadline and enqueue time survive dispatch.
        return key

    def remaining(self, key, now):
        return max(0, self.active[key][1] - now)

    def finish(self, key):
        if key not in self.active: raise ValueError("unknown active request")
        del self.active[key]

    def cancel(self, key):
        if key in self.active:
            raise ValueError("active cancellation requires kill/reap acknowledgement before finish")
        self.pending.pop(key, None)


class BoundedCache:
    def __init__(self, max_entries, max_bytes):
        positive(max_entries, zero=True); positive(max_bytes, zero=True)
        if bool(max_entries) != bool(max_bytes): raise ValueError("disable both limits")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.entries, self.reservations = {}, {}
        self.clock = self.evictions = self.bypasses = self.hits = self.misses = 0

    @property
    def used_bytes(self): return sum(x[0] for x in self.entries.values()) + self.reserved_bytes
    @property
    def reserved_bytes(self): return sum(self.reservations.values())

    def reserve(self, key, size):
        """Size includes canonical key + result/artifact + temporary-write charge."""
        positive(size)
        if not isinstance(key, str) or not key or key in self.reservations:
            raise ValueError("invalid/duplicate reservation")
        if not self.max_entries or size > self.max_bytes:
            self.bypasses += 1
            return False
        if (len(self.reservations) >= self.max_entries
            or self.reserved_bytes + size > self.max_bytes):
            self.bypasses += 1
            return False
        self.entries.pop(key, None)
        while self.used_bytes + size > self.max_bytes or len(self.entries) + len(self.reservations) >= self.max_entries:
            victim = min(self.entries, key=lambda k: (self.entries[k][1], k))
            del self.entries[victim]; self.evictions += 1
        self.reservations[key] = size
        return True

    def publish(self, key, value):
        if key not in self.reservations: raise ValueError("publication requires reservation")
        size = self.reservations.pop(key)
        self.clock += 1
        self.entries[key] = (size, self.clock, value)

    def abort(self, key): self.reservations.pop(key, None)

    def get(self, key):
        if key not in self.entries:
            self.misses += 1
            return None
        size, _, value = self.entries[key]
        self.clock += 1; self.hits += 1
        self.entries[key] = (size, self.clock, value)
        return value


class PipelineCapacity:
    """One role-wide ledger. Output is reserved before dispatch, held until commit."""
    def __init__(self, *, workers=1, pending=8, pending_bytes=32<<20,
                 active_bytes=16<<20, results=8, result_bytes=32<<20):
        if workers not in (1,2,4):
            raise ValueError('workers must be 1/2/4')
        for n in (pending,pending_bytes,active_bytes,results,result_bytes):
            positive(n)
        self.workers=workers;self.max_pending=pending;self.max_pending_bytes=pending_bytes
        self.max_active_bytes=active_bytes;self.max_results=results;self.max_result_bytes=result_bytes
        self.pending=OrderedDict();self.active={};self.results={};self.seen=set()

    @property
    def pending_bytes(self):
        return sum(r['input'] for r in self.pending.values())

    def enqueue(self,key,size,output,*,prompt,deadline,now):
        positive(size);positive(output)
        if len(self.seen)>=65536:
            raise ValueError('role logical job evidence capacity exhausted')
        if key in self.seen or type(prompt) is not int or prompt<0:
            raise ValueError('duplicate job or invalid stable affinity')
        if not math.isfinite(deadline) or not math.isfinite(now) or now>=deadline:
            return False
        held=[*self.pending.values(),*self.active.values(),*self.results.values()]
        if (len(self.pending)>=self.max_pending or self.pending_bytes+size>self.max_pending_bytes
            or size>self.max_active_bytes or len(held)>=self.max_results
            or sum(r['output'] for r in held)+output>self.max_result_bytes):
            return False
        self.pending[key]=dict(input=size,output=output,slot=prompt%self.workers,deadline=deadline,enqueued=now)
        self.seen.add(key);return True

    def dispatch(self,slot,*,now):
        if any(r['slot']==slot for r in self.active.values()):
            return None
        for key,row in self.pending.items():
            if row['slot']!=slot:
                continue
            if now>=row['deadline']:
                return None  # Owner must terminally cancel/charge the expired head.
            if sum(r['input'] for r in self.active.values())+row['input']>self.max_active_bytes:
                return None
            self.active[key]=self.pending.pop(key)
            self.active[key]['dispatched']=now
            return key
        return None

    def finish(self,key,*,acknowledged):
        if acknowledged is not True:
            raise ValueError('release requires positive termination acknowledgement')
        self.results[key]=self.active.pop(key)

    def commit(self,key):
        self.results.pop(key)

    def cancel_pending(self,key):
        self.pending.pop(key,None)
