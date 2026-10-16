"""Lightweight spawn targets for bounded worker fault injection.

Keep training/torch imports out of these targets: the child is constrained to
one GiB of address space by the same production worker implementation it tests.
"""


def oom_worker(channel, cache_bytes):
    from oeis_learn.sandbox.worker_pool import _worker
    from oeis_learn.sandbox.pipeline import Runtime

    original = Runtime.evaluate

    def injected(self, source, *args, **kwargs):
        if source == "i256.zero":
            raise MemoryError("bounded injected allocator failure")
        return original(self, source, *args, **kwargs)

    Runtime.evaluate = injected
    _worker(channel, cache_bytes)
