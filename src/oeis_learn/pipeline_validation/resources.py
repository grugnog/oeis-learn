"""Measured aggregate limits, separate from active peak reservations."""
import os
from pathlib import Path
from .identity import seal, number

MIB = 1 << 20
GIB = 1 << 30


def resolve(*, physical, available, limits=(), cpus=16, quota=None, cpuset=None, profile='workstation90', lower=None):
    number(physical, positive=True)
    ceilings = [int(physical * 9 // 10)]
    ceilings.extend(int(number(v, positive=True)) for v in limits if v is not None)
    ceiling = min(ceilings) // MIB * MIB
    profiles = {'historical16': 16 * GIB, 'capacity64': 64 * GIB, 'workstation90': ceiling}
    if profile not in profiles or profiles[profile] > ceiling:
        raise ValueError('resource profile exceeds authorized/effective ceiling')
    ceiling = profiles[profile]
    if lower is not None:
        if not 0 < number(lower) <= ceiling:
            raise ValueError('invocation memory may only lower the ceiling')
        ceiling = int(lower) // MIB * MIB
    if available is not None:
        number(available)
    count = min(number(v, positive=True) for v in (16, cpus, quota, cpuset) if v is not None)
    return seal(dict(version='pipeline-resource/v1', physical_bytes=physical, available_bytes=available,
        limit_bytes=list(limits), authorized_fraction=0.90, profile=profile, ceiling_bytes=ceiling,
        cpus=count, quota=quota, cpuset=cpuset, torch_intraop=2, torch_interop=1, native_threads=1,
        native_slot_rss=GIB, native_aggregate_rss=4*GIB,
        accounting='unavailable: device/shared-page accounting requires role observations'))


def reserve(policy, peak):
    number(peak, positive=True)
    headroom = max(GIB, (int(peak) + 9) // 10)
    required = peak + headroom
    if policy['available_bytes'] is None or required > min(policy['ceiling_bytes'], policy['available_bytes']):
        raise ValueError('peak plus headroom does not fit known current capacity')
    return dict(peak_bytes=peak, headroom_bytes=headroom, reserved_bytes=required)


def observe():
    memory = {}
    try:
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    except (OSError, ValueError):
        pass
    limits, quotas, sets = [], [], []
    # Walk the effective cgroup and visible ancestors; max does not erase a parent cap.
    try:
        relative = next(line.split('::', 1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if '::' in line)
        base = Path('/sys/fs/cgroup')
        current = base / relative.lstrip('/')
        current.resolve().relative_to(base)
        for directory in [current, *current.parents]:
            if directory != base and base not in directory.parents:
                break
            try:
                raw = (directory / 'memory.max').read_text().strip()
                if raw != 'max':
                    limits.append(int(raw))
            except (OSError, ValueError):
                pass
            try:
                quota, period = (directory / 'cpu.max').read_text().split()
                if quota != 'max':
                    quotas.append(int(quota) / int(period))
            except (OSError, ValueError, ZeroDivisionError):
                pass
        sets = list(os.sched_getaffinity(0))
    except (OSError, ValueError, StopIteration, AttributeError):
        pass
    return dict(physical=memory.get('MemTotal'), available=memory.get('MemAvailable'), limits=limits,
                cpus=os.cpu_count() or 1, quota=min(quotas) if quotas else None, cpuset=len(sets) or None)
