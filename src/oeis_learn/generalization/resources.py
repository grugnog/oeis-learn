"""Hard storage ceilings and a serial acceptance reservation ledger."""
from .budgets import integer

GIB=1024**3
CAPS=dict(study_bytes=64*GIB,checkpoint_bytes=32*GIB,shard_bytes=8*GIB,raw_bytes=4*GIB,sharing_bytes=512*1024**2)
MINUTES=dict(preflight=10,qualification=120,preparation=10,cases=75,packaging=15,margin=10)


def preflight(*,study_bytes,checkpoint_bytes,shard_bytes,raw_bytes,sharing_bytes,available_bytes):
    requested=locals().copy(); integer(available_bytes)
    for key,cap in CAPS.items():
        if integer(requested[key])>cap: raise ValueError('resource ceiling: '+key)
    if checkpoint_bytes+shard_bytes+raw_bytes>study_bytes or study_bytes>available_bytes: raise ValueError('insufficient storage reservation')
    return dict(minutes=sum(MINUTES.values()),reservations=dict(MINUTES),storage=requested,research_authorized=False)


def reserve_stage(stage,*,elapsed_seconds,requested_seconds):
    if stage not in MINUTES or stage=='margin': raise ValueError('unknown acceptance stage')
    integer(elapsed_seconds); integer(requested_seconds,1)
    if requested_seconds>MINUTES[stage]*60: raise ValueError('stage reservation exceeds ceiling')
    remaining=240*60-elapsed_seconds
    protected=10*60+(15*60 if stage!='packaging' else 0)
    if requested_seconds+protected>remaining: raise ValueError('insufficient time; preserve packaging and emergency margin')
    return dict(deadline_seconds=elapsed_seconds+requested_seconds,research_authorized=False)


def log_tail(text):
    encoded='\n'.join(text.splitlines()[-100:]).encode()[-1024**2:]
    return encoded.decode('utf-8',errors='ignore')


def compute_resources(*,physical_bytes,cgroup_bytes,effective_cpus,peak_bytes,headroom_bytes,native_bytes):
    for v in (physical_bytes,cgroup_bytes,effective_cpus,peak_bytes,headroom_bytes,native_bytes): integer(v,1)
    memory=min(physical_bytes*9//10,cgroup_bytes)
    if peak_bytes+headroom_bytes+native_bytes>memory or native_bytes>4*GIB: raise ValueError('measured peak plus reservations exceeds effective memory')
    return dict(memory_bytes=memory,cpus=min(16,effective_cpus),torch_intraop=2,torch_interop=1,native_threads=1,learners=1)
