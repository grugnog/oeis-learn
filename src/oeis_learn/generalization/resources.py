"""Hard storage ceilings and a serial acceptance reservation ledger."""
from .budgets import integer

GIB=1024**3
PREPARATION_SECONDS=24*3600  #operational work; separate from scientific learner exposure
CAPS=dict(study_bytes=64*GIB,checkpoint_bytes=32*GIB,shard_bytes=8*GIB,raw_bytes=4*GIB,sharing_bytes=512*1024**2)
# Acceptance-013 completed paired learning but exhausted post-training evaluation.
# Preserve scientific work limits; include evaluation startup, exit and recovery.
MINUTES=dict(preflight=45,qualification=120,preparation=10,cases=190,packaging=20,margin=10)
PROOF_SECONDS=180
CASE_SECONDS={'retained-eval':1500,'refresh':1500,'parameter-targets':1500,
              'replay-resume':6000,'failure-packaging':900}
SESSION_SECONDS=sum(MINUTES.values())*60


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
    remaining=SESSION_SECONDS-elapsed_seconds
    protected=MINUTES['margin']*60+(MINUTES['packaging']*60 if stage!='packaging' else 0)
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


def checkpoint_reservation(*,snapshot_bytes,weights_bytes,milestones,arms,seeds,acceptance_replay=False,selection=False):
    """Peak live upper bounds for serial learners and single channel projections.

    A retained run has initial/latest plus at most one state per milestone. One
    active resume projection and one atomic replacement coexist with those runs.
    Proofs project only each latest state plus initial weights, not entire stores.
    Acceptance replay measures policies without publishing unused selected copies.
    """
    for value in (snapshot_bytes,weights_bytes,arms,seeds):integer(value,1)
    integer(milestones)
    retained=milestones+2
    learners=arms*seeds+(2*seeds if acceptance_replay else 0)
    phases=dict(training=(learners+1)*retained+1,proof=learners*retained+2)
    if selection and not acceptance_replay:
        phases['selection']=(learners+2)*retained+2
    # Origins, live initializers, evaluation inputs and proof initial weights.
    weight_copies=learners+6
    peak=max(phases.values())
    return dict(required_checkpoint_bytes=peak*snapshot_bytes+weight_copies*weights_bytes,
                pinned_generations_reserved=peak,retained_per_run=retained,learner_runs=learners,
                phase_generations=phases,weight_copies_reserved=weight_copies,
                policy='serial-checkpoint-graph/v1')
