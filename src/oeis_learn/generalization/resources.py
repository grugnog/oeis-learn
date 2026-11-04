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


def _pac_arms(arms):
    if len(set(arms))!=len(arms) or not {'B','C'}<=set(arms)<=set('ABCD'):raise ValueError('complete predeclared PAC arms required')


def pac_campaign_reservation(arms,arm_reservations,*,preparation,qualification,acceptance,cleanup,finalization,existing):
    from .budgets import _seconds
    _pac_arms(arms)
    if set(arm_reservations)!=set(arms):raise ValueError('optional arm requires complete additional reservation')
    if preparation<86400 or finalization<3600:raise ValueError('preparation/finalization floor')
    total=0
    for value in (preparation,qualification,acceptance,cleanup,finalization,existing):_seconds(value)
    for arm,parts in arm_reservations.items():
        if set(parts)!={'preparation','training','diagnostics','checkpoints','generation','export'}:raise ValueError('complete arm reservation graph')
        for value in parts.values():_seconds(value,positive=True)
        if parts['training']<43200 or parts['generation']<23400:raise ValueError('learner/generation operational floor')
        total+=sum(parts.values())
    total+=preparation+qualification+acceptance+cleanup+finalization
    _seconds(total)
    return dict(version='pac-campaign-reservation/v1',seconds=max(existing,total),arms=list(arms),serial=True,
                automatic_extension=False,scientific_portfolio_seconds=60,semantic_probe_seconds=5)


def pac_diagnostic_reservation(updates,*,arm,costs,update_seconds,initial_final_overhead,existing):
    from .diagnostics import observation_forecast
    from .budgets import _seconds
    fields={'reward_probe','grade','common_loss_probe','semantic_attempt','mode_rng_restore','publication','cleanup'}
    if set(costs)!=fields:raise ValueError('complete observation cost calibration required')
    for value in costs.values():_seconds(value,positive=True)
    for value in (update_seconds,initial_final_overhead,existing):_seconds(value)
    n=observation_forecast(updates,arm=arm)
    forecast=sum(n[count]*costs[kind] for count,kind in (
        ('reward_probe_evaluations','reward_probe'),('grade_opportunities','grade'),
        ('common_loss_probes','common_loss_probe'),('semantic_attempts','semantic_attempt'),
        ('mode_rng_restorations','mode_rng_restore'),('publication_events','publication')))+costs['cleanup']
    _seconds(forecast)
    default=.2*update_seconds+initial_final_overhead
    return dict(counts=n,forecast_seconds=forecast,default_seconds=default,
                seconds=max(existing,default,3*forecast+initial_final_overhead),freeze_before_research=True,automatic_extension=False)


def pac_peak_storage(*,arms,snapshot_bytes,weights_bytes,data_bytes,data_copies,scratch_bytes,logs_bytes,package_bytes,milestones,available_bytes,scratch_kind):
    """Conservative simultaneous live graph; no original-input pruning authority."""
    _pac_arms(arms)
    for value in (snapshot_bytes,weights_bytes,data_bytes,data_copies,scratch_bytes,logs_bytes,package_bytes):integer(value,1)
    integer(milestones);integer(available_bytes)
    if scratch_kind!='disk':raise ValueError('private disk-backed scratch required')
    retained=len(arms)*(milestones+2)  # initial/latest/pins; coincident pins may deduplicate later
    common=dict(data=data_bytes*data_copies,weights=weights_bytes*(len(arms)+3),scratch=scratch_bytes,
                logs=logs_bytes,success_and_recovery_packages=2*package_bytes)
    phases={name:dict(common,snapshots=(retained+extra)*snapshot_bytes) for name,extra in (
        ('atomic_save_and_resume',2),('fresh_and_resumed_proof',2),('report_and_recovery',0))}
    required=max(sum(phase.values()) for phase in phases.values())
    if required>available_bytes:raise ValueError('insufficient peak live storage; preserve originals and complete generations')
    return dict(version='pac-peak-storage/v1',required_bytes=required,available_bytes=available_bytes,phases=phases,
                retained_generations=retained,prune_originals=False,scratch_kind='disk')
