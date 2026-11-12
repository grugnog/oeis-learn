"""Finite installed learner campaigns over explicitly supplied training inputs.

Diagnostic purpose is immutable. These campaigns cannot authorize research.
"""
import json
from pathlib import Path
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, record, reference, seal


def prepare(root, package, *, target_tokens, seconds=1800, arm='static'):
    from .progressive_acceptance_limits import measure, storage
    from .progressive_learner import compatibility, validate_input
    from oeis_learn.generalization.progressive.storage import preflight
    root = Path(root)
    if (type(target_tokens) is not int or target_tokens < 2 or
            type(seconds) is not int or not 60 <= seconds <= 3600 or
            arm not in {'static','refreshed'} or seal(package) != package or
            package.get('diagnostic_fault') is not None):
        raise ValueError('finite original diagnostic input required')
    root.mkdir(parents=True, exist_ok=False)
    measured = measure(package['profile'])
    limits = storage(root, measured, tasks=2)
    size = len(json.dumps(package).encode())
    limits['shard_bytes'] = max(limits['shard_bytes'], size*2)
    limits['package_bytes'] = max(limits['package_bytes'], size*2)
    limits['projection_bytes'] = max(limits['projection_bytes'], size*8)
    limits['required_bytes'] = sum(limits[k] for k in (
        'task_store_bytes','projection_bytes','journal_bytes','shard_bytes','package_bytes'))
    preflight(limits['required_bytes'], limits['required_inodes'])
    atomic_write(root/'training-input.json', package)
    channel = seal(dict(version='discovery-channel/v1',role='learner',kind='training_request',
        data=dict(membership='training',artifact=reference(root,'training-input.json'),
                  compatibility=compatibility(package),profile='bigint-body/v1')))
    atomic_write(root/'training-channel.json', channel)
    maximum = max(len(r['tokens']) for shard in package['shards'] for r in shard) * package['update_policy']['effective_batch']
    def caps(tokens, elapsed=seconds):
        return dict(tokens=tokens, draws=0, rejections=0, attempts=0, native_work=0,
                    elapsed_us=elapsed*1000000,disk_bytes=measured['task_bytes'],archive_bytes=0)
    seed = package['profile']['model']['seed']
    plans=[]; references=[]
    for i, target in enumerate((target_tokens//2, target_tokens)):
        name=('first','second')[i]
        initializer=dict(mode='random' if i==0 else 'full_state', artifact=None,
                         compatibility=compatibility(package))
        if i: initializer['parent_slot']='first'
        request=dict(operation='train',artifacts={'training':['training-channel.json']},
                     image=digest('diagnostic-container-recorded-by-outer-owner'),seconds=seconds,
                     reservation=caps(0)|{'disk_bytes':0})
        path=f'request-{i}.json';atomic_write(root/path,request)
        plans.append(dict(id=name,stage='capability' if i==0 else 'diversity',kind='training',
            arm=arm,seed=seed,initializer=initializer,token_target=target,
            caps=caps(target_tokens+2*maximum),startup_us=60_000_000,finalization_us=60_000_000,
            policy=reference(root,path)['sha256'],slot=name,output=name))
        references.append(dict(name=path,role='custodian',kind='policy',artifact=reference(root,path)))
    metadata=dict(created_at='2026-11-12T00:00:00Z',purpose='acceptance',track='synthetic_strict')
    exposure=record('exposure',dict(previous=None,events=[],
        detector_limits=['DIAGNOSTIC_ONLY original gradient support; no scientific inference']),**metadata)
    atomic_write(root/'exposure.json',exposure)
    references.extend([
        dict(name='original-exposure',role='custodian',kind='groups',artifact=reference(root,'exposure.json')),
        dict(name='training',role='learner',kind='prepared',artifact=reference(root,'training-channel.json'))])
    graph=dict(selected_stages=['capability','diversity'],nodes=[dict(id=p['id'],stage=p['stage'],
        kind='routine',dependencies=[] if i==0 else ['first'],maximum_visits=1,caps=p['caps'])
        for i,p in enumerate(plans)],edges=[dict(source='first',target='second',condition='COMPLETE',
        action='ADVANCE_REGISTERED')],diagnostic_pool=caps(0,0)|{'disk_bytes':0})
    campaign=record('campaign',dict(blueprint=digest('installed-learner-timing/v1'),
        runtime=digest('DIAGNOSTIC_ONLY_NOT_QUALIFICATION'),graph=graph,tasks=plans,references=references,
        caps=caps(sum(p['caps']['tokens'] for p in plans),seconds*2)|{'disk_bytes':limits['required_bytes']},
        storage=limits,target_policy=digest('diagnostic-no-scientific-authority')),**metadata)
    tasks=[record('task',dict(p,campaign=campaign['identity']),**metadata) for p in plans]
    for task in tasks:validate_input(package,task)
    atomic_write(root/'campaign.json',campaign)
    return campaign,tasks,channel
