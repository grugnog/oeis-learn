"""Fail-closed engineering authority, finite hierarchical reservations."""
from .identity import validate, number

FIELDS = {'version', 'scope', 'phases', 'stage_registry_version', 'authority', 'source', 'input_refs',
          'weights_ref', 'engines', 'resource_policy', 'workloads', 'comparisons', 'budgets', 'reuse_refs', 'excluded_authorities'}
DENIED = ['training', 'resume', 'T096_retry', 'assisted_data', 'final']


def parse(record):
    version=record.get('version') if isinstance(record,dict) else None
    variants={
        'pipeline-pilot-validation/v1':('engineering-pilot','pipeline-pilot-registry/v1',['learning-pilot-preflight/v1'],[]),
        'pipeline-soak-validation/v1':('engineering-soak','pipeline-soak-registry/v1',['full-partition-soak/v1'],['training-batch8-16-32/v1']),
        'pipeline-validation/v1':('engineering','pipeline-registry/v1',[],[]),
        'pipeline-replay-validation/v1':('engineering-replay','pipeline-replay-registry/v1',['cache-replay/v1'],[]),
        'pipeline-replay-comparison-validation/v1':('engineering-replay-comparison','pipeline-replay-comparison-registry/v1',['cache-replay/v1'],['cache-bookkeeping/v1']),
        'pipeline-replay-batching-validation/v1':('engineering-replay-batching','pipeline-replay-batching-registry/v1',['cache-replay-batching/v1'],['batch8-16/v1']),
        'pipeline-hardware-validation/v1':('engineering-hardware','pipeline-hardware-registry/v1',['replay-capacity128/v1'],['telemetry-overhead/v1','batch16-capacity/v1']),
        'pipeline-native-capacity-validation/v1':('engineering-native-capacity','pipeline-native-capacity-registry/v1',['native18-slot-waves/v1'],['native1-2-4/v1']),
        'pipeline-coschedule-validation/v1':('engineering-coschedule','pipeline-coschedule-registry/v1',['replay128-native144/v1'],['serial-overlap/v1','repeat10/v1']),
        'pipeline-model-sizing-validation/v1':('engineering-model-sizing','pipeline-model-sizing-registry/v1',['sizing-real32/v1'],['model8-32-100/v1']),
    }
    if version not in variants:raise ValueError('registered engineering version required')
    scope,registry,workloads,comparisons=variants[version]
    replay=version not in {'pipeline-validation/v1','pipeline-soak-validation/v1','pipeline-pilot-validation/v1'}
    validate(record, FIELDS, version)
    if record['scope'] != scope or record['stage_registry_version'] != registry:
        raise ValueError('engineering scope/registered version required')
    if record['authority'] != dict(training=False, resume=False, final=False, disposable_updates=True, operations_fixtures=True):
        raise ValueError('engineering authority cannot grant learning/resume/final access')
    if record['excluded_authorities'] != DENIED:
        raise ValueError('excluded authorities must remain explicit')
    if record['phases'] != ([1,2,4] if scope in {'engineering-coschedule','engineering-model-sizing'} else [1,3] if scope=='engineering-native-capacity' else [1,2] if replay else [1, 2, 3, 4, 5]):
        raise ValueError('registered engineering phases required')
    if record['workloads']!=workloads or record['comparisons']!=comparisons:
        raise ValueError('custom workload/comparison overrides are not registered; the static versioned adapters freeze these records')
    from oeis_learn.synthesis.state import require_hash
    if not isinstance(record['source'],dict):raise ValueError('closed source binding required')
    if record['source']:
        if set(record['source'])!={'sha256'}:raise ValueError('closed source snapshot hash required')
        require_hash(record['source']['sha256'])
    for refs in [record['input_refs'],record['reuse_refs']]:
        if not isinstance(refs,list) or len(refs)>16:raise ValueError('at most sixteen content-addressed references required')
        for ref in refs:
            if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not isinstance(ref['path'],str):raise ValueError('closed content reference required')
            require_hash(ref['sha256'])
    if record['weights_ref'] is not None:
        ref=record['weights_ref']
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not isinstance(ref['path'],str):raise ValueError('closed weights reference required')
        require_hash(ref['sha256'])
    budgets = record['budgets']
    fields = {'global_seconds', 'build_per_engine_seconds', 'phase_seconds', 'archive_seconds', 'cleanup_seconds', 'stream_bytes', 'transient_bytes', 'trace_bytes', 'zip_bytes', 'mandatory_reserve_bytes'}
    if scope=='engineering-pilot':fields.update({'free_disk_reserve_bytes','bundle_members'})
    if set(budgets) != fields or set(budgets['phase_seconds']) != {'1','2','3','4','5'}:
        raise ValueError('unknown/missing budget fields')
    for key, value in budgets.items():
        if key != 'phase_seconds':
            number(value, positive=True)
    for value in budgets['phase_seconds'].values():
        number(value, positive=True)
    caps=dict(build_per_engine_seconds=14400,archive_seconds=900,cleanup_seconds=300,
        stream_bytes=8<<20,transient_bytes=8<<30,trace_bytes=128<<20,zip_bytes=512<<20)
    if scope=='engineering-pilot':
        caps['transient_bytes']=16<<30
        caps['bundle_members']=4097
        if budgets['free_disk_reserve_bytes']<5<<30:raise ValueError('pilot requires fiveGiB free-space floor')
    if any(budgets[key]>maximum for key,maximum in caps.items()):raise ValueError('registered operational caps may only be lowered')
    phases={'1':3600,'2':10800,'3':7200,'4':18000,'5':14400}
    if scope=='engineering-pilot':phases={'1':10800,'2':1200,'3':1200,'4':28800,'5':12000}
    if any(budgets['phase_seconds'][key]>maximum for key,maximum in phases.items()):raise ValueError('owning phase caps may only be lowered')
    total = 2 * budgets['build_per_engine_seconds'] + sum(budgets['phase_seconds'].values()) + budgets['archive_seconds'] + budgets['cleanup_seconds']
    if total > budgets['global_seconds'] or budgets['global_seconds'] > 86400:
        raise ValueError('reservations must fit the parent')
    if budgets['mandatory_reserve_bytes'] < 16 << 20 or min(budgets['zip_bytes'],budgets['transient_bytes']) < budgets['mandatory_reserve_bytes']:
        raise ValueError('mandatory archive reserve required')
    if record['engines'] not in ['available', 'selected', 'both'] or record['resource_policy'] not in ['workstation90','capacity64','historical16']:
        raise ValueError('unknown engine/resource policy')
    return record
