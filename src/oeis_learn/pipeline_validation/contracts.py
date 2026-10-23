"""Fail-closed engineering authority, finite hierarchical reservations."""
from .identity import validate, number

FIELDS = {'version', 'scope', 'phases', 'stage_registry_version', 'authority', 'source', 'input_refs',
          'weights_ref', 'engines', 'resource_policy', 'workloads', 'comparisons', 'budgets', 'reuse_refs', 'excluded_authorities'}
DENIED = ['training', 'resume', 'T096_retry', 'assisted_data', 'final']


def parse(record):
    validate(record, FIELDS, 'pipeline-validation/v1')
    if record['scope'] != 'engineering' or record['stage_registry_version'] != 'pipeline-registry/v1':
        raise ValueError('engineering scope/registered version required')
    if record['authority'] != dict(training=False, resume=False, final=False, disposable_updates=True, operations_fixtures=True):
        raise ValueError('engineering authority cannot grant learning/resume/final access')
    if record['excluded_authorities'] != DENIED:
        raise ValueError('excluded authorities must remain explicit')
    if record['phases'] != [1, 2, 3, 4, 5]:
        raise ValueError('registered engineering phases required')
    budgets = record['budgets']
    fields = {'global_seconds', 'build_per_engine_seconds', 'phase_seconds', 'archive_seconds', 'cleanup_seconds', 'stream_bytes', 'transient_bytes', 'trace_bytes', 'zip_bytes', 'mandatory_reserve_bytes'}
    if set(budgets) != fields or set(budgets['phase_seconds']) != {'1','2','3','4','5'}:
        raise ValueError('unknown/missing budget fields')
    for key, value in budgets.items():
        if key != 'phase_seconds':
            number(value, positive=True)
    for value in budgets['phase_seconds'].values():
        number(value, positive=True)
    total = 2 * budgets['build_per_engine_seconds'] + sum(budgets['phase_seconds'].values()) + budgets['archive_seconds'] + budgets['cleanup_seconds']
    if total > budgets['global_seconds'] or budgets['global_seconds'] > 86400:
        raise ValueError('reservations must fit the parent')
    if budgets['mandatory_reserve_bytes'] < 16 << 20 or budgets['zip_bytes'] < budgets['mandatory_reserve_bytes']:
        raise ValueError('mandatory archive reserve required')
    if record['engines'] not in ['available', 'selected', 'both'] or record['resource_policy'] not in ['workstation90','capacity64','historical16']:
        raise ValueError('unknown engine/resource policy')
    return record
