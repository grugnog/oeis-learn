"""Training-only trajectory measurements for prospective whole-task feasibility.

Exact-input calibration still owns the complete-iteration safety margin. This
separate experiment measures mixed-length throughput, with no scoring panels.
"""
import math
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import digest, reference, seal
from .preparation_recovery import retain
from .progressive_research_inputs import checked_file, _json
from .progressive_runtime_timing import (
    FEASIBILITY_VERSION, FEASIBILITY_SECONDS, TARGET, LEARN_SECONDS,
    CALIBRATION_SECONDS, feasibility_scopes, validate_request, case_package,
)
from .progressive_workstation_state import read


def training_scope(package):
    """Rows and probes must match; the finite setting variants may differ."""
    from oeis_learn.generalization.progressive.shared_support import FrozenList
    rows=package['shards']
    identity=getattr(rows,'_trajectory_identity',None) if isinstance(rows,FrozenList) else None
    if identity is None:
        identity=digest(rows)
        if isinstance(rows,FrozenList):object.__setattr__(rows,'_trajectory_identity',identity)
    return digest({k: identity if k=='shards' else v for k, v in package.items() if k not in {
        'identity', 'profile', 'update_policy', 'refresh_by_batch', 'refresh_updates',
        'scope', 'diagnostic_fault'}})


def plan(root, output, cases, qualification):
    from .progressive_handoff import original_authority
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    output = Path(output)
    package = cases[0]['package']
    if any(training_scope(c['package']) != training_scope(package) for c in cases):
        raise ValueError('trajectory timing requires the same original training support and probes')
    q, _ = original_authority(root, qualification)
    retain(output/'private/training-input.json', package)
    inputs = seal(dict(qualification=qualification, training_scope=training_scope(package),
                       cases=sorted(c['compatibility'] for c in cases)))
    retain(output/'trajectory-inputs.json', inputs)
    req = seal(dict(version=FEASIBILITY_VERSION, source=digest(collect_snapshot(root)),
        image=q['images'][1], inputs=inputs['identity'], original_inputs=inputs['training_scope'],
        package=reference(output, output/'private/training-input.json'), cases=feasibility_scopes(),
        target_tokens=TARGET, learner_seconds=LEARN_SECONDS, calibration_seconds=CALIBRATION_SECONDS,
        seconds=FEASIBILITY_SECONDS, scope='DIAGNOSTIC_ONLY; no scientific selection or qualification'))
    validate_request(req)
    retain(output/'request.json', req)
    return req


def clean_stage(folder):
    process = _json(Path(folder)/'process.json')
    cleanup = _json(Path(folder)/'cleanup.json')
    if (process.get('returncode') != 0 or process.get('failure') or process.get('cleanup_failure')
            or cleanup.get('acknowledged') is not True or cleanup.get('within_bound') is not True
            or cleanup.get('failure')):
        raise ValueError('successful trajectory process and acknowledged cleanup required')


def replay(root, folder, *, cases, qualification):
    """Recompute projections from original installed receipts, never summary labels."""
    from .progressive_handoff import original_authority, validate_device_proofs
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    folder = Path(folder)
    req = validate_request(read(folder/'request.json'))
    inputs = read(folder/'trajectory-inputs.json')
    package = _json(checked_file(folder, req['package']), maximum=256 << 20)
    q, qroot = original_authority(root, qualification)
    source = digest(collect_snapshot(root))
    if (req['version'] != FEASIBILITY_VERSION or req['source'] != source or req['image'] != q['images'][1]
            or inputs['qualification'] != qualification or inputs['identity'] != req['inputs']
            or inputs['cases'] != sorted(c['compatibility'] for c in cases)
            or req['original_inputs'] != training_scope(package)
            or inputs['training_scope'] != training_scope(package)
            or any(training_scope(c['package']) != training_scope(package) for c in cases)):
        raise ValueError('current exact trajectory scope required')
    matrix, _ = validate_device_proofs(qroot, q, source=source, image=req['image'],
                                      engine=q['engine_security']['engine'])
    if _json(folder/'hardware.json') != matrix['hardware']:
        raise ValueError('trajectory hardware differs from qualification')
    clean_stage(folder)
    batch = read(folder/'runtime-cases.json')
    if (batch['request'] != req['identity'] or batch['status'] != 'PASS' or batch['device_block']
            or [r['index'] for r in batch['cases']] != list(range(len(req['cases'])))):
        raise ValueError('complete finite trajectory inventory required')
    rows = []
    for case, item in zip(req['cases'], batch['cases']):
        p = read(checked_file(folder, item['producer']))
        checked_file(folder, item['diagnostics'])
        if (item['status'] != 'PASS' or p['status'] != 'PASS' or p['request'] != req['identity']
                or p['case'] != case or p['device'] != 'actual_HIP'
                or p['package'] != case_package(package, case)['identity']
                or len(p['segments']) != 2
                or p['segments'][1]['result']['metrics']['initialization'] != 'resume'):
            raise ValueError('original successful canonical trajectory required')
        segments = p['segments']
        if any(s['result']['mechanical_status'] != 'COMPLETE' for s in segments):
            raise ValueError('complete trajectory segments required')
        windows = [w for s in segments for w in s['timing']['windows'] if not w['cold']]
        tokens = sum(w['tokens'] for w in windows)
        elapsed = sum(w['elapsed_us'] for w in windows)
        if not windows or any(w['tokens'] <= 0 or w['elapsed_us'] <= 0 for w in windows):
            raise ValueError('positive steady training windows required')
        snapshots = [s['timing']['components']['checkpoint_save'] for s in segments]
        if any(s['calls'] <= 0 or s['inclusive_us'] <= 0 for s in snapshots):
            raise ValueError('actual checkpoint publications required')
        if case['arm'] == 'refreshed' and max(s['timing']['maximum_shard'] for s in segments) < 1:
            raise ValueError('refreshed trajectory must consume new support')
        rows.append(dict(case=case, measured_tokens=tokens, measured_steady_us=elapsed,
            fixed_us=max(0, math.ceil(sum(s['process']['elapsed_us']-s['timing']['loop_us'] for s in segments)/2)),
            checkpoint_us=max(math.ceil(s['inclusive_us']/s['calls']) for s in snapshots)))
    return seal(dict(version=FEASIBILITY_VERSION, request=reference(root, folder/'request.json'),
        inputs=reference(root, folder/'trajectory-inputs.json'), batch=reference(root, folder/'runtime-cases.json'),
        cases=rows, status='MEASURED', interpretation='sampled training trajectory; 4x feasibility margin applied separately'))


def cost_context(cases, producers, trajectory):
    baseline = {}
    for c, p in zip(cases, producers):
        package = c['package']; u = package['update_policy']
        if (u['dropout'], u['lr_multiplier'], u['weight_decay'], u['schedule']) == (0, 1, .01, 'constant'):
            key = (package['profile']['shape'], u['effective_batch'], u['schedule_horizon_tokens'])
            baseline[key] = min(baseline.get(key, p['measurements']['update_us']), p['measurements']['update_us'])
    return dict(baseline=baseline, trajectory=trajectory)


def estimate(package, producer, context, *, delta, target, available, arm):
    """A sampled whole trajectory and an explicit guard, not a statistical bound."""
    from oeis_learn.generalization.progressive.milestones import MILESTONES
    from .generalization_persistence import policy as checkpoint_policy, INTERVAL
    from .research_limits import padded_us
    if checkpoint_policy(package) != 'hourly/v1':
        raise ValueError('trajectory feasibility requires hourly checkpoint policy')
    shape = package['profile']['shape']
    batch = package['update_policy']['effective_batch']
    rows = context['trajectory']['cases']
    def row(sh, ba, ar):
        matches = [r for r in rows if r['case'] == dict(id=f'{sh}-b{ba}-{ar}', kind='learner', shape=sh, batch=ba, arm=ar)]
        if len(matches) != 1:
            raise ValueError('measured model/batch trajectory required')
        return matches[0]
    selected = row(shape, batch, 'static')
    # Use the measured baseline calibration to account conservatively for any
    # setting-dependent overhead. Never grant a speedup from a noisy short case.
    baseline = context['baseline'].get((shape, batch, package['update_policy']['schedule_horizon_tokens']))
    if baseline is None:
        raise ValueError('matching exact baseline calibration required for trajectory')
    ratio = max(1., producer['measurements']['update_us']/baseline)
    refresh = 1.
    if arm == 'refreshed':
        a, b = row('reference', 16, 'static'), row('reference', 16, 'refreshed')
        refresh = max(1., (b['measured_steady_us']/b['measured_tokens'])/(a['measured_steady_us']/a['measured_tokens']))
    steady = math.ceil(delta*selected['measured_steady_us']/selected['measured_tokens']*ratio*refresh)
    checkpoint = max(selected['checkpoint_us'], producer['measurements']['checkpoint_us'])
    milestones = sum(target-delta < n <= target for n in MILESTONES)
    expected_snapshots = 2+steady//(INTERVAL*1000000)+milestones
    maximum_snapshots = 2+max(0, available)//(INTERVAL*1000000)+milestones
    # Steady windows already contain the periodic pair of probes. Do not charge
    # probe_us twice, or add the same probe work again on top of the trajectory.
    expected = steady+selected['fixed_us']+expected_snapshots*checkpoint
    guarded = padded_us(steady+selected['fixed_us']+maximum_snapshots*checkpoint)
    return dict(checkpoint_policy='hourly/v1', maximum_snapshots=maximum_snapshots,
        expected_work_us=expected, estimated_work_us=guarded, available_work_us=available,
        trajectory_case=selected['case']['id'], setting_cost_ratio=ratio, refresh_cost_ratio=refresh,
        measured_steady_tokens=selected['measured_tokens'], interpretation='sampled trajectory with 4x margin; original exact iteration guard retained')
