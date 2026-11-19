"""Finite training timing followed by a separately published recovery blueprint."""
import copy
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import reference, seal, validate, CAMPAIGN_DOCUMENT_BYTES
from .preparation_recovery import retain, Allowance
from .progressive_research_inputs import _json, checked_file
from .progressive_workstation_state import read, Blocked


def fenced_original(w):
    """Follow original ownership proofs through failed pre-HIP qualifications."""
    from .progressive_preparation_owner import VERSION, failure_proof
    node = read(w.s.folder/'preparation-owner.json')
    local = read(w.s.folder/'preparation-owner-proof.json')
    donor = w.args.reuse_calibration_session
    if node.get('receipt', {}).get('recipient') != str(w.s.folder.relative_to(w.root)):
        raise Blocked('calibration donor ownership belongs to another recipient')
    stable = lambda r: {k:v for k,v in r.items() if k not in {'identity','recipient','current_source'}}
    for depth in range(16):
        if node.get('version') != VERSION or seal(node) != node:
            raise Blocked('original calibration ownership history required')
        proof = read(checked_file(w.root, node['proof']))
        previous = read(checked_file(w.root, node['previous']))
        predecessor = previous['receipt'] if previous.get('version') == VERSION else previous
        if (seal(proof) != proof or seal(previous) != previous
                or proof.get('predecessor') != predecessor
                or stable(node['receipt']) != stable(predecessor)
                or proof.get('original_costs_retained') is not True
                or proof.get('scientific_allowance_transferred') is not False
                or (depth == 0 and proof != local)):
            raise Blocked('original calibration ownership proof differs')
        if proof.get('version') == 'progressive-preparation-owner-proof/v5':
            if predecessor['recipient'] != donor:
                raise Blocked('selected calibration donor differs from original fenced owner')
            path = checked_file(w.root, proof['blueprint'])
            if path != Path(w.root)/donor/'prepare-research/construction/blueprint.json':
                raise Blocked('original fenced calibration blueprint differs')
            return path.parent.parent
        # Only reviewed pre-science qualification, preparation or host-freeze
        # stops may sit between the complete calibration and this recipient.
        # Replay original termination evidence; scientific/unclean work blocks.
        if (proof.get('version') not in {'progressive-preparation-owner-proof/v1','progressive-preparation-owner-proof/v6','progressive-preparation-owner-proof/v7','progressive-preparation-owner-proof/v8','progressive-preparation-owner-proof/v9'}
                or failure_proof(w.root, predecessor) != proof):
            raise Blocked('only reviewed pre-science qualification, preparation or freeze stops may retain the calibration donor')
        node = previous
    raise Blocked('bounded calibration ownership history exhausted')


def prepare(w):
    from .progressive_calibration_reuse import build_proof, raw_prepared, bind_original_reservation, NAME
    from .progressive_calibration import selected_inputs
    from . import progressive_trajectory as timing
    from .progressive_runtime_timing import FEASIBILITY_SECONDS
    from .progressive_research_resources import policy
    base = w.base; folder = base/'training-trajectory'; output = base/'calibration-reuse'
    def headroom_refs():
        path=base/'preparation-headroom.json'
        return ([dict(name='preparation_headroom',role='custodian',kind='policy',artifact=reference(w.root,path))]
                if path.is_file() else [])
    def qualification():
        return reference(w.root, base/'qualification/qualification.json')
    def request():
        original_base = fenced_original(w)
        proof = build_proof(w.root, original_base, qualification())
        retain(output/'calibration-reuse.json', proof)
        blueprint = _json(base/'construction/blueprint.json', maximum=CAMPAIGN_DOCUMENT_BYTES)
        # Check the original debit before spending time on trajectory cases.
        bind_original_reservation(w.root, blueprint, proof)
        raw=raw_prepared(w.root,blueprint)
        cases, _ = selected_inputs(w.root,raw)
        value = timing.plan(w.root, folder, cases, qualification())
        # Separate diagnostic allowance, with no scientific exposure or refunds
        # of the original 72-hour calibration reservation.
        reservation = timing.reservation(w.root,[*raw['payload']['references'],*headroom_refs()],value,research=raw['purpose']=='research')
        retain(folder/'trajectory-reservation.json', reservation)
        return value, [output/'calibration-reuse.json', folder/'request.json', folder/'trajectory-inputs.json',
                       folder/'private/training-input.json', folder/'trajectory-reservation.json']
    req = w.s.step('trajectory-plan', request, dependencies=('prepare-research/construction',))
    def execute():
        from .progressive_handoff import stage
        from .supervision import diagnostic_program
        with Allowance(folder/'trajectory-allowance.json', req['identity'], FEASIBILITY_SECONDS).attempt() as (left, _):
            body = ('from oeis_learn.pipeline_validation.progressive_runtime_timing import run_inside;'
                    f"run_inside('/work/output',{max(1,left-120)!r})")
            stage(w.host(), w.root, req['image'], folder, ['python', '-c', diagnostic_program(body, 'training-trajectory')],
                  gpu=True, seconds=left)
        value = read(folder/'runtime-cases.json')
        if value['status'] != 'PASS':
            raise Blocked('trajectory cases incomplete; retained diagnostics; no calibration rerun')
        paths = [folder/'runtime-cases.json', folder/'hardware.json', folder/'process.json', folder/'cleanup.json']
        for row in value['cases']:
            paths.extend(checked_file(folder, row[k]) for k in ('producer', 'diagnostics'))
        return value, paths
    # No silent second attempt after a device/cleanup failure. The installed
    # runner retains every completed case and its original failure information.
    w.s.step('training-trajectory', execute, dependencies=('prepare-research/trajectory-plan',))
    def revision():
        blueprint = _json(base/'construction/blueprint.json', maximum=CAMPAIGN_DOCUMENT_BYTES)
        raw = raw_prepared(w.root, blueprint)
        cases, _ = selected_inputs(w.root, raw)
        report = timing.replay(w.root, folder, cases=cases, qualification=qualification())
        retain(folder/'trajectory-report.json', report)
        retain(output/'resource-policy.json', policy(trajectory=True))
        payload = copy.deepcopy(raw['payload']); payload.pop('blueprint')
        payload['references'] = [r for r in payload['references'] if r['name'] != 'research_resource_policy']
        payload['references'].extend(headroom_refs())
        for name, kind, path in [('research_resource_policy', 'policy', output/'resource-policy.json'),
                                  (NAME, 'policy', output/'calibration-reuse.json'),
                                  ('training_trajectory', 'source', folder/'trajectory-report.json'),
                                  ('trajectory_reservation', 'policy', folder/'trajectory-reservation.json')]:
            payload['references'].append(dict(name=name, role='custodian', kind=kind, artifact=reference(w.root, path)))
        retain(output/'preparation-plan.json', dict(version='progressive-preparation-plan/v1', payload=payload))
        revised = copy.deepcopy(blueprint)
        for r in revised['payload']['references']:
            if r['name'] == 'preparation_plan': r['artifact'] = reference(w.root, output/'preparation-plan.json')
        revised = bind_original_reservation(w.root, seal(revised), read(output/'calibration-reuse.json'))
        retain(output/'blueprint.json', revised)
        return str((output/'blueprint.json').relative_to(w.root)), [folder/'trajectory-report.json', output/'resource-policy.json',
            output/'preparation-plan.json', output/'blueprint.json']
    result = w.s.step('calibration-reuse-plan', revision, dependencies=('prepare-research/training-trajectory',))
    return result or str((output/'blueprint.json').relative_to(w.root))
