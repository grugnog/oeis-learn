"""Prospective setup headroom and early retained-calibration validation."""
from oeis_learn.generalization.progressive.contracts import reference, seal
from .preparation_recovery import retain
from .progressive_research_inputs import checked_file, _json

NAME = 'preparation_headroom'
# The command wrapper protects 120 seconds for cleanup. Leave additional time
# beyond the inner qualification/acceptance limits for startup and publication.
SETUP = dict(base_qualification=18000, qualify=4200, acceptance=7800)


def reservation(root, original):
    outer = _json(checked_file(root, original))
    if (seal(outer) != outer or outer.get('version') not in {
            'progressive-research-reservation/v1', 'progressive-research-reservation/v2'}
            or outer.get('status') != 'READY'):
        raise ValueError('original ready setup reservation required')
    setup = dict(outer['setup_seconds'])
    for key, seconds in SETUP.items():
        setup[key] = max(setup[key], seconds)
    extra = sum(setup[k] - outer['setup_seconds'][k] for k in SETUP)
    qualification = outer['ancillary_seconds']['qualification_acceptance'] + extra
    total = outer['total_seconds'] + extra
    if (qualification > outer['limits_seconds']['qualification_acceptance']
            or total > outer['selected_maximum_seconds']):
        raise ValueError('prospective setup headroom exceeds original outer limits')
    return seal(dict(version='progressive-preparation-headroom/v1',
        original_reservation=original, setup_seconds=setup, additional_seconds=extra,
        reserved_qualification_seconds=qualification, reserved_total_seconds=total,
        original_costs_retained=True, scientific_budgets_unchanged=True))


def verified(root, references):
    refs = [r['artifact'] for r in references if r['name'] == NAME]
    if not refs:
        return None
    if len(refs) != 1:
        raise ValueError('one original setup headroom reservation required')
    value = _json(checked_file(root, refs[0]))
    originals = [r['artifact'] for r in references if r['name'] in {
        'research-reservation', 'research-reservation-v2'}]
    if (len(originals)!=1
            or _json(checked_file(root,originals[0]))!=_json(checked_file(root,value['original_reservation']))
            or value != reservation(root,value['original_reservation'])):
        raise ValueError('original setup headroom reservation differs')
    return value


def prepare(w):
    """Run before expensive qualification. Metadata checks never grant HIP authority."""
    from .progressive_workstation_state import write
    original = reference(w.root, w.base/'roster/research-reservation.json')
    value = reservation(w.root, original)
    donor_refs = {}
    if getattr(w.args, 'reuse_calibration_session', None):
        from .progressive_calibration_reuse_workstation import fenced_original
        from .progressive_handoff import original_authority
        from .progressive_calibration_equivalence import validate
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        donor = fenced_original(w)
        qref = reference(w.root, donor/'qualification/qualification.json')
        q, qroot = original_authority(w.root, qref)
        changed = validate(_json(checked_file(qroot, q['source_closure'])), collect_snapshot(w.root))
        donor_refs = dict(calibration_donor=str(donor.relative_to(w.root)),
            original_qualification=qref, reviewed_host_changes=changed)
    retain(w.base/'preparation-headroom.json', value)
    result = seal(dict(version='progressive-execution-preflight/v1', status='COMPLETE',
        headroom=reference(w.root, w.base/'preparation-headroom.json'),
        current_qualification_required=True, **donor_refs))
    write(w.base/'execution-preflight.json', result)
    return value, [w.base/'preparation-headroom.json', w.base/'execution-preflight.json']
