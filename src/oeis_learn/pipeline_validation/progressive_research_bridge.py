"""Reuse verified registration data, never transfer execution/qualification authority."""
from pathlib import Path

from .identity import confined, seal
from .progressive_research_inputs import checked_file, validate_manifest, audit, reuse_exposure
from .progressive_workstation_state import read, write
from oeis_learn.generalization.progressive.contracts import reference, load

REGISTRATION_STEPS = ('source-inventory', 'project-source', 'register-originals',
                      'prospective-design', 'import-exposure', 'registration-status')


def adopt(root, session, output):
    root, output = Path(root), Path(output)
    folder = confined(root, session)
    if not folder.is_relative_to(root / 'run-state'):
        raise ValueError('registered workstation session required')
    binding, journal = read(folder / 'session.json'), read(folder / 'journal.json')
    originals = {}
    for name in REGISTRATION_STEPS:
        step = journal['steps'].get('register-inputs/' + name, {})
        if step.get('status') != 'COMPLETE' or not step.get('outputs'):
            raise ValueError('complete original registration step required: ' + name)
        for ref in step['outputs']:
            if ref['path'] in originals and originals[ref['path']] != ref:
                raise ValueError('conflicting registered original')
            originals[ref['path']] = ref
    for ref in originals.values():
        checked_file(root, ref)
    def owned(name):
        path = folder / 'register-inputs' / name
        ref = reference(root, path)
        if originals.get(ref['path']) != ref:
            raise ValueError('original journal does not bind ' + name)
        return read(path)
    status = owned('registration-status.json')
    if status.get('status') != 'INPUTS_REGISTERED' or status.get('research_ready') is not False:
        raise ValueError('data-only registration status required')
    source = owned('source/source-projection.json')
    if source.get('status') != 'COMPLETE' or source.get('failures'):
        raise ValueError('complete original source projection required')
    candidates = reference(root, checked_file(folder / 'register-inputs/source', source['candidates']))
    if originals.get(candidates['path']) != candidates:
        raise ValueError('original candidate roster not journal-bound')
    checked_file(root, source['snapshot'], maximum=128 << 20)
    manifest_ref = binding['binding']['input_manifest']
    manifest_path = confined(root, manifest_ref['path'])
    actual = reference(root, manifest_path)
    if actual['sha256'] != manifest_ref['sha256']:
        raise ValueError('registered resolution manifest changed')
    manifest = validate_manifest(load(manifest_path))
    current = audit(root, manifest, output / 'registration')
    if current['status'] != 'COMPLETE':
        raise ValueError('registered original data no longer passes current audit')
    donor = binding['binding'].get('reuse_exposure_session', session)
    _, exposure_files = reuse_exposure(root, donor, manifest, current, output / 'exposure-reuse')
    reuse = read(output / 'exposure-reuse/reuse.json')
    design = owned('design/research-design.json')
    profiles = owned('design/selected-profiles.json')
    result = seal(dict(version='progressive-registered-data/v1', status='COMPLETE',
        donor_session=reference(root, folder / 'session.json'),
        donor_journal=reference(root, folder / 'journal.json'), producer_source=binding['binding']['source'],
        registration=current['identity'], manifest=actual, source=source['snapshot'], candidates=candidates,
        exposure=reuse['index'], exposure_receipt=reuse['original_receipt'],
        recipe=manifest['retained']['recipe'], profiles=profiles, design=design,
        prior_cost_seconds=sum(s.get('elapsed_seconds', 0) for s in journal['steps'].values()),
        original_exposure_seconds=reuse['original_elapsed_seconds'],
        hardware_authority_transferred=False, research_ready=False))
    write(output / 'registered-data.json', result)
    return result, [output / 'registered-data.json', output / 'registration/registration.json',
                    *exposure_files, *[confined(root, r['path']) for r in originals.values()]]


def rehearsal(root, session, output, current_source):
    """Retain provenance and source identities; never mint a reusable PASS."""
    root, output = Path(root), Path(output)
    value = dict(version='progressive-rehearsal-reference/v1',
                 status='NOT_SUPPLIED', current_source=current_source,
                 hardware_authority_transferred=False, rerun_requested=False)
    if session:
        folder = confined(root, session)
        if not folder.is_relative_to(root / 'run-state'):
            raise ValueError('original rehearsal session required')
        binding, journal = read(folder / 'session.json'), read(folder / 'journal.json')
        files = {}
        for name in ('qualify', 'acceptance'):
            step = journal['steps'].get('rehearse/' + name, {})
            if step.get('status') != 'COMPLETE' or not step.get('outputs'):
                raise ValueError('completed original rehearsal ' + name + ' required')
            for ref in step['outputs']:
                files[ref['path']] = ref
        for ref in files.values():
            checked_file(root, ref)
        original_source = binding['binding']['source']
        value.update(status='RETAINED_FOR_SCOPE_VALIDATION' if original_source == current_source else 'HISTORICAL_SOURCE_DIFFERS',
                     original_source=original_source, session=reference(root, folder / 'session.json'),
                     journal=reference(root, folder / 'journal.json'), originals=list(files.values()),
                     scope='original evidence retained; current model/image/native authority must be separately validated')
    result = seal(value)
    write(output / 'rehearsal-reference.json', result)
    return result, [output / 'rehearsal-reference.json']


def request(data):
    """Finite native census before model scoring, calibration or task freezing."""
    if seal(data) != data or data.get('version') != 'progressive-registered-data/v1':
        raise ValueError('original registered-data receipt required')
    return seal(dict(version='progressive-research-census-request/v1', registration=data['identity'],
        candidates=data['candidates'], exposure=data['exposure'], recipe=data['recipe'],
        seed=20261112, training_draws=8192, evaluation_draws=16384, seconds=21600,
        shard_rows=512, per_case_seconds=4, required_per_stratum=128, minimum_groups=16,
        panels=['development','confirmation-diversity','assessment'],
        token_target=25000000, proposed_refresh_updates=8, max_shards=64,
        diagnostic=False, memory_bytes=64 << 30, disk_bytes=8 << 30))


def readiness(output, data, census, sensitivity=None, materials=None):
    """No passing data census can stand in for scientific or device acceptance."""
    deficits = list((materials or census).get('deficits', []))
    if not materials or materials.get('status') != 'COMPLETE':
        deficits.append('cohort_and_support_publication_not_completed')
    if not sensitivity or sensitivity.get('status') != 'COMPLETE':
        deficits.append('sensitivity_not_completed')
    deficits += ['current_source_selected_model_HIP_qualification',
                 'exact_input_update_checkpoint_restore_calibration',
                 'concrete_campaign_freeze_and_scoped_acceptance']
    value = seal(dict(version='progressive-research-readiness/v1', status='BLOCKED',
        registration=data['identity'], census=census['identity'], research_ready=False,
        sensitivity=sensitivity.get('identity') if sensitivity else None,
        materials=materials.get('identity') if materials else None,
        deficits=sorted(set(deficits)), training_started=False, assessment_exposed=False,
        stage_dispositions={
            'capability':'AWAITING_PREPARATION_AND_QUALIFICATION',
            'diversity':'AWAITING_SUPPORT_FEASIBILITY',
            'tuning':'OPTIONAL_DOMAIN_NOT_SELECTED',
            'scaling':'CONDITIONAL_ON_CONFIRMED_PREDECESSOR',
            'oeis':'CONDITIONAL_ON_ELIGIBLE_INITIALIZER_AND_SEPARATE_TRAINING_POOL',
            'components':'INDEPENDENT_ELIGIBILITY_REQUIRED',
            'source_pilot':'INDEPENDENT_ELIGIBILITY_REQUIRED',
            'discovery_review':'AWAITING_CANDIDATES',
            'assessment':'EXPLICIT_LOCKED_SELECTION_ONLY'},
        qualification='data reuse does not transfer historical HIP authority'))
    write(Path(output) / 'readiness.json', value)
    return value
