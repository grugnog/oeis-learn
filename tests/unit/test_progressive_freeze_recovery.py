"""Real hashes, journals and ownership; historical Q is an explicit test boundary."""
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import reference, seal, digest
from oeis_learn.pipeline_validation.progressive_workstation_state import write, read, Session
from oeis_learn.pipeline_validation import progressive_freeze_recovery as recovery
from oeis_learn.pipeline_validation.progressive_preparation_owner import failure_proof, claim_owner, VERSION


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    root = tmp_path; folder = root/'run-state/stopped'; base = folder/'prepare-research'
    receipt = seal(dict(version='progressive-preparation-data-reuse/v1', recipient='run-state/stopped',
        current_source=digest('old source'), census=digest('census')))
    binding = dict(source=receipt['current_source'], reuse_calibration_session='run-state/calibration',
        engine='podman', memory_gib=64, cpus=8, render_node=None)
    write(folder/'session.json', seal(dict(binding=binding))); (folder/'session.lock').touch()
    write(root/'support.json', {'unchanged': True}); write(root/'support-summary.json', {'rows': 2})
    original = seal(dict(status='COMPLETE', diagnostic=False, assessment_exposed=False, deficits=[],
        support=reference(root, 'support.json'), support_summary=reference(root, 'support-summary.json'), panel='old'))
    materials = seal(dict(original, panel='corrected'))
    write(base/'materials/materials.json', original)
    write(folder/'panel-revision/materials/materials.json', materials)
    report = seal(dict(status='PANELS_READY', source=binding['source'], deficits=[],
        training_support_unchanged=True, assessment_scored=False, model_calls=0, new_draws=0,
        original_materials=reference(root, base/'materials/materials.json'),
        materials=reference(root, folder/'panel-revision/materials/materials.json')))
    write(base/'qualification/qualification.json', {'historical_boundary_only': True})
    qref = reference(root, base/'qualification/qualification.json')
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_construction_recovery.prerequisites',
        lambda *a: (dict(qualification=qref), dict(historical_training_timing_boundary_only=True)))
    seen = []
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_calibration_reuse.source_authority',
        lambda root, proof, qualification: seen.append(qualification))
    write(root/'run-state/calibration/prepare-research/construction/blueprint.json', {'test': 'original'})
    reuse = seal(dict(blueprint=reference(root, root/'run-state/calibration/prepare-research/construction/blueprint.json')))
    write(base/'reuse.json', reuse)
    prepared = seal(dict(payload=dict(deficits=[], references=[dict(name='calibration_reuse', artifact=reference(root, base/'reuse.json'))])))
    output = base/'panel-campaign/prepared'; write(output/'prepared.json', prepared)
    prepared_result = seal(dict(operation='prepare', mechanical_status='COMPLETE',
        metrics=dict(prepared=reference(output, 'prepared.json'))))
    write(output/'result.json', prepared_result)
    write(base/'panel-campaign/calibration-reuse/blueprint.json', {'original': 'revised blueprint'})
    args = dict(blueprint=str((base/'panel-campaign/calibration-reuse/blueprint.json').relative_to(root)),
        prepared=str(output.relative_to(root)), qualification=str((base/'qualification').relative_to(root)),
        output=str((base/'frozen').relative_to(root)))
    trace = 'Traceback (most recent call last):\n'
    for name, function in [('pipeline_validation/progressive_study.py', 'validate_frozen'),
                           ('pipeline_validation/progressive_study.py', 'policy'),
                           ('generalization/progressive/contracts.py', 'verify_refs')]:
        trace += f'  File "{root}/src/oeis_learn/{name}", line 28, in {function}\n'
    trace += 'ValueError: artifact inventory cap\n'
    trace_path = base/'frozen/private/host-traceback.txt'; trace_path.parent.mkdir(parents=True)
    trace_path.write_text(trace)
    result = seal(dict(operation='freeze', mechanical_status='INVALID', reason='CONTRACT_MISMATCH',
        producer_error=dict(type='ValueError', message='artifact inventory cap', producer='host',
            local_traceback=reference(base/'frozen', trace_path))))
    write(base/'frozen/result.json', result)
    from oeis_learn.pipeline_validation.progressive_workstation import command
    logs = base/'panel-campaign/logs/freeze/7'
    write(logs/'command.json', dict(argv=command(root, SimpleNamespace(**{k:binding[k] for k in ('engine','memory_gib','cpus','render_node')}), 'freeze', **args)))
    write(logs/'process.json', dict(returncode=3, outcome='COMPLETE', failure=None, cleanup_failure=None))
    write(logs/'stdout.log', result)
    enrolled = seal(dict(test_boundary_only=True))
    write(base/'panel-campaign/continuation-allowance.json', seal(dict(active=None, binding=enrolled['identity'], spent_seconds=10, seconds=100)))
    journal = seal(dict(steps={
        'panel-revision/regroup': dict(status='COMPLETE', value=report, outputs=[report['materials']]),
        'prepare-research/prepare': dict(status='COMPLETE', outputs=[reference(root, output/'result.json')]),
        'prepare-research/freeze': dict(status='FAILED', binding=seal(dict(operation='freeze', arguments=args)), history=[dict(status='BLOCKED')]),
        'prepare-research/acceptance': dict(status='BLOCKED'),
        'prepare-research/panel-campaign-enroll': dict(status='COMPLETE', value=enrolled, outputs=[])}))
    write(folder/'journal.json', journal)
    return root, folder, receipt, journal, original, seen


@pytest.mark.parametrize('fault', [None, 'science', 'active', 'old-active', 'accepted', 'campaign',
    'result', 'trace', 'stdout', 'truncated', 'cleanup', 'command', 'allowance', 'support', 'scored', 'prepared'])
def test_only_original_clean_pre_science_freeze_transfers(stopped, fault):
    root, folder, receipt, journal, original, seen = stopped
    base = folder/'prepare-research'; logs = base/'panel-campaign/logs/freeze/7'
    if fault == 'science': journal['steps']['run-research/run'] = dict(status='COMPLETE')
    if fault == 'active': journal['steps']['prepare-research/freeze']['status'] = 'RUNNING'
    if fault == 'old-active': journal['steps']['prepare-research/freeze']['history'] = [dict(status='INTERRUPTED')]
    if fault == 'accepted': journal['steps']['prepare-research/acceptance']['status'] = 'COMPLETE'
    if fault == 'campaign': write(base/'frozen/campaign.json', {})
    if fault == 'result': write(base/'frozen/result.json', {})
    if fault == 'trace': (base/'frozen/private/host-traceback.txt').write_text('changed')
    if fault == 'stdout': (logs/'stdout.log').write_text('{}')
    if fault == 'truncated': write(logs/'process.json', dict(returncode=3, outcome='COMPLETE', omitted_bytes=dict(stdout=1)))
    if fault == 'cleanup': write(logs/'process.json', dict(returncode=3, outcome='COMPLETE', cleanup_failure='alive'))
    if fault == 'command': write(logs/'command.json', dict(argv=['python', 'other']))
    if fault == 'allowance': write(base/'panel-campaign/continuation-allowance.json', seal(dict(active=7, binding='x', spent_seconds=10, seconds=100)))
    if fault == 'support': write(root/'support.json', {'changed': True})
    if fault == 'scored': journal['steps']['panel-revision/regroup']['value']['assessment_scored'] = True
    if fault == 'prepared': write(base/'panel-campaign/prepared/prepared.json', {'changed': True})
    write(folder/'journal.json', seal(journal))
    if fault:
        with pytest.raises((ValueError, OSError, KeyError)):
            failure_proof(root, receipt)
        return
    proof = failure_proof(root, receipt)
    assert proof['version'] == recovery.VERSION and proof['current_source_requalification_required']
    assert seen and proof['original_costs_retained'] and not proof['scientific_allowance_transferred']
    before = (folder/'journal.json').read_bytes()
    claim = root/'run-state/ownership.json'; write(claim, receipt)
    recipient = root/'run-state/replacement'
    next_receipt = seal(dict(receipt, recipient='run-state/replacement', current_source=digest('new source')))
    claim_owner(root, claim, next_receipt)
    assert read(claim)['receipt'] == next_receipt and (folder/'journal.json').read_bytes() == before
    with pytest.raises(ValueError, match='superseded'):
        claim_owner(root, claim, receipt)
    values = dict(materials=original); outputs = {}
    recovery.adopt_panels(root, recipient, values, outputs)
    assert values['materials']['panel'] == 'corrected'
    assert values['materials']['support'] == original['support']
    first = (recipient/'panel-data-reuse.json').read_bytes()
    recovery.adopt_panels(root, recipient, dict(materials=original), {})
    assert (recipient/'panel-data-reuse.json').read_bytes() == first
    with pytest.raises(ValueError, match='original unscored data donor'):
        recovery.adopt_panels(root, recipient, dict(materials=dict(original, panel='wrong')), {})
    # The current source binding remains explicit, and all recovery metadata is
    # included in the normal failure/success ZIP, without raw private traceback.
    s = Session(root, 'run-state/replacement', dict(source=next_receipt['current_source']), 'prepare-research')
    with s.own(): pass
    import zipfile
    with zipfile.ZipFile(recipient/'share/prepare-research.zip') as archive:
        assert 'panel-data-reuse.json' in archive.namelist()
        assert not any('/private/' in n for n in archive.namelist())


def test_fenced_freeze_successor_keeps_the_first_calibration_donor(stopped):
    from oeis_learn.pipeline_validation.progressive_calibration_reuse_workstation import fenced_original
    root, folder, receipt, journal, original, _ = stopped
    original_receipt = seal(dict(receipt, recipient='run-state/calibration'))
    previous = root/'run-state/history/original-owner.json'; write(previous, original_receipt)
    old_proof = seal(dict(version='progressive-preparation-owner-proof/v5', predecessor=original_receipt,
        blueprint=reference(root, root/'run-state/calibration/prepare-research/construction/blueprint.json'),
        original_costs_retained=True, scientific_allowance_transferred=False))
    proof_path = root/'run-state/history/completed-calibration-proof.json'; write(proof_path, old_proof)
    owner = seal(dict(version=VERSION, receipt=receipt, previous=reference(root, previous), proof=reference(root, proof_path)))
    claim = root/'run-state/ownership.json'; write(claim, owner)
    successor = seal(dict(receipt, recipient='run-state/replacement', current_source=digest('corrected host')))
    claim_owner(root, claim, successor)
    workflow = SimpleNamespace(root=root, s=SimpleNamespace(folder=root/successor['recipient']),
        args=SimpleNamespace(reuse_calibration_session='run-state/calibration'))
    assert fenced_original(workflow) == root/'run-state/calibration/prepare-research'
    workflow.args.reuse_calibration_session = 'run-state/stopped'
    with pytest.raises(ValueError, match='selected calibration donor'):
        fenced_original(workflow)


def test_panel_revision_survives_an_intervening_failed_qualification(stopped):
    root, folder, receipt, journal, original, _ = stopped
    claim=root/'run-state/ownership.json'; write(claim,receipt)
    next_receipt=seal(dict(receipt,recipient='run-state/replacement',current_source=digest('fixed source')))
    claim_owner(root,claim,next_receipt)
    # An explicit historical CPU-failure boundary adds one owner node. The
    # immutable v8 proof behind it must still supply the corrected panels.
    previous=root/'run-state/history/freeze-owner.json'; write(previous,read(claim))
    boundary=seal(dict(version='progressive-preparation-owner-proof/v1',predecessor=next_receipt))
    boundary_path=root/'run-state/history/cpu-failure.json'; write(boundary_path,boundary)
    later=root/'run-state/later'
    write(later/'preparation-owner.json',seal(dict(version=VERSION,
        receipt=seal(dict(next_receipt,recipient='run-state/later')),
        previous=reference(root,previous),proof=reference(root,boundary_path))))
    values=dict(materials=original); outputs={}
    recovery.adopt_panels(root,later,values,outputs)
    assert values['materials']['panel']=='corrected'
    recorded=read(later/'panel-data-reuse.json')
    assert recorded['revision']['training_support_unchanged']
    assert read(root/recorded['original_proof']['path'])['version']==recovery.VERSION


from tests.unit.test_progressive_preparation_reuse import donor  # noqa: E402,F401


def test_canonical_data_adoption_selects_revised_panels_after_owner_transfer(stopped, donor):
    from oeis_learn.pipeline_validation.progressive_preparation_reuse import adopt
    root, folder, _, journal, original, _ = stopped
    _, donor_folder, _, donor_steps = donor
    donor_base = donor_folder/'prepare-research'
    write(donor_base/'materials/materials.json', original)
    donor_steps['prepare-research/materials']['outputs'] = [reference(root, donor_base/'materials/materials.json')]
    write(donor_folder/'journal.json', seal(dict(steps=donor_steps)))
    first = adopt(root, str(donor_folder.relative_to(root)), folder)[2]
    session = read(folder/'session.json'); session['binding']['source'] = first['current_source']
    write(folder/'session.json', seal(session))
    row = journal['steps']['panel-revision/regroup']
    row['value'] = seal(dict(row['value'], source=first['current_source']))
    write(folder/'journal.json', seal(journal))
    recipient = root/'run-state/successor'
    values, outputs, receipt = adopt(root, str(donor_folder.relative_to(root)), recipient)
    assert values['materials']['panel'] == 'corrected'
    assert values['materials']['support'] == original['support']
    assert (recipient/'panel-data-reuse.json') in outputs['materials']
    assert read(recipient/'preparation-owner-proof.json')['version'] == recovery.VERSION
    assert adopt(root, str(donor_folder.relative_to(root)), recipient)[0] == values
    with pytest.raises(ValueError, match='superseded'):
        adopt(root, str(donor_folder.relative_to(root)), folder)
