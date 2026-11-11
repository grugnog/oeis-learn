"""Original retained data, not metadata assertions, authorizes research assembly."""
import json

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal


def originals(root):
    from pathlib import Path
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.state import snapshot, STATE_FIELDS
    from oeis_learn.generalization.budgets import Ledger
    manifest = json.loads((Path(__file__).parents[2] /
        'specs/013-progressive-mathematical-discovery/validation/research-input-resolution.json').read_text())
    def put(path, value):
        atomic_write(root / path, value)
        return reference(root, path)
    recipe = resolve(dict(version='bigint-bootstrap/v1'))
    manifest['retained']['recipe'] = put('recipe.json', recipe)
    manifest['retained']['model'] = recipe['model']
    for arm in ('B', 'D'):
        folder = root / arm
        folder.mkdir()
        (folder / 'tensor.pt').write_bytes(b'bound but not deserialized by registration')
        tensor = reference(folder, 'tensor.pt')
        states = {key: tensor for key in STATE_FIELDS - {'rng', 'data'}}
        states.update(rng=dict.fromkeys(('python', 'numpy', 'torch', 'device'), tensor),
                      data=dict(cursor=1, pending_batch=None))
        compatibility = 'sha256:' + '1'*64
        state = snapshot(states, contract=compatibility, ledger=Ledger(100, consumed=10, updates=1))
        atomic_write(folder / 'state.json', state)
        descriptor = seal(dict(version='generalization-checkpoint/v1', kind='full_state',
            weights=tensor, state=reference(folder, 'state.json'), initialization='sha256:'+'2'*64,
            parent=None, tokens=10, updates=1, pins=['latest'], compatibility=compatibility))
        manifest['retained']['checkpoints'][arm] = dict(root=arm, descriptor=descriptor,
            roster=put(arm + '/retained.json', seal(dict(version='generalization-retained/v1', checkpoints=[descriptor]))),
            producer=put(arm + '/result.json', seal(dict(status='COMPLETE'))))
    row = dict(id='seen', indices=list(map(str, range(100))), values=list(map(str, range(100))),
               source='(program (return $index))')
    for key in ('original_plan', 'workflow', 'inventory'):
        manifest['prior_inputs'][key] = put('prior/' + key + '.json', {})
    for key in ('probes', 'observed'):
        manifest['prior_inputs'][key] = put('prior/' + key + '.json', [row])
    for item in manifest['prior_inputs']['originals']:
        item['count'] = 1
        item['ref'] = put('prior/' + item['name'] + '.json', [row])
    for key in ('plan', 'result', 'publication'):
        manifest['fresh_002'][key] = put('fresh/' + key + '.json', {})
    manifest['fresh_002']['targets'] = put('fresh/targets.json', [row])
    manifest['fresh_002']['roster'] = put('fresh/roster.json', {'seen': {}})
    manifest['fresh_002']['task_count'] = 1
    return seal(manifest)


def test_streaming_reference_check_rejects_mutation_and_alias(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'original tensor bytes')
    ref = reference(tmp_path, path)
    assert checked_file(tmp_path, ref) == path
    path.write_bytes(b'changed tensor bytes!')
    with pytest.raises(ValueError, match='identity'):
        checked_file(tmp_path, ref)
    path.unlink()
    (tmp_path / 'elsewhere').write_bytes(b'original tensor bytes')
    path.symlink_to(tmp_path / 'elsewhere')
    with pytest.raises(ValueError):
        checked_file(tmp_path, ref)


def test_registration_checks_all_inputs_and_preserves_first_failures(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import audit
    manifest = json.loads((__import__('pathlib').Path(__file__).parents[2] /
        'specs/013-progressive-mathematical-discovery/validation/research-input-resolution.json').read_text())
    result = audit(tmp_path, manifest, tmp_path / 'out')
    assert result['status'] == 'BLOCKED'
    assert result['selected_arm'] == 'D'
    assert {r['name'] for r in result['checks']} >= {'recipe', 'D.roster', 'B.roster',
        'fresh-002.targets', 'fresh-002.roster', 'prior.inventory', 'prior.probes', 'prior.observed'}
    assert len(result['deficits']) >= 25
    assert not result['resume_authorized']
    assert not result['research_ready']
    assert 'traceback' in result['checks'][0]
    assert (tmp_path / 'out/registration.json').is_file()


def test_private_exposure_index_detects_shift_and_structure_without_exporting_terms(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex
    row = dict(id='fresh-002/example', indices=list(map(str, range(100))),
               values=[str(i*i+3) for i in range(100)],
               source='(program (return (add (mul $index $index) (int 3))))')
    with ExposureIndex(tmp_path / 'private/exposure.sqlite') as index:
        index.add(row, origin='fresh-002', role='development')
        shifted = dict(row, id='renamed', values=[str(i*i+3) for i in range(1, 101)])
        assert 'window' in index.matches(shifted)
        changed = dict(row, values=[str(i*i+7) for i in range(100)],
                       source='(program (return (add (mul $index $index) (int 7))))')
        assert 'structure' in index.matches(changed)
        summary = index.summary()
    assert summary['rows'] == 1 and summary['origins'] == {'fresh-002': 1}
    assert 'values' not in json.dumps(summary)


def test_manifest_cannot_replace_selected_checkpoint_or_claim_registration(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import validate_manifest
    path = __import__('pathlib').Path(__file__).parents[2] / 'specs/013-progressive-mathematical-discovery/validation/research-input-resolution.json'
    original = json.loads(path.read_text())
    validate_manifest(original)
    for changes in ({'registration_applied': True}, {'research_ready': True}, {'executable': True}):
        with pytest.raises(ValueError):
            validate_manifest(seal(dict(original, **changes)))
    changed = json.loads(json.dumps(original))
    changed['retained']['selected_arm'] = 'B'
    with pytest.raises(ValueError):
        validate_manifest(seal(changed))


def test_complete_original_registration_and_private_exposure_import(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import audit, import_exposure
    manifest = originals(tmp_path)
    registered = audit(tmp_path, manifest, tmp_path / 'audit')
    assert registered['status'] == 'COMPLETE', registered['deficits']
    imported = import_exposure(tmp_path, manifest, registered, tmp_path / 'import')
    assert imported['status'] == 'COMPLETE', imported['failures']
    assert imported['summary']['rows'] == 19
    assert imported['summary']['origins']['fresh-002.targets'] == 1
    assert not registered['resume_authorized'] and not registered['research_ready']
    assert 'values' not in json.dumps(imported)


def test_missing_state_sidecar_blocks_but_other_arm_and_exposures_are_checked(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import audit
    manifest = originals(tmp_path)
    (tmp_path / 'D/tensor.pt').unlink()
    result = audit(tmp_path, manifest, tmp_path / 'audit')
    assert result['status'] == 'BLOCKED'
    assert any(r['name'].startswith('D.sidecar.') and r['status'] == 'BLOCKED' for r in result['checks'])
    assert any(r['name'] == 'B.weights' and r['status'] == 'PASS' for r in result['checks'])
    assert any(r['name'] == 'fresh-002.targets' and r['status'] == 'PASS' for r in result['checks'])


@pytest.mark.parametrize('writable', [False, True])
def test_custodian_file_mount_is_verified_readonly_before_cleanup(tmp_path, monkeypatch, writable):
    import runpy
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace
    from oeis_learn.container_runtime import Engine as RuntimeEngine
    stage = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/qualify_bigint_phase5.py'))['container_stage']
    source = tmp_path / 'source.json'
    source.write_text('{}')
    events = []
    class Engine(RuntimeEngine):
        def command(self, **kwargs):
            assert kwargs['mounts'][1] == (source, '/work/inputs/source.json', False)
            command = super().command(**kwargs)
            assert f'type=bind,src={source},dst=/work/inputs/source.json,readonly' in command
            return command
        def call(self, *args, **kwargs):
            events.append('inspect')
            return SimpleNamespace(stdout=json.dumps([dict(HostConfig=dict(NetworkMode='none',
                ReadonlyRootfs=True, Privileged=False), Mounts=[dict(Type='bind',
                Destination='/work/inputs/source.json', Source=str(source), RW=writable)])]).encode())
        def remove(self, name):
            events.append('remove')
    host = SimpleNamespace(engine=Engine('podman', executable='podman', details={}), policy=dict(memory_bytes=1<<30, cpus=1), gpu_lock=nullcontext)
    monkeypatch.setitem(stage.__globals__, 'launch', lambda *a, **k: None)
    if writable:
        with pytest.raises(RuntimeError, match='isolation'):
            stage(host, 'sha256:'+'a'*64, tmp_path / 'out', ['test'], readonly_inputs=[(source, '/work/inputs/source.json')])
    else:
        stage(host, 'sha256:'+'a'*64, tmp_path / 'out', ['test'], readonly_inputs=[(source, '/work/inputs/source.json')])
    assert events == ['inspect', 'remove']


def test_research_proposal_is_resolved_reference_shape_but_never_execution_authority(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import audit
    from oeis_learn.pipeline_validation.progressive_research_design import construct
    from oeis_learn.generalization.progressive.contracts import validate, load
    manifest = originals(tmp_path)
    registration = audit(tmp_path, manifest, tmp_path / 'audit')
    blueprint = construct(tmp_path, tmp_path / 'proposal', manifest, registration)
    validate('blueprint', blueprint)
    assert blueprint['purpose'] == 'research' and not blueprint['payload']['executable']
    assert blueprint['payload']['selected_stages'] == ['capability', 'diversity']
    assert blueprint['payload']['unresolved']
    selected = load(tmp_path / 'proposal/selected-profiles.json')
    assert len(selected['profiles']) == 2 and not selected['tuning_selected']
    assert all(p['model']['d_model'] == manifest['retained']['model']['d_model'] for p in selected['profiles'])
    design = load(tmp_path / 'proposal/research-design.json')
    assert design['initialization']['diversity'] == 'paired_fresh_random'
    assert design['diversity']['task_seconds'] <= 86400


def test_missing_workstation_inputs_collects_all_deficits_without_building(tmp_path, monkeypatch):
    import zipfile
    from pathlib import Path
    from oeis_learn.pipeline_validation import progressive_workstation as w
    from oeis_learn.experiments import source_snapshot
    original = Path(__file__).parents[2] / 'specs/013-progressive-mathematical-discovery/validation/research-input-resolution.json'
    (tmp_path / 'manifest.json').write_bytes(original.read_bytes())
    monkeypatch.setattr(source_snapshot, 'collect_snapshot', lambda _: {'source': 'test'})
    monkeypatch.setattr(w.Workflow, 'host', lambda _: pytest.fail('missing originals must not start engine work'))
    assert w.main(tmp_path, ['--engine','podman','--session','run-state/research',
        'register-inputs','--reference-campaign','prior','--input-manifest','manifest.json']) == 2
    with zipfile.ZipFile(tmp_path / 'run-state/research/share/register-inputs.zip') as archive:
        assert 'register-inputs/registration/registration.json' in archive.namelist()
        assert 'register-inputs/source-candidates.json' in archive.namelist()
        result = json.loads(archive.read('register-inputs/registration/registration.json'))
        assert len(result['deficits']) >= 25
        assert not result['research_ready']


def test_preparation_imports_originals_in_cpu_container_and_keeps_truth_out_of_zip(tmp_path, monkeypatch):
    import runpy
    import zipfile
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_workstation as w, supervision
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session, write
    from oeis_learn.pipeline_validation.progressive_research_inputs import import_exposure
    manifest = originals(tmp_path)
    atomic_write(tmp_path / 'manifest.json', manifest)
    s = Session(tmp_path, 'run-state/research', {'reference_campaign':'prior'}, 'register-inputs')
    args = SimpleNamespace(phase='register-inputs', input_manifest='manifest.json',
        engine='podman', memory_gib=64, cpus=8, render_node=None)
    flow = w.Workflow(s, args)
    monkeypatch.setattr(flow, 'host', lambda: SimpleNamespace())
    def launch(argv, *a, **kw):
        assert kw['seconds'] == 3650
        write(argv[-1], {'image':'sha256:'+'a'*64})
    monkeypatch.setattr(supervision, 'launch', launch)
    def stage(host, image, output, argv, **kw):
        assert not kw.get('gpu', False)
        assert len(kw['readonly_inputs']) == 20
        assert not any(str(src).endswith('.pt') for src, _ in kw['readonly_inputs'])
        value = json.loads((output / 'registration.json').read_text())
        result = import_exposure(tmp_path, manifest, value, output)
        assert result['status'] == 'COMPLETE'
    monkeypatch.setattr(runpy, 'run_path', lambda _: {'container_stage': stage})
    with s.own():
        flow.register_research()
    assert s.journal['steps']['register-inputs/import-exposure']['status'] == 'COMPLETE'
    assert s.code() == 2  # original registration cannot substitute for fresh source/panels
    with zipfile.ZipFile(s.folder / 'share/register-inputs.zip') as archive:
        assert 'register-inputs/exposure/exposure-import.json' in archive.namelist()
        assert 'register-inputs/design/research-design.json' in archive.namelist()
        assert not any('/private/' in p or p.endswith(('.sqlite','.pt')) for p in archive.namelist())
        assert not any('(program' in archive.read(p).decode(errors='ignore') for p in archive.namelist())


def test_source_inventory_reports_candidates_without_reading_private_truth(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import source_candidates
    atomic_write(tmp_path / 'data/original/manifest.json', dict(version='indexed-source/v1',
        release='immutable-original', records=[{'terms': 'PRIVATE'}]))
    atomic_write(tmp_path / 'run-state/private/manifest.json', dict(version='indexed-source/v1',
        release='not-an-approved-source', records=[]))
    result = source_candidates(tmp_path)
    assert len(result['candidates']) == 1
    assert result['candidates'][0]['artifact']['path'] == 'data/original/manifest.json'
    assert 'PRIVATE' not in json.dumps(result) and result['selected'] is None


def source_snapshot(root, *, count=3):
    entries = []
    for i in range(count):
        folder = root / 'data/source'
        folder.mkdir(parents=True, exist_ok=True)
        terms = folder / f'{i}.txt'
        terms.write_text(''.join(f'{n} {n*n+i}\n' for n in range(-2, 98)))
        atomic_write(folder / f'{i}.json', dict(offset=-2, formula='PRIVATE_FORMULA'))
        refs = [reference(folder, p) for p in (terms, folder / f'{i}.json')]
        entries.append(dict(record_id=f'OEIS:A{i:06d}',
            terms={k:refs[0][k] for k in ('path','sha256')},
            metadata={k:refs[1][k] for k in ('path','sha256')}))
    atomic_write(folder / 'manifest.json', dict(version='indexed-source/v1', release='immutable', records=entries))
    return 'data/source/manifest.json'


def test_source_projection_preserves_offsets_and_excludes_formulas(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_source import project
    path = source_snapshot(tmp_path)
    result = project(tmp_path, path, tmp_path / 'out')
    assert result['status'] == 'COMPLETE' and result['eligible'] == 3
    rows = json.loads((tmp_path / 'out/private/oeis-candidates.json').read_text())
    assert all(set(r) == {'id','indices','values'} for r in rows)
    assert all(r['indices'][0] == '-2' and r['indices'][-1] == '97' for r in rows)
    assert 'PRIVATE_FORMULA' not in json.dumps(result) + json.dumps(rows)


def test_source_projection_continues_after_corruption_and_never_replaces_member(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_source import project
    path = source_snapshot(tmp_path, count=4)
    first = project(tmp_path, path, tmp_path / 'first', limit=2)
    selected = json.loads((tmp_path / 'first/private/oeis-candidates.json').read_text())
    bad = int(selected[0]['id'].split('A')[1])
    (tmp_path / f'data/source/{bad}.txt').write_text('corrupted')
    result = project(tmp_path, path, tmp_path / 'second', limit=2)
    assert result['status'] == 'BLOCKED' and result['eligible'] == 1
    assert result['failures'][0]['member'] == selected[0]['id']
    rows = json.loads((tmp_path / 'second/private/oeis-candidates.json').read_text())
    assert [r['id'] for r in rows] == [selected[1]['id']]
    assert (tmp_path / 'second/source-progress.json').is_file()


def test_source_projection_bounds_aggregate_reads(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_source import project
    path = source_snapshot(tmp_path)
    result = project(tmp_path, path, tmp_path / 'out', maximum_bytes=1)
    assert result['status'] == 'BLOCKED' and result['eligible'] == 0
    assert len(result['failures']) == 3 and result['consumed_source_bytes'] == 0


def test_registration_preview_does_not_claim_qualification_or_freeze(tmp_path, capsys):
    from oeis_learn.pipeline_validation.progressive_workstation import main
    assert main(tmp_path, ['--engine','podman','--session','run-state/research','--dry-run',
        'register-inputs','--input-manifest','manifest.json']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['sequence'][-1] == 'registration-status'
    assert 'selected-Q' not in value['sequence'] and not value['launches']


def test_registered_source_can_be_supplied_after_inventory_without_repeating_import(tmp_path, monkeypatch):
    import runpy
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_workstation as w, supervision
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session, write
    from oeis_learn.pipeline_validation.progressive_research_inputs import import_exposure
    manifest = originals(tmp_path)
    atomic_write(tmp_path / 'manifest.json', manifest)
    args = SimpleNamespace(phase='register-inputs', input_manifest='manifest.json', oeis_snapshot=None,
        engine='podman', memory_gib=64, cpus=8, render_node=None)
    calls = []
    monkeypatch.setattr(w.Workflow, 'host', lambda _: SimpleNamespace())
    def launch(argv, *a, **kw):
        calls.append('image')
        write(argv[-1], {'image':'sha256:'+'a'*64})
    monkeypatch.setattr(supervision, 'launch', launch)
    def stage(host, image, output, argv, **kw):
        calls.append('exposure')
        import_exposure(tmp_path, manifest, json.loads((output/'registration.json').read_text()), output)
    monkeypatch.setattr(runpy, 'run_path', lambda _: {'container_stage':stage})
    binding = {'reference_campaign':'prior'}
    session = Session(tmp_path, 'run-state/research', binding, args.phase)
    with session.own():
        w.Workflow(session,args).register_research()
    assert session.code() == 2
    args.oeis_snapshot = source_snapshot(tmp_path)
    session = Session(tmp_path, 'run-state/research', binding, args.phase)
    with session.own():
        w.Workflow(session,args).register_research()
    assert session.code() == 0 and calls == ['image','exposure']
    status = session.journal['steps']['register-inputs/registration-status']['value']
    assert status['status'] == 'INPUTS_REGISTERED' and status['research_ready'] is False
    assert not any('qualify' in k or 'freeze' in k for k in session.journal['steps'])
    (tmp_path / 'B/tensor.pt').write_bytes(b'changed retained original')
    session = Session(tmp_path, 'run-state/research', binding, args.phase)
    with pytest.raises(ValueError, match='completed stage output changed'):
        with session.own():
            w.Workflow(session,args).register_research()


def test_source_inventory_checks_manifest_before_entry_cap_and_each_root_independently(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import source_candidates
    for base in ('data','run-state'):
        atomic_write(tmp_path / base / 'manifest.json', dict(version='indexed-source/v1', release=base, records=[]))
        for i in range(4):
            (tmp_path / base / str(i)).touch()
    atomic_write(tmp_path / 'tmp-oeis-bulk/fetch-plan.json', dict(selected=[]))
    result = source_candidates(tmp_path, maximum_files=2)
    assert {c['release'] for c in result['candidates']} == {'data','run-state'}
    assert result['truncated']
    assert result['bulk_caches'][0]['artifact']['path'] == 'tmp-oeis-bulk/fetch-plan.json'


def test_source_inventory_prunes_bulk_terms_but_finds_sibling_snapshot(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import source_candidates
    for i in range(30):
        p=tmp_path / f'run-state/a/terms/{i}'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    atomic_write(tmp_path / 'run-state/b/sampling/indexed-source/manifest.json',
        dict(version='indexed-source/v1', release='original', records=[]))
    result = source_candidates(tmp_path, maximum_files=15)
    assert len(result['candidates']) == 1 and not result['truncated']


@pytest.mark.parametrize('damage', [None, 'index', 'receipt', 'input', 'producer', 'incomplete'])
def test_cross_source_exposure_reuse_keeps_costs_and_rejects_changed_evidence(tmp_path, damage):
    from oeis_learn.pipeline_validation import progressive_research_inputs as inputs
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session, write
    manifest = originals(tmp_path)
    old = Session(tmp_path, 'run-state/original',
                  dict(source=next(iter(inputs.EXPOSURE_COMPATIBLE_SOURCES))), 'register-inputs')
    exposure = old.folder / 'register-inputs/exposure'
    with old.own():
        registration = inputs.audit(tmp_path, manifest, old.folder / 'register-inputs/registration')
        old.step('register-originals', lambda: (registration, []))
        receipt = inputs.import_exposure(tmp_path, manifest, registration, exposure)
        old.step('import-exposure', lambda: (str(exposure.relative_to(tmp_path)),
                    [exposure / 'exposure-import.json', exposure / 'private/exposure.sqlite']))
        old.journal['steps']['register-inputs/import-exposure']['elapsed_seconds'] = 5044.2325
        if damage == 'incomplete':
            old.journal['steps']['register-inputs/import-exposure']['status'] = 'FAILED'
        old.save()
    if damage == 'index':
        with (exposure / 'private/exposure.sqlite').open('ab') as stream:
            stream.write(b'changed')
    if damage == 'receipt':
        write(exposure / 'exposure-import.json', seal(dict(receipt, status='BLOCKED')))
    if damage == 'input':
        (tmp_path / 'prior/probes.json').write_text('[]')
    if damage == 'producer':
        write(old.folder / 'session.json', seal(dict(version='progressive-workstation-session/v1',
              binding=dict(source='unreviewed'))))
    if damage:
        with pytest.raises(ValueError):
            inputs.reuse_exposure(tmp_path, 'run-state/original', manifest, registration, tmp_path / 'new')
    else:
        value, files = inputs.reuse_exposure(tmp_path, 'run-state/original', manifest,
                                            registration, tmp_path / 'new')
        result = json.loads((tmp_path / 'new/reuse.json').read_text())
        assert result['original_elapsed_seconds'] == 5044.2325
        assert result['hardware_authority_transferred'] is False
        assert result['index']['path'].startswith('run-state/original/')
        assert value == str(exposure.relative_to(tmp_path))
        assert not (tmp_path / 'new/private/exposure.sqlite').exists()
        assert exposure / 'private/exposure.sqlite' in files


def test_registration_workflow_reuses_exposure_without_build_or_import(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_research_inputs as inputs
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session, write
    manifest = originals(tmp_path)
    write(tmp_path / 'manifest.json', manifest)
    old = Session(tmp_path, 'run-state/old',
                  dict(source=next(iter(inputs.EXPOSURE_COMPATIBLE_SOURCES))), 'register-inputs')
    exposure = old.folder / 'register-inputs/exposure'
    with old.own():
        registration = inputs.audit(tmp_path, manifest, old.folder / 'register-inputs/registration')
        old.step('register-originals', lambda: (registration, []))
        inputs.import_exposure(tmp_path, manifest, registration, exposure)
        old.step('import-exposure', lambda: (str(exposure.relative_to(tmp_path)),
                    [exposure / 'exposure-import.json', exposure / 'private/exposure.sqlite']))
    monkeypatch.setattr(inputs, 'import_exposure', lambda *a, **k: pytest.fail('duplicate import'))
    monkeypatch.setattr(Workflow, 'host', lambda *a: pytest.fail('engine/image requested for data reuse'))
    args = SimpleNamespace(phase='register-inputs', input_manifest='manifest.json',
                           oeis_snapshot=None, reuse_exposure_session='run-state/old')
    new = Session(tmp_path, 'run-state/new', dict(source='current-source'), 'register-inputs')
    with new.own():
        Workflow(new, args).register_research()
    assert new.journal['steps']['register-inputs/import-exposure']['status'] == 'COMPLETE'
    assert 'register-inputs/research-custodian-image' not in new.journal['steps']
    assert new.journal['steps']['register-inputs/project-source']['status'] == 'BLOCKED'
    assert 'rehearse/qualify' not in new.journal['steps']
