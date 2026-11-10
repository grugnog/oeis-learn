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
    stage = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/qualify_bigint_phase5.py'))['container_stage']
    source = tmp_path / 'source.json'
    source.write_text('{}')
    events = []
    class Engine:
        name = 'podman'
        def command(self, **kwargs):
            assert kwargs['mounts'][1] == (source, '/work/inputs/source.json', False)
            return ['boundary-only']
        def call(self, *args, **kwargs):
            events.append('inspect')
            return SimpleNamespace(stdout=json.dumps([dict(HostConfig=dict(NetworkMode='none',
                ReadonlyRootfs=True, Privileged=False), Mounts=[dict(Type='bind',
                Destination='/work/inputs/source.json', Source=str(source), RW=writable)])]).encode())
        def remove(self, name):
            events.append('remove')
    host = SimpleNamespace(engine=Engine(), policy=dict(memory_bytes=1<<30, cpus=1), gpu_lock=nullcontext)
    monkeypatch.setitem(stage.__globals__, 'launch', lambda *a, **k: None)
    if writable:
        with pytest.raises(RuntimeError, match='isolation'):
            stage(host, 'image', tmp_path / 'out', ['test'], readonly_inputs=[(source, '/work/inputs/source.json')])
    else:
        stage(host, 'image', tmp_path / 'out', ['test'], readonly_inputs=[(source, '/work/inputs/source.json')])
    assert events == ['inspect', 'remove']
