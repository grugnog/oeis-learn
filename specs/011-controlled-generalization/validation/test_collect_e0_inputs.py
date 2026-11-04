import importlib.util
import json
from pathlib import Path
import zipfile

import pytest
from oeis_learn.pipeline_validation.identity import seal, write
from oeis_learn.pipeline_validation.generalization_preparation import reference


def helper():
    spec = importlib.util.spec_from_file_location('collect_e0', Path(__file__).with_name('collect_e0_inputs.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root, monkeypatch):
    h = helper()
    imported = root / 'imported'
    source = root / 'old/inputs'
    for name in h.METADATA:
        write(imported / name, {})
    write(source / 'prompts.json', [{'id': 'historical'}])
    write(source / 'private/truth.json', [{'id': 'historical', 'values': ['1']}])
    prepared = seal(dict(files={n: reference(source / n, source)
                               for n in ('prompts.json', 'private/truth.json')}))
    write(source / 'prepared.json', prepared)
    write(imported / 'provenance/prepared.json', prepared, exclusive=False)
    descriptors = [seal({'label': label}) for label in ('best', 'latest')]
    for label, descriptor in zip(('best', 'latest'), descriptors):
        write(imported / (label + '-checkpoint.json'), descriptor, exclusive=False)
    write(imported / 'import.json', seal(dict(version='generalization-legacy-roster/v1', checkpoints=descriptors)), exclusive=False)
    # These tests isolate discovery/containment. Existing legacy tests cover the
    # canonical validator; the real uploaded metadata is also audited separately.
    import oeis_learn.generalization.legacy as legacy
    called = []
    monkeypatch.setattr(legacy, 'verify_legacy', lambda r, d: called.append(d))
    (source / 'private/sealed-final.json').write_text('never export')
    (source / 'private/weights.pt').write_bytes(b'never export')
    return h, source, called


def test_exact_pair_and_metadata_only_export(tmp_path, monkeypatch):
    h, source, called = fixture(tmp_path, monkeypatch)
    result = h.collect(tmp_path, 'imported', 'out.zip')
    assert result['source_root'] == 'old/inputs' and len(called) == 2
    with zipfile.ZipFile(tmp_path / 'out.zip') as z:
        assert set(z.namelist()) == set(h.METADATA) | {'historical/prompts.json', 'historical/truth.json', 'discovery.json'}
        assert z.read('historical/truth.json') == (source / 'private/truth.json').read_bytes()
    before = (tmp_path / 'out.zip').read_bytes()
    with pytest.raises(ValueError, match='fresh'):
        h.collect(tmp_path, 'imported', 'out.zip')
    assert (tmp_path / 'out.zip').read_bytes() == before


@pytest.mark.parametrize('name', ['prompts.json', 'private/truth.json', 'prepared.json'])
def test_corrupted_input_triple_never_exports(tmp_path, monkeypatch, name):
    h, source, _ = fixture(tmp_path, monkeypatch)
    (source / name).write_text('{}')
    with pytest.raises(ValueError, match='not found'):
        h.collect(tmp_path, 'imported', 'out.zip')
    assert not (tmp_path / 'out.zip').exists()


def test_search_bounds_and_symlinks(tmp_path, monkeypatch):
    h, source, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='bound'):
        h.collect(tmp_path, 'imported', 'out.zip', max_entries=0)
    (tmp_path / 'alias').symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        h.collect(tmp_path, 'imported', 'out.zip', search_roots=['alias'])
    assert not (tmp_path / 'out.zip').exists()
