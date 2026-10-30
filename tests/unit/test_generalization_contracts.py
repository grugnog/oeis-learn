import copy
import json
from pathlib import Path
import pytest
from oeis_learn.generalization.contracts import freeze, validate, verify_artifact, training_ancestry, resolve_study

FIXTURE = Path('tests/fixtures/generalization/authority.json')

def example(name):
    return copy.deepcopy(json.loads(FIXTURE.read_text())[name])

def test_identity_and_immutable_bytes():
    raw = example('preparation')
    frozen = freeze('preparation', raw)
    raw['numeric_law'] = 'changed'
    assert frozen.record()['numeric_law'] != 'changed'
    changed = frozen.record(); changed['numeric_law'] = 'changed'
    with pytest.raises(ValueError): validate('preparation', changed)
    changed = frozen.record(); changed['hidden_terms'] = []
    with pytest.raises(ValueError): freeze('preparation', changed)

def test_confined_verified_file(tmp_path):
    import hashlib
    (tmp_path / 'ok').write_bytes(b'abc')
    ref = dict(path='ok', bytes=3, sha256='sha256:'+hashlib.sha256(b'abc').hexdigest())
    assert verify_artifact(tmp_path, ref) == b'abc'
    for update in ({'bytes': 2}, {'path': '../ok'}, {'sha256': 'sha256:'+'0'*64}):
        with pytest.raises(ValueError): verify_artifact(tmp_path, dict(ref, **update))
    (tmp_path / 'link').symlink_to(tmp_path / 'ok')
    with pytest.raises(ValueError): verify_artifact(tmp_path, dict(ref, path='link'))

def test_ancestry_and_failed_preparation():
    training_ancestry(('mechanical', 'training'))
    for origin in ('oeis', 'selection', 'assessment', 'unknown', 'named_teacher'):
        with pytest.raises(ValueError): training_ancestry(('mechanical', origin))
    prep = freeze('preparation', example('preparation'))
    with pytest.raises(ValueError, match='shortfall'):
        resolve_study(prep, example('study'), (), shortfalls=('loop_active:2',))
