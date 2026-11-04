"""Independent peak-live storage counterexamples; no HIP execution."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fresh_pilot_storage as m
from oeis_learn.pipeline_validation.identity import write
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.resources import CAPS


def test_measured_preparation_fits_alone_but_not_validation_copy():
    # Uploaded original preparation: 7,620,788,693 cell bytes. Required
    # cohort/training/replay/prompt/ledger payloads are about 3.8 billion.
    original=7_620_788_693;required=3_803_289_860
    assert original<CAPS['shard_bytes']<original+required
    plan=m.peak_plan([dict(path='cells/a/training.json',bytes=required)],CAPS)
    assert plan['peak_shard_bytes']==2*required
    assert plan['peak_shard_bytes']<CAPS['shard_bytes']
    with pytest.raises(ValueError,match='shard'):m.peak_plan([dict(path='cells/a/training.json',bytes=(4<<30)+1)],CAPS)


def test_other_resource_categories_remain_bounded():
    with pytest.raises(ValueError,match='study'):
        m.peak_plan([dict(path='other.json',bytes=20<<30)],CAPS)
    with pytest.raises(ValueError,match='checkpoint'):
        m.peak_plan([dict(path='checkpoints/a.pt',bytes=1)],CAPS)


def test_projection_preserves_exact_authorities_and_originals(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_runtime as runtime
    base=tmp_path/'original';write(base/'cells/a/training.json',[{'source':'unchanged'}])
    write(base/'cells/a/private/selected.json',{'large':'audit only'})
    write(base/'study.json',{'identity':'frozen-study'});write(base/'runtime.json',{'identity':'frozen-runtime'})
    refs=[reference(base/'cells/a/training.json',base)]
    monkeypatch.setattr(runtime,'load',lambda *a:(base,{},dict(preparation=None,storage=CAPS)))
    monkeypatch.setattr(runtime,'execution_refs',lambda *a:refs)
    before={p:p.read_bytes() for p in base.rglob('*') if p.is_file()}
    path,receipt=m.project(tmp_path,'original/study.json','execution/private/prepared')
    dest=tmp_path/path
    assert dest.read_bytes()==(base/'study.json').read_bytes()
    assert dest.with_name('runtime.json').read_bytes()==(base/'runtime.json').read_bytes()
    assert not (dest.parent/'cells/a/private/selected.json').exists()
    assert receipt['original_manifest']==reference(base/'study.json',tmp_path)
    assert before=={p:p.read_bytes() for p in base.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):m.project(tmp_path,'original/study.json','execution/private/prepared')
    (base/'cells/a/training.json').write_text('tampered')
    with pytest.raises(ValueError,match='changed'):m.project(tmp_path,'original/study.json','another')


def test_real_guard_counts_original_plus_copy_and_accepts_execution_view(tmp_path):
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope,guard
    from oeis_learn.pipeline_validation.runner import ResourceBreach
    limits=dict(CAPS,shard_bytes=8000)
    original=tmp_path/'original/cells/data.json';original.parent.mkdir(parents=True);original.write_bytes(b'x'*7100)
    execution=tmp_path/'execution/cells/data.json';execution.parent.mkdir(parents=True);execution.write_bytes(b'x'*3500)
    with scope(tmp_path/'original',limits,extra_roots=[tmp_path/'execution']):
        with pytest.raises(ResourceBreach,match='shard_bytes'):guard(tmp_path/'disk-fail').check()
    with scope(tmp_path/'execution',limits):guard(tmp_path/'disk-pass').check()
