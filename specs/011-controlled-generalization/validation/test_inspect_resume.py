"""Local saved-state forensics tests; no native/container/GPU execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('inspect_resume',HERE/'inspect_resume.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_exact_tensor_mismatch_has_statistics_without_values():
    import torch
    result=helper.differences({'layer':torch.tensor([1.,2.])},{'layer':torch.tensor([1.,2.0001])})
    assert not result['equal']
    detail=result['details'][0]
    assert detail['different_elements']==1 and 0<detail['max_abs_finite']<.001
    assert set(detail)=={'path','reason','shape','dtype','different_elements','elements','nonfinite_pairs','max_abs_finite'}


def test_nested_rng_and_replay_differences_are_bounded():
    import numpy as np
    a={'rng':np.array([1,2]),'replay':{'private source (program)':list(range(200))}}
    b={'rng':np.array([1,3]),'replay':{'private source (program)':[900]*200}}
    result=helper.differences(a,b,limit=3)
    assert result['different_leaves']==201 and len(result['details'])==3 and result['omitted_details']==198
    assert 'private source' not in json.dumps(result)


def test_nan_never_silently_passes_or_breaks_json():
    import torch
    result=helper.differences(torch.tensor([float('nan')]),torch.tensor([float('nan')]))
    assert not result['equal'] and result['details'][0]['nonfinite_pairs']==1
    json.dumps(result,allow_nan=False)


def test_aggregate_differences_include_omitted_tensor_details():
    import torch
    result=helper.differences([torch.tensor([0.]),torch.tensor([0.])],
                              [torch.tensor([1.]),torch.tensor([5.])],limit=1)
    assert result['omitted_details']==1 and result['max_tensor_abs_finite']==5.
    assert result['tensor_different_elements']==2


@pytest.mark.parametrize('engine',['docker','podman'])
def test_actual_engine_command_is_read_only_cpu_and_has_no_socket(tmp_path,monkeypatch,engine):
    from oeis_learn.container_runtime import Engine
    adapter=Engine(engine,executable=engine,details={})
    monkeypatch.setattr(adapter,'user_args',lambda:[])
    monkeypatch.setattr(adapter,'device_args',lambda *_:pytest.fail('must never mount GPU'))
    paths={n:tmp_path/n for n in ('full','paused','resumed')}
    for path in paths.values():path.mkdir()
    helpers=tmp_path/'helper';helpers.mkdir();output=tmp_path/'out';output.mkdir()
    host=SimpleNamespace(engine=adapter,policy=dict(memory_bytes=16<<30,cpus=4))
    argv=helper.command(host,'sha256:'+'a'*64,helpers,output,paths,'inspection')
    mounts=[argv[i+1] for i,v in enumerate(argv) if v=='--mount']
    assert len(mounts)==5 and all('readonly' in m for m in mounts[:-1])
    assert '--network=none' in argv and '--read-only' in argv
    assert not any('sock' in m or '/dev/' in m for m in mounts)


def fixture(tmp_path,monkeypatch):
    import sys,torch
    monkeypatch.syspath_prepend(str(HERE))
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda _:None)
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.pipeline_validation.identity import seal
    roots={}
    for name in ('full','paused','resumed'):
        root=tmp_path/name;root.mkdir();roots[name]=root
        state=dict(weights={'encoder':{'weight':torch.tensor([1.])}},optimizer={'moment':torch.tensor([2. if name!='resumed' else 2.1])},
                   rng={'python':[1],'numpy':[2],'torch':torch.tensor([3]),'device':[]},
                   data=dict(cursor=1,pending_batch=None),replay={'cursor':4},
                   scheduler={'learner_seconds':10.},history={},pins={},selection={})
        store=GeneralizationCheckpointStore(root/'checkpoints','contract',100)
        ref=store.save(state,Ledger(100,consumed=10,updates=1))
        (root/'training.json').write_text(json.dumps({'checkpoint':ref,'compatibility':'contract'}))
        record=json.loads((store.root/ref['path']).read_text())
        (store.root/'retained.json').write_text(json.dumps(seal({'checkpoints':[{'pins':['initial'],'weights':record['states']['weights'],'state':ref}]})))
    return roots


def test_real_serialized_state_inspection_separates_mismatch_from_completion(tmp_path,monkeypatch):
    roots=fixture(tmp_path,monkeypatch)
    result=helper.inspect_saved(roots,tmp_path/'out')
    assert result['status']=='PASS' and not result['comparison_equal']
    assert not result['components']['optimizer']['equal']
    assert all(v['equal'] for k,v in result['components'].items() if k!='optimizer')
    assert result['initial_full_resumed']['equal']
    assert not result['feature_qualified']
    assert result['pause_boundary']['available']
    assert result['pause_boundary']['candidates'][0]['equal']


def test_missing_boundary_is_explicit_not_inferred_from_latest(tmp_path,monkeypatch):
    roots=fixture(tmp_path,monkeypatch)
    from oeis_learn.generalization.budgets import Ledger
    result=helper.pause_boundary(roots['full'],{'ledger':Ledger(100,consumed=20,updates=2)})
    assert result['available'] is False and result['candidates']==[]


def test_boundary_mismatch_is_detected_before_resumption(tmp_path,monkeypatch):
    roots=fixture(tmp_path,monkeypatch)
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    row=json.loads((roots['resumed']/'training.json').read_text())
    paused=GeneralizationCheckpointStore(roots['resumed']/'checkpoints','contract',100).load(row['checkpoint'])
    result=helper.pause_boundary(roots['full'],paused)
    assert not result['candidates'][0]['equal']
    assert not result['candidates'][0]['components']['optimizer']['equal']


def test_corrupted_private_payload_is_rejected_before_comparison(tmp_path,monkeypatch):
    roots=fixture(tmp_path,monkeypatch)
    row=json.loads((roots['resumed']/'training.json').read_text())
    path=roots['resumed']/'checkpoints'/row['checkpoint']['path']
    record=json.loads(path.read_text())
    (roots['resumed']/'checkpoints'/record['states']['optimizer']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):helper.inspect_saved(roots,tmp_path/'out')


def test_helpers_remain_outside_existing_qualification_snapshot():
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    assert not any('inspect_resume' in p for p in collect_snapshot(helper.ROOT))


def test_saved_run_discovery_selects_replay_arm_and_rejects_path_escape(tmp_path):
    from oeis_learn.pipeline_validation.identity import seal
    root=tmp_path/'replay-resume/case/private/replay';root.mkdir(parents=True)
    relative='abcd/execution/role/private/learner'
    for kind in ('training','paused','resumed'):
        directory=root/'7'/kind/relative;directory.mkdir(parents=True)
        (directory/'training.json').write_text('{}')
    record=dict(runs=[dict(seed=7,paired=dict(runs=[dict(arm='verified_replay',root=relative),dict(arm='continued_sft',root='not-replay')]))])
    path=root/'replay.json';path.write_text(json.dumps(seal(record)))
    result=helper.saved_runs(tmp_path)
    assert set(result)=={'full','paused','resumed'}
    assert result['full']==root/'7/training'/relative
    record['runs'][0]['paired']['runs'][0]['root']='../../escape'
    path.write_text(json.dumps(seal(record)))
    with pytest.raises(ValueError,match='traversal'):helper.saved_runs(tmp_path)
