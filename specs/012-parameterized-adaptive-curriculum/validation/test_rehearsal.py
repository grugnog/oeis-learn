"""Temporary harness stop/resume behavior with independent producer fixtures."""
from pathlib import Path
import importlib.util
import pytest
from oeis_learn.pipeline_validation.identity import digest,write
from oeis_learn.pipeline_validation.generalization_preparation import reference


def helper():
    spec=importlib.util.spec_from_file_location('r0_serial',Path(__file__).with_name('run.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_rehearsal_is_explicit_before_any_frozen_research_stage():
    h=helper();order=h.stage_order(research=True,arms=['B','C','D'],rehearsal=True)
    assert order[:6]==['admission','qualification','calibration','acceptance','rehearsal','baseline']
    assert 'rehearsal' not in h.stage_order()
    with pytest.raises(ValueError):h.stage_order(arms=['B','C'],rehearsal=True)


def test_stop_then_resume_reuses_all_gates_without_research_initialization(tmp_path,monkeypatch):
    h=helper();arms=['B','C','D'];order=h.stage_order(research=True,arms=arms,rehearsal=True)
    stages=[dict(name=n,operation='prepare',manifest='none',seconds=3600,extra=[],requires=[] if i==0 else [order[i-1]]) for i,n in enumerate(order)]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('s'),images=[digest('cpu'),digest('gpu')],research=True,arms=arms,rehearsal=True)
    calls=[]
    def run(stage,target):
        calls.append(stage['name']);write(target/'checked.json',dict(stage=stage['name']))
        return dict(status='PASS',producer=digest(stage),artifacts=[reference(target/'checked.json',target)],
                    process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(within_bound=True,acknowledged=True))
    # Isolate scheduler semantics from the independent archive tests.
    from oeis_learn.pipeline_validation import generalization_archive as archives
    def export(root,target):
        path=target.with_suffix('.zip');path.write_bytes(b'fixture');return path
    monkeypatch.setattr(h,'recover',export)
    monkeypatch.setattr(archives,'verify_archive',lambda path:dict(software_status='PARTIAL'))
    one=h.execute_plan(plan,tmp_path/'r0',workspace=tmp_path,runner=run,stop_after_rehearsal=True)
    assert one['status']=='PARTIAL' and not one['errors'] and calls==order[:5]
    old=(tmp_path/'r0/workflow.json').read_bytes();calls.clear()
    h.execute_plan(plan,tmp_path/'r1',workspace=tmp_path,runner=run,previous=tmp_path/'r0')
    assert calls==order[5:] and (tmp_path/'r0/workflow.json').read_bytes()==old
    assert all(r['reused'] for r in h._read(tmp_path/'r1/workflow.json')['receipts'][:5])


def test_stop_without_a_rehearsal_is_rejected_before_work(tmp_path):
    h=helper();order=h.stage_order();stages=[dict(name=n,operation='prepare',manifest='none',seconds=3600,extra=[],requires=[] if not i else [order[i-1]]) for i,n in enumerate(order)]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('s'),images=[digest('cpu'),digest('gpu')])
    with pytest.raises(ValueError):h.execute_plan(plan,tmp_path/'bad',workspace=tmp_path,runner=lambda *a:None,stop_after_rehearsal=True)
    assert not (tmp_path/'bad').exists()
