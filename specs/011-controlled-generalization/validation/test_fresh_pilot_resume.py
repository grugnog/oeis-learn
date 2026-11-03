import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fresh_pilot_resume as m
from oeis_learn.pipeline_validation.identity import seal,write,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference


def test_storage_gate_reports_actual_shortfall_and_keeps_margin(tmp_path,monkeypatch):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=65664630784))
    with pytest.raises(ValueError,match='224 GiB free; available 61.2 GiB'):m.disk_preflight(tmp_path)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=224<<30))
    assert m.disk_preflight(tmp_path)['required_free_bytes']==224<<30
    from oeis_learn.generalization.resources import CAPS
    assert m.REQUIRED_FREE_BYTES-2*CAPS['study_bytes']>=CAPS['study_bytes']+(32<<30)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.REQUIRED_FREE_BYTES-1))
    with pytest.raises(ValueError):m.disk_preflight(tmp_path)


@pytest.fixture
def original(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_runtime as runtime
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.generalization import feature_acceptance
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=256<<30))
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda _:{'production':'same'})
    feature_calls=[]
    monkeypatch.setattr(feature_acceptance,'feature_artifacts',lambda *a,**kw:feature_calls.append(kw))
    old=tmp_path/'runs/old';old.mkdir(parents=True)
    design=seal(dict(version='fresh-pilot-design/v2'))
    qualified=old/'acceptance/report/qualified'
    write(qualified/'source.json',{'production':'same'});write(qualified/'producer.json',seal(dict(status='PASS')))
    images=['sha256:'+'1'*64,'sha256:'+'2'*64]
    authority=seal(dict(version='generalization-qualification/v2',images=images,
        source_closure=reference(qualified/'source.json',qualified),resources=dict(memory_bytes=64<<30,cpus=4),
        producer=dict(receipt=reference(qualified/'producer.json',qualified))))
    write(qualified/'qualification.json',authority)
    prep=old/'research/prepare';base=prep/'preparation/role/private/prepared'
    write(base/'source.json',{'production':'same'});write(qualified/'feature.json',{})
    study=seal(dict(hypothesis='protocol '+design['identity'],source_closure=reference(base/'source.json',base)))
    sidecar=seal(dict(preparation=None,purpose='research',images=images,feature=reference(qualified/'feature.json',qualified)))
    write(base/'study.json',study);write(base/'runtime.json',sidecar)
    monkeypatch.setattr(runtime,'load',lambda *_:(base,study,sidecar))
    write(old/'research/design/design.json',design)
    write(old/'research/design/blueprint.json',dict(hypothesis=study['hypothesis']))
    def rel(p):return str(p.relative_to(tmp_path))
    job=seal(dict(version='fresh-pilot-smoke-job/v1',engine='podman',qualified=rel(qualified),manifest=rel(base/'study.json'),
        design=reference(old/'research/design/design.json',tmp_path),blueprint=reference(old/'research/design/blueprint.json',tmp_path)))
    write(old/'private/smoke-job.json',job)
    cells=[]
    for i in range(16):
        rows=[dict(id=str(i))];name='cell-'+str(i)
        write(base/'cells'/name/'private/selected.json',rows)
        cells.append(dict(name=name,status='FEASIBLE',selection=digest([str(i)])))
    feasible=seal(dict(status='FEASIBLE',protocol=design['identity'],cells=cells))
    write(old/'feasibility/feasibility.json',feasible)
    write(old/'feasibility/result.json',seal(dict(status='PASS',feasibility=reference(old/'feasibility/feasibility.json',old/'feasibility'))))
    def receipt(folder,stage,result):
        if not folder.exists():write(folder/'checked.json',{})
        inventory=[reference(p,folder) for p in folder.rglob('*') if p.is_file()]
        write(folder/'result.json',seal(dict(version='generalization-command-result/v2',status='PASS',stage=stage,original=None,
            inventory=inventory,result=result)))
    receipt(prep,'prepare',dict(manifest=job['manifest']))
    receipt(old/'research/preflight','preflight',dict(status='PASS'))
    stages=[dict(stage=name,status='PASS',exit_code=0,elapsed_seconds=10,originals=[str(path/'result.json')])
        for name,path in [('feasibility',old/'feasibility'),('preflight',old/'research/preflight'),('prepare',prep)]]
    stages.extend([dict(stage='smoke',status='FAIL',exit_code=1),dict(stage='controller',status='FAIL',exit_code=1)])
    workflow=seal(dict(version='fresh-pilot-workflow/v1',status='FAIL',assessment='SEALED',acceptance=dict(exit_code=0),stages=stages,throughput=digest('probe')))
    write(old/'workflow.json',workflow)
    return tmp_path,old,base,feature_calls


def test_read_only_prepared_reuse_verifies_original_bytes_and_selection(original):
    root,old,base,calls=original
    before={p:p.read_bytes() for p in old.rglob('*') if p.is_file()}
    context=m.prepared_inputs(root,'runs/old','runs/new','podman')
    assert context['manifest']==str((base/'study.json').relative_to(root)) and calls
    assert before=={p:p.read_bytes() for p in old.rglob('*') if p.is_file()}
    assert not (root/'runs/new').exists()


@pytest.mark.parametrize('change',['payload','source','stage','learner','symlink','engine','overlap','inventory','manifest','protocol'])
def test_bad_reuse_is_rejected_before_writes(original,change):
    root,old,base,_=original;output='runs/new';engine='podman'
    def amend(path,fn):
        value=json.loads(path.read_text());fn(value);write(path,seal(value),exclusive=False)
    if change=='payload':(base/'cells/cell-0/private/selected.json').write_text('[]')
    if change=='source':write(old/'acceptance/report/qualified/source.json',{'production':'changed'},exclusive=False)
    if change=='stage':amend(old/'workflow.json',lambda x:x['stages'].append(dict(stage='train-static')))
    if change=='learner':(old/'research/train-static').mkdir()
    if change=='symlink':
        p=base/'cells/cell-0/private/selected.json';p.rename(p.with_suffix('.original'));p.symlink_to(p.with_suffix('.original'))
    if change=='engine':engine='docker'
    if change=='overlap':output='runs/old/new'
    if change=='inventory':amend(old/'research/prepare/result.json',lambda x:x.update(inventory=[]))
    if change=='manifest':amend(old/'private/smoke-job.json',lambda x:x.update(manifest='elsewhere.json'))
    if change=='protocol':amend(old/'research/design/design.json',lambda x:x.update(version='other'))
    with pytest.raises((ValueError,OSError)):m.prepared_inputs(root,'runs/old',output,engine)
    assert not (root/output).exists()


@pytest.mark.parametrize('change',[None,'controller','source','other-helper','image','expired','metrics','protocol'])
def test_throughput_migration_only_allows_known_controller(original,monkeypatch,change):
    import time
    import fresh_pilot_throughput_host as h
    from test_fresh_pilot_throughput import measurements
    root,old,_,_=original
    context=m.prepared_inputs(root,'runs/old','runs/new','podman')
    probe=old.parent/'probe'
    report=measurements();report=seal(dict(report,protocol=context['design']['identity']))
    if change=='protocol':report=seal(dict(report,protocol=digest('other')))
    write(probe/'probe/measurement/metrics.json',report)
    write(probe/'probe/logs/process.json',dict(outcome='COMPLETE',returncode=0))
    write(probe/'probe/logs/cleanup.json',dict(acknowledged=True,within_bound=True))
    helpers={name:digest(name) for name in h.HELPERS};helpers['fresh_pilot.py']=m.PREDECESSOR_CONTROLLER
    expected=dict(helpers=helpers,image=context['images'][0],source=digest('source'))
    from fresh_pilot_throughput import assess
    receipt=seal(dict(version='fresh-pilot-throughput-receipt/v1',status='PASS',binding=expected,recorded_at=time.time(),verdict=assess(report),
        **{key:reference(probe/path,probe) for key,path in [('metrics','probe/measurement/metrics.json'),('process','probe/logs/process.json'),('cleanup','probe/logs/cleanup.json')]}))
    if change=='controller':receipt['binding']['helpers']['fresh_pilot.py']=digest('unapproved')
    if change=='expired':receipt['recorded_at']-=73*3600
    receipt=seal(receipt);write(probe/'result.json',receipt)
    context['workflow']['throughput']=receipt['identity']
    current=json.loads(json.dumps(expected));current['helpers']['fresh_pilot.py']=digest('new controller')
    if change=='source':current['source']=digest('changed')
    if change=='other-helper':current['helpers']['fresh_pilot_throughput.py']=digest('changed')
    if change=='image':current['image']=digest('changed')
    if change=='metrics':(probe/'probe/measurement/metrics.json').write_text('{}')
    host=SimpleNamespace(engine=SimpleNamespace(image=lambda x:x))
    monkeypatch.setattr(h,'host_for',lambda _:host)
    monkeypatch.setattr(h,'binding',lambda *a:current)
    if change:
        with pytest.raises(ValueError):m.throughput_proof(None,context,'podman')
    else:
        assert m.throughput_proof(None,context,'podman')[1]==receipt
