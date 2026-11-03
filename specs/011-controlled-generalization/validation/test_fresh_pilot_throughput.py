import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import fresh_pilot_throughput as m
from oeis_learn.pipeline_validation.identity import seal


def measurements(rate=.005):
    cells=[]
    for name,n,count in [('selection-parameter',2048,128),('selection-structural',2034,128),
        ('assessment-parameter',2048,128),('assessment-structural',2042,128),
        ('static',65536,32768),('refresh-06',2048,1024),('refresh-11',2048,1024)]:
        cells.append(dict(name=name,status='PASS',draws=n,admitted=n,native_rows=n,selected=count,
            families=32,minimum_family_rows=count//32,native_validation='PERFORMED',
            elapsed_seconds=n*rate,reference_seconds=n*rate*.1,native_seconds=n*rate*.8,selection_seconds=n*rate*.05))
    return seal(dict(version=m.POLICY,status='MEASURED',restart_equal=True,resources=m.RESOURCES,cells=cells,elapsed_seconds=1000))


def test_conservative_gate_uses_slowest_late_window_and_never_average():
    report=measurements();assert m.assess(report)['status']=='PASS'
    report['cells'][-1]['elapsed_seconds']=50
    result=m.assess(seal(report))
    assert result['status']=='BLOCKED' and result['conservative_seconds']>2*50/2048*12*65536


@pytest.mark.parametrize('change',[
    lambda r:r['cells'].pop(),lambda r:r.update(restart_equal=False),
    lambda r:r['cells'][4].update(families=31),lambda r:r['cells'][4].update(minimum_family_rows=1023),
    lambda r:r['cells'][4].update(native_rows=100),lambda r:r['cells'][0].update(native_validation='NOT_RUN'),
    lambda r:r['cells'][4].update(draws=64000),lambda r:r['cells'][4].update(elapsed_seconds=-1),
    lambda r:r['cells'][0].update(native_seconds=True),lambda r:r.update(status='INCOMPLETE'),
    lambda r:r['cells'][0].update(reference_seconds=9999)])
def test_incomplete_or_inconsistent_evidence_cannot_pass(change):
    report=measurements();change(report)
    with pytest.raises(ValueError):m.assess(seal(report))


def test_probe_keeps_original_first_shard_and_samples_late_cursors():
    from fresh_pilot_inputs import research_inputs
    design=research_inputs();before=json.dumps(design,sort_keys=True)
    cells=m.probe_cells(design)
    assert cells[4]==design['training'][0]
    assert [c['refresh']['cursor'] for c in cells[4:]]==[0,6*65536,11*65536]
    assert [c['refresh']['draws'] for c in cells[4:]]==[65536,2048,2048]
    assert all(sum(c['quotas'].values())==c['count'] for c in cells[4:])
    assert before==json.dumps(design,sort_keys=True)


def test_measure_calls_real_preparation_and_native_adapter_with_small_fixture(tmp_path,monkeypatch):
    from copy import deepcopy
    from fresh_pilot_inputs import research_inputs
    from oeis_learn.synthesis.reference import evaluate
    design=research_inputs();cells=m.probe_cells(design)
    for cell in cells[:4]:
        cell.update(rows=cell['rows'][:64],count=1,quotas={},structural=False)
    for cell in cells[4:]:
        cell['refresh']['draws']=256;cell['count']=32;cell['quotas']={k:1 for k in cell['quotas']}
    monkeypatch.setattr(m,'probe_cells',lambda d:deepcopy(cells))
    generations=[]
    class ReferenceDouble:
        def __init__(self):generations.append(self);self.calls=0;self.closed=False
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True
        def request(self,operation,rows,indices,*,deadline):
            assert operation=='execute' and len(rows)<=8 and len(indices)==100
            self.calls+=1
            return dict(complete=True,results=[dict(candidate_id=r['candidate_id'],outcome='ok',
                values=evaluate(r['source'],indices).values) for r in rows])
    report=m.measure(design,tmp_path/'probe',client_factory=ReferenceDouble,seconds=60)
    assert report['status']=='MEASURED' and report['restart_equal'] is True
    assert len(generations)==3 and all(g.closed and g.calls for g in generations)
    assert all(r['families']==32 for r in report['cells'][4:])
    # Reduced software fixture cannot masquerade as the registered full-scale probe.
    with pytest.raises(ValueError):m.assess(report)


@pytest.mark.parametrize('failure',[None,'timeout','cleanup','nonzero','missing','corrupt'])
def test_host_cpu_only_command_and_failure_archive(tmp_path,monkeypatch,failure):
    import time
    from types import SimpleNamespace
    from contextlib import nullcontext
    import fresh_pilot_throughput_host as h
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation import supervision,generalization_lifecycle
    from oeis_learn.pipeline_validation.identity import write,digest
    from oeis_learn.pipeline_validation.bundle import verify
    monkeypatch.setattr(h,'ROOT',tmp_path)
    adapter=Engine('podman',executable='podman',details={})
    monkeypatch.setattr(adapter,'user_args',lambda:[])
    monkeypatch.setattr(adapter,'remove',lambda *a,**kw:None)
    host=SimpleNamespace(engine=adapter)
    monkeypatch.setattr(h,'host_for',lambda engine:host)
    bound=dict(image=digest('image'),source=digest('source'),resources=m.RESOURCES)
    monkeypatch.setattr(h,'binding',lambda *a:bound)
    monkeypatch.setattr(generalization_lifecycle,'guard',lambda *a:nullcontext())
    captured=[]
    def launch(argv,logs,**kw):
        captured.extend(argv)
        assert 0<kw['seconds']<1800 and kw['cleanup_seconds']==30
        write(logs/'process.json',dict(outcome='FAIL' if failure in {'timeout','nonzero'} else 'COMPLETE',
            returncode=1 if failure=='nonzero' else -9 if failure=='timeout' else 0))
        (logs/'stderr.log').write_text('bounded diagnostic\n')
        if failure=='timeout':raise TimeoutError('fixture timeout')
        if failure=='missing':return
        report=measurements();report['protocol']=digest('protocol');report=seal(report)
        write(logs.parent/'measurement/metrics.json',report)
        if failure=='corrupt':(logs.parent/'measurement/metrics.json').write_text('{}')
    monkeypatch.setattr(supervision,'launch',launch)
    if failure=='cleanup':
        def reclaim(engine,name,logs,cleanup):
            write(logs/'cleanup.json',dict(acknowledged=False,within_bound=False))
            raise RuntimeError('fixture unacknowledged cleanup')
        monkeypatch.setattr(supervision,'reclaim',reclaim)
    code=h.run('podman','cpu','probe')
    assert '--device' not in captured and '/dev/kfd' not in str(captured)
    assert '--network=none' in captured
    assert '--cpus=4' in captured
    assert 'docker.sock' not in str(captured) and str(tmp_path)+',dst=' not in str(captured)
    root=tmp_path/'probe';verify(root/'report.zip')
    record=h.bounded(root/'result.json')
    assert record['status']==('PASS' if failure is None else 'FAIL')
    assert code==(0 if failure is None else 1)
    if failure is None:
        h.require_receipt(root,bound)
        with pytest.raises(ValueError):h.require_receipt(root,dict(bound,image=digest('changed')))
        with pytest.raises(ValueError):h.require_receipt(root,bound,now=time.time()+73*3600)
        with pytest.raises(ValueError):h.require_receipt(root,bound,now=0)
        path=root/'probe/measurement/metrics.json';path.write_text('{}')
        with pytest.raises(ValueError):h.require_receipt(root,bound)


def test_missing_throughput_stops_full_run_before_any_stage(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import fresh_pilot as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'invoke',lambda *a,**kw:pytest.fail('stage launched without throughput'))
    with pytest.raises(ValueError,match='throughput'):
        runner.run(SimpleNamespace(engine='podman',cpu_image='cpu',output='run'))
    assert not (tmp_path/'run').exists()


def test_installed_source_mismatch_fails_before_any_native_work(tmp_path,monkeypatch):
    import fresh_pilot_throughput_host as h
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.pipeline_validation.identity import write,digest
    monkeypatch.setattr(source_snapshot,'source_snapshot',lambda:{'installed':'old'})
    monkeypatch.setattr(m,'measure',lambda *a,**kw:pytest.fail('native work with old image'))
    write(tmp_path/'job.json',seal(dict(binding=dict(source=digest({'checkout':'new'})),work_seconds=10)))
    assert h.inside(tmp_path/'job.json',tmp_path/'out')==0
    report=h.bounded(tmp_path/'out/measurement/metrics.json')
    assert report['status']=='INCOMPLETE' and report['phase']=='source-binding'


def test_headroom_boundary_matches_nested_preparation_reserves():
    assert m.PREPARATION_WORK_SECONDS==21600-30-30-60
    for offset,status in [(-.1,'PASS'),(.1,'BLOCKED')]:
        report=measurements()
        protected=sum(r['elapsed_seconds'] for r in report['cells'][:4])
        rate=(m.PREPARATION_WORK_SECONDS+offset-3600-2*protected)/(2*12*65536)
        for row in report['cells'][4:]:
            row['elapsed_seconds']=row['draws']*rate
        assert m.assess(seal(report))['status']==status
