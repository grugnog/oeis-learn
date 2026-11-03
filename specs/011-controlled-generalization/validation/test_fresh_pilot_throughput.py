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
