"""Temporary representative CPU/native preparation gate; no scientific authority."""
import json
import math
from pathlib import Path
import time

from oeis_learn.generalization.resources import PREPARATION_SECONDS

POLICY='fresh-pilot-throughput/v2'
SECONDS=90*60
WORK_SECONDS=SECONDS-90  #60 seconds for terminal reporting,30 for external cleanup
PREPARATION_WORK_SECONDS=PREPARATION_SECONDS-120  #30 cleanup +30 process exit +60 study reporting
RESOURCES=dict(memory_bytes=64<<30,cpus=4,reference_workers=3,native_batch=8)
SAMPLE_DRAWS=2048


def save(path,value):
    from oeis_learn.pipeline_validation.identity import seal,write
    path=Path(path);pending=path.with_suffix('.next')
    write(pending,seal(value),exclusive=False);pending.replace(path)


def probe_cells(design):
    """Frozen original first shard, plus deterministic middle/final stream prefixes."""
    from copy import deepcopy
    if len(design['descriptors'])!=32 or len(design['training'])!=12:raise ValueError('expected frozen pilot roster')
    cells=deepcopy(design['cohorts'])+[deepcopy(design['training'][i]) for i in (0,6,11)]
    if [c['name'] for c in cells]!=['selection-parameter','selection-structural','assessment-parameter','assessment-structural','static','refresh-06','refresh-11']:
        raise ValueError('expected frozen cell order')
    for cell in cells[-2:]:
        cell['refresh']['draws']=SAMPLE_DRAWS
        cell['count']=1024
        cell['quotas']={key:32 for key in cell['quotas']}
    return cells


def assess(report):
    """Conservative engineering envelope; not a probabilistic confidence interval."""
    from oeis_learn.pipeline_validation.identity import seal
    if seal(report)!=report or report.get('version')!=POLICY or report.get('status')!='MEASURED':
        raise ValueError('complete original measurements required')
    if report.get('restart_equal') is not True or report.get('resources')!=RESOURCES:
        raise ValueError('native restart/resource evidence required')
    expected=[('selection-parameter',128,None),('selection-structural',128,None),
              ('assessment-parameter',128,None),('assessment-structural',128,None),
              ('static',32768,65536),('refresh-06',1024,2048),('refresh-11',1024,2048)]
    cells=report['cells']
    if len(cells)!=len(expected):raise ValueError('complete measurement roster required')
    for row,(name,selected,draws) in zip(cells,expected):
        if row['name']!=name or row['status']!='PASS' or row['selected']!=selected or row['native_validation']!='PERFORMED':
            raise ValueError('complete native cell required')
        for key in ('draws','admitted','native_rows','selected'):
            if type(row[key]) is not int or not 0<row[key]<=65536:raise ValueError('invalid cell count')
        if not row['selected']<=row['admitted']==row['native_rows']<=row['draws']:raise ValueError('inconsistent native coverage')
        for key in ('elapsed_seconds','reference_seconds','native_seconds','selection_seconds'):
            if type(row[key]) not in (int,float) or not math.isfinite(row[key]) or row[key]<0:raise ValueError('invalid cell time')
        if row['elapsed_seconds']<=0 or row['native_seconds']<=0:raise ValueError('positive measured time required')
        if sum(row[k] for k in ('reference_seconds','native_seconds','selection_seconds'))>row['elapsed_seconds']+.01:
            raise ValueError('phase time exceeds cell envelope')
        if draws is not None:
            if row['draws']!=draws or row['families']!=32 or row['minimum_family_rows']!=selected//32:
                raise ValueError('complete training-family coverage required')
        elif not selected<=row['draws']<=2048:raise ValueError('protected draw coverage')
    elapsed=report.get('elapsed_seconds')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not sum(r['elapsed_seconds'] for r in cells)<=elapsed<=WORK_SECONDS:
        raise ValueError('measurement exceeded or contradicted its work envelope')
    protected=sum(r['elapsed_seconds'] for r in cells[:4])
    slowest=max(r['elapsed_seconds']/r['draws'] for r in cells[4:])
    estimate=3*(protected+slowest*12*65536)+7200
    return dict(status='PASS' if estimate<=PREPARATION_WORK_SECONDS else 'BLOCKED',
        conservative_seconds=estimate,preparation_work_seconds=PREPARATION_WORK_SECONDS,
        headroom_seconds=PREPARATION_WORK_SECONDS-estimate,safety_multiplier=3,
        unmeasured_overhead_seconds=7200,slowest_training_seconds_per_draw=slowest,
        scope='preparation throughput only; no statistical confidence interval or campaign guarantee')


def measure(design,output,*,client_factory=None,seconds=WORK_SECONDS):
    from collections import Counter
    from oeis_learn.pipeline_validation.identity import seal,write,digest
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_cell,draw_candidates,reference
    from oeis_learn.pipeline_validation.generalization_role import native_batch
    from oeis_learn.generalization.exclusions import ExclusionIndex
    from oeis_learn.generalization.preparation import exclusion_record
    from oeis_learn.synthesis.client import SynthesisClient
    if seal(design)!=design:raise ValueError('intact frozen design required')
    if not 0<seconds<=WORK_SECONDS:raise ValueError('bounded diagnostic work required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();deadline=began+seconds;cells=probe_cells(design)
    record=dict(version=POLICY,status='INCOMPLETE',protocol=design['identity'],resources=RESOURCES,
        cells=[],restart_equal=False,scientific_status='UNMEASURED',feature_qualified=False)
    inputs=root/'private/inputs';inputs.mkdir(parents=True)
    write(inputs/'descriptors.json',design['descriptors']);descriptor=reference(inputs/'descriptors.json',inputs)
    for cell in cells:
        if 'rows' in cell:
            write(inputs/(cell['name']+'.json'),cell.pop('rows'))
            cell['candidates']=reference(inputs/(cell['name']+'.json'),inputs)
    exclusion=ExclusionIndex();seen=set()
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise TimeoutError('native throughput work budget exhausted')
        return value
    def execute_for(client):
        def execute(*args):raise AssertionError('production native batching required')
        execute.batch=lambda rows:native_batch(client,rows,deadline=deadline)
        return execute
    factory=client_factory or (lambda:SynthesisClient(digest(POLICY)))
    try:
        with factory() as client:
            for cell in cells:
                record['active_cell']=cell['name'];save(root/'metrics.json',record)
                started=time.monotonic();remaining()
                rows,_=draw_candidates(inputs,descriptor,cell)
                folder=root/'private/cells'/cell['name']
                ledger=prepare_cell(rows,**{k:cell[k] for k in ('role','holdout','count','quotas','structural')},
                    execute=execute_for(client),output=folder,exclusions=exclusion,unique_sources='refresh' in cell,
                    seen_sources=seen,workers=3,seconds=remaining(),length_boundary=cell['length_boundary'])
                progress=json.loads((folder/'progress.json').read_text())
                if seal(progress)!=progress:raise ValueError('progress identity mismatch')
                if ledger['status']!='PASS':raise ValueError('native preparation cell incomplete')
                selected=json.loads((folder/'private/selected.json').read_text())
                families=Counter(r['family'] for r in selected)
                if cell['role']=='training':
                    if set(families)!={d['family'] for d in design['descriptors']}:raise ValueError('training families missing')
                    seen.update(digest(r['source']) for r in selected)
                else:exclusion=exclusion.add_many(exclusion_record(r,cell['role'],cell['holdout']) for r in selected)
                record['cells'].append(dict(name=cell['name'],status=ledger['status'],draws=ledger['draws'],admitted=ledger['admitted'],
                    selected=len(selected),families=len(families),minimum_family_rows=min(families.values()),
                    native_validation=ledger['native_validation'],elapsed_seconds=time.monotonic()-started,
                    **{k:progress[k] for k in ('native_rows','reference_seconds','native_seconds','selection_seconds')}))
                save(root/'metrics.json',record)
                print(json.dumps(record['cells'][-1],sort_keys=True),flush=True)
        # New worker generations, unchanged deterministic inputs and exact values.
        rows,_=draw_candidates(inputs,descriptor,dict(cells[4],refresh=dict(cells[4]['refresh'],draws=64)))
        selections=[]
        for trial in range(2):
            with factory() as client:
                folder=root/'private'/f'restart-{trial}'
                ledger=prepare_cell(rows,role='training',holdout='parameter',count=32,quotas={},structural=False,
                    output=folder,execute=execute_for(client),workers=3,seconds=remaining(),length_boundary=16)
                if ledger['status']!='PASS':raise ValueError('native restart incomplete')
                selections.append((folder/'private/selected.json').read_bytes())
        record['restart_equal']=selections[0]==selections[1]
        record['status']='MEASURED'
    except (Exception,KeyboardInterrupt) as exc:
        record['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'INCOMPLETE'
        record['reason']=type(exc).__name__
    record['elapsed_seconds']=time.monotonic()-began
    save(root/'metrics.json',record)
    return seal(record)
