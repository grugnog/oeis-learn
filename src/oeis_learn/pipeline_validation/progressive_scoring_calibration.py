"""Measured positive scoring paths for prospective research resource planning."""
from pathlib import Path
from . import progressive_service_calibration as services
from .identity import digest,seal
from .progressive_research_inputs import checked_file,_json
from .progressive_workstation_state import read
from oeis_learn.generalization.progressive.contracts import atomic_write,reference


def prepare(output,support,*,source,image,native):
    output=Path(output)
    atomic_write(output/'private/training-services.json',services.training_samples(support))
    value=seal(dict(version=services.VERSION,source=source,image=image,native=native,
        inputs=digest(dict(support=support['identity'],scope='positive training service timing')),
        samples=reference(output,output/'private/training-services.json'),seconds=services.SECONDS,
        cases=[dict(index=i,sample=i//2,repeat=i%2) for i in range(8)],
        limits=dict(raw_seconds=4,fit_seconds=4,score_seconds=65,case_seconds=services.CASE_SECONDS),
        scope='training-only canonical-program service diagnostic; never learned capability'))
    services.validate_request(value)
    atomic_write(output/'request.json',value)
    return value


def request(root,evaluation):
    path=checked_file(root,evaluation['scoring_services']);value=services.validate_request(read(path))
    if (value['source']!=evaluation['source'] or value['image']!=evaluation['image']
            or value['inputs']!=digest(dict(support=evaluation['support'],scope='positive training service timing'))):
        raise ValueError('positive scoring services differ from original evaluation scope')
    samples=_json(checked_file(path.parent,value['samples']))
    if (len(samples)!=4 or len({r['id'] for r in samples})!=4 or
            any(set(r)!={'id','source','indices','values','training_row'} or
                len(r['indices'])!=100 or len(r['values'])!=100 for r in samples)):
        raise ValueError('four original full-horizon training witnesses required')
    return path.parent,value,samples


def measurements(root,evaluation):
    base,req,samples=request(root,evaluation)
    result=read(base/'service-cases.json')
    if (result.get('request')!=req['identity'] or result.get('status')!='PASS'
            or [r['index'] for r in result.get('cases',[])]!=list(range(8))):
        raise ValueError('complete positive scoring service measurement required')
    producers=[]
    for row,case in zip(result['cases'],req['cases']):
        p=read(checked_file(base,row['producer']))
        if (row['status']!='PASS' or p.get('status')!='PASS' or p.get('request')!=req['identity']
                or p.get('case')!=case or p.get('training_row')!=samples[case['sample']]['training_row']
                or p.get('native')!=req['native'] or p.get('errors') or p['fit'].get('complete') is not True
                or set(p['measurements'])!={'fit_us','raw_us','score_us'}
                or any(type(v)is not int or v<=0 for v in p['measurements'].values())
                or type(p['actual_elapsed_us'])is not int
                or not sum(p['measurements'].values())<=p['actual_elapsed_us']<=services.CASE_SECONDS*1000000
                or p['scoring'].get('complete') is not True
                or p['scoring'].get('outcome')!='success'
                or any(p['scoring'].get(k,{}).get('outcome')!='ok' or
                       p['scoring'].get(k,{}).get('coverage')!=100 for k in ('production','reference'))):
            raise ValueError('original complete independent positive scoring paths required')
        producers.append(p)
    return dict(score_us=max(p['measurements']['score_us'] for p in producers),
        owner_overhead_us=max(p['actual_elapsed_us']-sum(p['measurements'].values()) for p in producers),
        result=reference(root,base/'service-cases.json'))
