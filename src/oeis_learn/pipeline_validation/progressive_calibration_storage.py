"""Opt-in retirement of successful temporary calibration snapshots.

Training checkpoints are never eligible. The original exact-input producer,
verified snapshot identities, sizes and retirement records survive; a failed
case retains its files. This does not change or confer qualification authority.
"""
from pathlib import Path
import shutil

from .identity import confined
from oeis_learn.generalization.progressive.contracts import (
    atomic_write, load, reference, seal, verify_refs, _check, schema,
)

POLICY='verified-calibration-inventory/v1'
METADATA_PER_CASE=256 << 10


def required_bytes(cases,checkpoint_bytes,input_bytes,*,retention=None,metadata_bytes=16<<20):
    if retention not in (None,POLICY):
        raise ValueError('registered calibration snapshot retention policy required')
    # One serial case owns two live generations. All immutable input packages
    # and per-case inventories remain charged for the entire preparation.
    generations=2*(1 if retention==POLICY else cases)
    return generations*checkpoint_bytes+input_bytes+metadata_bytes+(
        cases*METADATA_PER_CASE if retention==POLICY else 0)


def retire(case_root, producer, *, maximum_bytes):
    case_root=Path(case_root)
    if (seal(producer)!=producer or producer.get('version') not in {'progressive-iteration-calibration/v1','progressive-iteration-calibration/v2'}
            or producer.get('status')!='PASS' or load(case_root/'producer.json')!=producer):
        raise ValueError('original successful exact-input calibration required for retirement')
    if type(maximum_bytes) is not int or maximum_bytes<1:
        raise ValueError('bounded original calibration storage required')
    target=confined(case_root,'checkpoints')
    files=[];directories=[];total=0
    for path in sorted(target.rglob('*')):
        confined(case_root,str(path.relative_to(case_root)))
        if path.is_dir():
            directories.append(str(path.relative_to(case_root)))
        elif path.is_file():
            total+=path.stat().st_size
            if total>maximum_bytes:
                raise ValueError('calibration retirement inventory exceeds original storage bound')
            files.append(reference(case_root,path))
        else:
            raise ValueError('regular owned calibration snapshot files required')
        if len(files)+len(directories)>256:
            raise ValueError('bounded calibration snapshot inventory required')
    if not files or not any(r['path'].endswith('.json') for r in files):
        raise ValueError('original verified calibration snapshot closure required')
    payload=dict(version='progressive-calibration-retirement/v1',policy=POLICY,
        producer=reference(case_root,case_root/'producer.json'),
        compatibility=producer['compatibility'],device=producer['device'],
        files=files,directories=directories,bytes=total,
        scope='temporary exact-input calibration snapshots only; no scientific or resumable training state')
    # Publish the entire hash inventory durably before removing owned temporary
    # files. A crash here cannot replace the original producer or erase the cost.
    intent=seal(dict(payload,status='INVENTORIED'))
    atomic_write(case_root/'checkpoint-retirement-intent.json',intent)
    shutil.rmtree(target)
    complete=seal(dict(payload,status='RETIRED',intent=reference(case_root,case_root/'checkpoint-retirement-intent.json')))
    atomic_write(case_root/'checkpoint-retirement.json',complete)
    return reference(case_root,case_root/'checkpoint-retirement.json')


def verify(case_root, ref, producer):
    import json
    case_root=Path(case_root)
    value=json.loads(verify_refs(case_root,[ref],max_bytes=METADATA_PER_CASE)[0])
    fields={'version','identity','policy','producer','compatibility','device','files','directories',
            'bytes','scope','status','intent'}
    if (set(value)!=fields or seal(value)!=value or value['version']!='progressive-calibration-retirement/v1'
            or value['policy']!=POLICY or value['status']!='RETIRED'
            or value['compatibility']!=producer['compatibility'] or value['device']!=producer['device']
            or value['producer']!=reference(case_root,case_root/'producer.json')
            or load(case_root/'producer.json')!=producer):
        raise ValueError('original calibration retirement binding differs')
    intent=json.loads(verify_refs(case_root,[value['intent']],max_bytes=METADATA_PER_CASE)[0])
    if intent!=seal(dict({k:v for k,v in value.items() if k not in {'identity','intent'}},status='INVENTORIED')):
        raise ValueError('original pre-cleanup calibration inventory differs')
    if (not isinstance(value['files'],list) or not value['files']
            or not isinstance(value['directories'],list)
            or len(value['files'])+len(value['directories'])>256):
        raise ValueError('bounded original calibration retirement inventory required')
    names=set()
    for item in value['files']:
        _check(schema('common')['$defs']['artifact'],item,schema('common'))
        if (not item['path'].startswith('checkpoints/') or item['path'] in names):
            raise ValueError('retirement inventory must contain distinct temporary snapshot files')
        confined(case_root,item['path'],exists=False)
        names.add(item['path'])
    for name in value['directories']:
        if not isinstance(name,str) or not name.startswith('checkpoints/') or name in names:
            raise ValueError('retirement directory inventory differs')
        confined(case_root,name,exists=False);names.add(name)
    if type(value['bytes']) is not int or value['bytes']!=sum(r['bytes'] for r in value['files']):
        raise ValueError('retirement byte inventory differs')
    if (case_root/'checkpoints').exists() or (case_root/'checkpoints').is_symlink():
        raise ValueError('calibration snapshot retirement did not complete')
    return value
