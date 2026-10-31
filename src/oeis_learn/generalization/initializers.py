"""Frozen, original-producer-bound retained initialization; no tensor loading.

As with qualification, hashes bind trusted canonical producer evidence, not an
attestation service. Each initializer document's references are relative to its
own directory, allowing bounded ancestors without rewriting immutable records.
"""
import json
from pathlib import Path
from .contracts import validate,verify_artifact,training_ancestry
from oeis_learn.pipeline_validation.identity import seal,confined,validate as closed


def _selection(record,descriptor):
    from .selection import select
    closed(record,{'version','status','scientific_status','targets','candidates','measurements','best_loss','selected','eligible','reason'},'generalization-checkpoint-selection/v1')
    if record['status']!='PASS' or record['selected']!=descriptor['identity']:
        raise ValueError('selection-designated initializer required')
    candidates=record['candidates'];measurements=record['measurements']
    if not isinstance(candidates,list) or not 1<=len(candidates)<=66 or not isinstance(measurements,list) or len(measurements)!=len(candidates):
        raise ValueError('complete frozen selection candidate set required')
    by_id={}
    for candidate in candidates:
        if set(candidate)!={'checkpoint','selections'}:raise ValueError('selection candidate fields')
        item=validate('checkpoint',candidate['checkpoint']).record()
        if item['identity'] in by_id:raise ValueError('duplicate selection candidate')
        by_id[item['identity']]=item
    if by_id.get(descriptor['identity'])!=descriptor:raise ValueError('selected checkpoint descriptor mismatch')
    measured=set();behavior=[]
    for row in measurements:
        if set(row)!={'checkpoint','loss','behavior'} or row['checkpoint'] not in by_id or row['checkpoint'] in measured:
            raise ValueError('unique measured selection candidate required')
        measured.add(row['checkpoint']);value=row['behavior']
        if value['id']!=row['checkpoint'] or value['tokens']!=by_id[row['checkpoint']]['tokens']:
            raise ValueError('selection exposure/identity mismatch')
        behavior.append(value)
    decision=select(behavior)
    if record['selected']!=decision.checkpoint_identity or record['eligible']!=list(decision.eligible) or record['reason']!=decision.reason:
        raise ValueError('selection result contradicts measured evidence')


def verify_initializer(root,study):
    study=validate('study',study).record()
    initialization=study['model']['initialization']
    if initialization=='random':return None
    if study['data']['target_format']!='literal':raise ValueError('retained initializer requires literal representation')
    seen=set();documents=[]
    def read(base,ref):return json.loads(verify_artifact(base,ref,max_bytes=1<<20))
    def visit(base,ref,expected,depth):
        if depth>=32:raise ValueError('initializer lineage depth ceiling')
        authority=read(base,ref)
        closed(authority,{'version','study','checkpoint','selection','producer','ancestry','parent'},'generalization-initializer/v1')
        if authority['identity'] in seen:raise ValueError('cyclic initializer lineage')
        seen.add(authority['identity'])
        directory=confined(base,ref['path']).parent
        source=validate('study',read(directory,authority['study'])).record()
        descriptor=validate('checkpoint',read(directory,authority['checkpoint'])).record()
        if descriptor['kind']!='full_state':
            raise ValueError('legacy weights-only import is not initializer selection evidence')
        selection=read(directory,authority['selection'])
        producer=read(directory,authority['producer'])
        training_ancestry(authority['ancestry'])
        if set(authority['ancestry'])!={'mechanical','training','random_initialization'} and set(authority['ancestry'])!={'mechanical','training','random_initialization','verified_training_replay'}:
            raise ValueError('complete mechanical initialization ancestry required')
        if source['track']!='generic_only' or source['data']['target_format']!='literal':
            raise ValueError('initializer source must be generic mechanical literal training')
        if source['precision']!=expected['precision'] or any(source['model'][k]!=expected['model'][k] for k in ('architecture','profile')):
            raise ValueError('initializer model/precision incompatibility')
        _selection(selection,descriptor)
        if source['model']['initialization']=='random':
            if authority['parent'] is not None or descriptor['parent'] is not None:
                raise ValueError('random initialization cannot hide imported parent')
            parent=None
        else:
            if authority['parent']!=source['model']['initialization']['artifact']:
                raise ValueError('initializer parent differs from origin study')
            parent=visit(directory,authority['parent'],source,depth+1)
            if descriptor['parent']!=parent['weights']['sha256']:
                raise ValueError('checkpoint parent weight identity mismatch')
        fields={'version','identity','stage','attempt','binding','state','reason','files'}
        optional={'observations','prerequisites','command','counts','cleanup','prerequisite_equivalence'}
        if not fields<=set(producer) or set(producer)-fields-optional or producer['version']!='pipeline-stage/v1' or seal(producer)!=producer:
            raise ValueError('original initializer producer receipt required')
        binding=dict(study=source['identity'],checkpoint=descriptor['identity'],ancestry=authority['ancestry'],parent=parent['authority'] if parent else None)
        if producer['state']!='PASS' or producer['stage']!='generalization.initializer' or producer['binding']!=binding or not isinstance(producer['attempt'],str) or not producer['attempt']:
            raise ValueError('original PASS initializer producer binding required')
        producer_dir=Path(authority['producer']['path']).parent
        # The producer must inventory the exact original documents and weights.
        for item in (authority['study'],authority['checkpoint'],authority['selection'],descriptor['weights']):
            try:key=str(Path(item['path']).relative_to(producer_dir))
            except ValueError as exc:raise ValueError('initializer payload outside producer') from exc
            if not isinstance(producer['files'],dict) or producer['files'].get(key)!=item['sha256']:
                raise ValueError('initializer payload absent from original producer')
            verify_artifact(directory,item,max_bytes=32<<30)
        documents.append(dict(root=str(directory),authority=authority,producer=producer))
        return dict(authority=authority['identity'],checkpoint=descriptor['identity'],weights=descriptor['weights'],root=str(directory))
    result=visit(Path(root),initialization['artifact'],study,0)
    return dict(result,lineage=documents)
