"""Frozen I/O projections for canonical generalization commands.

The runtime sidecar binds resolved recipe and role views to a sealed study. It
contains no executable callback or arbitrary container command. Host reads use
only the standard library; schema/data validation runs in the custodian role.
"""
import json
from pathlib import Path
from .identity import validate,confined,digest
from oeis_learn.generalization.contracts import verify_artifact


def load(root,manifest,runtime=None):
    root=Path(root);path=confined(root,manifest)
    if path.stat().st_size>1<<20:raise ValueError('study document cap')
    study=json.loads(path.read_text())
    sidecar=confined(root,runtime) if runtime else path.parent/'runtime.json'
    if sidecar.stat().st_size>1<<20:raise ValueError('runtime document cap')
    record=json.loads(sidecar.read_text())
    validate(record,{'version','study','purpose','recipe','images','preparation','training','evaluation','replay','assessment','feature','storage'},'generalization-runtime/v1')
    if record['study']!=study['identity']:raise ValueError('runtime differs from frozen study')
    import re
    if not isinstance(record['images'],list) or len(record['images'])!=2 or any(not re.fullmatch(r'sha256:[0-9a-f]{64}',x) for x in record['images']):
        raise ValueError('immutable qualification/GPU images required')
    if record['purpose'] not in {'research','acceptance'}:raise ValueError('registered runtime purpose required')
    from oeis_learn.generalization.resources import CAPS
    if not isinstance(record['storage'],dict) or set(record['storage'])!=set(CAPS):raise ValueError('closed storage reservations required')
    if record['purpose']=='acceptance' and (study['budgets']['target_tokens']>4096 or study['budgets']['seconds']>900 or len(study['seeds'])>1):
        raise ValueError('acceptance cannot authorize research-scale learning')
    return path.parent,study,record


def validate_execution(root,study,runtime):
    """Custodian-only schema and role verification before a learner is launched."""
    from oeis_learn.generalization.contracts import validate as authority,shape
    from oeis_learn.generalization.initializers import verify_initializer
    study=authority('study',study).record()
    if runtime['study']!=study['identity']:raise ValueError('runtime/study identity mismatch')
    recipe=json.loads(verify_artifact(root,runtime['recipe'],max_bytes=1<<20))
    model={k:v for k,v in recipe['model'].items() if k!='seed'}
    if study['model']['architecture']!=digest(model) or study['model']['profile']!=recipe['version']:
        raise ValueError('resolved model recipe differs from frozen architecture/profile')
    if study['precision']!='fp32':raise ValueError('qualified prospective precision is fp32')
    if runtime['preparation'] is not None:
        prep=authority('preparation',json.loads(verify_artifact(root,runtime['preparation'],max_bytes=1<<20))).record()
        if any(study.get(k)!=v for k,v in prep['study_settings'].items()) or any(study[k]!=prep[k] for k in ('seeds','budgets')):
            raise ValueError('preparation differs from frozen study settings')
        preparation_views(root,study,runtime,prep)
        verify_initializer(root,study)
        return dict(study=study['identity'],recipe=recipe,preparation=runtime['preparation'])
    training=runtime['training']
    if not isinstance(training,dict) or set(training)-set(study['arms']):raise ValueError('undeclared training arms')
    authorized=set();cohorts=[]
    for ref in study['data']['manifests']:
        cohort=authority('cohort',json.loads(verify_artifact(root,ref))).record();cohorts.append(cohort)
        if cohort['role']=='training':authorized.update(digest(r) for r in cohort['private_payloads'])
    for arm,view in training.items():
        if set(view)!={'shards','data_mode','batch_size'} or view['data_mode'] not in {'static','refresh'} or type(view['batch_size']) is not int or not 1<=view['batch_size']<=128:
            raise ValueError('closed bounded training view required')
        if not view['shards'] or len(view['shards'])>16 or any(digest(r) not in authorized for r in view['shards']):
            raise ValueError('training view outside frozen cohort inventory')
        for ref in view['shards']:
            rows=json.loads(verify_artifact(root,ref,max_bytes=8<<30))
            for row in rows:shape('training-target',row)
    if any(view['data_mode']=='refresh' for view in training.values()):
        from .generalization_preparation import draw_candidates
        source_plan=json.loads(verify_artifact(root,study['data']['generator'],max_bytes=1<<20))
        if source_plan.get('version')!='generalization-materialization-plan/v2':raise ValueError('refresh requires frozen descriptor/counter materialization plan')
        expected={}
        for cell in source_plan['cells']:
            if cell['role']!='training':continue
            if 'refresh' not in cell:raise ValueError('refresh training support must use fixed descriptors')
            rows,_=draw_candidates(root,study['data']['descriptors'],cell)
            expected.update({digest(['training',r['id'],r['source']]):r['source'] for r in rows})
        for view in training.values():
            for ref in view['shards']:
                for row in json.loads(verify_artifact(root,ref,max_bytes=8<<30)):
                    if expected.get(row['id'])!=row['source']:raise ValueError('training shard differs from frozen refresh draw stream')
    if study['progression']['stage']=='P3':
        if set(training)!={'literal','parameterized'}:raise ValueError('paired target views required')
        paired={}
        for arm,view in training.items():
            rows=[]
            for ref in view['shards']:
                if not ref['path'].endswith('/training-'+arm+'.json'):raise ValueError('target arm differs from prepared representation')
                rows.extend(json.loads(verify_artifact(root,ref,max_bytes=8<<30)))
            paired[arm]=[(r['id'],r['indices'],r['values']) for r in rows]
        if paired['literal']!=paired['parameterized']:raise ValueError('target arms must retain paired concrete tasks and ordering')
    if study['progression']['stage']=='P4':
        if set(training)!={'continued_sft','verified_replay'} or training['continued_sft']!=training['verified_replay'] or training['continued_sft']['data_mode']!='static':
            raise ValueError('paired replay requires identical static training views')
        replay=runtime['replay']
        if not isinstance(replay,dict) or set(replay)!={'tasks','seconds'} or not 0<replay['seconds']<=14400:
            raise ValueError('bounded one-round replay settings required')
        tasks=json.loads(verify_artifact(root,replay['tasks'],max_bytes=8<<30))
        if not 1<=len(tasks)<=256 or len({t['id'] for t in tasks})!=len(tasks):raise ValueError('bounded unique replay task roster required')
        wanted={t['id']:t for t in tasks};matched=set()
        from oeis_learn.generalization.contracts import training_ancestry
        for ref in training['continued_sft']['shards']:
            for row in json.loads(verify_artifact(root,ref,max_bytes=8<<30)):
                if row['id'] not in wanted:continue
                task=wanted[row['id']]
                if set(task)!={'id','role','ancestry','source','indices','values'} or task['role']!='training' or any(task[k]!=row[k] for k in ('source','indices','values')):
                    raise ValueError('replay task differs from prepared literal training')
                training_ancestry(task['ancestry']);matched.add(row['id'])
        if matched!=set(wanted):raise ValueError('replay task outside prepared training')
        if runtime['evaluation'] is None or runtime['evaluation']['role']!='selection':
            raise ValueError('paired current-policy selection evaluation required')
    evaluation=runtime['evaluation']
    if evaluation is not None:
        if set(evaluation)!={'role','prompts','targets','checkpoints','seconds','bank_seeds'} or evaluation['role'] not in {'selection','assessment'}:
            raise ValueError('closed evaluation role view required')
        if not evaluation['prompts'] or not evaluation['targets'] or not 0<evaluation['seconds']<=21600:
            raise ValueError('bounded nonempty evaluation views required')
        owned=[c for c in cohorts if c['role']==evaluation['role']]
        public={digest(r) for c in owned for r in c['public_payloads']}
        protected={digest(r) for c in owned for r in c['private_payloads']}
        if any(digest(r) not in public for r in evaluation['prompts']) or any(digest(r) not in protected for r in evaluation['targets']):
            raise ValueError('evaluation view outside its frozen role')
        for ref in evaluation['prompts']:
            for row in json.loads(verify_artifact(root,ref)):shape('visible-prompt',row)
        for ref in evaluation['targets']:
            for row in json.loads(verify_artifact(root,ref)):
                shape('scorer-target',row)
                if row['role']!=evaluation['role']:raise ValueError('cross-role scoring payload')
    initializer=verify_initializer(root,study)
    if initializer is not None:
        parent=Path(initializer['root']).relative_to(Path(root))
        initializer=dict(authority=initializer['authority'],checkpoint=initializer['checkpoint'],
            weights=dict(initializer['weights'],path=str(parent/initializer['weights']['path'])))
    return dict(study=study['identity'],recipe=recipe,initializer=initializer,training=training)


def preparation_views(root,study,runtime,prep):
    """Freeze which preparation cells each role will consume before publication."""
    plan=json.loads(verify_artifact(root,prep['generator'],max_bytes=1<<20))
    cells={c['name']:c for c in plan['cells']}
    def owned(names,role):
        if not isinstance(names,list) or not names or len(set(names))!=len(names) or any(n not in cells or cells[n]['role']!=role for n in names):
            raise ValueError('preparation view must name unique cells of its role')
    if not isinstance(runtime['training'],dict) or set(runtime['training'])!=set(study['arms']):
        raise ValueError('complete frozen training cell views required')
    for view in runtime['training'].values():
        if set(view)!={'cells','data_mode','batch_size'} or view['data_mode'] not in {'static','refresh'} or type(view['batch_size']) is not int or not 1<=view['batch_size']<=128:
            raise ValueError('closed preparation training view required')
        owned(view['cells'],'training')
    view=runtime['evaluation']
    if view is not None:
        if set(view)!={'role','cells','checkpoints','seconds','bank_seeds'} or view['role'] not in {'selection','assessment'} or not 0<view['seconds']<=21600:
            raise ValueError('closed preparation evaluation view required')
        owned(view['cells'],view['role'])
    if runtime['replay'] is not None:
        replay=runtime['replay']
        if set(replay)!={'cell','seconds'} or not 0<replay['seconds']<=14400:raise ValueError('closed preparation replay view required')
        owned([replay['cell']],'training')
    return cells


def publish_runtime(base,prepared,study,runtime):
    """Resolve frozen cell names to exact newly published role artifacts."""
    from .identity import seal,write
    from .generalization_preparation import reference
    from .generalization_commands import _copy
    prep=json.loads(verify_artifact(base,runtime['preparation'],max_bytes=1<<20))
    preparation_views(base,study,runtime,prep)
    resolved=json.loads((prepared/'study.json').read_text())
    _copy(prepared,[(base,runtime['recipe'])])
    training={}
    for arm,view in runtime['training'].items():
        filename='training-'+arm+'.json' if study['progression']['stage']=='P3' else 'training.json'
        training[arm]=dict(shards=[reference(prepared/'cells'/n/'private'/filename,prepared) for n in view['cells']],data_mode=view['data_mode'],batch_size=view['batch_size'])
    evaluation=runtime['evaluation']
    if evaluation is not None:
        _copy(prepared,[(base,r) for r in evaluation['checkpoints']])
        # Retained artifacts remain in their original relative directories.
        from .generalization_workflow import _checkpoints
        for descriptor,parent in _checkpoints(base,evaluation['checkpoints']) if evaluation['checkpoints'] else []:
            if (parent/descriptor['weights']['path']).is_file():
                target=prepared/parent.relative_to(base)
                _copy(target,[(parent,descriptor['weights'])])
        evaluation=dict(role=evaluation['role'],prompts=[reference(prepared/'cells'/n/'prompts.json',prepared) for n in evaluation['cells']],
            targets=[reference(prepared/'cells'/n/'private/targets.json',prepared) for n in evaluation['cells']],
            **{k:evaluation[k] for k in ('checkpoints','seconds','bank_seeds')})
    replay=runtime['replay']
    if replay is not None:replay=dict(tasks=reference(prepared/'cells'/replay['cell']/'private/replay-tasks.json',prepared),seconds=replay['seconds'])
    if runtime['assessment'] is not None:_copy(prepared,[(base,runtime['assessment'])])
    result=seal(dict(runtime,study=resolved['identity'],preparation=None,training=training,evaluation=evaluation,replay=replay))
    write(prepared/'runtime.json',result)
    return resolved,result


def initializer_refs(root,initialization):
    """Projection closure only; the custodian separately verifies all semantics."""
    if initialization=='random':return []
    seen={};pending=[initialization['artifact']]
    def add(ref):
        if ref['path'] in seen:
            if seen[ref['path']]!=ref:raise ValueError('conflicting initializer path')
            return
        verify_artifact(root,ref,max_bytes=32<<30);seen[ref['path']]=ref
    while pending:
        ref=pending.pop()
        if ref['path'] in seen:raise ValueError('repeated initializer ancestor')
        if len(seen)>160:raise ValueError('initializer closure bound')
        add(ref);authority=json.loads(verify_artifact(root,ref,max_bytes=1<<20))
        parent=Path(ref['path']).parent
        def relative(child):return dict(child,path=str(parent/child['path']))
        for key in ('study','checkpoint','selection','producer'):add(relative(authority[key]))
        descriptor=json.loads(verify_artifact(root,relative(authority['checkpoint']),max_bytes=1<<20))
        add(relative(descriptor['weights']))
        if authority['parent'] is not None:pending.append(relative(authority['parent']))
    return list(seen.values())


def execution_refs(root,study,runtime,*,preparation=False):
    """Exact custodian projection; never mount an entire operator output tree."""
    from oeis_learn.generalization.contracts import artifact_refs
    refs=[runtime['recipe'],study['source_closure'],*initializer_refs(root,study['model']['initialization'])]
    if preparation:
        if runtime['preparation'] is None:raise ValueError('preparation authority required')
        prep=json.loads(verify_artifact(root,runtime['preparation'],max_bytes=1<<20))
        refs.extend([runtime['preparation'],*artifact_refs(prep),study['assessment']['authorization'],study['assessment']['exposure_ledger']])
        plan=json.loads(verify_artifact(root,prep['generator'],max_bytes=1<<20))
        refs.extend(c['candidates'] for c in plan['cells'] if 'candidates' in c)
    else:
        refs.extend([study['data']['generator'],study['data']['descriptors']])
        refs.extend(study['data']['manifests'])
        for ref in study['data']['manifests']:
            cohort=json.loads(verify_artifact(root,ref,max_bytes=1<<20));refs.extend(artifact_refs(cohort))
        if runtime['replay'] is not None:refs.append(runtime['replay']['tasks'])
    unique={}
    for ref in refs:
        if ref['path'] in unique and unique[ref['path']]!=ref:raise ValueError('conflicting runtime artifact')
        verify_artifact(root,ref);unique[ref['path']]=ref
    if sum(r['bytes'] for r in unique.values())>64<<30:raise ValueError('runtime input storage ceiling')
    return list(unique.values())


def learner_config(study,runtime,recipe,arm,seed):
    if arm not in study['arms'] or seed not in study['seeds'] or arm not in runtime['training']:
        raise ValueError('undeclared arm/seed')
    view=runtime['training'][arm]
    return dict(study=study['identity'],arm=arm,seed=seed,recipe=recipe,shards=view['shards'],
        data_mode=view['data_mode'],batch_size=view['batch_size'],target_tokens=study['budgets']['target_tokens'],
        milestones=study['checkpoints']['milestones'],seconds=study['budgets']['seconds'],device='cuda')
