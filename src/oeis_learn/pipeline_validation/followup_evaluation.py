"""Bounded saved-weight evaluation with isolated hidden-truth scoring."""
import json
from collections import Counter
from .identity import write,seal,file_hash


def evaluate(owner,output,ordinal):
    data=owner.imported;p=owner.a.pilot;p.purpose='followup'
    p.manifest=data['manifest'];p.prepared=data['prepared']
    # Rebuild only the visible evaluation settings. Never import optimizer state.
    recipe=json.loads((data['prepared']/'recipe.json').read_text())
    # settings() expects pilot role options; the original recipe and roster remain bound.
    snapshot=data['snapshots'][ordinal]
    cfg=dict(recipe=recipe,contract_sha256=data['contract'],checkpoint=snapshot.get('pointer',{}),
        pilot=dict(work=data['manifest']['identity']),followup_weights=snapshot['weights_sha256'],
        followup_policy=owner.contract['evaluation_policy'],
        pilot_prompts_sha256=data['manifest']['files']['prompts.json']['sha256'],
        pilot_truth_sha256=data['manifest']['files']['private/truth.json']['sha256'])
    descriptor=output/'private/generation-settings.json';write(descriptor,cfg)
    generated,gen=p.invoke(output,'generate-'+str(ordinal),'pilot-generate',
        {'settings.json':descriptor,'weights.pt':snapshot['path'],'prompts.json':p.prepared/'prompts.json'},seconds=9000,ordinal=ordinal)
    cfg['pilot_selections_sha256']=file_hash(gen/'selections.json')
    descriptor=output/'private/scoring-settings.json';write(descriptor,cfg)
    scored,score=p.invoke(output,'score-'+str(ordinal),'pilot-score',
        {'settings.json':descriptor,'selections.json':gen/'selections.json','truth.json':p.prepared/'private/truth.json'},seconds=1500,ordinal=ordinal)
    outcomes=json.loads((score/'outcomes.json').read_text())
    selected=json.loads((gen/'selections.json').read_text())
    groups={}
    for split in sorted({r['strata']['split'] for r in selected.values()}):
        ids=[k for k,r in selected.items() if r['strata']['split']==split]
        for arm in ('raw','rescue','equal_budget','shuffled','absent','template'):
            values=[outcomes[k][arm] for k in ids];n=len(values)
            success=sum(v['success'] for v in values);complete=sum(v['complete'] for v in values)
            groups[split+'|'+arm]=dict(denominator=n,success=success,complete=complete,unresolved=n-complete,
                lower=success/n,upper=(success+n-complete)/n)
    result=seal(dict(status='COMPLETE',training=False,clock_complete=generated['clock_complete'],
        scientific_complete=scored['complete'],groups=groups,policy=owner.contract['evaluation_policy']))
    write(output/'evaluation.json',result)
    return result


def report(owner,output,invocation,receipts):
    points=[]
    for ordinal in owner.contract['snapshots']:
        root=invocation/'stages'/('diagnostics.snapshot-'+str(ordinal))
        path=root/('score-'+str(ordinal))/'outcomes.json'
        if not path.exists():continue
        outcomes=json.loads(path.read_text());selected=json.loads((root/('generate-'+str(ordinal))/'selections.json').read_text())
        groups={}
        for split in sorted({r['strata']['split'] for r in selected.values()}):
            ids=[k for k,r in selected.items() if r['strata']['split']==split]
            for arm in ('raw','rescue','equal_budget','shuffled','absent','template'):
                rows=[outcomes[k][arm] for k in ids];n=len(rows)
                success=sum(r['success'] for r in rows);complete=sum(r['complete'] for r in rows)
                groups[split+'|'+arm]=dict(denominator=n,success=success,complete=complete,unresolved=n-complete,
                    lower=success/n,upper=(success+n-complete)/n,reasons=dict(Counter(r['reason'] for r in rows)))
        points.append(dict(ordinal=ordinal,groups=groups))
    write(output/'evaluation.json',seal(dict(version='pipeline-followup-evaluation/v1',policy=owner.contract['evaluation_policy'],
        points=points,training=False,final_access='SEALED',promotion_requires='95% completion per split/arm and independent seed confirmation')))
    return dict(evaluation='COMPLETE' if len(points)==len(owner.contract['snapshots']) else 'PARTIAL',training=False)
