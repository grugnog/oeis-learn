"""Read-only legacy follow-up provenance. Never synthesize resume/initializer proof."""
import json,math
from oeis_learn.pipeline_validation.identity import seal,digest,validate as closed
from .contracts import verify_artifact


def audit_uniform(training,prepared,recipe,history,stage):
    from oeis_learn.pipeline_validation.followup_training import validate as profile
    if any(seal(r)!=r for r in (training,prepared,stage)):
        raise ValueError('legacy provenance seal mismatch')
    policy=profile(training['profile'])
    if (policy['arm']!='uniform' or training['resume'] is not False or training['final_access']!='SEALED'
        or training['work']!=prepared['identity'] or prepared['recipe']!=digest(recipe)):
        raise ValueError('original uniform corpus/recipe/weight record required')
    if (recipe['track']!=dict(kind='generic_only',assisted_manifest=None,foundation=None)
        or recipe['learning']['replay'] is not False or recipe['learning']['target_view']!='literal'
        or recipe['model']['precision']!='float32'):
        raise ValueError('original generic-only literal fp32 provenance required')
    if stage['version']!='pipeline-stage/v1' or stage['stage']!='followup.uniform' or stage['state']!='PASS':
        raise ValueError('original PASS uniform producer required')
    observed=stage['observations']['result']['result']
    if (stage['binding']['source']!=training['source'] or any(observed[k]!=training[k] for k in ('source','work','tokens','updates'))
        or observed['status']!='COMPLETE' or observed['training'] is not True or observed['resume'] is not False):
        raise ValueError('uniform producer differs from exported weights')
    if not isinstance(history,list) or not 1<=len(history)<=10000:raise ValueError('bounded original validation history required')
    previous=(-1,-1);losses=[]
    for row in history:
        exposure=(row['tokens'],row['updates'])
        if any(type(v) is not int or v<0 for v in exposure) or exposure[0]<previous[0] or exposure[1]<previous[1]:
            raise ValueError('monotone historical exposure required')
        previous=exposure;total=0.;tokens=0
        for key in ('parameter_development','structural_development'):
            split=row['splits'][key]
            if (split['complete'] is not True or type(split['tokens']) is not int or split['tokens']<=0
                or type(split['loss_sum']) not in (int,float) or not math.isfinite(split['loss_sum']) or split['loss_sum']<0):
                raise ValueError('complete finite historical loss required')
            total+=split['loss_sum'];tokens+=split['tokens']
        losses.append(total/tokens)
    if previous!=(training['tokens'],training['updates']):raise ValueError('latest history/weights exposure mismatch')
    best=history[min(range(len(history)),key=lambda i:losses[i])]
    return dict(best={k:best[k] for k in ('tokens','updates')},latest={k:training[k] for k in ('tokens','updates')},
        selection='historical_continuous_pooled_nonpadding_dev_loss')


def verify_legacy(root,descriptor):
    closed(descriptor,{'version','kind','weights','state','initialization','parent','tokens','updates','pins','compatibility','provenance'},'generalization-checkpoint/v2')
    if descriptor.get('version')!='generalization-checkpoint/v2' or descriptor.get('kind')!='weights_only':
        raise ValueError('explicit legacy weights-only descriptor required')
    proof=json.loads(verify_artifact(root,descriptor['provenance'],max_bytes=1<<20))
    closed(proof,{'version','training','prepared','recipe','history','producer','report','label'},'generalization-legacy-import/v1')
    documents={k:json.loads(verify_artifact(root,proof[k],max_bytes=4<<20)) for k in ('training','prepared','recipe','history','producer','report')}
    report=documents['report']
    if seal(report)!=report or report['version']!='pipeline-followup-study-result/v1' or documents['training'] not in report['arms']:
        raise ValueError('uniform record absent from original follow-up report')
    checked=audit_uniform(documents['training'],documents['prepared'],documents['recipe'],documents['history'],documents['producer'])
    label=proof['label']
    if label not in ('best','latest'):raise ValueError('historical best/latest label required')
    if documents['producer']['files'].get('train/validation-history.json')!=proof['history']['sha256']:
        raise ValueError('history absent from original producer inventory')
    if documents['prepared']['files']['recipe.json']['sha256']!=proof['recipe']['sha256']:
        raise ValueError('original recipe file mismatch')
    if (descriptor['weights']['sha256']!=documents['training'][label+'_weights']
        or any(descriptor[k]!=checked[label][k] for k in ('tokens','updates'))
        or descriptor['compatibility']!=digest(documents['recipe'])
        or descriptor['pins']!=['historical_'+label]
        or any(descriptor[k] is not None for k in ('state','parent','initialization'))):
        raise ValueError('legacy checkpoint identity/exposure/provenance mismatch')
    return dict(recipe=documents['recipe'],selection=checked['selection'],label=label)
