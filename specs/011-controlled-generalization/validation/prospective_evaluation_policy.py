"""Temporary outcome-blind cohort/exclusion policy; no learner or model imports."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

from oeis_learn.pipeline_validation.identity import seal,digest,write

PRIOR='sha256:725d54628ef32a36a1c315143287475700c7f2b6db3e14a5d12b9e58731cf537'
PARENT_PREPARATION='sha256:56955fda4ffba7eda116fbcfc26b174e3b0d381130db1185d1d8149b13073b7f'
SEED=20261104
LAW='independent-uniform-integers[-1024,1024]/v1'


from oeis_learn.generalization.structural_draws import design,draw,CandidateExclusions,roster

def prepare(protocol,originals,output,*,execute,seconds=14000,workers=3):
    """Run in the custodian only; original rows/targets never leave its mounts."""
    from oeis_learn.generalization.preparation import OrderedWorkers
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_cell,reference
    if not 0<seconds<=14000:raise ValueError('bounded preparation work required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+seconds
    candidates=draw(protocol);admitted=[];reference_rejections=0;seen=set()
    with OrderedWorkers(workers) as pool:
        for start in range(0,len(candidates),64):
            batch=candidates[start:start+64]
            for candidate,(value,error) in zip(batch,pool.map(batch,deadline=deadline)):
                if error:reference_rejections+=1;continue
                exact,_=value
                if exact['source'] in seen:continue
                seen.add(exact['source']);admitted.append(dict(exact,role='selection',holdout=candidate['holdout']))
    index=CandidateExclusions(admitted);original_count=0
    for item in originals:
        rows=json.loads(verify_artifact(item['root'],item['ref'],max_bytes=2<<30))
        if len(rows)!=item['count']:raise ValueError('original prepared cell count changed')
        index.observe(rows,role=item['role'],holdout=item['holdout'],deadline=deadline)
        original_count+=len(rows)
        print(json.dumps(dict(phase='exclusion',cell=item['name'],rows=len(rows))),flush=True)
        del rows
    bad=index.rejected();selected=[];ledgers=[]
    available=[]
    for cell in protocol['cells']:
        matching=[r for r in admitted if (r['holdout'],r['family'])==(cell['stratum'],cell['family'])]
        surviving=sum(r['id'] not in bad for r in matching)
        available.append(dict(stratum=cell['stratum'],family=cell['family'],required=cell['count'],
            reference_eligible=len(matching),surviving=surviving,deficit=max(0,cell['count']-surviving),
            exclusion_reasons=dict(Counter(reason for r in matching for reason in bad.get(r['id'],())))))
    write(root/'availability.json',seal(dict(version='prospective-family-availability/v1',design=protocol['identity'],
        families=available,original_rows=original_count,scope='pre-native availability; no model outcomes or protected terms')))
    for stratum in ('parameter','structural'):
        rows=[r for r in admitted if r['holdout']==stratum and r['id'] not in bad]
        quota={'family:'+c['family']:c['count'] for c in protocol['cells'] if c['stratum']==stratum}
        folder=root/'cells'/stratum
        ledger=prepare_cell(rows,role='selection',holdout=stratum,count=sum(quota.values()),quotas=quota,
            structural=False,output=folder,execute=execute,seconds=deadline-time.monotonic(),workers=workers,
            unique_sources=True,length_boundary=16)
        if ledger['status']!='PASS':raise ValueError('native cohort '+stratum+' shortfalls: '+', '.join(ledger['shortfalls'])[:1024])
        selected.extend(json.loads((folder/'private/selected.json').read_text()));ledgers.append(ledger)
    mapping=roster(selected,protocol)
    summary=seal(dict(version='prospective-native-cohort/v1',status='PASS',design=protocol['identity'],
        draws=len(candidates),reference_rejections=reference_rejections,excluded=len(bad),original_rows=original_count,
        exclusion_rules=dict(Counter(rule for rules in bad.values() for rule in rules)),
        selected=len(selected),roster=mapping,native_ledgers=[l['identity'] for l in ledgers],
        files=[reference(root/'cells'/s/n,root) for s in ('parameter','structural')
               for n in ('prompts.json','private/targets.json','private/selected.json','ledger.json')],
        assessment='exclusion-only in custodian; no scoring',
        originals=[{k:v for k,v in item.items() if k!='root'} for item in originals]))
    write(root/'summary.json',summary)
    return summary


def evaluation_identity(parent,runtime,*,source,recipe,feature,images,protocol,cohort,selection,marker,descriptors,exclusions):
    """A new E0 binding; no old receipt becomes authority for changed execution."""
    from copy import deepcopy
    from oeis_learn.generalization.execution import bind,validate_policy
    study=deepcopy(parent)
    study.update(recipe='E0-evaluation-extension/v1',hypothesis='Prospective retained-pilot selection '+protocol['identity'],
        changing_factors=['evaluation'],arms=['retained','controls','shared_absent_bank'],source_closure=source)
    study['data'].update(generator=descriptors,descriptors=descriptors,exclusions=exclusions,manifests=[cohort])
    study['selection']['cohort']=selection
    study['assessment'].update(cohort=marker,authorization=marker,exposure_ledger=marker)
    study['budgets'].update(target_tokens=0,seconds=21600,draws=8192)
    study['checkpoints']['milestones']=[]
    study['progression'].update(stage='E0')
    sidecar=dict(runtime,recipe=recipe,images=images,preparation=None,training={},evaluation=dict(
        role='selection',prompts=protocol['prompts'],targets=protocol['targets'],checkpoints=protocol['checkpoint_manifests'],seconds=21600,
        bank_seeds=protocol['bank_seeds']),replay=None,assessment=None,feature=feature)
    study,sidecar=bind(study,sidecar);validate_policy(study,sidecar)
    return study,sidecar


def lineage(workspace,pilot,prepared,protocol):
    """Hash exact original inputs without reading assessment contents on the host."""
    from oeis_learn.pipeline_validation.identity import confined,file_hash
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.generalization.contracts import verify_artifact
    from fresh_pilot import read,latest_checkpoint
    root=Path(workspace);pilot=confined(root,pilot);prepared=confined(root,prepared)
    original=prepared/'research/prepare';receipt=read(original/'result.json')
    if (seal(receipt)!=receipt or receipt['identity']!=PARENT_PREPARATION or receipt['status']!='PASS'
        or receipt['stage']!='prepare' or receipt['original'] is not None):raise ValueError('reviewed original native preparation required')
    inventory={r['path']:r for r in receipt['inventory']}
    if len(inventory)!=len(receipt['inventory']):raise ValueError('duplicate original inventory path')
    cells=[('selection-parameter','selection','parameter',128),('selection-structural','selection','structural',128),
           ('assessment-parameter','assessment','parameter',128),('assessment-structural','assessment','structural',128)]
    cells.extend(('static' if i==0 else f'refresh-{i:02}','training','parameter',32768) for i in range(12))
    originals=[]
    for name,role,holdout,count in cells:
        path='preparation/role/private/prepared/cells/'+name+'/private/selected.json'
        ref=inventory.get(path)
        if ref is None:raise ValueError('original exclusion cell missing: '+name)
        p=confined(original,path)
        if ref['bytes']>2<<30 or not p.is_file() or p.stat().st_size!=ref['bytes'] or file_hash(p)!=ref['sha256']:
            raise ValueError('original exclusion input changed: '+name)
        originals.append(dict(name=name,role=role,holdout=holdout,count=count,ref=reference(p,root)))
    base=pilot/'execution/private/prepared';study=read(base/'study.json');runtime=read(base/'runtime.json')
    prior=json.loads((Path(__file__).parent/'observable-evaluation-protocol.json').read_text())
    if study['identity']!=prior['original_study'] or seal(study)!=study or seal(runtime)!=runtime or runtime['study']!=study['identity']:
        raise ValueError('reviewed completed pilot required')
    checkpoints={}
    for arm in ('static','refreshed'):
        folder=pilot/'execution'/('train-'+arm);terminal=read(folder/'result.json')
        if (seal(terminal)!=terminal or terminal['status']!='PASS' or terminal['study']!=study['identity']
            or terminal['stage']!='train' or terminal['original'] is not None or len(terminal['result']['runs'])!=1):
            raise ValueError('original successful paired learner required')
        run=terminal['result']['runs'][0]
        if run['arm']!=arm or run['seed']!=20261016:raise ValueError('original learner arm/seed changed')
        learner=confined(folder/'training',run['root']);result=run['result']
        if seal(read(learner/'training.json'))!=result:raise ValueError('original training result changed')
        checkpoint,weights=latest_checkpoint(learner,result,study=study['identity'],arm=arm)
        if checkpoint['weights']!=protocol['checkpoints'][arm]:raise ValueError('reviewed retained weights changed')
        checkpoints[arm]=dict(checkpoint=checkpoint,weights_root=str(weights.relative_to(root)),
            producer=reference(folder/'result.json',root))
    if len({c['checkpoint']['initialization'] for c in checkpoints.values()})!=1:raise ValueError('paired initializer mismatch')
    recipe=json.loads(verify_artifact(base,runtime['recipe']))
    return dict(originals=originals,checkpoints=checkpoints,study=study,runtime=runtime,recipe=recipe,
        preparation=reference(original/'result.json',root),parent_manifest=reference(base/'study.json',root))


def summarize(mapping,scores,controls):
    """Frozen denominators, macro rates, paired discordances and censoring."""
    if set(scores)!={'static','refreshed'} or any(set(s)!=set(mapping) for s in scores.values()):
        raise ValueError('complete paired score roster required')
    cells={};eligible=True
    for category in ('certified','ambiguous','unresolved'):
        for stratum in ('parameter','structural'):
            ids=[k for k,r in mapping.items() if (r['category'],r['stratum'])==(category,stratum)]
            groups=defaultdict(list)
            for k in ids:groups[mapping[k]['family']].append(k)
            result={}
            for control in controls:
                arms={}
                for arm in ('static','refreshed'):
                    values=[scores[arm][k][control] for k in ids]
                    if any(type(v['success']) is not bool or type(v['complete']) is not bool or v['success'] and not v['complete'] for v in values):
                        raise ValueError('invalid outcome')
                    complete=sum(v['complete'] for v in values);success=sum(v['success'] for v in values)
                    macro=sum(sum(scores[arm][k][control]['success'] for k in keys)/len(keys) for keys in groups.values())/len(groups) if groups else None
                    arms[arm]=dict(tasks=len(ids),families=len(groups),success=success,complete=complete,
                        unresolved=len(ids)-complete,family_macro=macro)
                    if category=='certified' and control=='equal_budget':eligible &= bool(ids) and complete/len(ids)>=.95
                discordance=Counter((scores['static'][k][control]['success'],scores['refreshed'][k][control]['success']) for k in ids)
                result[control]=dict(arms=arms,both=discordance[True,True],static_only=discordance[True,False],
                    refreshed_only=discordance[False,True],neither=discordance[False,False])
            cells[category+'/'+stratum]=result
    return seal(dict(version='prospective-retained-summary/v1',cells=cells,
        primary_interpretable=eligible,scientific_status='EXPLORATORY_SELECTION' if eligible else 'INCONCLUSIVE_COMPLETION',
        assessment='SEALED',automatic_advancement=False))
