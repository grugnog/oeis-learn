"""Bounded materialization and immutable per-cell publication adapters."""
from pathlib import Path
import time
from oeis_learn.generalization.cohorts import project,publish
from oeis_learn.generalization.exclusions import ExclusionIndex
from oeis_learn.generalization.contracts import digest
from .identity import write,seal,file_hash


def reference(path,root):
    return dict(path=str(path.relative_to(root)),sha256=file_hash(path),bytes=path.stat().st_size)


def draw_candidates(root,descriptors,cell):
    """Adapt a frozen refresh cell directly to the Phase B numeric stream."""
    import json
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.generalization.refresh import RefreshPlan
    if 'refresh' not in cell:return json.loads(verify_artifact(root,cell['candidates'],max_bytes=8<<30)),None
    settings=cell['refresh']
    if cell['role']!='training' or set(settings)!={'seed','capacity','cursor','draws'}:
        raise ValueError('closed training-only refresh draw projection required')
    plan=RefreshPlan(json.loads(verify_artifact(root,descriptors,max_bytes=1<<20)),seed=settings['seed'],capacity=settings['capacity'])
    rows,cursor=plan.take(settings['cursor'],settings['draws'])
    return list(rows),seal(dict(version='generalization-refresh-draws/v1',plan=plan.identity,start=settings['cursor'],next_cursor=cursor,draws=len(rows)))


def prepare_cell(candidates,*,role,holdout,count,quotas,structural,output,execute,
                 exclusions=None,draw_cap=1_000_000,seconds=21600,length_boundary=32,unique_sources=False,
                 seen_sources=None,workers=1):
    """Only native-validated cells can publish learner/scorer payloads."""
    from oeis_learn.generalization.preparation import scan
    if not callable(execute):raise ValueError('native validation callback required')
    if type(length_boundary) is not int or not 1<=length_boundary<=4096:raise ValueError('bounded frozen length bin required')
    output=Path(output)
    if output.exists():raise ValueError('fresh preparation output required')
    output.mkdir(parents=True,exist_ok=False)
    choice,ledger=scan(candidates,role=role,holdout=holdout,count=count,quotas=quotas,
        structural=structural,execute=execute,exclusions=exclusions,draw_cap=draw_cap,
        seconds=seconds,length_boundary=length_boundary,unique_sources=unique_sources,
        seen_sources=seen_sources,workers=workers)
    write(output/'ledger.json',ledger)
    if not choice.complete:return ledger
    visible=[];protected=[]
    for row in choice.rows:
        # Hash-based opaque id contains no protected labels.
        row=dict(row,id=digest([role,row['id'],row['source']]))
        prompt,target=project(row);visible.append(prompt);protected.append(target)
    write(output/'private/selected.json',list(choice.rows))
    write(output/'prompts.json',visible)
    write(output/'private/targets.json',protected)
    write(output/'inventory.json',seal(dict(version='generalization-cell-files/v1',files=[reference(output/name,output) for name in ('ledger.json','prompts.json','private/targets.json')])))
    return ledger


def publish_study(choices,preparation,proposed,manifests,*,root,output):
    """Resolve only after all cells and exclusion commitments pass Phase B."""
    resolved=publish(choices,preparation,proposed,manifests,root=root)
    write(Path(output)/'study.json',resolved.record())
    return resolved


def prepare_study(root,preparation,proposed,*,output,execute,workers=1,seconds=None):
    """Materialize a frozen cell plan and publish only after all role cells pass.

    The native callback is supplied by the isolated preparation role. Its enclosing
    container deadline bounds both native and independent-reference work.
    """
    began=time.monotonic()
    if seconds is not None and not 0<seconds<=21600:raise ValueError('preparation allowance')
    import json
    from collections import Counter
    from oeis_learn.generalization.contracts import validate,verify_artifact,artifact_refs,freeze
    from oeis_learn.generalization.cohorts import CohortChoice
    from oeis_learn.decoder.bigint_codec import encode_body
    from .identity import confined
    prep=validate('preparation',preparation.record()).record()
    if prep['version']=='generalization-preparation/v3' and (seal(proposed)!=proposed or prep['proposed_study']!=proposed['identity']):
        raise ValueError('changed proposed study authority')
    plan=json.loads(verify_artifact(root,prep['generator']))
    if set(plan)!={'version','cells'} or plan['version'] not in {'generalization-materialization-plan/v1','generalization-materialization-plan/v2'}:
        raise ValueError('frozen materialization plan required')
    from oeis_learn.generalization.refresh import LAWS
    cells=plan['cells']
    if not 3<=len(cells)<=16 or len({c['name'] for c in cells})!=len(cells):raise ValueError('bounded unique cells required')
    for cell in cells:
        mode='refresh' if 'refresh' in cell else 'candidates'
        if set(cell)-{'length_boundary'}!={'name','role','holdout','count','quotas','structural',mode} or mode=='refresh' and plan['version']!='generalization-materialization-plan/v2':
            raise ValueError('unknown cell authority')
        if not cell['name'].replace('-','').replace('_','').isalnum():raise ValueError('invalid cell name')
        if mode=='candidates':verify_artifact(root,cell['candidates'],max_bytes=8<<30)
        elif prep['numeric_law'] not in LAWS:raise ValueError('unknown refresh numeric law')
    if {c['role'] for c in cells}!=set(prep['roles']):raise ValueError('complete role roster required')
    quota=Counter()
    for c in cells:quota.update(c['quotas'])
    if dict(quota)!=prep['quotas']:raise ValueError('changed preparation quotas')
    output=Path(output)
    if output.exists():raise ValueError('fresh preparation root required')
    output.mkdir(parents=True,exist_ok=False)
    proposed=json.loads(json.dumps(proposed))
    # Inputs are operator-private prepared state. Public bundles use an explicit
    # evidence allowlist; this directory is never a report-role input wholesale.
    from .generalization_runtime import initializer_refs
    incoming=[*artifact_refs(prep),*initializer_refs(root,proposed['model']['initialization']),proposed['source_closure'],proposed['assessment']['authorization'],proposed['assessment']['exposure_ledger']]
    copied={}
    for ref in incoming:
        data=verify_artifact(root,ref)
        if ref['path'] in copied:
            if copied[ref['path']]!=ref:raise ValueError('conflicting artifact paths')
            continue
        path=confined(output,ref['path'],exists=False);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
        copied[ref['path']]=ref
    exclusions=ExclusionIndex(json.loads(verify_artifact(root,prep['exclusions'])))
    descriptors=json.loads(verify_artifact(root,prep['descriptors']))
    if any(d['numeric_law']!=prep['numeric_law'] for d in descriptors):raise ValueError('refresh numeric law differs from frozen preparation')
    training_capacity=0;training_sources=set()
    choices={};manifests=[];by_role={role:[] for role in prep['roles']}
    allowance=min(21600,prep['budgets']['seconds'],seconds if seconds is not None else 21600)
    deadline=began+allowance-min(60,allowance/5);draws=0
    for cell_number,cell in enumerate(cells):
        remaining=deadline-time.monotonic();available=min(1_000_000,prep['budgets']['draws'])-draws
        if remaining<=0 or available<=0:
            result=seal(dict(status='BLOCKED',reason='BUDGET_EXHAUSTED',study=None))
            write(output/'preparation-result.json',result);return result
        folder=output/'cells'/cell['name']
        candidates,refresh_record=draw_candidates(root,prep['descriptors'],cell)
        if refresh_record is not None and refresh_record['draws']>available:raise ValueError('refresh draw reservation exceeds remaining preparation budget')
        remaining=deadline-time.monotonic()
        if remaining<=0:
            result=seal(dict(status='BLOCKED',reason='BUDGET_EXHAUSTED',cell=cell['name'],study=None))
            write(output/'preparation-result.json',result);return result
        ledger=prepare_cell(candidates,
            **{k:cell[k] for k in ('role','holdout','count','quotas','structural')},
            output=folder,execute=execute,exclusions=exclusions,seconds=remaining,draw_cap=available,unique_sources='refresh' in cell,
            length_boundary=cell.get('length_boundary',32),workers=workers,
            seen_sources=training_sources if cell['role']=='training' else None)
        draws+=ledger['draws']
        if refresh_record is not None:write(folder/'refresh.json',refresh_record)
        if ledger['status']!='PASS':
            reason='NATIVE_VALIDATION_FAILED' if 'native_validation_failed' in ledger['shortfalls'] else 'BUDGET_EXHAUSTED' if 'preparation_deadline' in ledger['shortfalls'] else 'QUOTA_SHORTFALL'
            result=seal(dict(status='BLOCKED',reason=reason,cell=cell['name'],study=None))
            write(output/'preparation-result.json',result);return result
        rows=json.loads((folder/'private/selected.json').read_text())
        choices[cell['name']]=CohortChoice(json.dumps(rows,sort_keys=True).encode(),())
        if cell['role']=='training':training_sources.update(digest(row['source']) for row in rows)
        # Same-role windows never exclude. Once protected cells are complete,
        # retain training source hashes only, not millions of unused window tuples.
        if any(c['role']!=cell['role'] for c in cells[cell_number+1:]):
            exclusions=exclusions.add_many(dict(id=row['id'],role=cell['role'],family=row['family'],holdout=cell['holdout'],
                source=row['source'],values=row['values'],group=row['group'],variant=row['variant']) for row in rows)
        protected='private/targets.json'
        extra_payloads=[]
        if cell['role']=='training':
            training=[];dispositions=[];literal=[];parameterized=[];tasks=[];pair_proofs=[]
            for row in rows:
                prompt_id=digest([cell['role'],row['id'],row['source']])
                target_source=row['source']
                if proposed['data']['target_format']=='parameterized' or proposed['progression']['stage']=='P3':
                    from oeis_learn.generalization.targets import pair,learning_target
                    paired=pair(row['source'],row['parameter_source'],row['assignment'],indices=tuple(row['indices']),sharing=proposed['data']['sharing_policy'])
                    pair_proofs.append(dict(id=prompt_id,**paired))
                    dispositions.append(dict(id=prompt_id,disposition=paired['disposition'],family=row['family'],stratum=cell['holdout']))
                    learning_source,learning_tokens=learning_target(paired)
                    if proposed['data']['target_format']=='parameterized':target_source=learning_source
                    parameterized.append(dict(id=prompt_id,role='training',source=learning_source,tokens=learning_tokens,indices=row['indices'],values=row['values']))
                training.append(dict(id=prompt_id,role='training',source=target_source,tokens=encode_body(target_source),indices=row['indices'],values=row['values']))
                literal.append(dict(id=prompt_id,role='training',source=row['source'],tokens=encode_body(row['source']),indices=row['indices'],values=row['values']))
                tasks.append(dict(id=prompt_id,role='training',ancestry=['mechanical','training'],source=row['source'],indices=row['indices'],values=row['values']))
            from oeis_learn.generalization.budgets import charge_tokens
            training_capacity+=charge_tokens([r['tokens'] for r in training],bos=1,pad=0)
            write(folder/'private/training.json',training)
            write(folder/'private/target-dispositions.json',dispositions);protected='private/training.json'
            write(folder/'private/replay-tasks.json',tasks)
            extra_payloads.append(reference(folder/'private/replay-tasks.json',output))
            if proposed['progression']['stage']=='P3':
                write(folder/'private/target-pairs.json',pair_proofs)
                extra_payloads.append(reference(folder/'private/target-pairs.json',output))
                write(folder/'private/training-literal.json',literal)
                write(folder/'private/training-parameterized.json',parameterized)
                extra_payloads.extend(reference(folder/('private/training-'+name+'.json'),output) for name in ('literal','parameterized'))
        coverage=Counter(tag for row in rows for tag in row['tags'])
        manifest=freeze('cohort',dict(version='generalization-cohort/v1',role=cell['role'],family_rule='fixed-descriptor/v1',
            sources=[digest(row['source']) for row in rows],variants=sorted({row['variant'] for row in rows}),groups=[digest(row['group']) for row in rows],
            quotas=cell['quotas'],coverage=dict(coverage),exclusions=prep['exclusions'],indices=rows[0]['indices'],
            public_payloads=[reference(folder/'prompts.json',output)],private_payloads=[reference(folder/protected,output)]+extra_payloads+([reference(folder/'private/target-dispositions.json',output)] if cell['role']=='training' else []),
            preparation_ledger=reference(folder/'ledger.json',output)))
        write(folder/'cohort.json',manifest.record());ref=reference(folder/'cohort.json',output)
        manifests.append(ref);by_role[cell['role']].append(ref)
        used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if used>min(prep['budgets']['storage_bytes'],8<<30):
            result=seal(dict(status='BLOCKED',reason='STORAGE_EXHAUSTED',study=None))
            write(output/'preparation-result.json',result);return result
    if any(arm in {'refresh','refreshed'} for arm in prep['study_settings']['arms']) and training_capacity<prep['budgets']['target_tokens']:
        result=seal(dict(status='BLOCKED',reason='INSUFFICIENT_REFRESH_CAPACITY',study=None,available_target_tokens=training_capacity))
        write(output/'preparation-result.json',result);return result
    for role in ('selection','assessment'):
        path=output/(role+'-roster.json')
        write(path,seal(dict(version='generalization-cohort-roster/v1',role=role,cells=by_role[role])))
        proposed[role]['cohort']=reference(path,output)
    resolved=publish_study(choices,preparation,proposed,manifests,root=output,output=output)
    result=seal(dict(status='PASS',study=resolved.identity,draws=draws))
    write(output/'preparation-result.json',result);return result
