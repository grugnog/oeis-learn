"""Bounded materialization and immutable per-cell publication adapters."""
from pathlib import Path
import time
from oeis_learn.data.bigint_programs import materialize
from oeis_learn.generalization.coverage import observe
from oeis_learn.generalization.cohorts import choose,project,publish
from oeis_learn.generalization.exclusions import ExclusionIndex
from oeis_learn.generalization.contracts import digest
from .identity import write,seal,file_hash


def reference(path,root):
    return dict(path=str(path.relative_to(root)),sha256=file_hash(path),bytes=path.stat().st_size)


def prepare_cell(candidates,*,role,holdout,count,quotas,structural,output,execute,
                 exclusions=None,draw_cap=1_000_000,seconds=14400,length_boundary=32):
    """execute is the qualified native client's externally deadline-bounded adapter."""
    if role not in {'training','selection','assessment'} or holdout not in {'parameter','structural'}:
        raise ValueError('invalid cohort role/holdout')
    if not 0<seconds<=14400 or not 1<=draw_cap<=1_000_000: raise ValueError('preparation limits')
    output=Path(output)
    if output.exists(): raise ValueError('fresh preparation output required')
    output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+seconds;admitted=[];rejections=[];seen=0
    exclusions=exclusions if exclusions is not None else ExclusionIndex()
    for row in candidates:
        if seen>=draw_cap or time.monotonic()>=deadline: break
        seen+=1
        try:
            if row['origin']!='generic_random': raise ValueError('nonmechanical origin')
            exact=materialize(row)
            native=execute(exact['source'],tuple(exact['indices']))
            if native.get('outcome')!='ok' or tuple(native['values'])!=tuple(exact['values']): raise ValueError('native/reference disagreement')
            observation=observe(exact['source'],tuple(exact['indices']),stage=row['stage'])
            tags=[row['stage'],'long' if observation.nodes>=length_boundary else 'short','above64' if observation.above_64_bits else 'within64']
            if observation.loop_active:tags.append('loop_active')
            if observation.branch_active:tags.append('branch_active')
            exclusion=dict(id=exact['id'],role=role,family=row['family'],holdout=holdout,source=exact['source'],values=exact['values'],group=row['group'],variant=row['variant'])
            if exclusions.reasons(exclusion): raise ValueError('cross-role exclusion')
            admitted.append(dict(exact,role=role,tags=tags))
        except ValueError as exc:
            rejections.append(dict(draw=seen,reason=str(exc)))
    choice=choose(admitted,count=count,quotas=quotas,structural=structural,draw_cap=draw_cap)
    ledger=seal(dict(version='generalization-cell-preparation/v1',role=role,draws=seen,rejections=rejections,shortfalls=list(choice.shortfalls),status='PASS' if choice.complete else 'BLOCKED'))
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


def prepare_study(root,preparation,proposed,*,output,execute):
    """Materialize a frozen cell plan and publish only after all role cells pass.

    The native callback is supplied by the isolated preparation role. Its enclosing
    container deadline bounds both native and independent-reference work.
    """
    import json
    from collections import Counter
    from oeis_learn.generalization.contracts import validate,verify_artifact,artifact_refs,freeze
    from oeis_learn.generalization.cohorts import CohortChoice
    from oeis_learn.decoder.bigint_codec import encode_body
    from .identity import confined
    prep=validate('preparation',preparation.record()).record()
    plan=json.loads(verify_artifact(root,prep['generator']))
    if set(plan)!={'version','cells'} or plan['version']!='generalization-materialization-plan/v1':
        raise ValueError('frozen materialization plan required')
    cells=plan['cells']
    if not 3<=len(cells)<=16 or len({c['name'] for c in cells})!=len(cells):raise ValueError('bounded unique cells required')
    for cell in cells:
        if set(cell)!={'name','role','holdout','count','quotas','structural','candidates'}:
            raise ValueError('unknown cell authority')
        if not cell['name'].replace('-','').replace('_','').isalnum():raise ValueError('invalid cell name')
        verify_artifact(root,cell['candidates'],max_bytes=8<<30)
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
    incoming=[*artifact_refs(prep),proposed['source_closure'],proposed['assessment']['authorization'],proposed['assessment']['exposure_ledger']]
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
    choices={};manifests=[];by_role={role:[] for role in prep['roles']}
    deadline=time.monotonic()+min(14400,prep['budgets']['seconds']);draws=0
    for cell in cells:
        remaining=deadline-time.monotonic();available=min(1_000_000,prep['budgets']['draws'])-draws
        if remaining<=0 or available<=0:
            result=seal(dict(status='BLOCKED',reason='BUDGET_EXHAUSTED',study=None))
            write(output/'preparation-result.json',result);return result
        folder=output/'cells'/cell['name']
        ledger=prepare_cell(json.loads(verify_artifact(root,cell['candidates'])),
            **{k:cell[k] for k in ('role','holdout','count','quotas','structural')},
            output=folder,execute=execute,exclusions=exclusions,seconds=remaining,draw_cap=available)
        draws+=ledger['draws']
        if ledger['status']!='PASS':
            result=seal(dict(status='BLOCKED',reason='QUOTA_SHORTFALL',cell=cell['name'],study=None))
            write(output/'preparation-result.json',result);return result
        rows=json.loads((folder/'private/selected.json').read_text())
        choices[cell['name']]=CohortChoice(json.dumps(rows,sort_keys=True).encode(),())
        for row in rows:
            exclusions=exclusions.add(dict(id=row['id'],role=cell['role'],family=row['family'],holdout=cell['holdout'],
                source=row['source'],values=row['values'],group=row['group'],variant=row['variant']))
        protected='private/targets.json'
        if cell['role']=='training':
            training=[]
            for row in rows:
                prompt_id=digest([cell['role'],row['id'],row['source']])
                target_source=row['source']
                if proposed['data']['target_format']=='parameterized':
                    from oeis_learn.generalization.targets import pair
                    paired=pair(row['source'],row['parameter_source'],row['assignment'],indices=tuple(row['indices']),sharing=proposed['data']['sharing_policy'])
                    if paired['disposition']!='supported':
                        write(folder/'unsupported.json',dict(id=prompt_id,disposition=paired['disposition']))
                        result=seal(dict(status='BLOCKED',reason='UNSUPPORTED_TARGET',cell=cell['name'],study=None))
                        write(output/'preparation-result.json',result);return result
                    target_source=paired['parameter_source']
                training.append(dict(id=prompt_id,role='training',source=target_source,tokens=encode_body(target_source),indices=row['indices'],values=row['values']))
            write(folder/'private/training.json',training);protected='private/training.json'
        coverage=Counter(tag for row in rows for tag in row['tags'])
        manifest=freeze('cohort',dict(version='generalization-cohort/v1',role=cell['role'],family_rule='fixed-descriptor/v1',
            sources=[digest(row['source']) for row in rows],variants=sorted({row['variant'] for row in rows}),groups=[digest(row['group']) for row in rows],
            quotas=cell['quotas'],coverage=dict(coverage),exclusions=prep['exclusions'],indices=rows[0]['indices'],
            public_payloads=[reference(folder/'prompts.json',output)],private_payloads=[reference(folder/protected,output)],
            preparation_ledger=reference(folder/'ledger.json',output)))
        write(folder/'cohort.json',manifest.record());ref=reference(folder/'cohort.json',output)
        manifests.append(ref);by_role[cell['role']].append(ref)
        used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if used>min(prep['budgets']['storage_bytes'],8<<30):
            result=seal(dict(status='BLOCKED',reason='STORAGE_EXHAUSTED',study=None))
            write(output/'preparation-result.json',result);return result
    for role in ('selection','assessment'):
        path=output/(role+'-roster.json')
        write(path,seal(dict(version='generalization-cohort-roster/v1',role=role,cells=by_role[role])))
        proposed[role]['cohort']=reference(path,output)
    resolved=publish_study(choices,preparation,proposed,manifests,root=output,output=output)
    result=seal(dict(status='PASS',study=resolved.identity,draws=draws))
    write(output/'preparation-result.json',result);return result
