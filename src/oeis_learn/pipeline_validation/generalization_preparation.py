"""Bounded materialization and immutable per-cell publication adapters."""
from pathlib import Path
import time
from oeis_learn.generalization.cohorts import project,publish
from oeis_learn.generalization.exclusions import ExclusionIndex
from oeis_learn.generalization.contracts import digest
from oeis_learn.generalization.resources import PREPARATION_SECONDS
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
    last_progress=[None];reported_phases=set()
    def progress(value):
        # One atomic snapshot on first phase entry, then every30 seconds while scanning.
        # It is deliberately outside the immutable cohort/ledger identity.
        now=value['elapsed_seconds']
        if value['phase'] in reported_phases and last_progress[0] is not None and now-last_progress[0]<30 and value['phase'] not in {'selection','complete','blocked'}:return
        record=seal(dict(version='generalization-cell-progress/v1',role=role,**value))
        pending=output/'progress.json.next'
        write(pending,record,exclusive=False);pending.replace(output/'progress.json')
        last_progress[0]=now;reported_phases.add(value['phase'])
        import sys,json
        print('preparation-progress: '+json.dumps(dict(cell=output.name,**value),sort_keys=True),file=sys.stderr,flush=True)
    choice,ledger=scan(candidates,role=role,holdout=holdout,count=count,quotas=quotas,
        structural=structural,execute=execute,exclusions=exclusions,draw_cap=draw_cap,
        seconds=seconds,length_boundary=length_boundary,unique_sources=unique_sources,
        seen_sources=seen_sources,workers=workers,observer=progress)
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
    if seconds is not None and not 0<seconds<=PREPARATION_SECONDS:raise ValueError('preparation allowance')
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
    allowance=min(PREPARATION_SECONDS,prep.get('work_seconds',prep['budgets']['seconds']),seconds if seconds is not None else PREPARATION_SECONDS)
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


def prepare_pac_training(rows,probes,*,output,execute,final_exclusions,seed,batch_size,purpose):
    """Custodian publication of paired rows and independent training feedback."""
    from collections import Counter,defaultdict
    from oeis_learn.generalization.preparation import pac_exact_row
    from oeis_learn.generalization.cohorts import pac_inventory
    from oeis_learn.generalization.exclusions import pac_probe_exclusions
    from oeis_learn.decoder.bigint_codec import encode_body
    if purpose not in {'research','acceptance'}:raise ValueError('PAC preparation purpose')
    if any(r.get('role')!='training' for r in rows+probes):raise ValueError('training-only PAC feedback')
    counts=Counter(r['family'] for r in rows)
    if purpose=='research' and (len(rows)!=32768 or len(counts)!=32 or len(final_exclusions)!=4):
        raise ValueError('original 32-family pool and all final exclusions required')
    if not 2<=len(counts)<=32 or min(counts.values())<max(4,batch_size):raise ValueError('PAC distinct family batch capacity')
    if len({r['id'] for r in rows+probes})!=len(rows)+len(probes):raise ValueError('PAC unique gradient/probe identities')
    if len(probes)>256*len(counts):raise ValueError('PAC finite probe draw cap')
    literal=[];parameterized=[];probe_rows=[];pairs=[];metadata=[];draws=[];per_family=Counter()
    protected_sources={r['source'] for cell in final_exclusions for r in cell}
    protected_windows={tuple(r['values'][i:i+20]) for cell in final_exclusions for r in cell for i in range(81)}
    collisions=set()
    for is_probe,items in ((False,rows),(True,probes)):
        for offset,row in enumerate(items):
            pair,values=pac_exact_row(row,execute)
            basic=dict(id=row['id'],role='training',indices=row['indices'],values=values)
            target=dict(basic,source=pair['target_source'],tokens=pair['target_tokens'])
            meta=dict(id=row['id'],family=row['family'],source_hash=digest(row['source']),
                output_hash=digest(values),window_hash=digest(values[:20]),role='training')
            if is_probe:
                if row['family'] not in counts:raise ValueError('PAC probe outside training support')
                draws.append(dict(meta,draw_index=per_family[row['family']],native_hash=digest(values)));per_family[row['family']]+=1
                probe_rows.append(target)
                if row['source'] in protected_sources or any(tuple(values[i:i+20]) in protected_windows for i in range(81)):collisions.add(row['id'])
            else:
                literal.append(dict(basic,source=row['source'],tokens=encode_body(row['source'])))
                parameterized.append(target);pairs.append(dict(id=row['id'],**pair))
                metadata.append(dict(meta,offset=offset,target_stratum='supported' if pair['disposition']=='supported' else 'fallback',
                                     length_stratum='short' if len(target['tokens'])<=32 else 'long'))
    if purpose=='research':
        inventory=pac_inventory(metadata,draws,pool_identity=digest(metadata),draw_identity=digest(draws),
            seed=seed,batch_size=batch_size,final_collision_ids=collisions,exclusions=digest(final_exclusions))
    else:
        bad,shared=pac_probe_exclusions(metadata,draws,final_collision_ids=collisions)
        families=[]
        for f in sorted(counts):
            ids=[r['id'] for r in metadata if r['family']==f]
            panel=[r['id'] for r in draws if r['family']==f and r['id'] not in bad][:8]
            if len(panel)!=8:raise ValueError('PAC acceptance probe quota shortfall')
            families.append(dict(id=f,rows=ids,anchors=sorted(ids)[:4],probes=panel,missing_strata=[]))
        inventory=seal(dict(version='pac-acceptance-inventory/v1',role='training',families=families,rows=metadata,
            sidecar=digest(metadata),probe_seed=seed,shared_training_windows=shared))
    from .generalization_reporting import pac_representation_report
    representation=pac_representation_report(inventory,literal,parameterized,{})
    wanted={x for f in inventory['families'] for x in f['probes']}
    output=Path(output)
    if output.exists():raise ValueError('fresh PAC prepared output required')
    output.mkdir(parents=True)
    artifacts={}
    for name,value in [('literal',literal),('parameterized',parameterized),('probes',[r for r in probe_rows if r['id'] in wanted]),('inventory',inventory),('pairs',pairs)]:
        path=output/'private'/(name+'.json');write(path,value);artifacts[name]=reference(path,output)
    bundle=seal(dict(version='pac-prepared/v1',role='training',**{k:[v] if k in {'literal','parameterized'} else v for k,v in artifacts.items()}))
    write(output/'prepared.json',bundle)
    write(output/'coverage.json',seal(dict(version='pac-coverage/v1',rows=len(rows),probes=len(wanted),
        families=len(counts),dispositions=dict(Counter(p['disposition'] for p in pairs)),purpose=purpose)))
    write(output/'representation.json',representation)
    return bundle


def validate_pac_preparation(root,plan,experiment,runtime):
    import json
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.generalization.structural import original_cells,verify
    fields={'version','originals','rows','probes','protocol','seed','batch_size'}
    if set(plan)!=fields or plan['version']!='pac-preparation/v1':raise ValueError('closed PAC preparation required')
    for key in ('rows','probes'):verify_artifact(root,plan[key],max_bytes=8<<30)
    if plan['seed']!=experiment['seed'] or type(plan['batch_size']) is not int or not 1<=plan['batch_size']<=128:raise ValueError('PAC frozen seed/batch')
    if runtime['purpose']=='research':
        metadata=[{k:v for k,v in item.items() if k!='ref'} for item in plan['originals']]
        if sorted(metadata,key=lambda r:r['name'])!=sorted(original_cells(),key=lambda r:r['name']):raise ValueError('all 16 original PAC exclusion inputs required')
        verify(plan['protocol'])
    elif plan['originals'] or plan['protocol'] is not None:raise ValueError('acceptance fixture cannot expose final truth')
    for item in plan['originals']:
        rows=json.loads(verify_artifact(root,item['ref'],max_bytes=2<<30))
        if len(rows)!=item['count']:raise ValueError('original PAC exclusion count mismatch')
    return plan


def prepare_pac(root,experiment,runtime,*,output,execute,seconds):
    """Installed native custodian. A quota failure publishes no learner authority."""
    import json
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.generalization import structural
    from oeis_learn.generalization.structural_draws import CandidateExclusions
    from oeis_learn.generalization.preparation import reference_candidate
    from .generalization_execution import bind_pac
    from collections import Counter
    plan=validate_pac_preparation(root,runtime['preparation'],experiment,runtime)
    deadline=time.monotonic()+seconds;output=Path(output);output.mkdir(parents=True,exist_ok=False)
    def bounded(source,indices):
        if time.monotonic()>=deadline:raise TimeoutError('PAC preparation work exhausted')
        return execute(source,indices)
    originals=[];final=[];roster=None
    for item in plan['originals']:
        rows=json.loads(verify_artifact(root,item['ref'],max_bytes=2<<30));originals.append((item,rows))
        if item['role']!='training':final.append(rows)
    if runtime['purpose']=='research':
        draws=structural.draw(plan['protocol']);exact=[];results={};by_id={r['id']:r for r in draws};started=time.monotonic();rejections=Counter()
        for row in draws:
            value,error=reference_candidate(row)
            if error:rejections[error]+=1;continue
            materialized,_=value;exact.append(dict(materialized,role='selection',holdout='structural'))
        index=CandidateExclusions(exact)
        for item,rows in originals:index.observe(rows,role=item['role'],holdout=item['holdout'],deadline=deadline)
        bad=index.rejected();available=Counter(r['family'] for r in exact if r['id'] not in bad)
        before=Counter(r['family'] for r in exact);raw=Counter(r['family'] for r in draws)
        availability=dict(version='pac-structural-availability/v1',
            families=[dict(family=c['family'],draws=raw[c['family']],before_exclusion=before[c['family']],available=available[c['family']],required=c['count'],deficit=max(0,c['count']-available[c['family']])) for c in plan['protocol']['cells']],
            reference_rejections=dict(rejections),exclusion_reasons=dict(Counter(reason for reasons in bad.values() for reason in reasons)),
            reference_and_exclusion_seconds=time.monotonic()-started,native_completed=0,native_seconds=0.)
        write(output/'availability.json',seal(availability));native_started=time.monotonic()
        for row in exact:
            if row['id'] in bad:continue
            native=bounded(row['source'],row['indices'])
            if native.get('outcome')!='ok':raise ValueError('PAC structural native execution incomplete')
            results[row['id']]=dict(reference=row['values'],native=native['values'])
        availability.update(native_completed=len(results),native_seconds=time.monotonic()-native_started)
        write(output/'availability.json',seal(availability),exclusive=False)
        roster=structural.admit(plan['protocol'],draws,[by_id[r['id']] for r in exact],results,
            exclusion_cells=[{k:v for k,v in i.items() if k!='ref'} for i,_ in originals],rejected_ids=set(bad))
        # Original full protected records remain custodian-only.
        selected=[r for r in exact if digest(['selection',r['id'],r['source']]) in roster]
        write(output/'private/structural-targets.json',selected)
    rows=json.loads(verify_artifact(root,plan['rows'],max_bytes=8<<30));probes=json.loads(verify_artifact(root,plan['probes'],max_bytes=8<<30))
    if runtime['purpose']=='research':
        static=next(r for i,r in originals if i['name']=='static')
        if [(r['id'],r['source']) for r in rows]!=[(r['id'],r['source']) for r in static]:raise ValueError('original ordered static PAC corpus changed')
    bundle=prepare_pac_training(rows,probes,output=output/'training',execute=bounded,final_exclusions=final,
        seed=plan['seed'],batch_size=plan['batch_size'],purpose=runtime['purpose'])
    # Paths in the bundle are rooted at the published study root.
    from oeis_learn.generalization.contracts import artifact_refs
    for key in ('literal','parameterized'):bundle[key]=[dict(r,path='training/'+r['path']) for r in bundle[key]]
    for key in ('probes','inventory','pairs'):bundle[key]=dict(bundle[key],path='training/'+bundle[key]['path'])
    bundle=seal(bundle);write(output/'training.json',bundle)
    admission=seal(dict(version='pac-admission/v2',status='PASS',purpose=runtime['purpose'],
        experiment=experiment['identity'],preparation=digest(plan),training=bundle['identity'],
        structural=reference(output/'private/structural-targets.json',output) if runtime['purpose']=='research' else None,
        source=experiment['source'],images=runtime['images'],originals=plan['originals'],roster=roster,parameter='DEFERRED'))
    write(output/'admission.json',admission)
    for ref in [runtime['recipe'],runtime['source']]:
        target=output/ref['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(verify_artifact(root,ref))
    resolved=dict(runtime,preparation=None,prepared=dict(training=reference(output/'training.json',output),admission=reference(output/'admission.json',output)))
    # Admission records its immutable proposed identity; the resolved binding owns
    # those exact receipt bytes and cannot be edited independently.
    experiment,resolved=bind_pac(experiment,resolved)
    write(output/'study.json',experiment);write(output/'runtime.json',resolved)
    return dict(status='PASS',study=experiment['identity'],manifest='private/prepared/study.json')


def verify_pac_prepared(root,experiment,runtime):
    import json
    from oeis_learn.generalization.contracts import verify_artifact,pac_record
    from oeis_learn.generalization.role_contracts import pac_training_refs
    refs=runtime['prepared']
    admission=json.loads(verify_artifact(root,refs['admission'],max_bytes=16<<20))
    bundle=json.loads(verify_artifact(root,refs['training'],max_bytes=16<<20))
    if (seal(admission)!=admission or set(admission)!={'version','status','purpose','experiment','preparation','training','source','images','originals','roster','parameter','structural','identity'}
        or admission['version']!='pac-admission/v2' or admission['status']!='PASS' or admission['purpose']!=runtime['purpose']
        or admission['source']!=experiment['source'] or admission['images']!=runtime['images'] or admission['training']!=bundle['identity']):raise ValueError('verified PAC admission closure required')
    if seal(bundle)!=bundle:raise ValueError('PAC training publication changed')
    inventory=json.loads(verify_artifact(root,bundle['inventory']))
    if runtime['purpose']=='research':
        from oeis_learn.generalization.structural import original_cells
        from oeis_learn.generalization.metrics import _pac_roster
        if sorted(({k:v for k,v in x.items() if k!='ref'} for x in admission['originals']),key=lambda x:x['name'])!=sorted(original_cells(),key=lambda x:x['name']):raise ValueError('all original admission inputs required')
        _pac_roster(admission['roster']);pac_record('inventory',inventory)
        verify_pac_structural(root,admission)
        if inventory['identity']!=experiment['inventory']:raise ValueError('PAC frozen inventory mismatch')
    else:
        if inventory['version']!='pac-acceptance-inventory/v1' or admission['roster'] is not None or admission['structural'] is not None:raise ValueError('separate acceptance training authority required')
    for arm in experiment['arms']:
        for ref in pac_training_refs(bundle,arm):verify_artifact(root,ref,max_bytes=8<<30)
    from oeis_learn.generalization.targets import paired_learning_view
    literal=[r for ref in bundle['literal'] for r in json.loads(verify_artifact(root,ref))]
    parameterized=[r for ref in bundle['parameterized'] for r in json.loads(verify_artifact(root,ref))]
    proofs=json.loads(verify_artifact(root,bundle['pairs'],max_bytes=8<<30))
    metadata={r['id']:r for r in inventory['rows']}
    if len(literal)!=len(parameterized) or len(proofs)!=len(literal):raise ValueError('complete paired PAC rows required')
    for a,b,p in zip(literal,parameterized,proofs):
        if a['id']!=b['id'] or p['id']!=a['id'] or a['indices']!=b['indices'] or a['values']!=b['values']:raise ValueError('PAC paired concrete order mismatch')
        checked=paired_learning_view(a['source'],p['parameter_source'],p['assignment'],indices=tuple(a['indices']),sharing='explicit')
        if dict(checked,id=a['id'])!=p or checked['target_tokens']!=b['tokens'] or checked['target_source']!=b['source']:raise ValueError('PAC paired proof changed')
        m=metadata[a['id']]
        if m['source_hash']!=digest(a['source']) or m['output_hash']!=digest(a['values']):raise ValueError('PAC sidecar hash mismatch')
    if runtime['purpose']=='research':
        actual=dict(pool=inventory['sidecar'],order=digest([r['id'] for r in literal]),
            targets=digest(parameterized),literal_targets=digest(literal),probes=bundle['probes']['sha256'])
        if any(experiment[k]!=v for k,v in actual.items()):raise ValueError('PAC frozen pool/target/probe/order identity mismatch')
    return bundle


def verify_pac_structural(root,admission):
    """Bind native-admitted private truth bytes before publishing visible views."""
    import json
    from oeis_learn.generalization.contracts import verify_artifact
    if admission.get('version')!='pac-admission/v2' or admission.get('structural') is None:
        raise ValueError('hash-bound structural admission required')
    rows=json.loads(verify_artifact(root,admission['structural'],max_bytes=16<<20))
    ids=[digest(['selection',r['id'],r['source']]) for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(admission['roster']):raise ValueError('structural truth/roster identity mismatch')
    return rows
