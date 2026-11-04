"""Public projections of recorded selected-policy outcomes; never an oracle union."""
import json
from pathlib import Path
from oeis_learn.pipeline_validation.identity import seal,write,digest,confined
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.interfaces import EXIT_CODES

ARMS=('static','refreshed')
CONTROLS=('raw','rescue','equal_budget','shuffled','absent','template','bank')
REQUIRED=('feasibility','preflight','prepare','smoke','train-static','train-refreshed','generate-static','score-static','generate-refreshed','score-refreshed','controller')
PUBLIC_CAP=500<<20  # reserve archive/manifest overhead within the canonical512MiB cap


def preparation_diagnostics(root):
    """Read only original cell ledgers; exclude target/source/term payloads."""
    import re
    from collections import Counter
    base=Path(root)/'research/prepare/preparation/role/private/prepared'
    result=dict(version='fresh-pilot-preparation-diagnostics/v1',cells=[],qualification_evidence=False)
    def bounded(path):
        path=confined(root,path)
        if not path.is_file() or path.stat().st_size>32<<20:raise ValueError('cell control size cap')
        value=json.loads(path.read_bytes())
        if seal(value)!=value:raise ValueError('cell control identity mismatch')
        return value
    def progress_record(path):
        import math
        value=bounded(path)
        counts=('draws','admitted','native_rows')
        timings=('elapsed_seconds','reference_seconds','native_seconds','selection_seconds')
        if set(value)!={'version','identity','role','phase',*counts,*timings} or value['version']!='generalization-cell-progress/v1':
            raise ValueError('original bounded progress required')
        if value['role'] not in {'training','selection','assessment'} or value['phase'] not in {'reference','native','selection','complete','blocked'}:
            raise ValueError('progress role/phase')
        for key in counts:
            if type(value[key]) is not int or not 0<=value[key]<=1_000_000:raise ValueError('progress count')
        if value['admitted']>value['draws'] or value['native_rows']>value['draws']:raise ValueError('progress coverage')
        for key in timings:
            if type(value[key]) not in (int,float) or not math.isfinite(value[key]) or not 0<=value[key]<=86400:raise ValueError('progress time')
        return dict(**{k:value[k] for k in ('role','phase',*counts,*timings)},original=reference(path,root))
    terminal=base/'preparation-result.json'
    if terminal.exists():
        value=bounded(terminal)
        result['terminal']={k:value[k] for k in ('status','reason','cell') if k in value and
            isinstance(value[k],str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',value[k])}
    cells=base/'cells'
    known={'native/reference disagreement','cross-role exclusion','duplicate refresh source',
        'excluded:input_integer_limit','descriptor family mismatch','concrete descriptor grounding mismatch','nonmechanical origin'}
    if cells.is_dir():
        paths=sorted(cells.iterdir())
        if len(paths)>16:raise ValueError('cell count cap')
        for folder in paths:
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',folder.name):raise ValueError('cell name')
            ledger=folder/'ledger.json'
            progress=progress_record(folder/'progress.json') if (folder/'progress.json').exists() else None
            if not ledger.exists():
                if progress is not None:result['cells'].append(dict(cell=folder.name,status='INCOMPLETE',progress=progress))
                continue
            value=bounded(ledger)
            if value.get('version')!='generalization-cell-preparation/v1':raise ValueError('original cell ledger required')
            if value.get('status') not in {'PASS','BLOCKED'} or value.get('role') not in {'training','selection','assessment'}:
                raise ValueError('cell status/role')
            counts=Counter()
            for row in value['rejections']:
                reason=row.get('reason','')
                counts[reason if reason in known else 'reference exclusion' if reason.startswith('excluded:') else 'OTHER']+=1
            shortfalls=[]
            for member in value['shortfalls']:
                if member in {'preparation_deadline','native_validation_failed'}:
                    shortfalls.append(member);continue
                if not isinstance(member,str) or not re.fullmatch(r'(?:rows|[ABCD]|short|long|within64|above64|loop_active|branch_active|loop_active_families|branch_active_families|family:sha256:[a-f0-9]{64}):[0-9]{1,7}',member):
                    raise ValueError('unknown shortfall label')
                shortfalls.append(member)
            if type(value['draws']) is not int or not 0<=value['draws']<=1_000_000:raise ValueError('draw count')
            result['cells'].append(dict(cell=folder.name,status=value['status'],role=value['role'],draws=value['draws'],
                rejected=sum(counts.values()),rejection_counts=dict(counts),shortfalls=shortfalls,
                original=reference(ledger,root)))
            if progress is not None:result['cells'][-1]['progress']=progress
    return seal(result)


def feasibility_diagnostics(root):
    """Metadata-only projection, including the last completed cell after timeout."""
    import re
    folder=Path(root)/'feasibility'
    for name in ('feasibility.json','progress.json'):
        path=folder/name
        if not path.exists():continue
        path=confined(root,path)
        if path.stat().st_size>1<<20:raise ValueError('feasibility metadata cap')
        value=json.loads(path.read_bytes())
        if seal(value)!=value or value.get('version')!='fresh-pilot-feasibility/v1':raise ValueError('original source feasibility required')
        if value['status'] not in {'FEASIBLE','BLOCKED','INTERRUPTED'}:raise ValueError('feasibility status')
        result=dict(version='fresh-pilot-feasibility-diagnostics/v1',status=value['status'],
            qualification_evidence=False,cells=[],original=reference(path,root))
        if len(value['cells'])>16:raise ValueError('feasibility cell cap')
        for cell in value['cells']:
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',cell['name']):raise ValueError('cell name')
            row=dict(name=cell['name'])
            for key in ('draws','admitted','selected','tokens'):
                if key in cell:
                    n=cell[key]
                    if type(n) is not int or not 0<=n<=1_000_000_000:raise ValueError('feasibility count')
                    row[key]=n
            if cell['status'] not in {'FEASIBLE','BLOCKED'}:raise ValueError('cell status')
            row['status']=cell['status'];row['shortfalls']=[]
            for item in cell['shortfalls']:
                if item not in {'preparation_deadline','native_validation_failed'} and not re.fullmatch(r'(?:rows|[ABCD]|short|long|within64|above64|loop_active|branch_active|loop_active_families|branch_active_families|family:sha256:[a-f0-9]{64}):[0-9]{1,7}',item):raise ValueError('shortfall label')
                row['shortfalls'].append(item)
            row['rejection_counts']={}
            known={'cross-role exclusion','duplicate refresh source','native/reference disagreement',
                'excluded:input_integer_limit','descriptor family mismatch','concrete descriptor grounding mismatch','nonmechanical origin'}
            for reason,n in cell.get('rejection_counts',{}).items():
                if type(n) is not int or not 0<=n<=1_000_000:raise ValueError('rejection count')
                label=reason if reason in known else 'OTHER'
                row['rejection_counts'][label]=row['rejection_counts'].get(label,0)+n
            result['cells'].append(row)
        if type(value.get('training_tokens')) is int:result['training_tokens']=value['training_tokens']
        return seal(result)
    return None


def training_timing(value):
    """Bounded aggregate projection; never export arbitrary nested worker payloads."""
    import math,re
    if value.get('version')!='generalization-training-timing/v1':raise ValueError('training timing version required')
    def numbers(row,keys):
        result={}
        for key in keys:
            if key not in row:continue
            n=row[key]
            if n is not None and (type(n) not in (int,float) or not math.isfinite(n)):raise ValueError('finite numeric progress required')
            result[key]=n
        return result
    fields=('checkpoint_seconds','update_seconds','checkpoint_calls','journal_writes','invocation_seconds',
        'max_checkpoint_seconds','publication_seconds','publication_reserve_seconds','last_loss','last_gradient_norm')
    result=dict(version=value['version'],**numbers(value,fields))
    if 'checkpoint_policy' in value:
        if value['checkpoint_policy'] not in {'hourly/v1','per-update/v1'}:raise ValueError('registered persistence policy required')
        result['checkpoint_policy']=value['checkpoint_policy']
    for key,limit in (('windows',64),('trend',128)):
        if key not in value:continue
        if not isinstance(value[key],list) or len(value[key])>limit:raise ValueError('timing window bound')
        result[key]=[numbers(w,('updates','tokens','seconds','through_updates','loss_sum','loss_observations','mean_loss')) for w in value[key]]
    if 'checkpoint_io' in value:result['checkpoint_io']=numbers(value['checkpoint_io'],('serialized_bytes','reused_tensor_bytes','verified_bytes','verification_cache_hits'))
    if 'progress' in value:
        result['progress']=numbers(value['progress'],('tokens','charged_tokens','discarded_tokens','updates','recoverable_tokens','recoverable_updates'))
        sha=value['progress'].get('checkpoint',{}).get('sha256')
        if sha is not None:
            if not re.fullmatch(r'sha256:[0-9a-f]{64}',sha):raise ValueError('checkpoint digest required')
            result['progress']['checkpoint_sha256']=sha
    return result


def diagnostics(root,destination):
    """Bounded control-only walk for this temporary multi-stage layout."""
    import os,stat
    from oeis_learn.pipeline_validation.generalization_diagnostics import control_fields,terminal_fields
    root=Path(root);destination=Path(destination);destination.mkdir(parents=True)
    skip={'public','recovery','diagnostics','acceptance-inputs','design','inputs','checkpoints',
          'cells','ledgers','scorer','visible','generation','selection','qualified','selected','.runtime','native-cache','command-export'}
    files=[];used=0;entries=0
    try:
        summary=preparation_diagnostics(root)
        target=destination/'preparation-summary.json';write(target,summary);used=target.stat().st_size
        if summary['cells'] or 'terminal' in summary:
            files.append(dict(source='research-preparation-ledgers',status='PROJECTED',artifact=reference(target,destination)))
        else:target.unlink();used=0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        files.append(dict(source='research-preparation-ledgers',status='OMITTED',reason=type(exc).__name__))
    try:
        summary=feasibility_diagnostics(root)
        if summary is not None:
            target=destination/'feasibility-summary.json';write(target,summary);used+=target.stat().st_size
            files.append(dict(source='source-feasibility',status='PROJECTED',artifact=reference(target,destination)))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        files.append(dict(source='source-feasibility',status='OMITTED',reason=type(exc).__name__))
    try:
        from fresh_pilot_checkpoint import collect
        summary=collect(root)
        if summary['runs']:
            target=destination/'checkpoint-summary.json';write(target,summary);used+=target.stat().st_size
            files.append(dict(source='original-learner-checkpoints',status='PROJECTED',artifact=reference(target,destination)))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        files.append(dict(source='original-learner-checkpoints',status='OMITTED',reason=type(exc).__name__))
    for parent,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in skip and not (Path(parent)/d).is_symlink())
        entries+=len(dirs)+len(names)
        if entries>16384:
            files.append(dict(status='OMITTED',reason='diagnostic discovery bound'));break
        for name in sorted(names):
            if name not in {'result.json','process.json','cleanup.json','disk.json','timing.json','stdout.log','stderr.log'}:continue
            path=Path(parent)/name;source=str(path.relative_to(root));row=dict(source=source);files.append(row)
            try:
                if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):raise ValueError('regular control required')
                if used>=4<<20:raise ValueError('diagnostic byte cap')
                size=path.stat().st_size
                if name.endswith('.log'):
                    with path.open('rb') as stream:
                        data=stream.read(32768)
                        if size>65536:stream.seek(-32768,2);data+=b'\n[opening and final tail]\n'+stream.read(32768)
                        else:data+=stream.read(32768)
                else:
                    if size>256<<10:raise ValueError('control JSON cap')
                    value=json.loads(path.read_bytes())
                    data=(json.dumps((training_timing if name=='timing.json' else terminal_fields if name=='result.json' else control_fields)(value),sort_keys=True)+'\n').encode()
                if used+len(data)>4<<20:raise ValueError('diagnostic byte cap')
                target=destination/(digest(source)[7:]+'-'+name);target.write_bytes(data);used+=len(data)
                row.update(status='PROJECTED' if name.endswith('.json') else 'COPIED' if size<=65536 else 'TRUNCATED',
                    original_bytes=size,artifact=reference(target,destination))
            except (OSError,ValueError,TypeError) as exc:row.update(status='OMITTED',reason=str(exc)[:256])
    write(destination/'index.json',seal(dict(version='fresh-pilot-diagnostics/v1',files=files,bytes=used,qualification_evidence=False)))
    return [(p,'fresh/diagnostics/'+p.name) for p in sorted(destination.iterdir())]


def row_key(arm,control,key):return digest([arm,control,key])


def measurement_plan(study,targets):
    """Freeze one selected-policy outcome per row/control, before learner work.

    The single evidence slot is not a proposal count. Inference still uses the
    registered eight-draw controls and one-draw raw/rescue diagnostic. Original
    attempt/cost ledgers accompany these deployment-outcome projections.
    """
    from collections import Counter
    if study['seeds']!=[20261016] or len({r['id'] for r in targets})!=256:
        raise ValueError('complete fixed research roster required')
    if any(r['role']!='selection' for r in targets) or Counter(r['stratum'] for r in targets)!=dict(parameter=128,structural=128):
        raise ValueError('two 128-row selection strata required; assessment remains sealed')
    return dict(version='generalization-measurement-plan/v1',study=study['identity'],required_stages=list(REQUIRED),
        roster={row_key(arm,control,row['id']):dict(family=row['family'],seed=20261016,
            cell=arm+'/'+control+'/'+row['stratum'],attempts=1)
            for arm in ARMS for control in CONTROLS for row in targets})


def measurements(plan,targets,scored):
    ids={r['id'] for r in targets};records=[]
    if set(scored)-set(ARMS):raise ValueError('unregistered measurement arm')
    for arm,report in scored.items():
        outcomes=report['result']['outcomes']
        if set(outcomes)-ids:raise ValueError('outcome outside frozen roster')
        for key,controls in outcomes.items():
            if set(controls)-set(CONTROLS):raise ValueError('unregistered measurement control')
            for control,result in controls.items():
                if type(result.get('complete')) is not bool or type(result.get('success')) is not bool:
                    raise ValueError('explicit boolean completion/outcome required')
                row=row_key(arm,control,key)
                if row not in plan['roster']:raise ValueError('measurement outside frozen roster')
                records.append(dict(row=row,ordinal=0,outcome=result['success'] if result['complete'] else None,
                    queue_us=None,service_us=None,cleanup_us=None))
    return dict(version='generalization-attempts/v1',attempts=records)


def terminal_status(code,original):
    if code==130:return 'INTERRUPTED'
    status=original.get('status') if isinstance(original,dict) else None
    return status if status in EXIT_CODES and EXIT_CODES[status]==code else 'FAIL'


def assemble_public(output,*,study,plan,targets,scored,stages,qualified,selection,extra=()):
    """Copy exact original proof files and derive only public aggregate projections."""
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.generalization.bundles import reconcile_ledger
    from oeis_learn.generalization.reconciliation import reconcile_measurements
    from oeis_learn.pipeline_validation.generalization_qualification_io import public_closure
    from oeis_learn.generalization.feature_acceptance import feature_artifacts
    from oeis_learn.pipeline_validation.bundle import public
    output=Path(output);qualified=Path(qualified)
    if output.exists():raise ValueError('fresh public evidence root required')
    output.mkdir(parents=True)
    used=0;members=0
    def reserve(size):
        nonlocal used,members
        if size<0 or used+size>PUBLIC_CAP or members>=9900:raise ValueError('public evidence budget exceeded')
        used+=size;members+=1
    def put(name,value):
        reserve(len(json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode())+1)
        write(output/name,value);return reference(output/name,output)
    def copy(source,name):
        import shutil,stat
        if not public(Path(name)):raise ValueError('private or unsupported public evidence path')
        source=Path(source)
        if any(p.is_symlink() for p in (source,*source.parents)) or not stat.S_ISREG(source.stat().st_mode):
            raise ValueError('regular original evidence required')
        before=source.stat()
        if before.st_size>PUBLIC_CAP:raise ValueError('public evidence budget exceeded')
        ref=reference(source,source.parent)
        destination=confined(output,name,exists=False);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            other=reference(destination,destination.parent)
            if (other['bytes'],other['sha256'])!=(ref['bytes'],ref['sha256']):raise ValueError('conflicting evidence path')
        else:
            reserve(before.st_size)
            with source.open('rb') as incoming,destination.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing,1<<20)
        after=source.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('evidence changed during copy')
        verify_artifact(destination.parent,dict(ref,path=destination.name),max_bytes=PUBLIC_CAP)
        return reference(destination,output)
    authority=json.loads((qualified/'qualification.json').read_text())
    feature=json.loads((qualified/'feature.json').read_text())
    producer=json.loads(verify_artifact(qualified,authority['producer']['receipt']))
    refs=[reference(qualified/'qualification.json',qualified),*public_closure(qualified,authority),
          reference(qualified/'feature.json',qualified),*feature_artifacts(feature,qualification=producer['identity'],root=qualified)]
    for ref in refs:
        verify_artifact(qualified,ref,max_bytes=512<<20);copy(qualified/ref['path'],ref['path'])
    ledger=dict(study=study['identity'],required=list(REQUIRED),receipts=[])
    for stage in stages:
        name=stage['stage']
        if name not in REQUIRED:raise ValueError('unregistered public stage')
        originals=[]
        for i,path in enumerate(stage.get('originals',[])):
            source=Path(path)
            if source.stat().st_size>16<<20:raise ValueError('original terminal size cap')
            original=json.loads(source.read_bytes())
            if seal(original)!=original or terminal_status(stage['exit_code'],original)!=stage['status']:
                raise ValueError('stage differs from original terminal result')
            originals.append(copy(Path(path),f'fresh/originals/{name}-{i}.json'))
        if not originals and stage['status'] not in {'FAIL','INTERRUPTED'}:
            raise ValueError('nonfailure stage requires its original terminal result')
        record=dict(version='fresh-pilot-stage/v1',study=study['identity'],stage=name,
            status=stage['status'],exit_code=stage['exit_code'],elapsed_seconds=stage['elapsed_seconds'],
            command=stage['command'],original_artifacts=originals)
        if stage.get('execution')=='REUSED':
            record.update(execution='REUSED',parent_workflow=stage['parent_workflow'])
            if 'source_transition' in stage:record['source_transition']=stage['source_transition']
        record=seal(record)
        put(f'fresh/stages/{name}.json',record)
        ledger['receipts'].append(dict(stage=name,status=stage['status'],study=study['identity'],original=record['identity']))
    for source,name in extra:copy(Path(source),name)
    attempts=measurements(plan,targets,scored)
    projections=reconcile_measurements(plan,attempts,study)
    projected={name:put('fresh/'+name+'.json',value) for name,value in projections.items()}
    for arm,report in scored.items():put('fresh/scoring-'+arm+'.json',report)
    record=dict(version='generalization-evidence/v1',study=put('fresh/study.json',study),
        effective_invocation=put('fresh/measurement-plan.json',plan),qualification_producers=[reference(output/'qualification.json',output)],
        stage_ledger=put('fresh/stage-ledger.json',ledger),identities=[study['identity'],feature['identity']],
        attempts=put('fresh/attempts.json',attempts),selection=put('fresh/selection.json',selection),
        exposure=put('fresh/exposure.json',dict(assessment='SEALED',accesses=0,scope='selection supervision only')),
        **projected,software_status=reconcile_ledger(ledger,study=study['identity']),scientific_status='INCONCLUSIVE',
        interpretation='Exploratory single-seed fresh screen. One recorded outcome slot per selected policy, not an oracle union or proposal count. Actual proposal/cost ledgers are separate. Missing/censored results retain fixed denominators; unequal/partial training exposure cannot support a matched-cap claim. No confirmation, assessment exposure or automatic advancement.',
        omissions=['private optimizer/RNG and tensor state','sealed assessment sources and terms','selection target sources and terms','private runtime caches'])
    record['inventory']=[reference(p,output) for p in sorted(output.rglob('*')) if p.is_file()]
    write(output/'evidence.json',seal(record))
    return record
