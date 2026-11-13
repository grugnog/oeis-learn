"""Stdlib inventory and narrowly scoped retirement of completed timing tensors.

No whole-run deletion. Only weights.pt/optimizer.pt inside successful, explicitly
DIAGNOSTIC_ONLY runtime/training timing cases can be retired. All original JSON,
inputs, journals, ZIPs, source records and other checkpoint state remain in place.
"""
from pathlib import Path
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import zipfile
from .identity import confined,seal
from .progressive_workstation_state import read,write
from .progressive_research_inputs import checked_file
from oeis_learn.generalization.progressive.contracts import reference

VERSION='discovery-timing-cleanup/v1'


def safe(root,path,*,exists=True):
    return confined(root,path,exists=exists)


def stamp(path):
    s=path.lstat()
    if not stat.S_ISREG(s.st_mode):raise ValueError('regular file required: '+str(path))
    return dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,
                ctime_ns=s.st_ctime_ns,links=s.st_nlink,blocks=s.st_blocks)


@contextlib.contextmanager
def locks(root,sessions):
    with contextlib.ExitStack() as stack:
        for p in [root/'run-state/gpu.lock',*[safe(root,s)/'session.lock' for s in sorted(sessions)]]:
            p=safe(root,p,exists=False);p.parent.mkdir(parents=True,exist_ok=True)
            stream=stack.enter_context(p.open('a'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/"run-state/gpu-pending.json").exists():raise ValueError("unreconciled GPU owner; cleanup blocked")
        yield


def owned(root,folder,phase):
    journal=read(safe(root,folder/'journal.json'))
    steps={k.split('/',1)[1]:v for k,v in journal['steps'].items() if k.startswith(phase+'/')}
    needed=('runtime-measurements','runtime-review') if phase=='measure-runtime' else ('training-measurements','runtime-review')
    if any(steps.get(n,{}).get('status')!='COMPLETE' for n in needed):
        raise ValueError('completed timing measurements and review required: '+str(folder))
    if any(v.get('status') in {'RUNNING','INTERRUPTED'} for v in journal['steps'].values()):
        raise ValueError('active or interrupted session cannot be cleaned')
    refs={r['path']:r for s in steps.values() for r in s.get('outputs',[])}
    for ref in refs.values():checked_file(root,ref)
    return refs


def candidates(root,session):
    folder=safe(root,session)
    if not folder.is_relative_to(root/'run-state/discovery'):
        raise ValueError('explicit discovery timing session required')
    binding=read(safe(root,folder/'session.json'))
    if seal(binding)!=binding:raise ValueError('original session seal differs')
    runtime=folder/'measure-runtime/timing';training=folder/'measure-training/timing'
    if runtime.is_dir()==training.is_dir():raise ValueError('one recognized timing phase required')
    phase='measure-runtime' if runtime.is_dir() else 'measure-training';base=runtime if runtime.is_dir() else training
    refs=owned(root,folder,phase)
    req=read(safe(root,base/'request.json'))
    if reference(root,base/'request.json') not in refs.values():raise ValueError('request not journal-bound')
    if seal(req)!=req or not req.get('scope','').startswith('DIAGNOSTIC_ONLY'):
        raise ValueError('explicit diagnostic request required')
    roots=[];evidence=[reference(root,folder/'session.json'),reference(root,folder/'journal.json'),reference(root,base/'request.json')]
    if phase=='measure-runtime':
        from .progressive_runtime_timing import validate_request
        validate_request(req)
        for i,case in enumerate(req['cases']):
            case_root=base/'cases'/str(i);path=safe(root,case_root/'producer.json');p=read(path)
            if (seal(p)!=p or p.get('status')!='PASS' or p.get('scope')!='DIAGNOSTIC_ONLY'
                    or p.get('request')!=req['identity'] or p.get('case')!=case):
                raise ValueError('complete original diagnostic case required')
            if reference(root,path) not in refs.values():raise ValueError('case not journal-bound')
            evidence.append(reference(root,path))
            checked_file(case_root,p['diagnostics']);evidence.append(reference(root,case_root/p['diagnostics']['path']))
            if case['kind']=='learner':
                campaign_path=safe(root,case_root/'private/learner/campaign.json');campaign=read(campaign_path)
                if seal(campaign)!=campaign or campaign['purpose']!='acceptance':raise ValueError('diagnostic campaign required')
                evidence.append(reference(root,campaign_path))
                from oeis_learn.generalization.progressive.contracts import record
                for task in campaign['payload']['tasks']:
                    bound=record('task',dict(task,campaign=campaign['identity']),
                        created_at=campaign['created_at'],purpose=campaign['purpose'],track=campaign['track'])
                    roots.append(case_root/'private/learner/learning'/bound['identity'][7:]/'checkpoints')
            else:roots.append(case_root/'private/calibration/checkpoints')
    else:
        from .progressive_training_timing import validate_request
        validate_request(req)
        path=safe(root,base/'measurement/producer.json');p=read(path)
        if seal(p)!=p or p.get('status')!='PASS' or not p.get('scope','').startswith('DIAGNOSTIC_ONLY'):
            raise ValueError('complete diagnostic training producer required')
        if reference(root,path) not in refs.values():raise ValueError('producer not journal-bound')
        evidence.append(reference(root,path));roots.append(base/'measurement/checkpoints')
    files=[]
    for checkpoint_root in roots:
        checkpoint_root=safe(root,checkpoint_root,exists=False)
        if not checkpoint_root.exists():continue
        for generation in sorted(checkpoint_root.iterdir()):
            if generation.is_symlink():raise ValueError('checkpoint alias rejected')
            if not generation.is_dir():continue
            if not re.fullmatch(r'generalization-[0-9a-f]{32}',generation.name):continue
            state=safe(root,generation/'state.json');doc=read(state)
            if seal(doc)!=doc or doc.get('version')!='generalization-learner-state/v1':raise ValueError('original sealed checkpoint manifest required')
            evidence.append(reference(root,state))
            for name in ('weights','optimizer'):
                path=safe(root,generation/(name+'.pt'),exists=False)
                if not path.exists():continue
                ref=doc.get('states',{}).get(name,{})
                if ref.get('path')!=generation.name+'/'+name+'.pt':raise ValueError('original checkpoint tensor binding differs')
                files.append(dict(path=str(path.relative_to(root)),original_sha256=ref['sha256'],**stamp(path)))
    return files,evidence


def inventory(root):
    """Metadata-only scan; never classify unknown runs as disposable."""
    groups=[];largest=[];errors=[]
    paths=[]
    for name in ('data','runs','tmp-validation','tmp-fresh-inputs-extended-002'):
        if (root/name).exists():paths.append(root/name)
    for parent in ('run-state/discovery','run-state/generalization'):
        if (root/parent).is_dir():paths.extend(p for p in (root/parent).iterdir() if p.is_dir())
    for folder in sorted(paths):
        size=0;allocated=0;count=0
        if folder.is_symlink():errors.append(str(folder)+': symlink skipped');continue
        def error(exc):errors.append(str(exc))
        for base,dirs,files in os.walk(folder,followlinks=False,onerror=error):
            dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink()]
            for name in files:
                p=Path(base)/name
                try:
                    s=p.lstat()
                    if not stat.S_ISREG(s.st_mode):continue
                    size+=s.st_size;allocated+=s.st_blocks*512;count+=1
                    if s.st_size>=128<<20:largest.append(dict(path=str(p.relative_to(root)),bytes=s.st_size,links=s.st_nlink))
                except OSError as exc:errors.append(str(exc))
        groups.append(dict(path=str(folder.relative_to(root)),bytes=size,allocated_bytes=allocated,files=count,
            disposition='KEEP_OR_REVIEW; no whole-run deletion authorized by this tool'))
    usage=shutil.disk_usage(root)
    return dict(filesystem=dict(total_bytes=usage.total,used_bytes=usage.used,free_bytes=usage.free),
        allocation_note='per-file allocated totals may double-count hardlinks; reclaimable plan lower bound excludes multiply-linked files',
        groups=groups,largest_files=sorted(largest,key=lambda r:r['bytes'],reverse=True)[:200],errors=errors,
        protected=['data','run-state/generalization/pac-012-bcd-r1-002','all research inputs, panels, exposures and scientific runs'],
        limitation='directory names/size do not prove that a historical run has no descendants')


def bundle(output):
    with zipfile.ZipFile(output/'cleanup-report.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('inventory.json','plan.json','retirement-intent.json','retirement.json','failure.json'):
            if (output/name).is_file():z.write(output/name,name)


def plan(root,sessions,output):
    root=Path(root).resolve();output=safe(root,output,exists=False)
    if not sessions or len(sessions)!=len(set(sessions)):raise ValueError('distinct explicit timing sessions required')
    if any(output.is_relative_to(safe(root,s)) for s in sessions):raise ValueError('separate maintenance output required')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():raise ValueError('plan already exists; use apply or a new output directory')
    with locks(root,sessions):
        files=[];evidence=[]
        for session in sessions:
            selected,refs=candidates(root,session);files+=selected;evidence+=refs
        value=seal(dict(version=VERSION,sessions=sessions,files=files,evidence=evidence,
            logical_bytes=sum(r['bytes'] for r in files),
            reclaimable_bytes_lower_bound=sum(r['blocks']*512 for r in files if r['links']==1),
            consequence='timing models become non-resumable; original measurements, history and all research inputs remain',
            root=str(root)))
        write(output/'inventory.json',inventory(root));write(output/'plan.json',value);bundle(output)
        return value


def apply(root,output):
    root=Path(root).resolve();output=safe(root,output);value=read(output/'plan.json')
    if seal(value)!=value or value.get('version')!=VERSION or value.get('root')!=str(root):raise ValueError('original cleanup plan required')
    with locks(root,value['sessions']):
        # Revalidate allowlist, original producers and evidence before any deletion.
        allowed=[]
        for session in value['sessions']:allowed+=candidates(root,session)[0]
        by_path={r['path']:r for r in value['files']}
        actual={r['path'] for r in allowed}
        existing={r['path'] for r in value['files'] if safe(root,r['path'],exists=False).exists()}
        if actual!=existing:raise ValueError('timing checkpoint inventory changed')
        for ref in value['evidence']:checked_file(root,ref)
        intent_path=output/'retirement-intent.json'
        if intent_path.exists():
            intent=read(intent_path)
            if seal(intent)!=intent or intent.get('plan')!=value['identity']:raise ValueError('retirement intent differs')
        else:
            # Hash the entire deletion set first. Any changed tensor aborts all deletion.
            refs=[]
            for row in value['files']:
                path=safe(root,row['path'])
                if stamp(path)!={k:row[k] for k in stamp(path)}:raise ValueError('planned tensor changed')
                ref=reference(root,path)
                if ref['sha256']!=row['original_sha256']:raise ValueError('original tensor hash differs')
                refs.append(ref)
            intent=seal(dict(version=VERSION,plan=value['identity'],files=refs,status='INVENTORIED'))
            write(intent_path,intent)
        originals={r['path']:r for r in intent['files']}
        if set(originals)!=set(by_path):raise ValueError('retirement inventory differs')
        for row in value['files']:
            path=safe(root,row['path'],exists=False)
            if path.exists():
                s=stamp(path)
                if any(s[k]!=row[k] for k in ('bytes','device','inode','mtime_ns')):raise ValueError('planned tensor changed')
                # On retry verify remaining bytes against durable pre-deletion hashes.
                if reference(root,path)!=originals[row['path']]:raise ValueError('retirement tensor changed')
        for row in value['files']:
            path=safe(root,row['path'],exists=False)
            if path.exists():
                path.unlink()
                fd=os.open(path.parent,os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
        receipt=seal(dict(version=VERSION,plan=value['identity'],inventory=reference(output,intent_path),
            status='RETIRED',files=len(value['files']),logical_bytes=value['logical_bytes'],
            scientific_history_deleted=False,timing_models_resumable=False))
        write(output/'retirement.json',receipt);bundle(output);return receipt


def main(root,argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('plan','apply'))
    p.add_argument('--session',action='append',default=[]);p.add_argument('--output',required=True)
    a=p.parse_args(argv);output=safe(Path(root),a.output,exists=False)
    try:
        result=plan(root,a.session,output) if a.action=='plan' else apply(root,output)
        print(json.dumps(result if a.action=='apply' else {k:result[k] for k in ('logical_bytes','reclaimable_bytes_lower_bound','consequence')},indent=2))
        print('Report:',output/'cleanup-report.zip')
        return 0
    except Exception as exc:
        output.mkdir(parents=True,exist_ok=True)
        write(output/'failure.json',dict(error=type(exc).__name__,message=str(exc)));bundle(output)
        raise
