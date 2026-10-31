"""Bounded public control copies, with metadata-only discovery of nested roles."""
import hashlib
import os
from pathlib import Path
import stat
from .identity import seal,write

JSON_CAP=256<<10
XML_CAP=2<<20
LOG_CAP=64<<10
TOTAL_CAP=4<<20
CASES=('parameter-targets','refresh','retained-eval','replay-resume')
CONTROL=('process.json','cleanup.json','stdout.log','stderr.log')
ROLE_SCAN_ENTRIES=8192
ROLE_SCAN_DIRS=2048
ROLE_SCAN_ROLES=256
ROLE_SCAN_DEPTH=32
ROLE_SKIP=frozenset({'input','inputs','checkpoints','.runtime','native-cache',
                     'cache','caches','selected','cells'})


def nested_controls(root):
    """Discover only role/supervision controls; never read discovered payloads."""
    pending=[];names=[];omissions=[];entries=directories=roles=0
    for case in (*CASES,'failure-packaging'):
        relative=Path(case)/'case/private'
        if any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)):
            omissions.append(dict(source=str(relative),status='OMITTED',reason='symlink'))
        elif (root/relative).is_dir():pending.append((relative,0))
    bounded=False
    while pending and not bounded:
        relative,depth=pending.pop();directories+=1
        if directories>ROLE_SCAN_DIRS or depth>ROLE_SCAN_DEPTH:
            bounded=True;break
        try:
            # Recheck queued directories; never descend links or role payloads.
            if any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)):
                omissions.append(dict(source=str(relative),status='OMITTED',reason='symlink'));continue
            with os.scandir(root/relative) as children:
                for child in children:
                    entries+=1
                    if entries>ROLE_SCAN_ENTRIES:bounded=True;break
                    if child.name in ROLE_SKIP:continue
                    if child.is_symlink():
                        omissions.append(dict(source=str(relative/child.name),status='OMITTED',reason='symlink'));continue
                    if not child.is_dir(follow_symlinks=False):continue
                    path=relative/child.name
                    if child.name=='role':
                        if roles>=ROLE_SCAN_ROLES:bounded=True;break
                        roles+=1
                        names.extend(str(path/'supervision'/name) for name in CONTROL)
                    else:pending.append((path,depth+1))
        except OSError as exc:
            omissions.append(dict(source=str(relative),status='OMITTED',reason=type(exc).__name__))
    if bounded:
        omissions.append(dict(source='nested-role-discovery',status='OMITTED',reason='nested role discovery bound'))
    return sorted(names),omissions


def candidates(*,nested=()):
    # Fixed public paths plus the four controls of bounded discovered roles.
    # Prioritize terminal explanations before potentially large suite logs/XML.
    for stage in ('preflight','prepare',*CASES,'failure-packaging','report','verify-bundle'):
        for name in ('result.json','summary.json'):yield f'{stage}/{name}'
    from oeis_learn.generalization.feature_acceptance import OBLIGATIONS
    for case,(_,labels) in OBLIGATIONS.items():
        for name in ('receipt.json','case-result.json',*(label+'.json' for label in labels)):
            yield f'{case}/case/{name}'
    yield 'replay-resume/bootstrap.json'
    yield from nested
    for operation in ('report/packaging','verify-bundle/verification'):
        yield operation+'/role/result.json'
        for name in CONTROL:yield operation+'/role/supervision/'+name
    for stage in ('prepare',*CASES,'failure-packaging'):
        base=f'{stage}/qualification'
        for position in range(2):
            yield f'{base}/inspection/image-{position}/cleanup.json'
            for name in CONTROL:yield f'{base}/inspection/image-{position}/logs/{name}'
        for name in CONTROL:yield f'{base}/inspection/device/logs/{name}'
        yield f'{base}/verification/role/result.json'
        for name in CONTROL:yield f'{base}/verification/role/supervision/{name}'
    for stage in ('preflight','prepare'):
        for case in CASES:
            role=f'{stage}/{case}/validation/role'
            yield role+'/result.json'
            for name in CONTROL:yield role+'/supervision/'+name
    for case in CASES:
        role=f'{case}/validation/role'
        yield role+'/result.json'
        for name in CONTROL:yield role+'/supervision/'+name
    for stage in ('image.qualification','cpu.semantic','image.gpu','hip.preflight'):
        yield f'qualify/pipeline/stages/{stage}/receipt.json'
        yield f'qualify/pipeline/stages/{stage}/inspection/cleanup.json'
        for name in CONTROL:yield f'qualify/pipeline/stages/{stage}/inspection/logs/{name}'
    for stage,suite in (('cpu.semantic','bigint-cpu'),('hip.preflight','bigint-gpu')):
        for step in ('environment',suite+'-regressions'):
            for name in (*CONTROL,'receipt.json','results.xml'):
                yield f'qualify/pipeline/stages/{stage}/{step}/{name}'
    yield 'report/inspection/cleanup.json'
    for name in CONTROL:yield 'report/inspection/logs/'+name


def collect(root,destination,*,cap=TOTAL_CAP):
    """Snapshot controls with explicit omissions; originals and status are untouched."""
    if type(cap) is not int or not 0<=cap<=TOTAL_CAP:raise ValueError('diagnostic byte cap')
    root=Path(root).resolve(strict=True);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    nested,omissions=nested_controls(root);nested=set(nested)
    rows=[];used=0
    for name in candidates(nested=sorted(nested)):
        relative=Path(name)
        # Archive paths must never expose private directories as public payloads.
        export_path=(str(Path('nested-roles')/hashlib.sha256(str(relative.parent).encode()).hexdigest()/relative.name)
                     if name in nested else name)
        path=root/name;row=dict(source=name,export_path=export_path);rows.append(row)
        try:
            if any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)):
                raise ValueError('symlink')
            if not path.exists():row.update(status='MISSING');continue
            # Nonblocking/no-follow also rejects FIFO/link replacements at open.
            fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
            with os.fdopen(fd,'rb') as stream:
                before=os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):raise ValueError('nonregular file')
                size=before.st_size;row['original_bytes']=size
                limit=LOG_CAP if path.suffix=='.log' else XML_CAP if path.suffix=='.xml' else JSON_CAP
                if path.suffix!='.log' and size>limit:raise ValueError(('XML' if path.suffix=='.xml' else 'JSON')+' byte cap')
                if min(size,limit)>cap-used:raise ValueError('total byte cap')
                if size>limit:
                    marker=b'\n...[diagnostic log truncated: opening and final tail]...\n'
                    half=(limit-len(marker))//2
                    data=stream.read(half)+marker
                    stream.seek(-(limit-half-len(marker)),os.SEEK_END)
                    data+=stream.read(limit-half-len(marker))
                    status='TRUNCATED'
                else:data=stream.read(limit+1);status='COPIED'
                after=os.fstat(stream.fileno())
                if len(data)>limit or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise ValueError('file changed during capture')
            target=destination/export_path;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(data)
            used+=len(data)
            row.update(status=status,exported_bytes=len(data),sha256='sha256:'+hashlib.sha256(data).hexdigest())
        except (OSError,ValueError) as exc:
            row.update(status='OMITTED',reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__)
    rows.extend(omissions)
    result=seal(dict(version='generalization-diagnostics/v1',qualification_evidence=False,
        exported_bytes=used,payload_cap=cap,json_cap=JSON_CAP,xml_cap=XML_CAP,log_cap=LOG_CAP,files=rows))
    write(destination/'index.json',result)
    return result


def export(root,destination):
    """Package existing allowed records without executing or mutating the run."""
    from .bundle import pack,verify
    root=Path(root).resolve(strict=True);destination=Path(destination).resolve()
    archive=destination.with_suffix('.zip')
    if (not root.is_dir() or destination==root or root in destination.parents or
        destination in root.parents or destination.exists() or archive.exists()):
        raise ValueError('fresh disjoint diagnostic output and archive required')
    destination.mkdir(parents=True,exist_ok=False)
    index=collect(root,destination/'diagnostics')
    write(destination/'summary.json',seal(dict(selected_scope='PARTIAL',
        reason='DIAGNOSTIC_EXPORT_ONLY',feature_qualified=False,scientific_status='UNMEASURED')))
    mandatory=['summary.json','diagnostics/index.json']
    mandatory.extend('diagnostics/'+r['export_path'] for r in index['files'] if r['status'] in {'COPIED','TRUNCATED'})
    archive=pack(destination,mandatory=mandatory)
    verify(archive)
    return dict(archive=str(archive),qualification_evidence=False,exported_bytes=index['exported_bytes'])


def main(argv=None):
    import argparse,json
    parser=argparse.ArgumentParser(description='Export bounded existing diagnostics; no container or HIP execution.')
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:result=export(args.run,args.output)
    except (OSError,ValueError) as exc:
        parser.exit(1,str(exc)+'\n')
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
