"""Bounded copies of explicitly public control records, never role input/state trees."""
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


def candidates():
    # Fixed paths: do not recursively discover similarly named private files.
    for stage in ('preflight','prepare'):
        for name in ('result.json','summary.json'):yield f'{stage}/{name}'
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
    rows=[];used=0
    for name in candidates():
        path=root/name;row=dict(source=name);rows.append(row)
        try:
            relative=Path(name)
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
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(data)
            used+=len(data)
            row.update(status=status,exported_bytes=len(data),sha256='sha256:'+hashlib.sha256(data).hexdigest())
        except (OSError,ValueError) as exc:
            row.update(status='OMITTED',reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__)
    result=seal(dict(version='generalization-diagnostics/v1',qualification_evidence=False,
        exported_bytes=used,payload_cap=cap,json_cap=JSON_CAP,xml_cap=XML_CAP,log_cap=LOG_CAP,files=rows))
    write(destination/'index.json',result)
    return result
