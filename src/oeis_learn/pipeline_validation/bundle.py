"""Mandatory-first bounded public evidence export and member verification."""
import json
import os
from pathlib import Path
import stat
import zipfile
from .identity import seal, file_hash, confined, write, validate

EXCLUDED = {'input','inputs','private','cache','caches','.runtime','pytest','__pycache__'}


def public(path):
    return path.suffix in {'.json','.jsonl','.xml','.log','.md','.txt','.csv','.html'} and not (set(path.parts)&EXCLUDED or path.suffix in {'.pt','.pth','.bin','.ckpt','.zip','.lock'} or 'truth' in path.name or 'weights' in path.name)


def pack(root, *, mandatory, cap=512<<20, optional_cap=128<<20):
    root=Path(root)
    archive=root.with_suffix('.zip')
    if archive.exists():
        raise ValueError('archive exists; use a fresh invocation')
    members, omissions=[],[]
    used=0
    for name in dict.fromkeys(mandatory):
        path=confined(root,name)
        info=path.stat()
        if not public(Path(name)) or not stat.S_ISREG(info.st_mode) or info.st_mode & 0o444 == 0:
            raise ValueError('unreadable/unsafe mandatory file: '+str(path))
        size=info.st_size
        if used+size > cap-(1<<20):
            raise ValueError('mandatory evidence exceeds archive cap: '+str(path))
        members.append(dict(path=name,bytes=size,sha256=file_hash(path),mandatory=True));used+=size
    names={r['path'] for r in members}
    optional=0
    for path in sorted(root.rglob('*')):
        name=str(path.relative_to(root))
        if name in names or name=='manifest.json':
            continue
        if path.is_symlink():
            omissions.append(dict(path=name,reason='symlink'));continue
        if not path.is_file() or not public(Path(name)):
            continue
        size=path.stat().st_size
        if used+size>cap-(1<<20) or optional+size>optional_cap:
            omissions.append(dict(path=name,reason='optional byte cap'));continue
        members.append(dict(path=name,bytes=size,sha256=file_hash(path),mandatory=False));used+=size;optional+=size
    summary=json.loads((root/'summary.json').read_text())
    manifest=seal(dict(version='pipeline-bundle/v1',members=members,omissions=omissions,
                       summary_state=summary['selected_scope'],excluded=sorted(EXCLUDED)))
    write(root/'manifest.json',manifest)
    try:
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for member in members:
                z.write(confined(root,member['path']),member['path'])
            z.write(root/'manifest.json','manifest.json')
        if archive.stat().st_size>cap:
            raise ValueError('compressed archive exceeds cap')
        verify(archive)
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    return archive


def verify(archive):
    import hashlib
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):
            raise ValueError('duplicate archive members')
        manifest=json.loads(z.read('manifest.json'))
        validate(manifest,{'version','members','omissions','summary_state','excluded'},'pipeline-bundle/v1')
        expected={row['path'] for row in manifest['members']}|{'manifest.json'}
        if set(names)!=expected:
            raise ValueError('archive inventory mismatch')
        for row in manifest['members']:
            if set(row)!={'path','bytes','sha256','mandatory'}:
                raise ValueError('unknown bundle member fields')
            path=Path(row['path'])
            info=z.getinfo(row['path'])
            if path.is_absolute() or '..' in path.parts or not public(path) or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('unsafe archive member')
            if info.file_size!=row['bytes'] or info.file_size>512<<20:
                raise ValueError('archive member size mismatch')
            h=hashlib.sha256()
            with z.open(info) as stream:
                for block in iter(lambda:stream.read(1<<20),b''):
                    h.update(block)
            if 'sha256:'+h.hexdigest()!=row['sha256']:
                raise ValueError('archive member hash mismatch')
        summary=json.loads(z.read('summary.json'))
        if summary.get('stages'):
            from .report import summarize
            verdict=summarize(summary['stages'],interrupted=summary['selected_scope']=='INTERRUPTED')['selected_scope']
            if verdict!=summary['selected_scope']:
                raise ValueError('summary contradicts stage outcomes')
        if summary['selected_scope']!=manifest['summary_state']:
            raise ValueError('contradictory summary')
    return manifest
