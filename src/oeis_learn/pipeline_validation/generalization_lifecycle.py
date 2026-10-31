"""Bounded inventories and verified fresh-root command restarts."""
import json
from pathlib import Path
from .identity import seal,write,confined
from .generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact


def invocation(args,study,runtime):
    selected={k:getattr(args,k) for k in ('arms','seeds','checkpoints','proposal_bank','case') if hasattr(args,k)}
    return seal(dict(version='generalization-command-invocation/v1',study=study['identity'],runtime=runtime['identity'],operation=args.operation,selected=selected,resume=getattr(args,'resume',None)))


def inventory(root,*,cap=64<<30):
    refs=[];total=0
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():raise ValueError('symlink in command output')
        if not path.is_file() or path==Path(root)/'result.json':continue
        total+=path.stat().st_size
        if total>cap or len(refs)>=10000:raise ValueError('command output inventory cap')
        refs.append(reference(path,Path(root)))
    return refs


def reuse(previous,output,effective):
    """Recheck every original output byte before copying a compatible PASS."""
    from .generalization_commands import _copy
    previous=Path(previous);output=Path(output)
    record=json.loads((previous/'result.json').read_text())
    if seal(record)!=record or record.get('version')!='generalization-command-result/v2':raise ValueError('verified original command result required')
    if record['status']!='PASS' or record['invocation']!=effective or record['original'] is not None:
        raise ValueError('reuse requires compatible original PASS command')
    refs=record['inventory']
    if not isinstance(refs,list) or len(refs)>10000 or sum(r['bytes'] for r in refs)>64<<30:raise ValueError('reuse inventory cap')
    # A failed or missing output never silently reruns a learner. The operator
    # supplies a fresh invocation/full-state resume with the reason preserved.
    for ref in refs:verify_artifact(previous,ref)
    _copy(output,[(previous,r) for r in refs if r['path'] not in {'summary.json','invocation.json'} and not r['path'].startswith('qualification/')])
    write(output/'original-result.json',record)
    return record


from contextlib import contextmanager,nullcontext
from contextvars import ContextVar
_SCOPE=ContextVar('generalization_storage_scope',default=None)


@contextmanager
def scope(root,storage,*,extra_roots=()):
    token=_SCOPE.set((Path(root),storage,tuple(Path(p) for p in extra_roots)))
    try:yield
    finally:_SCOPE.reset(token)


def guard(output):
    current=_SCOPE.get()
    if current is None:return nullcontext()
    root,limits,extra=current
    from .disk import DiskWatchdog
    from .runner import ResourceBreach
    class StudyDisk(DiskWatchdog):
        def check(self):
            super().check()
            categories=dict(checkpoint_bytes=0,shard_bytes=0,raw_bytes=0)
            from itertools import chain
            seen=set()
            for path in chain.from_iterable(folder.rglob('*') for folder in (root,*extra) if folder.exists()):
                if path.is_symlink():raise ResourceBreach('symlink in study output')
                try:
                    if not path.is_file():continue
                    parts=path.parts;info=path.stat();key=(info.st_dev,info.st_ino)
                    if key in seen:continue
                    seen.add(key);size=info.st_size
                except FileNotFoundError:continue
                if 'checkpoints' in parts or 'selected' in parts or path.suffix in {'.pt','.pth','.safetensors'}:categories['checkpoint_bytes']+=size
                if 'cells' in parts and path.suffix=='.json':categories['shard_bytes']+=size
                if any(k in parts for k in ('logs','supervision','attempts','traces')):categories['raw_bytes']+=size
            for key,size in categories.items():
                if size>limits[key]:raise ResourceBreach('study reservation exceeded: '+key)
    return StudyDisk(root,output,cap=limits['study_bytes'],trace_cap=limits['raw_bytes'],free_reserve=16<<20,extra_roots=extra)
