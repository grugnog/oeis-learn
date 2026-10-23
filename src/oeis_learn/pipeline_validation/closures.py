"""Explicit content closure with conservative fallback for unmapped changes."""
from pathlib import Path
from .identity import seal, file_hash

RULE = 'pipeline-closure/v1'


def closure(root, paths, prerequisites=()):
    root=Path(root)
    files={}
    for name in sorted(set(paths)):
        path=root/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('missing/unsafe closure input: '+name)
        files[name]=file_hash(path)
    return seal(dict(version=RULE,files=files,prerequisites=list(prerequisites),
                     exclusions={'docs':'presentation only; source/package inspection still mandatory'}))


def compatible_changes(old, current, changed):
    for path in changed:
        if path.startswith(('docs/','specs/')) and '/contracts/' not in path:
            continue
        if path not in old['files'] or old['files'].get(path) != current['files'].get(path):
            return False
    return old['files'] == current['files'] and old['prerequisites'] == current['prerequisites']
