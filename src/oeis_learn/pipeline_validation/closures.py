"""Versioned conservative dependency closure; unmapped changes invalidate reuse."""
from pathlib import Path
from .identity import seal, file_hash, confined, validate

RULE='pipeline-closure/v1'


def presentation(path):
    path=Path(path)
    return str(path)=='README.md' or (path.parts[0]=='docs' and path.suffix=='.md')


def dependency_closure(snapshot, *, prerequisites):
    # Start with every mapped source/build/config/test/adapter byte. Narrower
    # closures require a separately reviewed rule, not inferred file extensions.
    files={str(p):h for p,h in sorted(snapshot.items()) if not presentation(p)}
    return seal(dict(version=RULE,files=files,prerequisites=prerequisites,
        exclusions={'README.md':'presentation only','docs/*.md':'presentation only',
            'configs/foundation/pipeline_studies/*.json':'prospective authority inputs independently hashed by study admission, not compiled runtime code'},
        unmapped='invalidate',whole_package_inspection='mandatory current image'))


def closure(root,paths,prerequisites=()):
    files={}
    for name in sorted(set(paths)):
        path=confined(root,name)
        if not path.is_file():raise ValueError('regular closure input required: '+name)
        files[name]=file_hash(path)
    return dependency_closure(files,prerequisites=list(prerequisites))


def compatible_closure(old,current):
    fields={'version','files','prerequisites','exclusions','unmapped','whole_package_inspection'}
    try:
        validate(old,fields,RULE);validate(current,fields,RULE)
        return old==current
    except (ValueError,TypeError):
        return False


def compatible_changes(old,current,changed):
    for path in changed:
        if presentation(path):continue
        if path not in old['files'] or old['files'].get(path)!=current['files'].get(path):
            return False
    return compatible_closure(old,current)
