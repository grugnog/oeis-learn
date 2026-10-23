"""Successful stage imports validate every bound mandatory byte, never whole runs."""
from .identity import seal, confined, file_hash


def compatible(receipt, root, binding, prerequisites):
    try:
        return (seal(receipt)==receipt and receipt['state']=='PASS' and receipt['binding']==binding
            and receipt.get('prerequisites')==prerequisites and bool(receipt['files'])
            and all(file_hash(confined(root,p))==h for p,h in receipt['files'].items()))
    except (KeyError,TypeError,ValueError,OSError):
        return False


def validate_import(root):
    import json
    from pathlib import Path
    from .identity import validate
    root=Path(root)
    manifest=json.loads(confined(root,'manifest.json').read_text())
    validate(manifest,{'version','members','omissions','summary_state','excluded'},'pipeline-bundle/v1')
    names=set()
    for member in manifest['members']:
        if set(member)!={'path','bytes','sha256','mandatory'} or member['path'] in names:
            raise ValueError('invalid imported bundle inventory')
        names.add(member['path'])
        path=confined(root,member['path'])
        if path.stat().st_size!=member['bytes'] or file_hash(path)!=member['sha256']:
            raise ValueError('imported bundle artifact mismatch: '+str(path))
    return manifest


def find_stage(roots,stage,binding,prerequisites):
    import json
    from pathlib import Path
    for root in roots:
        folder=Path(root)/'stages'/stage
        if not (folder/'receipt.json').is_file():continue
        receipt=json.loads(confined(folder,'receipt.json').read_text())
        if compatible(receipt,folder,binding,prerequisites):
            return folder,receipt
    return None
