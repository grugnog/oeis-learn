"""Successful stage imports validate every bound mandatory byte, never whole runs."""
from .identity import seal, confined, file_hash


def compatible(receipt, root, binding, prerequisites, equivalence=None):
    try:
        return (seal(receipt)==receipt and receipt['state']=='PASS' and receipt['binding']==binding
            and (receipt.get('prerequisites')==prerequisites or equivalence is not None and receipt.get('prerequisite_equivalence')==equivalence) and bool(receipt['files'])
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
    summary=json.loads(confined(root,'summary.json').read_text())
    inventory={m['path']:m for m in manifest['members']}
    stages=summary.get('stages',{})
    for stage,row in stages.items():
        if seal(row)!=row:raise ValueError('changed imported terminal receipt')
        path=f'stages/{stage}/receipt.json'
        if path not in inventory or json.loads(confined(root,path).read_text())!=row:
            raise ValueError('imported summary/receipt mismatch')
        for name,sha in row['files'].items():
            member=inventory.get(f'stages/{stage}/{name}')
            if member is None or member['sha256']!=sha:raise ValueError('missing bound stage artifact')
        for parent,sha in row.get('prerequisites',{}).items():
            predecessor=stages.get(parent,{})
            if parent not in stages or sha not in {predecessor.get('identity'),predecessor.get('reused_from')}:
                raise ValueError('broken imported prerequisite link')
            if row['state'] in {'PASS','REUSED'} and predecessor['state'] not in {'PASS','REUSED'}:
                raise ValueError('successful imported stage has a failed prerequisite')
    return manifest


def dependency_key(receipt):
    # Only immutable image artifacts may substitute for a fresh image receipt's
    # timestamps/attempt UUID. The current image inspection still executes.
    result=receipt.get('observations',{}).get('result',{})
    if receipt.get('stage','').startswith('image.') and result.get('image'):
        from .identity import digest
        return digest(dict(stage=receipt['stage'],binding=receipt['binding'],image=result['image']))
    return receipt.get('reused_from',receipt['identity'])


def find_stage(roots,stage,binding,prerequisites,equivalence=None):
    import json
    from pathlib import Path
    for root in roots:
        folder=Path(root)/'stages'/stage
        if not (folder/'receipt.json').is_file():continue
        receipt=json.loads(confined(folder,'receipt.json').read_text())
        if compatible(receipt,folder,binding,prerequisites,equivalence):
            return folder,receipt
    return None
