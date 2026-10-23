"""Successful stage imports validate every bound mandatory byte, never whole runs."""
from .identity import seal, confined, file_hash


def compatible(receipt, root, binding, prerequisites):
    try:
        return (seal(receipt)==receipt and receipt['state']=='PASS' and receipt['binding']==binding
            and receipt.get('prerequisites')==prerequisites and bool(receipt['files'])
            and all(file_hash(confined(root,p))==h for p,h in receipt['files'].items()))
    except (KeyError,TypeError,ValueError,OSError):
        return False
