"""Reconcile fixed provenance and denominators without inventing private weights."""
from .metrics import summarize


def reconcile(records, *, study, source, roster):
    outcomes={}
    for record in records:
        if record['study']!=study or record['source']!=source or record['roster']!=roster:
            raise ValueError('incompatible evidence authority/denominator')
        summarize(roster,record['outcomes'])
        for row,value in record['outcomes'].items():
            if row in outcomes and outcomes[row]!=value:
                raise ValueError('conflicting row outcome')
            outcomes[row]=value
    return summarize(roster,outcomes)


def checkpoint_roster(frozen, available_hashes):
    if not frozen or any(not value for value in frozen.values()):
        raise ValueError('explicit best/latest checkpoint identities required')
    return {label:'available' if identity in available_hashes else 'unavailable' for label,identity in frozen.items()}
