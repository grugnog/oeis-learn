"""Exact disk-backed exposure checks at custodian publication/freeze boundaries.

The original indexes remain private. Public exposure records retain their exact
identities; they do not imply that an offline verifier has inspected private data.
"""
from contextlib import ExitStack, contextmanager
import json

from .contracts import digest, verify_refs


@contextmanager
def open_indexes(root, exposures):
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex, checked_file

    refs = {}
    for exposure in exposures:
        for ref in exposure['payload'].get('indexes', []):
            if ref['path'] in refs and refs[ref['path']] != ref:
                raise ValueError('conflicting original exposure index')
            refs[ref['path']] = ref
    with ExitStack() as stack:
        indexes = [stack.enter_context(ExposureIndex(checked_file(root, ref), readonly=True))
                   for ref in refs.values()]
        yield indexes


def check_panel(root, panel, indexes):
    """Recompute original indexed values; never trust a supplied freshness bit."""
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex
    from .contracts import validate

    if not indexes:
        return
    panel=validate('cohort',panel)
    if panel['payload']['role']=='training':
        return

    for row in panel['payload']['rows']:
        truth = json.loads(verify_refs(root, [row['truth']], max_bytes=4 << 20)[0])
        keys = set(ExposureIndex.keys(truth))
        keys.add(('structure', row['structure']))
        keys.update(('group', digest(group)) for group in row['groups'])
        # Parameter transfer deliberately retains structure, but never an exact
        # sequence, prefix/shift, identity or declared relation group.
        if row['stratum'] in {'parameter','retention'} and panel['payload']['group_policy'] == 'parameter_structure_sharing':
            keys = {(kind, value) for kind, value in keys if kind != 'structure'}
        for index in indexes:
            if any(index.db.execute('SELECT 1 FROM keys WHERE kind=? AND value=?',key).fetchone()
                   for key in keys):
                raise ValueError('indexed original exposure overlaps panel member: '+row['id'])
