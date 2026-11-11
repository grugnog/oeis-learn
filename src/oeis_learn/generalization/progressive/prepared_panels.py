"""Custodian projection of outcome-blind reservations into canonical cohorts.

Private native source witnesses stay in the original reservoir. Generator
channels contain only indexed visible values; scorer channels receive the exact
reserved truth. This conversion never fills a short panel with replacements.
"""
from pathlib import Path

from .contracts import atomic_write, digest, record, reference, seal, validate
from .cohorts import DEFAULT_PROFILE, STRATA


LIMITS = [
    'Exact finite values, 20-term windows, normalized AST structures and declared source relations only.',
    'OEIS entries without an original program have no inferred AST, operation coverage or work measurement.',
    'Composition identity is the normalized proper-subtree/operator inventory; no semantic independence claim.',
]


def composition(row):
    if row['stratum']=='oeis':
        return digest(['unavailable-source-composition',row['id']])
    from oeis_learn.pipeline_validation.progressive_research_census import features
    f=features(row['source'])
    return digest([sorted(f['proper']),sorted(f['operators'])])


def cohort(root, output, rows, *, panel, role, decision, exposure,
           required=128, minimum_groups=16, purpose='research', created_at):
    """Write private raw inputs and return a closed cohort for canonical publish."""
    validate('exposure',exposure)
    root,output=Path(root),Path(output)
    if not output.is_relative_to(root) or output.exists():
        raise ValueError('fresh confined panel projection directory required')
    from oeis_learn.pipeline_validation.identity import confined
    output=confined(root,str(output.relative_to(root)),exists=False)
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('unique original reservation identities required')
    projected=[]
    private=output/'private';private.mkdir(parents=True,mode=0o700)
    for row in rows:
        if row.get('role')!=panel:
            raise ValueError('reservation belongs to another panel')
        if 'draw_identity' in row and (row.get('native_verified') is not True and purpose=='research'):
            raise ValueError('native original mechanical witness required')
        source='oeis' if row['stratum']=='oeis' else 'mechanical'
        name=digest(row['id'])[7:]
        truth={k:row[k] for k in ('id','indices','values')}
        prompt=dict(id=truth['id'],indices=truth['indices'][:20],values=truth['values'][:20])
        atomic_write(private/(name+'-truth.json'),truth)
        atomic_write(private/(name+'-visible.json'),prompt)
        # Original group/family keys survive component assignment. Otherwise a
        # publication could erase a relation present in the retained index.
        groups=sorted(set([row['group'],*row.get('declared_groups',[])]))
        is_retention=panel.startswith('retention')
        projected.append(dict(id=row['id'],source=source,offset=row['indices'][0],
            stratum='retention' if is_retention else row['stratum'],groups=groups,
            structure=row.get('structure',digest(['unavailable-source-structure',row['id']])),
            behavior=digest(row['values']),composition=composition(row),
            operations=row.get('operations',[]),
            difficulty=dict(depth=0,nodes=0,work=0),
            visible=reference(root,private/(name+'-visible.json')),
            truth=reference(root,private/(name+'-truth.json'))))
    result=record('cohort',dict(role=role,panel=panel,decision=decision,
        profile=dict(DEFAULT_PROFILE),required_per_stratum=required,minimum_groups=minimum_groups,
        group_policy='parameter_structure_sharing',detector_limits=LIMITS,
        exposure=exposure['identity'],rows=projected,
        strata=['retention'] if panel.startswith('retention') else list(STRATA)),
        created_at=created_at,purpose=purpose,track='synthetic_strict')
    atomic_write(private/'projection.json',seal(dict(version='progressive-panel-projection/v1',
        reservation=digest(rows),cohort=result['identity'],replacement_draws=0,
        source_witnesses='retained in original private panel reservations')))
    return result
