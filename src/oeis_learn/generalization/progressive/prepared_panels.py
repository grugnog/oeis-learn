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


def screening(rows, *, seed):
    """Select two members of sixteen original groups per development stratum."""
    if len({r['id'] for r in rows})!=len(rows) or any(r['role']!='development' for r in rows):
        raise ValueError('unique original development reservations required')
    selected=[];counts={};deficits=[]
    for stratum in STRATA:
        groups={}
        for row in rows:
            if row['stratum']==stratum:
                groups.setdefault(row['group'],[]).append(row)
        eligible=[group for group,members in groups.items() if len(members)>=2]
        chosen=sorted(eligible,key=lambda g:digest([seed,stratum,g]))[:16]
        for group in chosen:
            for row in sorted(groups[group],key=lambda r:digest([seed,r['id']]))[:2]:
                selected.append(dict(id=row['id'],group=group,stratum=stratum,role='development'))
        counts[stratum]=dict(eligible_groups=len(eligible),selected_groups=len(chosen),tasks=2*len(chosen))
        if len(chosen)!=16:
            deficits.append('screening_support:'+stratum)
    if not deficits:
        from .tuning import validate_screening_panel
        validate_screening_panel(selected)
    return selected,seal(dict(version='progressive-screening-publication/v1',
        reservation=digest(rows),seed=seed,status='INCOMPLETE' if deficits else 'COMPLETE',
        roster=digest(selected),strata=counts,deficits=deficits,
        selection='outcome-blind fixed original groups; no replacements or reduced quotas'))


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
        # Downstream comparison contracts use the first sorted group as the
        # statistical unit. Preserve all original relation keys for exclusion,
        # but give every member of the reserved connected component the same
        # primary alias; a member-specific family label must not split it.
        component='0000000000000000:component:'+row['group']
        if any(g<=component for g in groups):
            raise ValueError('original relation key collides with reserved component namespace')
        groups=[component,*groups]
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
