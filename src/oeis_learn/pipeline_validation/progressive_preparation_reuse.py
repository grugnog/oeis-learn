"""Reuse completed unscored native data, never historical execution authority."""
from pathlib import Path
import fcntl
from .identity import confined,seal,digest
from .progressive_research_inputs import checked_file,_json
from .preparation_recovery import retain
from oeis_learn.generalization.progressive.contracts import reference


def adopt(root,donor,recipient):
    root=Path(root);folder=confined(root,donor);recipient=Path(recipient)
    if not folder.is_relative_to(root/'run-state') or recipient.resolve().is_relative_to(folder):
        raise ValueError('separate original preparation donor required')
    with confined(root,folder/'session.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=_json(folder/'journal.json');binding=_json(folder/'session.json')
        if (any(k.startswith(('run-research/','assess/')) for k in journal['steps'])
                or any((folder/'prepare-research'/n).exists() for n in ('frozen','prepared','construction'))):
            raise ValueError('data reuse must precede campaign construction and scientific exposure')
        values={};outputs={}
        for name,rel in [('registered-data','registered/registered-data.json'),
                         ('native-census','census/census.json'),('materials','materials/materials.json'),
                         ('sensitivity','sensitivity/sensitivity.json')]:
            step=journal['steps'].get('prepare-research/'+name,{})
            if step.get('status')!='COMPLETE':raise ValueError('completed original data required: '+name)
            outputs[name]=[checked_file(root,r) for r in step['outputs']]
            path=folder/'prepare-research'/rel
            if reference(root,path) not in step['outputs']:raise ValueError('original data not journal-bound')
            values[name]=_json(path,maximum=64<<20)
        census=values['native-census'];materials=values['materials']
        if (census.get('diagnostic') is not False or census.get('training_started') is not False
                or census.get('assessment_exposed') is not False
                or materials.get('status')!='COMPLETE' or materials.get('diagnostic') is not False
                or values['sensitivity'].get('status')!='COMPLETE'):
            raise ValueError('complete outcome-blind original data required')
        from .progressive_handoff import original_authority
        q,qroot=original_authority(root,reference(root,folder/'prepare-research/qualification/qualification.json'))
        previous=_json(checked_file(qroot,q['source_closure']),maximum=64<<20)
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        current=collect_snapshot(root)
        if digest(previous)!=binding['binding']['source']:raise ValueError('original data source closure differs')
        prefixes=('src/oeis_learn/synthesis/','src/oeis_learn/sandbox/','src/oeis_learn/decoder/',
                  'crates/','src/oeis_learn/generalization/progressive/prepared_',
                  'src/oeis_learn/generalization/progressive/training_rows')
        names={'Dockerfile','pyproject.toml','uv.lock',*[f'src/oeis_learn/pipeline_validation/{n}.py' for n in (
            'progressive_research_census','progressive_research_materials','progressive_research_inputs',
            'progressive_research_revision','progressive_research_bridge','progressive_research_sensitivity')]}
        # Reviewed change adds hourly package policy/recadence; the native
        # support producer and every original row remain byte-identical.
        support='src/oeis_learn/generalization/progressive/prepared_support.py'
        reviewed=(previous.get(support),current.get(support))==(
            'sha256:dc53f0150ef28fbaf64f1d6cb9da69e984267016d25a3fde5b68dd4bc0aa287c',
            'sha256:7b331fd4b9feb844456add718f8e7b1b7397e95e255b264e59b1315eb2093471')
        changed=[k for k in set(current)|set(previous) if (k in names or k.startswith(prefixes)) and current.get(k)!=previous.get(k) and not (reviewed and k==support)]
        if changed:raise ValueError('data producer changed; fresh review required: '+', '.join(sorted(changed)))
        from .progressive_research_bindings import inputs
        from .progressive_research_design import scientific_round
        inputs(root,values['registered-data'],scientific_round(values['registered-data'],expanded=True),materials)
        receipt=seal(dict(version='progressive-preparation-data-reuse/v1',
            donor=reference(root,folder/'session.json'),journal=reference(root,folder/'journal.json'),
            original_source=binding['binding']['source'],current_source=digest(current),
            census=census['identity'],materials=materials['identity'],
            recipient=str(recipient.relative_to(root)),hardware_authority_transferred=False,
            original_costs_retained=True,assessment_exposed=False))
        claim=root/'run-state/discovery-preparation-revisions'/('runtime-'+census['identity'][7:]+'.json')
        retain(claim,receipt)
        return values,outputs,receipt
