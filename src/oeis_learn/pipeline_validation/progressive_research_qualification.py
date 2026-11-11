"""Current-source research model qualification through canonical installed roles."""
from pathlib import Path

from .identity import seal
from .progressive_research_inputs import checked_file, _json
from .progressive_workstation_state import write
from oeis_learn.generalization.progressive.contracts import record, reference


def selected_profiles(root, data, design):
    from oeis_learn.generalization.progressive.models import resolve_profile
    from .progressive_research_design import scientific_round
    if design!=scientific_round(data):
        raise ValueError('original prospective scientific round required')
    recipe=_json(checked_file(root,data['recipe']))
    # A model seed is not a distinct arithmetic/update capability. Actual cases
    # bind the selected seed; the validated runtime model scope omits it.
    profiles=[resolve_profile(recipe,shape=shape,dropout=dropout,seed=design['seeds'][0])
        for shape in design['tuning']['shapes'] for dropout in design['tuning']['space']['dropout']]
    return seal(dict(version='progressive-research-qualification-profiles/v1',
        design=design['identity'],recipe=data['recipe'],profiles=profiles,
        purpose='current research model scope; no historical authority transfer'))


def qualification_blueprint(root, output, *, matrix, base):
    from oeis_learn.generalization.progressive.budgets import KEYS
    output=Path(output)
    plan=seal(dict(version='progressive-qualification-plan/v1',
        matrix=reference(root,matrix),base_qualification=reference(root,Path(base)/'qualification.json'),
        seconds=3600))
    write(output/'qualification-plan.json',plan)
    caps=dict.fromkeys(KEYS,0)
    caps.update(elapsed_us=12*3600*1000000,disk_bytes=64<<30,archive_bytes=512<<20)
    value=record('blueprint',dict(executable=False,selected_stages=['capability','diversity','tuning'],
        references=[dict(name='qualification_plan',role='custodian',kind='policy',
                         artifact=reference(root,output/'qualification-plan.json'))],
        unresolved=[],proposed_caps=caps,reference_campaign=str(output.relative_to(root))),
        created_at='2026-11-11T00:00:00Z',purpose='research',track='synthetic_strict')
    write(output/'qualification-blueprint.json',value)
    return output/'qualification-blueprint.json'


def qualify(workflow, data, design):
    """No tiny rehearsal call; obtain base/current research scope only once."""
    w=workflow;s=w.s;root=w.root;base=w.base
    def profiles():
        value=selected_profiles(root,data,design)
        write(base/'matrix/profiles.json',value)
        return value,[base/'matrix/profiles.json']
    s.step('qualification-profiles',profiles,dependencies=('prepare-research/registered-data',
                                                         'prepare-research/census-plan'))
    built=w.images()
    def foundation():
        from .generalization_qualification_io import produce
        result=produce(w.host(),root,base/'base-qualification',
            images=[built[t] for t in ('qualification','gpu')],contract=s.binding['source'])
        if result['status']!='PASS':
            raise RuntimeError('current-source base qualification failed; original receipts retained')
        return str((base/'base-qualification').relative_to(root)),[base/'base-qualification/qualification.json']
    s.step('base-qualification',foundation,dependencies=('prepare-research/image-qualification','prepare-research/image-gpu'))
    body=("import os;from oeis_learn.pipeline_validation.progressive_qualification import matrix,hardware_facts;"
          "from oeis_learn.experiments.source_snapshot import source_snapshot;"
          "from oeis_learn.generalization.progressive.contracts import atomic_write,digest,load;"
          "p=load('/work/output/profiles.json');"
          "atomic_write('/work/output/matrix.json',matrix(p['profiles'],source=digest(source_snapshot()),"
          "image=os.environ['OEIS_QUALIFICATION_IMAGE'],engine=os.environ['OEIS_QUALIFICATION_ENGINE'],hardware=hardware_facts()))")
    w.installed('matrix',built['gpu'],body,gpu=True,seconds=120,outputs=('matrix.json',),
        dependencies=('prepare-research/base-qualification','prepare-research/qualification-profiles'))
    def plan():
        path=qualification_blueprint(root,base/'plans',matrix=base/'matrix/matrix.json',base=base/'base-qualification')
        return str(path.relative_to(root)),[path,base/'plans/qualification-plan.json']
    blueprint=s.step('qualification-plan',plan,dependencies=('prepare-research/matrix',))
    return w.run_command('qualify','qualify',blueprint=blueprint or 'missing',
        output=str((base/'qualification').relative_to(root)),seconds=3700,
        dependencies=('prepare-research/qualification-plan',))
