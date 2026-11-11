"""Full-size construction over explicit synthetic data/qualification boundaries.

These tests do not execute research, claim native provenance or qualify HIP.
"""
import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write,digest,load,record,reference,seal
from oeis_learn.pipeline_validation import progressive_research_bindings as b


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    from unittest.mock import patch
    from tests.unit.test_progressive_research_roster import fixture
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture as learning_fixture
    from oeis_learn.pipeline_validation.progressive_research_design import scientific_round
    root=tmp_path_factory.mktemp('research-construction')
    learning_fixture(root/'original',interrupt=False)
    package=load(root/'original/training-input.json')
    # The native data producer is a named test boundary; no such mutation is
    # available in the production compiler or registered-data adapter.
    shards=[[seal(dict(r,native_verified=True)) for r in shard] for shard in package['shards']]
    rule=seal(dict(batches=[16,32,64]))
    support=seal(dict(version='progressive-prepared-support/v1',policy=rule['identity'],diagnostic=False,
        shards=shards,layout=package['layout'],refresh_updates=1000000))
    summary=seal(dict(status='COMPLETE',diagnostic=False,support=support['identity'],policy=rule,
        maximum_target_tokens=max(len(r['tokens']) for s in shards for r in s)))
    atomic_write(root/'support.json',support)
    atomic_write(root/'recipe.json',package['profile']['base_recipe'])
    (root/'weights.pt').write_bytes(b'EXPLICIT TEST BOUNDARY: never loaded as model weights')
    atomic_write(root/'state.json',dict(test_boundary_only=True))
    manifest=dict(retained=dict(checkpoints=dict(D=dict(root='.',descriptor=dict(
        weights=reference(root,'weights.pt'),state=reference(root,'state.json'))))))
    atomic_write(root/'manifest.json',manifest)
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19])),
        recipe=reference(root,'recipe.json'),manifest=reference(root,'manifest.json')))
    design=scientific_round(data)
    _,rosters,screening=fixture();panels={};reservations={};pubs=[]
    exposure=record('exposure',dict(previous=None,events=[],detector_limits=['TEST BOUNDARY']),
        created_at='2026-11-11T00:00:00Z',purpose='research',track='synthetic_strict')
    atomic_write(root/'exposure.json',exposure)
    from oeis_learn.generalization.progressive.prepared_panels import cohort
    for name,rows in rosters.items():
        reserved=[dict(id=r['id'],group=r['group'],stratum='parameter' if name.startswith('retention') else r['stratum'],
            role=name,indices=list(map(str,range(100))),values=list(map(str,range(100))),
            source=None if r['stratum']=='oeis' else '(program (return $index))',operations=[]) for r in rows]
        role='assessment' if name=='assessment' else 'development' if name in ('development','retention-development') else 'confirmation'
        panel=cohort(root,root/name,reserved,panel=name,role=role,decision='test-boundary',
            exposure=exposure,created_at=exposure['created_at'])
        atomic_write(root/name/'cohort.json',panel)
        panels[name]=panel;reservations[name]=reserved
        pubs.append(dict(panel=name,cohort=reference(root,root/name/'cohort.json')))
    bound=dict(roster=b.construct(design,rosters,screening),panels=panels,reservations=reservations,
               screening=screening,support=support,support_summary=summary,exposure=exposure)
    materials=seal(dict(panels=pubs,support=reference(root,'support.json'),exposure=reference(root,'exposure.json')))
    atomic_write(root/'qualification/source.json',{'test_boundary_only':True})
    q=dict(source_closure=reference(root/'qualification','source.json'),images=[digest('cpu'),digest('hip')],
           runtime={'execution_policy':digest('TEST EXECUTION POLICY BOUNDARY')})
    atomic_write(root/'qualification/qualification.json',q)
    with patch.object(b,'inputs',return_value=bound):
        args=b.compile_plan(root,root/'compiled',data,design,materials,qualification=root/'qualification',checkpoint_bytes=64<<20)
        result=b.finish_plan(*args,data=data,design=design,materials=materials)
    return root,args,result


def test_full_registered_plan_has_real_closed_requests_and_exact_cases(compiled):
    root,(compiler,bound,study,*_),summary=compiled
    assert summary['tasks']==376 and summary['exact_calibration_cases']==1296
    assert summary['executable'] is False and summary['assessment_exposed'] is False
    from oeis_learn.pipeline_validation.progressive_outputs import validate_requests
    from oeis_learn.pipeline_validation.progressive_study import validate_frozen
    plan=load(root/'compiled/preparation-plan.json',maximum=128<<20)['payload']
    blueprint=load(root/'compiled/blueprint.json')
    record('prepared',dict(plan,blueprint=blueprint['identity']),created_at=blueprint['created_at'],
           purpose='research',track='synthetic_strict')
    assert 4096<len(plan['references'])<16384
    # Literal request validation needs one task per identity, not an expanded
    # list repeating each ID hundreds of times for its finite alternatives.
    validate_requests(root,plan['task_plans'],plan['references'],plan['graph'],purpose='research')
    validate_frozen(root,dict(purpose='research',track='synthetic_strict',payload=dict(tasks=plan['task_plans'],references=plan['references'])))
    requests=[load(root/r['artifact']['path']) for r in plan['references'] if r['kind']=='policy' and r['name'].startswith('requests/')]
    assert len(requests)==376
    from oeis_learn.generalization.progressive.budgets import role_work_requirements
    audit=next(r for r in requests if r['operation']=='audit-language')
    need=role_work_requirements('audit-language',[load(root/p) for p in audit['artifacts']['sources']])
    assert all(audit['reservation'][k]>=v for k,v in need.items())
    assert not any(r['operation']=='assess' for r in requests)
    assert all(not (isinstance(v,str) and '/assessment/' in v) for r in requests for vs in r['artifacts'].values() for v in vs)
    # Independent same-panel baseline + candidate + retention for every pair.
    for item in bound['roster']['comparisons']:
        request=load(root/f"compiled/requests/{item['id']}.json")
        template=load(root/request['artifacts']['public'][0])['data']
        assert len(template['roster'])==640
        assert {r['stratum'] for r in template['roster']}=={'parameter','composition','unseen_structure','oeis','retention'}
    assert (root/'compiled/calibration-plan.json').stat().st_size<4096


def test_compiler_retains_complete_original_finite_domain_and_reference_controls(compiled):
    root,(compiler,bound,study,*_),_=compiled
    rows={t['id']:t for t in compiler.plans}
    assert len(rows['trial-00-1000000-train']['trial_choices'])==162
    assert len(rows['trial-15-4000000-train']['trial_choices'])==162
    assert len(rows['confirmation-1-17']['trial_choices'])==648
    assert len(rows['confirmation-0-17']['trial_choices'])==1
    assert 'trial_choices' not in rows['trial-00-retention-baseline-generate']
    assert rows['confirmation-baseline-1-17-generate']['trial']['ordinal']==0
    assert rows['confirmation-baseline-1-17-generate']['slot']=='tuning-confirmation-1'
    packages=list((root/'compiled/private/packages').glob('*.json'))
    assert packages and all('weights' not in load(p) for p in packages)
