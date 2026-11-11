"""Whole prospective research roster, without engines or scientific outcomes."""
import pytest

from oeis_learn.generalization.progressive.contracts import seal
from oeis_learn.generalization.progressive.research_roster import construct
from oeis_learn.pipeline_validation.progressive_research_design import scientific_round


def fixture():
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19]))))
    design=scientific_round(data)
    names=['development','confirmation-diversity',*[f'confirmation-tuning-{i}' for i in range(4)],
           'retention-development','retention-diversity','retention-tuning','assessment']
    panels={name:[dict(id=f'{name}-{s}-{i:03d}',group=f'{name}-{s}-g{i//8}',stratum=s)
                  for s in (['retention'] if name.startswith('retention') else ['parameter','composition','unseen_structure','oeis'])
                  for i in range(128)] for name in names}
    screening=[dict(r,role='development') for r in panels['development'] if int(r['id'].split('-')[-1])%8<2]
    return design,panels,screening


def test_whole_round_reserves_all_roles_before_results_and_never_assessment():
    design,panels,screening=fixture();r=construct(design,panels,screening)
    tasks=r['tasks'];by_id={t['id']:t for t in tasks}
    assert len(tasks)==376 and r['executable'] is False and r['assessment_exposed'] is False
    assert {t['stage'] for t in tasks}=={'capability','diversity','tuning'}
    assert all(t.get('panel')!='assessment' and t['operation']!='assess' for t in tasks)
    for control in ('retained_D','random'):
        rows=[t for t in tasks if t.get('control')==control and t['stage']=='capability']
        assert len(rows)==8 and all(len(t['members'])==64 and t['all_methods'] for t in rows)
        assert sorted(x for t in rows for x in t['members'])==sorted(x['id'] for x in panels['development'])
    for ordinal in range(16):
        selected=[t for t in tasks if t['trial'] and t['trial']['ordinal']==ordinal and t['trial']['phase'] not in {'selection','confirmation'}]
        assert len(selected)==18
        assert sum(t['seconds'] for t in selected)+1800<=7200
        learned=[t for t in selected if t['operation']=='train']
        assert [t['token_target'] for t in learned]==[1000000,2000000,4000000]
        assert [t['parent'] for t in learned]==[None,learned[0]['id'],learned[1]['id']]
        scores=[t for t in selected if t['trial']['phase']=='scoring']
        assert [t['trial']['coordinate'] for t in scores]==[1000000,2000000,4000000]
        for score in scores:
            assert set(score['members'])=={x['id'] for x in screening}
            e=r['study_evaluations'][score['id']]
            assert by_id[e['retention_baseline_scoring']]['panel']=='retention-development'
    training=[t for t in tasks if t['operation']=='train']
    assert len(training)==60 and sum(t['seconds'] for t in training)<=536*3600
    assert sum(t['seconds'] for t in tasks if t['operation']!='train')<=136*3600
    assert len(r['comparisons'])==4
    for c in r['comparisons'][1:]:
        for seed in design['seeds']:
            pair=[p for p in c['producers'] if p['seed']==seed]
            assert {p['arm'] for p in pair}=={'baseline','candidate'}
            assert {by_id[p['primary']['generation']]['panel'] for p in pair}=={c['panel']}
            baseline=next(p for p in pair if p['arm']=='baseline')
            assert baseline['checkpoint']==f'confirmation-0-{seed}'
    assert construct(design,panels,screening)==r


@pytest.mark.parametrize('change',['quota','screening','panel','domain','look'])
def test_roster_rejects_missing_support_or_silently_reduced_design(change):
    design,panels,screening=fixture()
    if change=='quota':panels['confirmation-diversity'].pop()
    if change=='screening':screening[0]['group']='foreign'
    if change=='panel':panels.pop('assessment')
    if change=='domain':design['tuning']['learning_trials']=2;design=seal(design)
    if change=='look':design['tuning']['screening_looks'].pop();design=seal(design)
    with pytest.raises(ValueError):construct(design,panels,screening)
