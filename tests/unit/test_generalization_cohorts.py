import json
from pathlib import Path
import pytest
from oeis_learn.generalization.coverage import observe
from oeis_learn.generalization.cohorts import choose, project

CASES=json.loads(Path('tests/fixtures/generalization/mechanical-cases.json').read_text())

def test_executed_not_syntactic_coverage():
    active='(program (local $b0 bigint) (local $c0 control) (loop $c0 (control 2) (set $b0 (add (get $b0) (int 1)))) (return (get $b0)))'
    a=observe(active,tuple(map(str,range(100))),stage='C')
    z=observe(active.replace('(control 2)','(control 0)'),tuple(map(str,range(100))),stage='C')
    assert a.loop_iterations==200 and a.loop_active and not z.loop_active
    huge=observe('(program (return (int 18446744073709551616)))',('0',),stage='B')
    assert huge.above_64_bits

def test_quota_shortfall_and_projection():
    rows=[dict(id=str(i),family='f'+str(i),tags=['loop_active','branch_active'],role='selection',indices=list(map(str,range(100))),values=['0']*100,source='private') for i in range(4)]
    result=choose(rows,count=4,quotas={'loop_active':2,'branch_active':2},structural=True)
    assert result.complete and len(result.rows)==4
    failure=choose(rows[:1],count=4,quotas={'loop_active':2,'branch_active':2},structural=True)
    assert not failure.complete and failure.shortfalls
    visible,scorer=project(rows[0])
    assert set(visible)=={'id','indices','values'} and len(visible['values'])==20
    assert scorer['family']=='f0'
    with pytest.raises(ValueError): project(dict(rows[0],role='unknown'))

def test_both_branch_paths_and_longer_program_bins():
    from oeis_learn.generalization.coverage import distribution
    src='(program (local $b0 bigint) (if (lt $index (int 1)) (then (set $b0 (int 2))) (else (set $b0 (int 3)))) (return (get $b0)))'
    active=observe(src,('0','1'),stage='D')
    inactive=observe(src,('1','2'),stage='D')
    assert active.branch_active and not inactive.branch_active
    assert distribution([active],length_boundary=1)['long']==1
    assert not choose([dict(id='x',family='same',tags=['loop_active','branch_active'])],count=1,quotas={},structural=True).complete
