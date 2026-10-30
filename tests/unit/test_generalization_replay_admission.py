import pytest
from oeis_learn.decoder.bigint_codec import encode_body
from oeis_learn.generalization.replay_admission import admit

ORIGINAL='(program (return (add (int 0) (int 0))))'
FINAL='(program (return (int 0)))'

def task(role='training'):
    return dict(id='task',role=role,ancestry=('mechanical','training'),source=ORIGINAL,indices=tuple(map(str,range(100))),values=('0',)*100)

def production(source,indices): return ('0',)*len(indices)

def test_full_horizon_shorter_and_forbidden_sources():
    entry=admit(task(),original_source=FINAL,final_source=FINAL,assignment={},tokens=encode_body(FINAL),origin_checkpoint='checkpoint',production=production,production_identity='runtime',order=0)
    assert entry.record()['verification']['terms']==100
    for t in (task('selection'),dict(task(),ancestry=('mechanical','oeis')),dict(task(),values=('0',)*20+('1',)*80)):
        with pytest.raises(ValueError): admit(t,original_source=FINAL,final_source=FINAL,assignment={},tokens=encode_body(FINAL),origin_checkpoint='checkpoint',production=production,production_identity='runtime',order=0)
    with pytest.raises(ValueError): admit(task(),original_source=FINAL,final_source=FINAL,assignment={},tokens=encode_body(FINAL),origin_checkpoint='checkpoint',production=lambda s,i:('1',)*100,production_identity='runtime',order=0)
    with pytest.raises(ValueError): admit(task(),original_source=FINAL,final_source=FINAL,assignment={},tokens=[1],origin_checkpoint='checkpoint',production=production,production_identity='runtime',order=0)
