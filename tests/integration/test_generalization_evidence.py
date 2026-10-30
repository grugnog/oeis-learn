import hashlib
import pytest
from oeis_learn.pipeline_validation.generalization_evaluation import evaluate_checkpoint
from oeis_learn.pipeline_validation.generalization_reporting import recorded_report


def test_verified_checkpoint_visible_only_and_missing(tmp_path):
    raw=b'actual checkpoint fixture';(tmp_path/'weights').write_bytes(raw)
    ref=dict(path='weights',bytes=len(raw),sha256='sha256:'+hashlib.sha256(raw).hexdigest())
    prompts=[dict(id='opaque',indices=list(map(str,range(20))),values=['0']*20)]
    calls=[]
    def load(path): calls.append(('load',path.read_bytes()));return 'model'
    def generate(model,prompt,control):
        assert set(prompt)=={'id','indices','values'}
        calls.append((control,prompt['id']));return [dict(source='candidate',seconds=1,control=control)]
    def fit(source,prompt): return dict(status='unresolved',reason='deadline',queue_seconds=1,service_seconds=2,cleanup_seconds=0)
    result=evaluate_checkpoint(tmp_path,ref,prompts,load=load,generate=generate,fit=fit,controls=('raw','rescue'))
    assert result['availability']=='available' and len(result['attempts'])==2
    assert calls[0]==('load',raw)
    assert evaluate_checkpoint(tmp_path,dict(ref,path='missing'),prompts,load=load,generate=generate,fit=fit)['availability']=='unavailable'
    with pytest.raises(ValueError): evaluate_checkpoint(tmp_path,dict(ref,sha256='sha256:'+'0'*64),prompts,load=load,generate=generate,fit=fit)


def test_recorded_report_preserves_missing_fields():
    roster={'a':'small','b':'large','c':'large'}
    record=dict(study='study',source='source',roster=roster,outcomes={'a':True,'b':False})
    report=recorded_report([record],study='study',source='source',roster=roster,checkpoints={'best':'a','latest':'b'},available={'a'})
    assert report['summary']['lower']=='1/3' and report['summary']['upper']=='2/3'
    assert report['checkpoints']['latest']=='unavailable'
