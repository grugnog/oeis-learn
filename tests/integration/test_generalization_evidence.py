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


def test_best_and_latest_load_their_exact_tensor_payloads(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.pipeline_validation.generalization_evaluation import load_retained
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    def models(recipe,device):return torch.nn.Linear(1,1),torch.nn.Linear(1,1),{}
    monkeypatch.setattr(bigint_worker,'models',models)
    refs={}
    for name,value in [('best',1.),('latest',2.)]:
        encoder,decoder,_=models({},'cpu')
        with torch.no_grad():encoder.bias.fill_(value)
        path=tmp_path/(name+'.pt');torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict()),path)
        refs[name]=reference(path,tmp_path)
    best=load_retained(tmp_path,refs['best'],{},device='cpu')[0]
    latest=load_retained(tmp_path,refs['latest'],{},device='cpu')[0]
    assert best.bias.item()==1 and latest.bias.item()==2
    with pytest.raises(ValueError):load_retained(tmp_path,dict(refs['best'],path='latest.pt'),{},device='cpu')


def test_bank_generation_is_shared_and_recipient_independent(monkeypatch):
    torch=pytest.importorskip('torch')
    from oeis_learn.pipeline_validation.generalization_evaluation import build_bank
    from oeis_learn.decoder import bigint_generation
    from oeis_learn.generalization.proposals import apply_bank,bank_cost
    import time
    decoder=torch.nn.Linear(1,2);decoder.d_model=2;draws=[]
    def sample(decoder,requests,**kwargs):
        request=requests[0];draws.append(request.seed)
        assert request.memory.tolist()==[[[1.,0.]]]
        return [dict(source=f'(program (return (int {request.seed})))',complete=True)]
    monkeypatch.setattr(bigint_generation,'sample_batch',sample)
    bank=build_bank(decoder,'checkpoint',list(range(8)),deadline=time.monotonic()+60)
    calls=[]
    for value in ('1','2'):
        prompt=dict(id=value,indices=list(map(str,range(20))),values=[value]*20)
        apply_bank(bank,prompt,lambda source,indices,values:calls.append((source,values)))
    assert draws==list(range(8)) and [c[0] for c in calls[:8]]==[c[0] for c in calls[8:]]
    assert calls[0][1]!=calls[8][1]
    cost=bank_cost(bank,2);assert cost['amortized_seconds']==cost['cold_seconds']/2


def test_production_evaluation_uses_matched_drain_and_single_proposal_diagnostics(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_evaluation as evaluation,pilot_clocks
    from oeis_learn.fresh import bigint_diagnostics
    (tmp_path/'weights').write_bytes(b'fixture')
    prompts=[dict(id='opaque',indices=list(map(str,range(20))),values=['0']*20)]
    monkeypatch.setattr(evaluation,'load_retained',lambda *a,**k:('encoder','decoder',{}))
    class Evidence:
        def __init__(self,*args,**kwargs):self.clock_summary={'cost_complete':True,'identity':'clock'}
        def finish(self):pass
    monkeypatch.setattr(pilot_clocks,'FullClockEvidence',Evidence)
    def generate(encoder,decoder,rows,recipe,client,**kwargs):
        assert rows==prompts
        assert kwargs['evaluation_policy']==pilot_clocks.DRAIN
        assert kwargs['diagnostic_attempts']==dict(correct=8,shuffled=8,absent=8,template=8)
        return {'records':{}}
    monkeypatch.setattr(bigint_diagnostics,'generate',generate)
    result=evaluation.evaluate_retained(tmp_path,dict(path='weights',sha256='identity',bytes=7),prompts,{},tmp_path/'result',
        client=None,device='cpu',study='study',source='source')
    assert result['primary_attempts']['correct']==8 and result['raw_rescue_attempts']==1
