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


@pytest.mark.parametrize('load_seconds',[20,101])
def test_retained_evaluation_charges_load_before_generation(tmp_path,monkeypatch,load_seconds):
    import time
    from oeis_learn.pipeline_validation import generalization_evaluation as evaluation,pilot_clocks
    from oeis_learn.fresh import bigint_diagnostics
    clock=[100.];monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    (tmp_path/'weights').write_bytes(b'fixture')
    prompts=[dict(id='opaque',indices=list(map(str,range(20))),values=['0']*20)]
    def load(*a,**kw):clock[0]+=load_seconds;return 'encoder','decoder',{}
    monkeypatch.setattr(evaluation,'load_retained',load)
    class Evidence:
        def __init__(self,*a,**kw):self.clock_summary={'cost_complete':True,'identity':'clock'}
        def finish(self):pass
    monkeypatch.setattr(pilot_clocks,'FullClockEvidence',Evidence)
    def generate(*a,**kw):
        assert load_seconds<100, 'no work may start after loading exhausted the scope'
        assert kw['deadline']==200., 'loading must not reset the deadline'
        return {'records':{}}
    monkeypatch.setattr(bigint_diagnostics,'generate',generate)
    args=(tmp_path,dict(path='weights',sha256='identity',bytes=7),prompts,{},tmp_path/'out')
    kw=dict(client=None,device='cpu',study='study',source='source',seconds=100)
    if load_seconds>100:
        with pytest.raises(TimeoutError):evaluation.evaluate_retained(*args,**kw)
    else:assert evaluation.evaluate_retained(*args,**kw)['status']=='PASS'


def test_generator_setup_is_charged_before_retained_evaluation(tmp_path,monkeypatch):
    import time
    from contextlib import contextmanager
    from oeis_learn.pipeline_validation import generalization_role as role,generalization_evaluation as evaluation,sizing_execution
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.generalization import contracts
    clock=[100.];monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    def verify(*a,**kw):clock[0]+=5;return '[]'
    @contextmanager
    def client(*a,**kw):clock[0]+=10;yield None
    monkeypatch.setattr(contracts,'verify_artifact',verify)
    monkeypatch.setattr(bigint_worker,'client',client)
    from contextlib import nullcontext
    monkeypatch.setattr(sizing_execution,'execution_context',nullcontext)
    def evaluate(*a,**kw):assert kw['seconds']==85;return {'status':'PASS'}
    monkeypatch.setattr(evaluation,'evaluate_retained',evaluate)
    settings=dict(checkpoint={},prompts={},recipe={},study='study',source='source',device='cpu',bank_seeds=None)
    assert role.generator(dict(visible_prompts=tmp_path,frozen_weights=tmp_path),tmp_path/'out',settings,seconds=100)['status']=='PASS'


@pytest.mark.parametrize('mode',['complete','incomplete','empty','unsupported','wrong_hash'])
def test_bank_preserves_exact_original_without_mutating_fitter_records(tmp_path,monkeypatch,mode):
    import copy,json
    from oeis_learn.pipeline_validation import generalization_evaluation as evaluation,pilot_clocks
    from oeis_learn.fresh import bigint_diagnostics
    from oeis_learn.generalization.proposals import make_bank
    from oeis_learn.synthesis.admission import source_identity
    (tmp_path/'weights').write_bytes(b'fixture')
    prompts=[dict(id=str(n),indices=list(map(str,range(20))),values=[str(n)]*20) for n in (3,5)]
    originals=['(program (return (int '+str(n)+')))' for n in range(8)]
    monkeypatch.setattr(evaluation,'load_retained',lambda *a,**k:('encoder','decoder',{}))
    monkeypatch.setattr(evaluation,'build_bank',lambda *a,**k:make_bank('weights',list(range(8)),lambda _,seed,*a:originals[seed],elapsed_seconds=0))
    class Evidence:
        def __init__(self,*a,**kw):self.clock_summary={'cost_complete':True,'identity':'clock'}
        def finish(self):pass
    monkeypatch.setattr(pilot_clocks,'FullClockEvidence',Evidence)
    monkeypatch.setattr(bigint_diagnostics,'generate',lambda *a,**k:dict(records={p['id']:dict(selections={},arm_evidence={}) for p in prompts}))
    retained=[];before=[]
    class Client:
        def fit(self,candidate,prompt,**kw):
            if mode=='unsupported':raise ValueError('unsupported')
            final='(program (return (int '+prompt.values[0]+')))'
            item=dict(source=final,source_sha256=source_identity(final),
                original_source_sha256=source_identity(candidate) if mode!='wrong_hash' else 'sha256:'+'0'*64,
                transformed_source='(program (return (param.get p0)))',assignment={'p0':prompt.values[0]},
                verification=dict(scope='visible_only',admitted=True,indices=list(prompt.indices),values=list(prompt.values),
                                  prompt_sha256=prompt.sha256,source_sha256=source_identity(final)))
            result=dict(complete=mode!='incomplete',outcome='unsat_in_scope' if mode=='empty' else 'verified_solution',
                        admitted=[] if mode=='empty' else [item,copy.deepcopy(item)])
            retained.append(result);before.append(copy.deepcopy(result));return result
    args=(tmp_path,dict(path='weights',sha256='weights',bytes=7),prompts,{},tmp_path/'out')
    kw=dict(client=Client(),device='cpu',study='study',source='source',bank_seeds=list(range(8)))
    if mode=='wrong_hash':
        with pytest.raises(ValueError,match='bank original source'):evaluation.evaluate_retained(*args,**kw)
    else:
        evaluation.evaluate_retained(*args,**kw)
        rows=json.loads((tmp_path/'out/selections.json').read_text())
        for p in prompts:
            selected=rows[p['id']]['selections']['bank']
            if mode in {'empty','unsupported'}:assert selected is None
            else:
                assert selected['original_source'] in originals
                assert source_identity(selected['original_source'])==selected['original_source_sha256']
                assert selected['original_source']!=selected['transformed_source']
                assert selected['verification']['values']==p['values']
                assert rows[p['id']]['arm_evidence']['bank']['complete']==(mode=='complete')
    assert retained==before, 'fitter/cache-owned result objects must remain unchanged'


def test_pac_export_reconstructs_every_family_total_pair_and_cost(tmp_path):
    import importlib.util,json
    from fractions import Fraction
    from pathlib import Path
    from tests.unit.test_generalization_metrics import pac_roster
    from tests.unit.test_generalization_execution import pac_setup
    from tests.integration.test_generalization_archive import pac_record
    from oeis_learn.pipeline_validation.identity import write,seal
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_finalization import publish_pac
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('pac_analysis_evidence',root/'specs/012-parameterized-adaptive-curriculum/validation/analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    roster=pac_roster();exp,control,profiles,_=pac_setup(optional_d=True)
    outcomes={arm:{key:(None if i%31==0 else (i+offset)%3==0) for i,key in enumerate(roster)} for arm,offset in [('B',0),('C',1),('D',2)]}
    costs={arm:dict(allowed_tokens=100,useful_tokens=80,reserved_tokens=5,discarded_tokens=10,unspent_tokens=5,examples=12,updates=6,seconds=9) for arm in outcomes}
    report=module.analyze(roster,outcomes,experiment=exp,control=control,profiles=profiles,costs=costs)
    import subprocess,sys,os
    payload=dict(roster=roster,outcomes=outcomes,experiment=exp,control=control,profiles=profiles,costs=costs)
    write(tmp_path/'reconciliation-input.json',payload)
    portable=subprocess.run([sys.executable,'-S','-c',
        'import importlib.util,json,sys; s=importlib.util.spec_from_file_location("a",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); p=json.load(open(sys.argv[2])); m.analyze(p.pop("roster"),p.pop("outcomes"),**p)',
        str(root/'specs/012-parameterized-adaptive-curriculum/validation/analysis.py'),str(tmp_path/'reconciliation-input.json')],
        cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src')),capture_output=True,text=True,timeout=30)
    assert portable.returncode==0,portable.stderr
    assert report['comparisons']['C-D']['attribution']=='review_substitution'
    write(tmp_path/'analysis.json',report)
    packet=seal(dict(pac_record('PARTIAL'),inventory=[reference(tmp_path/'analysis.json',tmp_path)]))
    archive=publish_pac(packet,tmp_path/'export',payload_root=tmp_path)
    assert verify_archive(archive)['software_status']=='PARTIAL'
    import zipfile
    with zipfile.ZipFile(archive) as z:exported=json.loads(z.read('analysis.json'))
    assert exported==report
    report=exported;roster=report['reconstruction']['roster'];outcomes=report['reconstruction']['outcomes']
    for arm,values in outcomes.items():
        for category in ('certified','ambiguous','unresolved'):
            ids=[k for k,r in roster.items() if r['category']==category]
            observed=report['arms'][arm][category]
            assert observed['successes']==sum(values[k] is True for k in ids)
            assert observed['complete']==sum(values[k] is not None for k in ids)
            groups={roster[k]['family'] for k in ids}
            macro=sum((Fraction(sum(values[k] is True for k in ids if roster[k]['family']==f),sum(roster[k]['family']==f for k in ids)) for f in groups),Fraction())/len(groups)
            assert observed['family_macro_lower']==str(macro)
        actual=report['costs']['arms'][arm]
        assert actual['allowed_tokens']==sum(actual[k] for k in ('useful_tokens','reserved_tokens','discarded_tokens','unspent_tokens'))
        assert (actual['examples'],actual['updates'],actual['seconds'])==(12,6,9)
