import copy
import json
import zipfile
from types import SimpleNamespace

import pytest

from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.generalization.selector_run_diagnostics import summarize,interval_union


def fixture():
    records={}
    for i in range(4):
        d=dict(position=0,seed=i,source_sha256='source',tokens=[1]*1024,
            generation_batch_id='batch',generation_batch_size=4,generation_seconds=2.,
            generation=dict(complete=False,reason='token_cap_without_eos'),
            fitting=dict(outcome='not_attempted',clock=dict(dispatched=10+i,completed=12+i)))
        records[str(i)]=dict(details=dict(correct=dict(details=[d])))
    clock=seal(dict(cost_complete=True,clocks=[seal(dict(start=0,finish=20))]))
    return records,clock


def test_shared_batches_and_overlapping_service_are_not_counted_per_row():
    records,clock=fixture();m=summarize(records,clock)
    assert m['decode_seconds']==2 and m['decode_batches']==1
    assert m['native_service_union_seconds']==5
    assert m['cap_batch_seconds']==2 and m['cap_batches']==1
    assert m['generation_failures']=={'token_cap_without_eos':4}
    assert m['scientific_status']=='DIAGNOSTIC_ONLY' and not m['freeze_eligible']
    changed=copy.deepcopy(records);changed['0']['details']['correct']['details'][0]['tokens']=[2]
    assert summarize(changed,clock)['attempt_behavior']!=m['attempt_behavior']


def test_missing_decode_clock_is_explicit_and_partial_batches_are_rejected():
    records,clock=fixture()
    for row in records.values():row['details']['correct']['details'][0]['generation_batch_id']=None
    m=summarize(records,clock)
    assert m['decode_seconds'] is None and m['unidentified_decode_rows']==4
    records,clock=fixture();records.pop('0')
    with pytest.raises(ValueError,match='batch accounting'):summarize(records,clock)
    with pytest.raises(ValueError):interval_union([(2,1)])


def test_collector_exports_all_declared_evidence_but_no_private_payload(tmp_path):
    from oeis_learn.pipeline_validation.selector_reliability_commands import collect
    root=tmp_path/'output';gen=root/'probes/B/generation/role/generation';gen.mkdir(parents=True)
    (gen/'selections.json').write_text('{}')
    private=root/'probes/B/generation/role/private';private.mkdir()
    (private/'weights.pt').write_bytes(b'private weights')
    (private/'targets.json').write_text('private targets')
    with zipfile.ZipFile(collect(root)) as z:
        names=z.namelist();assert 'probes/B/generation/role/generation/selections.json' in names
        assert not any('private' in n or n.endswith('.pt') for n in names)
        inventory=json.loads(z.read('inventory.json'))
        assert inventory['missing'] and all(r['sha256'].startswith('sha256:') for r in inventory['files'])


def test_collection_failure_does_not_print_a_successful_report(tmp_path,monkeypatch,capsys):
    from oeis_learn.pipeline_validation import selector_reliability_commands as c
    monkeypatch.setattr(c,'source',lambda _: 'source')
    def failure(*a):raise ValueError('original qualification failure')
    monkeypatch.setattr(c,'probes',failure)
    monkeypatch.setattr(c,'collect',lambda _: (_ for _ in ()).throw(OSError('disk failure')))
    args=SimpleNamespace(output='out',campaign='original')
    assert c.main(args,tmp_path,lambda _:None)==1
    assert 'original qualification failure' in (tmp_path/'out/error.txt').read_text()
    assert '"status": "PASS"' not in capsys.readouterr().out


def test_benchmark_rejects_unknown_policy_before_touching_gpu(tmp_path):
    from oeis_learn.pipeline_validation.selector_reliability_roles import benchmark
    with pytest.raises(ValueError):benchmark({},tmp_path,dict(benchmark_policy='unknown',bank_seeds=None),seconds=10)


def test_single_command_qualifies_probes_before_abba_benchmarks_and_collects(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import selector_reliability_commands as c
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    calls=[];root=tmp_path/'out'
    recipe=tmp_path/'recipe.json';recipe.write_text('{}')
    weights=tmp_path/'weights';weights.mkdir();(weights/'weights.pt').write_bytes(b'fixture')
    descriptor=dict(identity='checkpoint',weights=reference(weights/'weights.pt',weights))
    plan=seal(dict(engine='test',source='source',inputs=dict(recipe=reference(recipe,tmp_path),
        checkpoints={a:dict(descriptor=descriptor,root='weights') for a in ('B','D')})))
    def probes(args,workspace,host):
        calls.append('qualified-probes')
        p=workspace/args.output;(p/'publication/role/visible').mkdir(parents=True)
        c.write(p/'plan.json',plan)
        c.write(p/'result.json',seal(dict(scientific_status='INCOMPLETE')))
        c.write(p/'publication/role/visible/prompts.json',[])
    def invoke(host,workspace,p,target,role,operation,artifacts,roots,settings,kind,**kw):
        assert calls[0]=='qualified-probes' and operation=='selector-benchmark' and kw['gpu']
        assert set(artifacts)=={'visible_prompts','frozen_weights'} and settings['bank_seeds'] is None
        calls.append(settings['benchmark_policy'])
        records,clock=fixture();gen=target/'role/generation';gen.mkdir(parents=True)
        c.write(gen/'selections.json',records);c.write(gen/'attempts/portfolio-clocks.json',clock)
        c.write(target/'role/benchmark-telemetry.json',seal(dict(status='PASS')))
        return dict(clock=clock['identity'],benchmark_artifacts=dict(
            records=reference(gen/'selections.json',target/'role'),
            clock=reference(gen/'attempts/portfolio-clocks.json',target/'role'),
            telemetry=reference(target/'role/benchmark-telemetry.json',target/'role')))
    monkeypatch.setattr(c,'probes',probes);monkeypatch.setattr(c,'_invoke',invoke)
    monkeypatch.setattr(c,'source',lambda _: 'source')
    assert c.main(SimpleNamespace(output='out',campaign='original'),tmp_path,
        lambda _:SimpleNamespace(engine=SimpleNamespace(name='test')))==0
    assert calls[1:]==[c.DRAIN,c.MATCHED,c.MATCHED,c.DRAIN]*2
    result=c.read(root/'result.json',sealed=True)
    assert result['probe_status']=='INCOMPLETE' and not result['freeze_eligible']
    assert (root/'diagnostics.zip').is_file() and len(result['runs'])==8


def test_benchmark_role_binds_records_and_collects_bounded_telemetry(tmp_path,monkeypatch):
    import torch
    from oeis_learn.pipeline_validation import generalization_role
    from oeis_learn.pipeline_validation.selector_reliability_roles import benchmark
    from oeis_learn.pipeline_validation.selector_validation_commands import read,check_ref
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.pilot_clocks import DRAIN
    def generate(inputs,output,settings,**kwargs):
        assert kwargs['prompt_limit']==16 and kwargs['evaluation_policy']==DRAIN
        records,clock=fixture()
        write(output/'generation/selections.json',records)
        write(output/'generation/attempts/portfolio-clocks.json',clock)
        return seal(dict(status='PASS',clock=clock['identity']))
    monkeypatch.setattr(generalization_role,'generator',generate)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:123)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:456)
    settings=dict(checkpoint={},prompts={},recipe={},study='study',source='source',
        device='cuda',bank_seeds=None,benchmark_policy=DRAIN)
    result=benchmark({},tmp_path,settings,seconds=10)
    assert result==seal(result)
    for ref in result['benchmark_artifacts'].values():check_ref(tmp_path,ref)
    telemetry=read(check_ref(tmp_path,result['benchmark_artifacts']['telemetry']),sealed=True)
    assert telemetry['gpu_peak_allocated_bytes']==123
    assert (tmp_path/'telemetry/capability.json').is_file()
    assert (tmp_path/'telemetry/telemetry.json').is_file()
