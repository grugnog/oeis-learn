import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.pipeline_validation.identity import seal,write,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference
from tests.unit.test_selector_fit_replay import fixture,PROMPT


def test_preparation_keeps_full_training_truth_out_of_fitter_channels(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import selector_fit_replay_role as role
    from oeis_learn.generalization import selector_validation
    records,targets=fixture()
    probes=[dict(PROMPT,role='training',values=list(map(str,range(100))),indices=list(map(str,range(100))))]
    monkeypatch.setattr(selector_validation,'probe_views',lambda rows,inventory:([PROMPT],rows,{}))
    for name,data in [('generation',records),('probes',probes),('inventory',{})]:write(tmp_path/(name+'.json'),data)
    refs={n:reference(tmp_path/(n+'.json'),tmp_path) for n in ('generation','probes','inventory')}
    output=tmp_path/'out';result=role.prepare(dict(mechanical_sources=tmp_path),output,dict(refs,targets=targets),seconds=30)
    visible=json.loads((output/result['prompts']['path']).read_text())
    candidates=json.loads((output/result['candidates']['path']).read_text())
    assert len(visible[0]['values'])==20 and set(visible[0])=={'id','indices','values'}
    assert len(candidates['attempts'])==8 and 'values' not in candidates['attempts'][0]
    assert not (output/'private/targets.json').exists()


def test_host_matrix_is_cpu_only_and_original_probe_is_not_rewritten(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import selector_fit_replay_commands as host,generalization_roles,engines
    from oeis_learn import container_validation
    write(tmp_path/'original.json',seal(dict(status='PASS',freeze_eligible=False)));before=(tmp_path/'original.json').read_bytes()
    monkeypatch.setattr(host,'inputs',lambda *a:dict(targets={'case':['x']*8},generation={},probes={},inventory={},recipe={}))
    monkeypatch.setattr(host,'source',lambda *a:'current-source')
    monkeypatch.setattr(engines.Images,'ensure',lambda *a,**kw:'image')
    monkeypatch.setattr(container_validation,'inspect_current_image',lambda *a,**kw:None)
    monkeypatch.setattr(container_validation,'_stages',lambda *a,**kw:('image',{'native':dict(status='PASS')}))
    hosts=[];launches=[]
    def factory(args):
        value=SimpleNamespace(engine=SimpleNamespace(name='podman'),cpus=args.cpus,memory=args.memory_gib)
        hosts.append(value);return value
    def invoke(host,image,workspace,role,artifacts,**kw):
        launches.append((host.cpus,role,artifacts,kw))
        assert not kw.get('gpu',False)
        assert kw['envelope']['work']>=7200
        if role=='preparation':return dict(status='PASS',packet='packet',cases=1,attempts=8,prompts={},candidates={})
        assert role=='fitter' and set(artifacts)=={'visible_prompts','generated_candidates'}
        assert kw['settings']['recipe']=={}
        return dict(status='PASS',packet='packet',runs=[dict(wall_seconds=1,attempts=[dict(complete=True)]*8,
            cases={'case':dict(fit_search_complete=True)})])
    monkeypatch.setattr(generalization_roles,'project_and_invoke',invoke)
    args=SimpleNamespace(probes='original',output='diagnostics',compare_cpus=8,repetitions=3)
    assert host.main(args,tmp_path,factory)==0
    assert [h.cpus for h in hosts]==[4,8] and all(h.memory==16 for h in hosts)
    assert [(cpu,role) for cpu,role,*_ in launches]==[(4,'preparation'),(4,'fitter'),(4,'fitter'),(8,'fitter')]
    result=json.loads((tmp_path/'diagnostics/result.json').read_text())
    assert result['freeze_eligible'] is False and result['scientific_status']=='DIAGNOSTIC_ONLY'
    assert (tmp_path/'original.json').read_bytes()==before and (tmp_path/'diagnostics/report.zip').exists()


def test_observer_records_real_supervised_worker_transport(tmp_path):
    import time
    from tests.integration.test_synthesis_worker import fake_worker,request
    from oeis_learn.synthesis.worker_pool import WorkerPool
    from oeis_learn.pipeline_validation.selector_fit_replay_role import ObservedWorker
    command=fake_worker(tmp_path,"send(dict(base,results=[dict(candidate_id=k['candidate_id'],outcome='unsat_in_scope') for k in r['candidates']],unstarted=[],complete=True))")
    with WorkerPool(command=command,contract_sha256=request()['contract_sha256']) as worker:
        observer=ObservedWorker(worker);observer.local.calls=[]
        result=observer.call(dict(request(),operation='fit'),deadline=time.monotonic()+3)
        assert result['complete'] and len(result['results'])==2
        timing=observer.local.calls[0]
        assert timing['complete'] and timing['wall_seconds']>0 and timing['queue_wait_seconds']>=0
        assert timing['metrics']['batches']==1
