"""Independent convergence counterexamples; no hardware proof is synthesized."""
import json
from pathlib import Path
import pytest
from oeis_learn.pipeline_validation.identity import seal,write,digest


@pytest.mark.parametrize('pac',[True,False])
def test_crash_budget_completion_preserves_charges_and_pac_wrapper_continues(tmp_path,monkeypatch,pac):
    from tests.integration.test_generalization_runtime import pac_config
    from tests.integration.test_generalization_curriculum import models
    from oeis_learn.pipeline_validation import generalization_training as training,generalization_curriculum as curriculum
    from oeis_learn.fresh import bigint_worker
    config=pac_config(tmp_path)
    if not pac:config.pop('pac')
    monkeypatch.setattr(bigint_worker,'models',lambda recipe,device:(*models(),{}))
    monkeypatch.setattr(curriculum,'semantic_probe',lambda *a,**k:None)
    original=training.torch_update
    def injected(*args,**kwargs):
        step=original(*args,**kwargs);count=0
        def update(batch):
            nonlocal count
            result=step(batch);count+=1
            if count==3:raise RuntimeError('optimizer died')
            return result
        return update
    monkeypatch.setattr(training,'torch_update',injected)
    with pytest.raises(RuntimeError,match='optimizer died'):training.run_torch(tmp_path/'data',tmp_path/'crash',config)
    folder=tmp_path/'crash/checkpoints';pointer=json.loads((folder/'generalization-latest.json').read_text())
    monkeypatch.setattr(training,'torch_update',original)
    result=training.run_torch(tmp_path/'data',tmp_path/'recovered',config,resume=(folder,pointer),resume_elapsed=30)
    assert result['discarded_tokens']>0 and result['learner_seconds']>=30
    assert result['tokens']+result['discarded_tokens']==result['charged_tokens']<=config['target_tokens']
    assert result['reason']=='RECOVERED_WITH_DISCARDED_EXPOSURE'
    assert result['status']==('PASS' if pac else 'PARTIAL')
    if not pac:return
    import importlib.util
    path=Path('specs/012-parameterized-adaptive-curriculum/validation/run.py')
    spec=importlib.util.spec_from_file_location('converged_wrapper',path);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    write(tmp_path/'study.json',{'fixture':'canonical transport'})
    stages=[dict(name=n,operation='train' if n.startswith('train-') else 'evaluate',manifest='study.json',seconds=3600,extra=[],requires=[] if not i else [h.stage_order(research=True)[i-1]]) for i,n in enumerate(h.stage_order(research=True))]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('source'),images=[digest('cpu'),digest('gpu')],research=True)
    calls=[]
    def invoke(engine,operation,manifest,output,**kwargs):
        target=tmp_path/output;target.mkdir(parents=True)
        value=seal(dict(status=result['status'] if operation=='train' else 'PASS',stage=operation,inventory=[],result=result if operation=='train' else {}))
        write(target/'result.json',value);write(target.parent/'execution-supervision/process.json',dict(outcome='COMPLETE',returncode=0,cleanup_failure=None))
        return value
    monkeypatch.setattr(h,'invoke',invoke)
    def canonical(stage,target):
        calls.append(stage['name']);return h._canonical_command_stage(plan,stage,target,tmp_path,{})
    h.execute_plan(plan,tmp_path/'campaign',workspace=tmp_path,runner=canonical)
    assert calls==h.stage_order(research=True) and calls[-1]=='analysis'
    calls.clear();h.execute_plan(plan,tmp_path/'resumed-campaign',workspace=tmp_path,runner=canonical,previous=tmp_path/'campaign')
    assert calls==[]

