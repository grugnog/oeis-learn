import json
from pathlib import Path
import pytest


def test_selected_initializer_and_sanitized_pins_from_real_cpu_store(tmp_path,monkeypatch):
    import torch
    from tests.integration.test_generalization_runtime import fixture
    from oeis_learn.pipeline_validation.generalization_training import run_torch
    from oeis_learn.pipeline_validation.generalization_runtime import learner_config
    from oeis_learn.pipeline_validation.generalization_selection_io import remember_origin,publish_selected,persist_selected
    from oeis_learn.pipeline_validation import generalization_roles as roles,generalization_role as handlers
    from oeis_learn.pipeline_validation.identity import seal,digest
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.generalization.initializers import verify_initializer
    from oeis_learn.generalization.selection import select
    from oeis_learn.generalization.role_contracts import Role
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    study,runtime,_=fixture(tmp_path);recipe=json.loads((tmp_path/'recipe.json').read_text())
    def models(recipe,device):
        torch.manual_seed(recipe['model']['seed']);return torch.nn.Linear(1,1),torch.nn.Linear(1,1),{}
    def update(encoder,decoder,batch,device,**kw):
        optimizer=kw['optimizer'];optimizer.zero_grad();loss=decoder(encoder(torch.ones(len(batch),1))).square().mean();loss.backward();optimizer.step()
    monkeypatch.setattr(bigint_worker,'models',models);monkeypatch.setattr(bigint_worker,'_batch_loss',update)
    config=learner_config(study,runtime,recipe,'static',study['seeds'][0]);config['device']='cpu'
    trained=run_torch(tmp_path,tmp_path/'trained',config)
    parent=tmp_path/'trained/checkpoints';remember_origin(tmp_path,parent,study,'static')
    roster=json.loads((parent/'retained.json').read_text());available=[(c,parent) for c in roster['checkpoints']]
    behavior=[dict(id=c['identity'],tokens=c['tokens'],role='selection',structural=({'s':'f'},{'s':c['identity']==roster['latest']}),parameter=({'p':'g'},{'p':c['identity']==roster['latest']})) for c,_ in available]
    chosen=select(behavior)
    decision=seal(dict(version='generalization-checkpoint-selection/v1',status='PASS',scientific_status='UNMEASURED',targets=reference(tmp_path/'training.json',tmp_path),
        candidates=[dict(checkpoint=c,selections=reference(tmp_path/'empty.json',tmp_path)) for c,_ in available],
        measurements=[dict(checkpoint=b['id'],loss=1.,behavior=b) for b in behavior],best_loss=available[0][0]['identity'],selected=chosen.checkpoint_identity,eligible=list(chosen.eligible),reason=chosen.reason))
    calls=[]
    def invoke(host,image,workspace,name,artifacts,**kw):
        calls.append((name,kw['operation']));output=workspace/kw['output']/'role';output.mkdir(parents=True)
        if name=='learner':
            assert kw['operation']=='pin-selection' and artifacts['training_targets']==[]
            settings=kw['settings'];inputs=kw['source_roots']['verified_training_replay']
            pins=json.loads((inputs/settings['decision']['path']).read_text())
            assert 'measurements' not in pins and 'candidates' not in pins
        return handlers.HANDLERS[(Role(name),kw['operation'])](kw['source_roots'],output,kw['settings'],seconds=kw['seconds'])
    monkeypatch.setattr(roles,'project_and_invoke',invoke)
    publication=publish_selected(None,tmp_path,tmp_path/'published',tmp_path,study,runtime,decision,available)
    assert publication['status']=='PASS'
    consumer=seal(dict(study,version='generalization-study/v2',recipe='P4-replay/v1',arms=['continued_sft','verified_replay'],
        progression=dict(study['progression'],stage='P4'),model=dict(study['model'],initialization=dict(kind='retained',artifact=publication['artifact']))))
    assert verify_initializer(tmp_path/'published/publication/role',consumer)['checkpoint']==roster['latest']
    results=persist_selected(None,tmp_path,tmp_path/'pinned',runtime,decision,available,seconds=60)
    assert len(results)==1 and results[0]['status']=='PASS'
    result=results[0];root=tmp_path/result['output']/result['root']
    restored=GeneralizationCheckpointStore(root,trained['compatibility'],config['target_tokens']).load(result['checkpoint'])
    assert restored['states']['selection']['checkpoint']==roster['latest']
    assert calls==[('preparation','publish-initializer'),('learner','pin-selection')]
