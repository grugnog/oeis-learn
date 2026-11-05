"""Exercise the complete installed calibration handler with explicit CPU adapters.

Actual tensors, 64 updates, feedback, checkpoint save/load and measured arithmetic
are retained. HIP gates/synchronization and native semantic execution are replaced;
this fixture cannot qualify production device performance.
"""
import copy
from contextlib import contextmanager
import json
import pytest


@pytest.fixture
def calibration_case(tmp_path,monkeypatch):
    import torch
    from tests.integration.test_generalization_curriculum import prepared,models as tiny_models
    from oeis_learn.pipeline_validation import generalization_training as training,generalization_curriculum as curriculum,generalization_execution as execution
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.pipeline_validation.identity import digest
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.generalization.contracts import verify_artifact
    batches,bundle=prepared(tmp_path)
    literal_rows=[r for ref in bundle['literal'] for r in json.loads(verify_artifact(tmp_path/'data',ref))]
    expected={tuple(r['tokens']):r['values'] for r in literal_rows}
    native_calls=[];runs=[]
    # Only the calibration entry gate sees a synthetic device capability.
    monkeypatch.setattr(execution,'verify_execution',lambda:None)
    monkeypatch.setattr(torch.version,'hip','CPU-ADAPTER-NOT-DEVICE-EVIDENCE')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    def models(recipe,device):
        encoder,decoder=tiny_models();return encoder,decoder,{}
    monkeypatch.setattr(bigint_worker,'models',models)
    original_run=training.run_torch;original_step=training.torch_update;original_observe=curriculum.observe_targets
    def run(root,output,config,**kwargs):
        assert config['device']=='cuda' and config['pac']['purpose']=='research' and kwargs==dict(stop_after_updates=64)
        adapted=copy.deepcopy(config);adapted['device']='cpu';adapted['pac'].update(purpose='acceptance',observation_seconds=120)
        with monkeypatch.context() as m:
            m.setattr(torch.cuda,'is_available',lambda:False)
            result=original_run(root,output,adapted,**kwargs)
        runs.append(result);return result
    monkeypatch.setattr(training,'run_torch',run)
    monkeypatch.setattr(training,'torch_update',lambda encoder,decoder,optimizer,**kw:original_step(encoder,decoder,optimizer,device='cpu'))
    monkeypatch.setattr(curriculum,'observe_targets',lambda encoder,decoder,rows,**kw:original_observe(encoder,decoder,rows,device='cpu'))
    monkeypatch.setattr(curriculum,'semantic_probe',lambda *a,**kw:None)
    class Native:
        fail=False
        def execute(self,source,indices,**kwargs):
            result=evaluate(source,indices)
            assert result.outcome=='ok' and list(result.values)==expected[tuple(encode_body(source))]
            native_calls.append(source)
            return dict(outcome='timeout' if self.fail else result.outcome,values=list(result.values))
    native=Native()
    @contextmanager
    def client(settings,output):yield native
    monkeypatch.setattr(bigint_worker,'client',client)
    recipe=resolve(dict(version='bigint-bootstrap/v1'))
    settings=dict(recipe=recipe,source=digest('source'),environment=digest('CPU adapter'),bundle=bundle,
        study=dict(identity=digest('fixture-study'),seed=7,initializer=digest('random'),model=digest('tiny-model'),arms=['B','C']))
    assert all('source' not in r for r in batches.rows.values())
    return settings,native,native_calls,runs


def test_complete_calibration_uses_literal_tokens_and_real_update_save_load(tmp_path,calibration_case):
    from oeis_learn.pipeline_validation.generalization_pac_calibration import calibrate,verify_calibration,REQUIRED
    settings,native,calls,runs=calibration_case
    result=calibrate(dict(training_targets=tmp_path/'data'),tmp_path/'calibration',settings,seconds=300)
    assert len(calls)==4 and len(runs)==1 and runs[0]['updates']==64
    assert result['valid_rewards']>=6 and result['tokens']>0
    assert set(result['samples'])==set(REQUIRED) and all(result['samples'].values())
    assert result['samples']['save']==[runs[0]['timing']['max_checkpoint_seconds']]
    assert len(result['samples']['short_update'])==len(result['samples']['long_update'])==2
    assert result['weights_are_research_initializers'] is False
    assert verify_calibration(result,source=settings['source'],environment=settings['environment'],arms=['B','C'],model=settings['study']['model'])==result
    assert json.loads((tmp_path/'calibration/calibration.json').read_text())==result


def test_incomplete_native_calibration_stops_before_learner(tmp_path,calibration_case):
    from oeis_learn.pipeline_validation.generalization_pac_calibration import calibrate
    settings,native,calls,runs=calibration_case;native.fail=True
    with pytest.raises(ValueError,match='representative native execution incomplete'):
        calibrate(dict(training_targets=tmp_path/'data'),tmp_path/'calibration',settings,seconds=300)
    assert len(calls)==1 and not runs
    assert not (tmp_path/'calibration/calibration.json').exists()
