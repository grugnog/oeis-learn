"""Qualification proof. This test must execute on actual HIP, never CPU fallback."""
import os
import math
from pathlib import Path
import pytest

pytestmark=pytest.mark.gpu


def test_generalization_device_load_and_finite_update(tmp_path):
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':pytest.fail('actual HIP required')
        pytest.skip('actual HIP required')
    from tests.integration.test_bigint_workflow import _fixture
    from oeis_learn.fresh.bigint_worker import models,_batch_loss
    from oeis_learn.fresh.common import file_hash
    from oeis_learn.pipeline_validation.identity import seal,write
    _,_,settings,examples=_fixture('cuda',tmp_path)
    encoder,decoder,_=models(settings['recipe'],'cuda')
    saved=tmp_path/'qualification-weights.pt'
    torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict()),saved)
    loaded=torch.load(saved,map_location='cpu',weights_only=True)
    encoder.load_state_dict(loaded['encoder']);decoder.load_state_dict(loaded['decoder'])
    parameters=list(encoder.parameters())+list(decoder.parameters())
    before=[p.detach().clone() for p in parameters]
    optimizer=torch.optim.AdamW(parameters,lr=settings['recipe']['learning']['learning_rate'])
    loss=_batch_loss(encoder,decoder,examples['training'][:2],'cuda',optimizer=optimizer)
    assert math.isfinite(loss['loss']) and math.isfinite(loss['gradient_norm'])
    assert all(torch.isfinite(p).all() for p in parameters)
    assert any(p.grad is not None for p in parameters)
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    assert any(not torch.equal(a,b) for a,b in zip(before,parameters))
    # supervised_step checks finite loss; persist the actual loaded checkpoint hash.
    record=seal(dict(version='generalization-device-evidence/v1',device='hip',loaded_weights=file_hash(saved),
        finite_forward=True,finite_backward=True,parameters_changed=True))
    destination=os.environ.get('OEIS_GENERALIZATION_DEVICE_EVIDENCE')
    if destination:write(Path(destination),record)
