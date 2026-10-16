"""Full matched profiles on actual HIP, including restored cached generation."""

import pytest
import torch
from oeis_learn.experiments.bootstrap_models import bootstrap_profile, construct_bootstrap_models
from oeis_learn.decoder.program_codec import BOS_ID, PAD_ID, encode_body
from oeis_learn.rl.sft_trainer import supervised_step
from oeis_learn.evaluation.search import ModelProposer, SearchLedger

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.version.hip,
        reason="actual HIP accelerator required",
    ),
]


@pytest.mark.parametrize("kind", ["tri_stream", "hybrid", "integer_bytes"])
def test_full_bootstrap_hip_update_checkpoint_and_decoder_cache(kind, tmp_path):
    torch.manual_seed(71)
    torch.use_deterministic_algorithms(True)
    encoder, decoder, _ = construct_bootstrap_models(bootstrap_profile(kind), device="cuda")
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()], lr=0.0001, foreach=False
    )
    values = [[2**200 + n for n in range(20)]]
    tokens = torch.tensor([[BOS_ID] + encode_body("i256.const 7")], device="cuda")
    before = decoder.lm_head.weight.detach().clone()
    result = supervised_step(
        encoder, decoder, optimizer, values, tokens, device="cuda", pad_id=PAD_ID
    )
    assert not torch.equal(before, decoder.lm_head.weight)
    assert result["loss"] > 0
    checkpoint = tmp_path / "weights.pt"
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        memory = encoder.forward_from_sequences(values, device="cuda")
        full = decoder(tokens, memory)
        cached, _ = decoder.prefill(tokens, memory, capacity=32)
        torch.testing.assert_close(full, cached, atol=2e-5, rtol=2e-4)
    _, evidence = ModelProposer(encoder, decoder, context_limit=17).stochastic(
        values[0],
        7,
        SearchLedger(max_attempts=1, max_tokens=16, max_expansions=16, budget_ns=30 * 10**9),
        greedy=True,
    )
    assert evidence["tokens"]
    assert decoder.lm_head.weight.device.type == "cuda"
