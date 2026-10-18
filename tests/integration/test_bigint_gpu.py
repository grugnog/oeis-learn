import pytest
import torch

pytestmark = pytest.mark.gpu


def test_actual_hip_large_input_update():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("actual HIP device required; CPU is not evidence")
    from oeis_learn.encoder.hybrid import ChunkHybridEncoder

    model = ChunkHybridEncoder(dropout=0).cuda()
    assert next(model.parameters()).is_cuda
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.chunk_encoder.projection.weight.detach().clone()
    values = [0, 1, -((1 << 4096) - 1), (1 << 4096) - 1]
    out = model.forward_from_sequences([values])
    assert out.is_cuda and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.chunk_encoder.projection.weight)


def test_actual_hip_parameter_decoder_cache_and_position_growth():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("actual HIP device required")
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder

    options = dict(
        vocab_size=codec.FOUNDATION_VOCAB_SIZE,
        d_model=256,
        n_heads=4,
        n_decoder_layers=4,
        d_ff=1024,
        dropout=0,
        max_seq_len=4,
        codec_sha256=codec.codec_digest(),
        attention_backend="eager",
        input_version="decoder-inputs/v1",
        position_scale=16,
    )
    cpu = WatTransformerDecoder(**options).eval()
    body = codec.encode_body("(program (return (add (param.get p0) $index)))")
    tokens = torch.tensor([[codec.BOS_ID] + body[:-1]])
    memory = torch.ones(1, 3, 256)
    with torch.no_grad():
        cpu(tokens, memory)  # grow before device movement/checkpoint load
    gpu = WatTransformerDecoder(**options).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    assert gpu.pos_encoder.pe.is_cuda
    with torch.no_grad():
        for prompt in (memory, memory.flip(1), torch.ones(1, 1, 256)):
            prompt = prompt.cuda()
            ids = tokens.cuda()
            full = gpu(ids, prompt)
            cached, cache = gpu.prefill(ids[:, :1], prompt, capacity=1024)
            outputs = [cached]
            for token in ids[:, 1:].unbind(1):
                outputs.append(gpu.decode_step(token, cache))
            torch.testing.assert_close(torch.cat(outputs, 1), full, atol=2e-5, rtol=2e-4)
