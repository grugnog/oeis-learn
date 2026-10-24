import pytest


def test_actual_hip_diagnostic_checkpoint_roundtrip(tmp_path):
    if not torch.cuda.is_available() or not torch.version.hip:
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':
            pytest.fail('actual HIP device required; CPU is not evidence')
        pytest.skip('actual HIP device required')
    from tests.unit.test_pipeline_profile import check_diagnostic_checkpoint_device_roundtrip
    check_diagnostic_checkpoint_device_roundtrip('cuda',tmp_path)
import os
import torch
from tests.integration.test_bigint_decoding import check_generation_batch_parity

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("prompt", ["correct", "shuffled", "absent"])
@pytest.mark.parametrize("temperature", [0, 0.8])
def test_actual_hip_mixed_prompt_batched_generation_parity(prompt, temperature):
    if not torch.cuda.is_available() or not torch.version.hip:
        if os.environ.get("OEIS_REQUIRE_BIGINT_GPU") == "1":
            pytest.fail("actual HIP device required")
        pytest.skip("actual HIP device required")
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder

    torch.manual_seed(20261018)
    decoder = (
        WatTransformerDecoder(
            vocab_size=codec.FOUNDATION_VOCAB_SIZE,
            codec_sha256=codec.codec_digest(),
            d_model=256,
            n_heads=4,
            n_decoder_layers=4,
            d_ff=1024,
            dropout=0,
            attention_backend="eager",
            input_version="decoder-inputs/v1",
            position_scale=16,
        )
        .cuda()
        .eval()
    )
    check_generation_batch_parity(decoder, "cuda", prompt, temperature)


def test_actual_hip_large_input_update():
    if not torch.cuda.is_available() or not torch.version.hip:
        if os.environ.get("OEIS_REQUIRE_BIGINT_GPU") == "1":
            pytest.fail("actual HIP device required; CPU is not evidence")
        pytest.skip("actual HIP device required; CPU is not evidence")
    from oeis_learn.encoder.hybrid import ChunkHybridEncoder

    model = ChunkHybridEncoder(dropout=0).cuda()
    assert next(model.parameters()).is_cuda
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.chunk_encoder.projection.weight.detach().clone()
    values = [
        [0, (1 << bits) - 2, -((1 << bits) - 1), (1 << bits) - 1] for bits in (256, 512, 1024, 4096)
    ]
    out = model.forward_from_sequences(values)
    assert out.is_cuda and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.chunk_encoder.projection.weight)
    if os.environ.get("OEIS_TEST_OUTPUT"):
        import hashlib
        import json
        from pathlib import Path

        output = Path(os.environ["OEIS_TEST_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        report = dict(
            device=torch.cuda.get_device_name(),
            hip=torch.version.hip,
            before_sha256=hashlib.sha256(before.cpu().numpy().tobytes()).hexdigest(),
            after_sha256=hashlib.sha256(
                model.chunk_encoder.projection.weight.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            widths=[256, 512, 1024, 4096],
            difference_bits=4097,
            finite=True,
            updated=True,
        )
        (output / "encoder-update.json").write_text(json.dumps(report, indent=2) + "\n")


def test_actual_hip_parameter_decoder_cache_and_position_growth():
    if not torch.cuda.is_available() or not torch.version.hip:
        if os.environ.get("OEIS_REQUIRE_BIGINT_GPU") == "1":
            pytest.fail("actual HIP device required")
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
    memory = torch.randn(2, 3, 256)
    tokens = tokens.expand(2, -1)
    with torch.no_grad():
        cpu(tokens, memory)  # grow before device movement/checkpoint load
    gpu = WatTransformerDecoder(**options).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    assert gpu.pos_encoder.pe.is_cuda
    with torch.no_grad():
        from oeis_learn.fresh.conditioning_controls import condition_memory

        for mode in ("correct", "shuffled", "absent"):
            prompt, _ = condition_memory(
                memory,
                torch.zeros(2, 3, dtype=torch.bool),
                mode,
                order=(1, 0) if mode == "shuffled" else None,
            )
            cpu_full = cpu(tokens, prompt)
            prompt = prompt.cuda()
            ids = tokens.cuda()
            full = gpu(ids, prompt)
            torch.testing.assert_close(full.cpu(), cpu_full, atol=2e-5, rtol=2e-4)
            cached, cache = gpu.prefill(ids[:, :1], prompt, capacity=1024)
            outputs = [cached]
            for token in ids[:, 1:].unbind(1):
                outputs.append(gpu.decode_step(token, cache))
            torch.testing.assert_close(torch.cat(outputs, 1), full, atol=2e-5, rtol=2e-4)

            from oeis_learn.decoder.sampler import WatProgramSampler

            generated = []
            for use_cache in (False, True):
                sampler = WatProgramSampler(
                    gpu,
                    max_length=24,
                    temperature=0,
                    codec_profile="bigint-codec/v1",
                    use_cache=use_cache,
                )
                generated.append(sampler.sample_trajectories(prompt, [17, 19]))
            assert [r.tokens for r in generated[0]] == [r.tokens for r in generated[1]]
            assert [r.complete for r in generated[0]] == [r.complete for r in generated[1]]


@pytest.mark.gpu
def test_hoisted_bookkeeping_actual_hip_replay():
    from tests.integration.test_bigint_decoding import model
    from oeis_learn.decoder import bigint_codec as codec
    assert torch.version.hip and torch.cuda.is_available(), 'actual HIP required'
    decoder=model().to('cuda')
    memory=torch.randn(8,3,16,device='cuda')
    token=torch.full((8,),codec.BOS_ID,device='cuda',dtype=torch.long)
    with torch.no_grad():
        decoder.cache_bookkeeping='baseline'
        _,a=decoder.prefill(token[:,None],memory,capacity=1024)
        decoder.cache_bookkeeping='hoisted/v1'
        _,b=decoder.prefill(token[:,None],memory,capacity=1024)
        for position in range(1,1024):
            expected=decoder.decode_step(token,a)
            actual=decoder.decode_step(token,b)
            if position+1 in (32,128,513,1024):
                torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('batch',[1,8])
@pytest.mark.parametrize('backend',['eager','sdpa'])
def test_actual_hip_hoisted_long_context_parity(batch,backend):
    if not torch.cuda.is_available() or not torch.version.hip:
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':
            pytest.fail('actual HIP device required; CPU is not evidence')
        pytest.skip('actual HIP device required')
    from tests.integration.test_bigint_decoding import check_hoisted_cache_parity
    check_hoisted_cache_parity('cuda',batch,backend)


def test_actual_hip_baseline_replay_reference_and_frozen_roles(tmp_path):
    if not torch.cuda.is_available() or not torch.version.hip:
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':
            pytest.fail('actual HIP device required; CPU is not replay evidence')
        pytest.skip('actual HIP device required')
    from tests.unit.test_pipeline_profile import check_baseline_replay_role
    check_baseline_replay_role('cuda',tmp_path)
