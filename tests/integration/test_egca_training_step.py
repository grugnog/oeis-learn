"""Integration test for complete EGCA-GRPO training iteration."""

import torch
import pytest
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.data.models import SequenceRecord
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.trainer import EgcaGrpoTrainer


def test_egca_training_step_executes_and_updates_gradients():
    encoder = TriStreamEncoder(d_model=64, n_heads=2, n_encoder_layers=2, d_ff=128)
    decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2, d_ff=128)
    scheduler = CurriculumScheduler(initial_stage=1)

    trainer = EgcaGrpoTrainer(
        encoder=encoder,
        decoder=decoder,
        scheduler=scheduler,
        rollout_group_size=4,
        asymmetric_penalty_weight=1.0,
    )

    record = SequenceRecord(
        oeis_id="A000027",
        name="Positive integers: 1, 2, 3, 4, 5...",
        terms=[n for n in range(1, 30)],
        curriculum_stage=1,
    )

    metrics = trainer.train_step_for_prompt(record)

    assert "loss" in metrics
    assert "pass_rate" in metrics
    assert "mean_reward" in metrics
    assert "entropy" in metrics
    assert "acr" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))
    assert len(trainer.telemetry.records) == 1


def test_original_policy_checkpoint_reproduces_next_rollout_update(tmp_path):
    from oeis_learn.tracking.training_checkpoint import learning_state_digest

    torch.manual_seed(123)
    encoder = TriStreamEncoder(d_model=16, n_heads=2, n_encoder_layers=1, d_ff=32, dropout=0)
    decoder = WatTransformerDecoder(d_model=16, n_heads=2, n_decoder_layers=1, d_ff=32, dropout=0)
    trainer = EgcaGrpoTrainer(
        encoder, decoder, CurriculumScheduler(), rollout_group_size=2, max_program_length=24
    )
    record = SequenceRecord("A000027", "", list(range(20)), curriculum_stage=1)
    trainer.train_step_for_prompt(record)
    checkpoint = tmp_path / "policy.pt"
    trainer.save_checkpoint(checkpoint, epoch=1)
    expected = trainer.train_step_for_prompt(record)
    actions = [r["actions"] for r in trainer.last_rollouts]
    state = learning_state_digest(trainer.decoder.state_dict())
    trainer.load_checkpoint(checkpoint)
    actual = trainer.train_step_for_prompt(record)
    assert actual["loss"] == expected["loss"] and actual["entropy"] == expected["entropy"]
    assert [r["actions"] for r in trainer.last_rollouts] == actions
    assert learning_state_digest(trainer.decoder.state_dict()) == state
    assert len(trainer.scheduler.prompt_histories[record.oeis_id]) == 2


@pytest.mark.parametrize("objective", ["reinforce", "grpo"])
def test_generic_training_prompt_uses_real_body_codec_policy_and_execution(tmp_path, objective):
    from oeis_learn.encoder.integer_bytes import IntegerByteEncoder
    from oeis_learn.decoder.program_codec import (
        FOUNDATION_VOCAB_SIZE,
        PAD_ID,
        EOS_ID,
        TOKEN_TO_ID,
        codec_digest,
    )
    from oeis_learn.rl.trainer import TrainingPrompt
    from oeis_learn.sandbox.runner import FoundationRunner

    torch.manual_seed(7)
    encoder = IntegerByteEncoder(16, 2, 1, 32)
    decoder = WatTransformerDecoder(
        vocab_size=FOUNDATION_VOCAB_SIZE,
        d_model=16,
        n_heads=2,
        n_decoder_layers=1,
        d_ff=32,
        dropout=0,
        pad_idx=PAD_ID,
        codec_sha256=codec_digest(),
    )
    # This fixed policy makes complete sampled programs likely; it is not an
    # archive teacher or a substituted rollout. The grammar and sampler decide.
    with torch.no_grad():
        decoder.lm_head.weight.zero_()
        decoder.lm_head.bias.zero_()
        decoder.lm_head.bias[TOKEN_TO_ID["i256.zero"]] = 6
        decoder.lm_head.bias[EOS_ID] = 8
    with FoundationRunner(tmp_path / "runtime", backend="python_wasmtime") as runner:
        trainer = EgcaGrpoTrainer(
            encoder,
            decoder,
            CurriculumScheduler(),
            wasm_runner=runner,
            objective=objective,
            seed=7,
            rollout_group_size=2,
            max_program_length=16,
        )
        metrics = trainer.train_step_for_prompt(TrainingPrompt("training-archive-key", (0,) * 20))
    assert metrics["pass_count"] == 2 and torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["prompt_kind"] == "generic_training"
    assert len(trainer.last_rollouts) == 2
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in list(encoder.parameters()) + list(decoder.parameters())
    )
