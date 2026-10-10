"""Integration test for complete EGCA-GRPO training iteration."""

import torch
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
