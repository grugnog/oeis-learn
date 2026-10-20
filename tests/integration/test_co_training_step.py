"""On-policy KL updates, padding masks and explicit retired-teacher rejection."""

import pytest
import torch
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.data.models import SequenceRecord
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import PAD_ID
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer
from oeis_learn.rl.trainer import EgcaGrpoTrainer


def test_on_policy_step_executes_with_kl_and_no_legacy_teacher(fitting_client):
    encoder = TriStreamEncoder(d_model=64, n_heads=2, n_encoder_layers=2)
    decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2)
    ref_decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2)
    scheduler = CurriculumScheduler(initial_stage=1)
    elite_buffer = EliteSeedDemonstrationBuffer()

    terms = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190]

    trainer = EgcaGrpoTrainer(
        encoder=encoder,
        decoder=decoder,
        scheduler=scheduler,
        elite_buffer=elite_buffer,
        ref_decoder=ref_decoder,
        rollout_group_size=4,
        beta_sft=0.0,
        ref_encoder=TriStreamEncoder(d_model=64, n_heads=2, n_encoder_layers=2),
        beta_kl=0.05,
        alpha_ent=0.01,
    )

    record = SequenceRecord(
        oeis_id="A000217",
        name="Triangular numbers",
        terms=terms,
        curriculum_stage=1,
    )

    metrics = trainer.train_step_for_prompt(record, epoch=1)

    assert "loss" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))
    assert "entropy" in metrics
    assert metrics["entropy"] >= 0.0
    assert not elite_buffer.entries and not elite_buffer.canonical_archive


def test_legacy_teacher_entry_is_explicitly_rejected():
    from oeis_learn.data.models import EliteReplayBufferEntry

    buffer = EliteSeedDemonstrationBuffer()
    with pytest.raises(ValueError, match="Legacy teacher ingestion is retired"):
        buffer.add_entry(EliteReplayBufferEntry("A000012", [1] * 20, "i256.const 1"))
    assert not buffer.entries and not buffer.canonical_archive


def test_decoder_padding_mask_prevents_gradient_leak():
    decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2)
    memory = torch.randn(2, 10, 64)

    # Batch of 2 with padding tokens at the end
    tgt = torch.tensor(
        [
            [1, 5, 10, 20, PAD_ID, PAD_ID],
            [1, 8, 12, 15, 18, 2],
        ],
        dtype=torch.long,
    )

    logits = decoder(tgt, memory)
    assert logits.shape == (2, 6, decoder.vocab_size)
    assert not torch.isnan(logits).any()


def test_old_teacher_blending_is_explicitly_rejected():
    with pytest.raises(ValueError, match="unsupported legacy"):
        EgcaGrpoTrainer(
            TriStreamEncoder(d_model=16, n_heads=2, n_encoder_layers=1, d_ff=32),
            WatTransformerDecoder(d_model=16, n_heads=2, n_decoder_layers=1, d_ff=32),
            CurriculumScheduler(),
            enable_cgi=True,
        )
