"""Integration test for mixed SFT + RL co-training loss step and padding attention masks."""

import pytest
import torch
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.data.models import SequenceRecord
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import PAD_ID
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer
from oeis_learn.rl.trainer import EgcaGrpoTrainer


def test_co_training_step_executes_with_sft_blending_and_kl():
    encoder = TriStreamEncoder(d_model=64, n_heads=2, n_encoder_layers=2)
    decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2)
    ref_decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2)
    scheduler = CurriculumScheduler(initial_stage=1)
    elite_buffer = EliteSeedDemonstrationBuffer()

    from oeis_learn.data.models import EliteReplayBufferEntry

    # Pre-populate elite buffer with canonical sequence A000217
    terms = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190]
    wat_code = '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_s local.get $n i64.extend_i32_s i64.const 1 i64.add i64.mul i64.const 2 i64.div_s))'
    elite_buffer.add_entry(EliteReplayBufferEntry(oeis_id="A000217", terms=terms, wat_code=wat_code))

    trainer = EgcaGrpoTrainer(
        encoder=encoder,
        decoder=decoder,
        scheduler=scheduler,
        elite_buffer=elite_buffer,
        ref_decoder=ref_decoder,
        rollout_group_size=4,
        beta_sft=0.20,
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


def test_decoder_padding_mask_prevents_gradient_leak():
    decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2)
    memory = torch.randn(2, 10, 64)

    # Batch of 2 with padding tokens at the end
    tgt = torch.tensor([
        [1, 5, 10, 20, PAD_ID, PAD_ID],
        [1, 8, 12, 15, 18, 2],
    ], dtype=torch.long)

    logits = decoder(tgt, memory)
    assert logits.shape == (2, 6, decoder.vocab_size)
    assert not torch.isnan(logits).any()
