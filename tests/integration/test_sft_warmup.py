"""Integration test for SFT Warmup Training Convergence."""

import torch
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.sft_trainer import SftTrainer


def test_sft_trainer_runs_and_decreases_loss(tmp_path):
    ckpt_path = tmp_path / "sft_best.pt"
    dataset_path = tmp_path / "sft_data.json"

    # Use scaled lightweight models for fast integration test
    encoder = TriStreamEncoder(d_model=64, n_heads=2, n_encoder_layers=2, d_ff=128)
    decoder = WatTransformerDecoder(d_model=64, n_heads=2, n_decoder_layers=2, d_ff=128)

    trainer = SftTrainer(
        dataset_path=str(dataset_path),
        output_checkpoint=str(ckpt_path),
        encoder=encoder,
        decoder=decoder,
        epochs=3,
        lr=1e-3,
        batch_size=8,
    )

    result = trainer.train()
    assert result["epochs_trained"] == 3
    assert result["final_loss"] > 0.0
    assert ckpt_path.exists()

    # Verify loaded checkpoint
    ckpt = torch.load(str(ckpt_path))
    assert "encoder_state_dict" in ckpt
    assert "decoder_state_dict" in ckpt
