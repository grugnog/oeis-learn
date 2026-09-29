"""Unit tests for Checkpoint v2 format validation, strict FP32, and legacy conversion."""

from __future__ import annotations

import os
import tempfile
import pytest
import torch
from oeis_learn.decoder.wat_grammar import TOKEN_TO_ID
from oeis_learn.evaluation.checkpoint import (
    CheckpointProvenance,
    compute_file_sha256,
    compute_vocabulary_sha256,
    convert_legacy_checkpoint,
    load_checkpoint_v2,
    save_checkpoint_v2,
)


@pytest.fixture
def sample_models():
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder

    enc_cfg = {
        "d_model": 64,
        "n_heads": 2,
        "n_encoder_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "primes": [3, 5, 7, 11],
        "max_valuation": 16,
        "use_film": True,
    }
    dec_cfg = {
        "d_model": 64,
        "n_heads": 2,
        "n_decoder_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
    }
    encoder = TriStreamEncoder(**enc_cfg)
    decoder = WatTransformerDecoder(**dec_cfg)
    return encoder, decoder, enc_cfg, dec_cfg


def test_checkpoint_v2_save_and_load_roundtrip(sample_models):
    encoder, decoder, enc_cfg, dec_cfg = sample_models
    vocab_hash = compute_vocabulary_sha256(TOKEN_TO_ID)

    with tempfile.NamedTemporaryFile(suffix=".v2.pt", delete=False) as f:
        ckpt_path = f.name

    try:
        provenance = save_checkpoint_v2(
            checkpoint_path=ckpt_path,
            encoder=encoder,
            decoder=decoder,
            encoder_config=enc_cfg,
            decoder_config=dec_cfg,
            epoch=10,
            producer_version="test-1.0.0",
        )

        assert provenance.format_version == "2.0"
        assert provenance.precision == "fp32"
        assert provenance.vocabulary_sha256 == vocab_hash
        assert provenance.checkpoint_sha256.startswith("sha256:")

        loaded_enc, loaded_dec, loaded_prov = load_checkpoint_v2(
            ckpt_path, device=torch.device("cpu")
        )
        assert loaded_prov.epoch == 10
        assert loaded_prov.checkpoint_sha256 == provenance.checkpoint_sha256
        assert loaded_prov.precision == "fp32"
        # Weights should match
        for p1, p2 in zip(encoder.parameters(), loaded_enc.parameters()):
            assert torch.equal(p1, p2)
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_checkpoint_v2_rejects_precision_mismatch(sample_models):
    encoder, decoder, enc_cfg, dec_cfg = sample_models
    with tempfile.NamedTemporaryFile(suffix=".v2.pt", delete=False) as f:
        ckpt_path = f.name

    try:
        save_checkpoint_v2(
            checkpoint_path=ckpt_path,
            encoder=encoder,
            decoder=decoder,
            encoder_config=enc_cfg,
            decoder_config=dec_cfg,
            epoch=1,
            producer_version="test-1.0.0",
        )
        # Tamper payload precision
        payload = torch.load(ckpt_path, weights_only=False)
        payload["provenance"]["precision"] = "fp16"
        torch.save(payload, ckpt_path)

        with pytest.raises(ValueError, match="Strict FP32 precision required"):
            load_checkpoint_v2(ckpt_path, device=torch.device("cpu"))
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_checkpoint_v2_rejects_checksum_tampering(sample_models):
    encoder, decoder, enc_cfg, dec_cfg = sample_models
    with tempfile.NamedTemporaryFile(suffix=".v2.pt", delete=False) as f:
        ckpt_path = f.name

    try:
        save_checkpoint_v2(
            checkpoint_path=ckpt_path,
            encoder=encoder,
            decoder=decoder,
            encoder_config=enc_cfg,
            decoder_config=dec_cfg,
            epoch=1,
            producer_version="test-1.0.0",
        )
        # Expected hash check
        fake_sha = "sha256:" + "0" * 64
        with pytest.raises(ValueError, match="Checksum mismatch"):
            load_checkpoint_v2(ckpt_path, expected_sha256=fake_sha, device=torch.device("cpu"))
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_legacy_checkpoint_conversion(sample_models):
    encoder, decoder, enc_cfg, dec_cfg = sample_models
    with tempfile.NamedTemporaryFile(suffix=".legacy.pt", delete=False) as f_leg, \
         tempfile.NamedTemporaryFile(suffix=".v2.pt", delete=False) as f_v2:
        leg_path = f_leg.name
        v2_path = f_v2.name

    try:
        # Create legacy checkpoint format without provenance
        legacy_data = {
            "epoch": 60,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "metadata": {"pass_rate": 0.12},
        }
        torch.save(legacy_data, leg_path)
        leg_sha = compute_file_sha256(leg_path)

        # Convert to v2
        prov = convert_legacy_checkpoint(
            legacy_checkpoint_path=leg_path,
            output_v2_path=v2_path,
            encoder_config=enc_cfg,
            decoder_config=dec_cfg,
            producer_version="conversion-script-v1",
        )

        assert prov.format_version == "2.0"
        assert prov.source_checkpoint_sha256 == leg_sha
        assert prov.epoch == 60

        # Now load with load_checkpoint_v2
        loaded_enc, loaded_dec, loaded_prov = load_checkpoint_v2(v2_path, device=torch.device("cpu"))
        assert loaded_prov.source_checkpoint_sha256 == leg_sha
        for p1, p2 in zip(encoder.parameters(), loaded_enc.parameters()):
            assert torch.equal(p1, p2)
    finally:
        if os.path.exists(leg_path):
            os.remove(leg_path)
        if os.path.exists(v2_path):
            os.remove(v2_path)
