"""Checkpoint v2 format management, architecture reconstruction, and legacy conversion."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, Optional, Tuple
import torch
from oeis_learn.data.models import CheckpointIdentity, CheckpointProvenance
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import ID_TO_TOKEN, TOKEN_TO_ID
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 digest over file bytes, prefixed with 'sha256:'."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def compute_vocabulary_sha256(token_to_id: Dict[str, int]) -> str:
    """Computes deterministic digest over the token vocabulary ordered by token ID."""
    ordered_tokens = sorted(token_to_id.keys(), key=lambda t: token_to_id[t])
    raw = json.dumps(ordered_tokens, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def get_runtime_environment_info() -> Dict[str, Any]:
    """Captures runtime versions and host platform metadata."""
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def save_checkpoint_v2(
    checkpoint_path: str,
    encoder: TriStreamEncoder,
    decoder: WatTransformerDecoder,
    encoder_config: Dict[str, Any],
    decoder_config: Dict[str, Any],
    epoch: int,
    producer_version: str = "oeis-learn-0.1.0",
    source_checkpoint_sha256: Optional[str] = None,
    runtime_environment: Optional[Dict[str, Any]] = None,
) -> CheckpointProvenance:
    """Saves model weights along with strict Checkpoint v2 provenance metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    vocab_hash = compute_vocabulary_sha256(TOKEN_TO_ID)
    runtime_env = runtime_environment or get_runtime_environment_info()

    payload = {
        "format_version": "2.0",
        "precision": "fp32",
        "epoch": epoch,
        "producer_version": producer_version,
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
        "vocabulary_sha256": vocab_hash,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "runtime_environment": runtime_env,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
    }

    # Save to disk first to obtain exact file digest
    torch.save(payload, checkpoint_path)
    file_sha = compute_file_sha256(checkpoint_path)

    # Attach provenance with the computed file sha256
    prov = CheckpointIdentity(
        format_version="2.0",
        checkpoint_sha256=file_sha,
        producer_version=producer_version,
        epoch=epoch,
        precision="fp32",
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        vocabulary_sha256=vocab_hash,
        source_checkpoint_sha256=source_checkpoint_sha256,
        runtime_environment=runtime_env,
    )
    payload["provenance"] = prov.to_dict()
    torch.save(payload, checkpoint_path)
    final_sha = compute_file_sha256(checkpoint_path)

    return CheckpointIdentity(
        format_version="2.0",
        checkpoint_sha256=final_sha,
        producer_version=producer_version,
        epoch=epoch,
        precision="fp32",
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        vocabulary_sha256=vocab_hash,
        source_checkpoint_sha256=source_checkpoint_sha256,
        runtime_environment=runtime_env,
    )


def load_checkpoint_v2(
    checkpoint_path: str,
    expected_sha256: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TriStreamEncoder, WatTransformerDecoder, CheckpointProvenance]:
    """Loads and reconstructs TriStreamEncoder and WatTransformerDecoder strictly from Checkpoint v2."""
    dev = device or torch.device("cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    actual_sha = compute_file_sha256(checkpoint_path)
    if expected_sha256 is not None and expected_sha256 != actual_sha:
        raise ValueError(
            f"Checksum mismatch for {checkpoint_path}: expected {expected_sha256}, got {actual_sha}"
        )

    payload = torch.load(checkpoint_path, map_location=dev, weights_only=False)

    prov_dict = payload.get("provenance", {})
    format_version = prov_dict.get("format_version", payload.get("format_version"))
    if format_version != "2.0":
        raise ValueError(f"Expected checkpoint format_version '2.0', got '{format_version}'")

    precision = prov_dict.get("precision", payload.get("precision"))
    if precision != "fp32":
        raise ValueError(f"Strict FP32 precision required, checkpoint has '{precision}'")

    expected_vocab = compute_vocabulary_sha256(TOKEN_TO_ID)
    vocab_hash = prov_dict.get("vocabulary_sha256", payload.get("vocabulary_sha256"))
    if vocab_hash != expected_vocab:
        raise ValueError(
            f"Vocabulary hash mismatch: expected {expected_vocab}, got {vocab_hash}"
        )

    enc_cfg = prov_dict.get("encoder_config", payload.get("encoder_config"))
    dec_cfg = prov_dict.get("decoder_config", payload.get("decoder_config"))
    if not enc_cfg or not dec_cfg:
        raise ValueError("Missing encoder_config or decoder_config in checkpoint metadata")

    # Reconstruct architecture strictly from metadata
    encoder = TriStreamEncoder(**enc_cfg)
    decoder = WatTransformerDecoder(**dec_cfg)

    encoder.load_state_dict(payload["encoder_state_dict"])
    decoder.load_state_dict(payload["decoder_state_dict"])

    encoder.to(dev)
    decoder.to(dev)
    encoder.eval()
    decoder.eval()

    provenance = CheckpointIdentity(
        format_version="2.0",
        checkpoint_sha256=actual_sha,
        producer_version=prov_dict.get("producer_version", payload.get("producer_version", "unknown")),
        epoch=int(prov_dict.get("epoch", payload.get("epoch", 0))),
        precision="fp32",
        encoder_config=enc_cfg,
        decoder_config=dec_cfg,
        vocabulary_sha256=vocab_hash,
        source_checkpoint_sha256=prov_dict.get(
            "source_checkpoint_sha256", payload.get("source_checkpoint_sha256")
        ),
        runtime_environment=prov_dict.get("runtime_environment", payload.get("runtime_environment", {})),
    )

    return encoder, decoder, provenance


def convert_legacy_checkpoint(
    legacy_checkpoint_path: str,
    output_v2_path: str,
    encoder_config: Dict[str, Any],
    decoder_config: Dict[str, Any],
    producer_version: str = "legacy-converter-v1",
) -> CheckpointProvenance:
    """Converts a legacy model checkpoint to strict Checkpoint v2 format."""
    if not os.path.exists(legacy_checkpoint_path):
        raise FileNotFoundError(f"Legacy checkpoint not found: {legacy_checkpoint_path}")

    source_sha = compute_file_sha256(legacy_checkpoint_path)
    legacy_payload = torch.load(
        legacy_checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )

    encoder = TriStreamEncoder(**encoder_config)
    decoder = WatTransformerDecoder(**decoder_config)

    encoder.load_state_dict(legacy_payload["encoder_state_dict"])
    decoder.load_state_dict(legacy_payload["decoder_state_dict"])

    epoch = int(legacy_payload.get("epoch", 0))

    return save_checkpoint_v2(
        checkpoint_path=output_v2_path,
        encoder=encoder,
        decoder=decoder,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        epoch=epoch,
        producer_version=producer_version,
        source_checkpoint_sha256=source_sha,
    )
