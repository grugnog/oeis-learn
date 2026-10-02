"""Checkpoint v2 format management, architecture reconstruction, and legacy conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from oeis_learn.experiments.artifacts import ArtifactPath, compute_canonical_digest, load_json
from oeis_learn.experiments.config import load_config
from oeis_learn.experiments.models import validate_artifact
from oeis_learn.experiments.profiles import profile_digests
from oeis_learn.evaluation.foundation_cohort import exact_keys

import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, Optional, Tuple
import torch
from oeis_learn.data.models import CheckpointIdentity, CheckpointProvenance
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import TOKEN_TO_ID
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
        raise ValueError(f"Vocabulary hash mismatch: expected {expected_vocab}, got {vocab_hash}")

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
        producer_version=prov_dict.get(
            "producer_version", payload.get("producer_version", "unknown")
        ),
        epoch=int(prov_dict.get("epoch", payload.get("epoch", 0))),
        precision="fp32",
        encoder_config=enc_cfg,
        decoder_config=dec_cfg,
        vocabulary_sha256=vocab_hash,
        source_checkpoint_sha256=prov_dict.get(
            "source_checkpoint_sha256", payload.get("source_checkpoint_sha256")
        ),
        runtime_environment=prov_dict.get(
            "runtime_environment", payload.get("runtime_environment", {})
        ),
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


# Strict foundation loading is deliberately separate from legacy v2 conversion.
# This reader does not write checkpoints or claim to implement T041 recovery.


@dataclass
class FoundationCheckpoint:
    manifest: dict
    manifest_sha256: str
    contract: dict
    config: dict
    encoder: TriStreamEncoder
    decoder: WatTransformerDecoder
    device: torch.device
    payload: dict | None = None


def load_foundation_checkpoint(manifest_path, *, device="cpu", include_payload=False) -> FoundationCheckpoint:
    """Load exact weights from a complete, externally hashed foundation bundle.

    Fixed sidecars are part of this reader contract: config.yaml,
    effective-config.json, contract.json, pool.json and runtime.json, all under
    the manifest's immutable parent. GPU selection never falls back to CPU.
    torch.load uses its restricted weights-only unpickler, not executable pickle.
    """
    import random
    import numpy as np

    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path.read_bytes())
    validate_artifact(manifest)
    if manifest["kind"] != "checkpoint_manifest":
        raise ValueError("checkpoint manifest required")
    root = manifest_path.parent
    if manifest["codec_sha256"] != profile_digests()["codec"]:
        raise ValueError("checkpoint codec identity mismatch")
    sidecars = {}
    for name, field in [
        ("contract", "contract_sha256"),
        ("pool", "pool_sha256"),
        ("runtime", "runtime_sha256"),
        ("effective-config", "effective_config_sha256"),
    ]:
        path = ArtifactPath(root, name + ".json").as_path()
        value = load_json(path.read_bytes())
        if compute_canonical_digest(value) != manifest[field]:
            raise ValueError(f"{name} identity mismatch")
        sidecars[name] = value
    contract = sidecars["contract"]
    exact_keys(
        contract,
        (
            "schema_version",
            "kind",
            "purpose",
            "run_id",
            "profiles",
            "track",
            "initialization",
            "objective",
            "constitution_version",
            "observed_terms",
            "total_terms",
            "index_policy",
            "enabled_subsystems",
            "run_provenance",
            "contract_id",
        ),
        "experiment contract",
    )
    if (
        contract["contract_id"] != compute_canonical_digest(contract, "contract_id")
        or contract["schema_version"] != "foundation/v1"
        or contract["kind"] != "experiment_contract"
        or contract["purpose"] not in ("diagnostic", "training")
        or contract["run_id"] != manifest["run_id"]
        or contract["profiles"] != profile_digests()
        or contract["track"] != "strict_generic"
        or contract["initialization"] != "random"
        or contract["objective"] != "sft"
        or contract["constitution_version"] != "2.0.0"
    ):
        raise ValueError("incompatible checkpoint purpose/profiles/adopted contract")
    if (
        type(contract["observed_terms"]) is not int
        or contract["observed_terms"] != 20
        or type(contract["total_terms"]) is not int
        or contract["total_terms"] != 100
        or contract["index_policy"] != "prefix_rebased_zero"
        or contract["enabled_subsystems"] != ["sft"]
    ):
        raise ValueError("incompatible checkpoint horizon/index/subsystems")
    # Diagnostic fixtures explicitly lack production image/source qualification.
    # A training bundle must carry these pins even though qualification itself
    # remains the responsibility of later hardware/provenance gates.
    provenance = contract["run_provenance"]
    provenance_keys = (
        "source_revision",
        "dependency_lock",
        "base_image_digest",
        "final_image_id",
        "environment_report",
        "dataset_manifest",
        "benchmark_manifest",
        "evaluation_protocol",
        "effective_config",
    )
    exact_keys(provenance, provenance_keys, "run provenance")
    from oeis_learn.experiments.models import check_digest
    import re

    for key, value in provenance.items():
        if value is None and contract["purpose"] == "diagnostic":
            continue
        if key == "source_revision":
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ValueError("resolved source revision required")
        else:
            check_digest(value, key)
    for key, field in (
        ("dataset_manifest", "pool_sha256"),
        ("effective_config", "effective_config_sha256"),
    ):
        if provenance[key] != manifest[field]:
            raise ValueError("run provenance disagrees with checkpoint manifest")
    cfg = load_config(ArtifactPath(root, "config.yaml").as_path())
    effective = cfg.to_dict()
    effective.update(
        model=cfg.effective_model_profile(),
        constructors=cfg.model_constructor_kwargs(),
        training=cfg.effective_training_profile(),
    )
    if sidecars["effective-config"] != effective:
        raise ValueError("effective model/config constructor mismatch")
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda") or (dev.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested model device unavailable; no CPU fallback")
    blob = ArtifactPath(root, manifest["blob_path"]).as_path()
    if blob.stat().st_size > 40 << 30:
        raise ValueError("checkpoint exceeds 40 GiB limit")
    # Hash and decode the same open file, then detect in-place mutation. A rename
    # between integrity checking and loading cannot substitute different weights.
    with blob.open("rb") as stream:
        actual = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != manifest["blob_sha256"]:
            raise ValueError("checkpoint blob digest mismatch")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        stream.seek(0)
        if "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest() != actual:
            raise ValueError("checkpoint mutated during loading")
    exact_keys(payload, manifest["payload_state_keys"], "complete checkpoint payload")
    exact_keys(payload["model"], ("encoder", "decoder"), "model state")
    if payload["scheduler"] is not None or payload["scaler"] is not None:
        raise ValueError("inactive scheduler/scaler state must be explicit null")
    counters = payload["counters"]
    fields = (
        "run_id",
        "generation",
        "completed_update",
        "contract_sha256",
        "pool_sha256",
        "codec_sha256",
        "runtime_sha256",
        "effective_config_sha256",
        "ledger_sequence",
        "charged_budget_ns",
    )
    exact_keys(counters, fields, "checkpoint counters")
    if any(
        type(counters[key]) is not type(manifest[key]) or counters[key] != manifest[key]
        for key in fields
    ):
        raise ValueError("checkpoint payload/manifest counters disagree")
    order = payload["data_order"]
    exact_keys(
        order,
        ("epoch", "permutation", "cursor", "next_sample_ids", "accumulation_step"),
        "data order",
    )
    if (
        type(order["epoch"]) is not int
        or order["epoch"] < 0
        or type(order["cursor"]) is not int
        or not isinstance(order["permutation"], list)
        or not 0 <= order["cursor"] <= len(order["permutation"])
        or order["accumulation_step"] != 0
        or type(order["accumulation_step"]) is not int
        or order["next_sample_ids"] != manifest["next_sample_ids"]
        or order["permutation"][order["cursor"] : order["cursor"] + len(order["next_sample_ids"])]
        != order["next_sample_ids"]
    ):
        raise ValueError("incomplete checkpoint data cursor/update boundary")
    from oeis_learn.experiments.models import check_digest

    for identity in order["permutation"]:
        check_digest(identity, "sample identity")
    try:
        random.Random().setstate(payload["python_rng"])
        n = payload["numpy_rng"]
        exact_keys(
            n, ("algorithm", "keys", "position", "has_gauss", "cached_gaussian"), "numpy RNG"
        )
        if (
            n["algorithm"] != "MT19937"
            or len(n["keys"]) != 624
            or any(type(v) is not int or not 0 <= v < 2**32 for v in n["keys"])
        ):
            raise ValueError("invalid numpy RNG state")
        np.random.RandomState().set_state(
            (
                n["algorithm"],
                np.array(n["keys"], dtype=np.uint32),
                n["position"],
                n["has_gauss"],
                n["cached_gaussian"],
            )
        )
        torch.Generator(device="cpu").set_state(payload["torch_cpu_rng"])
        if (
            not isinstance(payload["torch_gpu_rng"], dict)
            or not isinstance(payload["named_rng"], dict)
            or not payload["named_rng"]
        ):
            raise ValueError("explicit GPU and named RNG states required")
        for state in [*payload["torch_gpu_rng"].values(), *payload["named_rng"].values()]:
            if (
                not isinstance(state, torch.Tensor)
                or state.dtype != torch.uint8
                or state.ndim != 1
                or state.numel() == 0
            ):
                raise ValueError("invalid RNG tensor")
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError("invalid/incomplete RNG state") from exc
    constructors = cfg.model_constructor_kwargs()
    # Reconstructing must not consume the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        encoder = TriStreamEncoder(**constructors["encoder"])
        decoder = WatTransformerDecoder(**constructors["decoder"])
    for name, model in [("encoder", encoder), ("decoder", decoder)]:
        state = payload["model"][name]
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError(f"{name} state keys mismatch")
        for key, tensor in state.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected[key].shape
                or tensor.dtype != expected[key].dtype
            ):
                raise ValueError(f"{name} architecture/dtype mismatch at {key}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError("nonfinite model weights")
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(model.state_dict()[key], tensor) for key, tensor in state.items()):
            raise ValueError("checkpoint tensors were silently changed or ignored during loading")
        model.to(dev).eval()
    if not isinstance(payload["optimizer"], dict) or set(payload["optimizer"]) != {
        "state",
        "param_groups",
    }:
        raise ValueError("complete optimizer state required")
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()])
    if manifest["completed_update"] > 0 and not payload["optimizer"]["state"]:
        raise ValueError("completed checkpoint lacks optimizer moment state")
    try:
        optimizer.load_state_dict(payload["optimizer"])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("invalid optimizer state") from exc
    parameters = [*encoder.parameters(), *decoder.parameters()]
    ids = [i for group in payload["optimizer"]["param_groups"] for i in group["params"]]
    if len(ids) != len(set(ids)) or len(ids) != len(parameters):
        raise ValueError("invalid optimizer parameter identities")
    param_map = dict(zip(ids, parameters))
    for identity, state in payload["optimizer"]["state"].items():
        if identity not in param_map or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("incomplete optimizer parameter state")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != param_map[identity].shape
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError("invalid optimizer moment state")
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not torch.isfinite(step).all()
            or not 0 <= step.item() <= manifest["completed_update"]
        ):
            raise ValueError("invalid optimizer update step")
    return FoundationCheckpoint(
        manifest,
        compute_file_sha256(str(manifest_path)),
        contract,
        effective,
        encoder,
        decoder,
        dev,
        payload if include_payload else None,
    )
