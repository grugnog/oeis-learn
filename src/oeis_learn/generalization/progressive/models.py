"""Closed research shapes over the retained input/model recipe, fresh weights only.

Hardware-only m8/m32/m100 records remain a separate qualification namespace.
This module resolves on stdlib hosts; model libraries are imported at construction.
"""

import copy
import math

from oeis_learn.pipeline_validation.identity import seal

from .contracts import digest
from .metrics import _seed

SHAPES = ("reference", "smaller_width", "shallower", "larger")
VERSION = "progressive-research-model/v1"


def resolve_profile(recipe, *, shape="reference", seed=20261016, dropout=0):
    from oeis_learn.decoder.bigint_codec import codec_digest
    from oeis_learn.fresh.bigint_config import resolve

    if resolve(recipe) != recipe:
        raise ValueError("fully resolved retained model recipe required")
    if shape not in SHAPES or type(dropout) not in (int, float) or dropout not in (0, 0.1, 0.2):
        raise ValueError("registered research shape/dropout required")
    _seed(seed)
    m = copy.deepcopy(recipe["model"])
    m["seed"] = seed
    m["dropout"] = float(dropout)
    d, h = m["d_model"], m["heads"]
    if shape == "smaller_width":
        m["d_model"] = max(h, d // 2 // h * h)
    elif shape == "shallower":
        m["encoder_layers"] = max(1, m["encoder_layers"] // 2)
        m["decoder_layers"] = max(1, m["decoder_layers"] // 2)
    elif shape == "larger":
        m["d_model"] = math.ceil(3 * d / (2 * h)) * h
        m["encoder_layers"] += 1
        m["decoder_layers"] += 1
    m["ff"] = 4 * m["d_model"]
    if (
        m["precision"] != "float32"
        or m["attention"] != "eager"
        or m["d_model"] % h
        or m["d_model"] % 2
        or not 2 <= m["d_model"] <= 2048
        or not 1 <= m["encoder_layers"] <= 12
        or not 1 <= m["decoder_layers"] <= 12
    ):
        raise ValueError("valid FP32 head-divisible research dimensions required")
    return seal(
        {
            "version": VERSION,
            "shape": shape,
            "base_recipe": copy.deepcopy(recipe),
            "model": m,
            "inputs": copy.deepcopy(recipe["inputs"]),
            "encoder_positions": 128,
            "decoder_positions": 1024,
            "codec": codec_digest(),
            "initialization": "fresh_random",
            "qualification": "PENDING",
        }
    )


def validate_profile(profile):
    fields = {
        "version",
        "identity",
        "shape",
        "base_recipe",
        "model",
        "inputs",
        "encoder_positions",
        "decoder_positions",
        "codec",
        "initialization",
        "qualification",
    }
    if (
        not isinstance(profile, dict)
        or set(profile) != fields
        or profile["version"] != VERSION
        or seal(profile) != profile
    ):
        raise ValueError("closed sealed research model profile required")
    m = profile["model"]
    if not isinstance(m, dict) or "seed" not in m or "dropout" not in m:
        raise ValueError("explicit research model seed/dropout required")
    expected = resolve_profile(
        profile["base_recipe"], shape=profile["shape"], seed=m["seed"], dropout=m["dropout"]
    )
    if expected != profile:
        raise ValueError("research dimensions/positions/input policy changed")
    return copy.deepcopy(profile)


def construct(profile, *, device):
    p = validate_profile(profile)
    import random

    import numpy as np
    import torch

    if device not in {"cpu", "cuda"}:
        raise ValueError("explicit research device required")
    if device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("actual HIP required; CPU fallback prohibited")
    if device == "cuda":
        from oeis_learn.pipeline_validation.generalization_execution import verify_execution
        verify_execution()
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("FP32 default required for seeded construction")
    m = p["model"]
    seed = m["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.encoder.ablation import InputMask
    from oeis_learn.encoder.hybrid import ChunkHybridEncoder
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder

    inputs = p["inputs"]
    mask = InputMask(**inputs["mask"])
    if inputs["profile"] == "hybrid-chunks/v1":
        encoder = ChunkHybridEncoder(
            research_profile=p,
            dropout=m["dropout"],
            group_mask=mask,
            ablation_mode=inputs["ablation_mode"],
        )
    else:
        if inputs["profile"] != "tri-stream/v1" or mask.bits() != (True, True, True, False):
            raise ValueError("compatible retained tri-stream input profile required")
        encoder = TriStreamEncoder(
            d_model=m["d_model"],
            n_heads=m["heads"],
            n_encoder_layers=m["encoder_layers"],
            d_ff=m["ff"],
            dropout=m["dropout"],
            max_seq_len=p["encoder_positions"],
            primes=inputs["primes"],
            base_moduli=inputs["primes"],
            max_valuation=inputs["max_valuation"],
            input_profile="hybrid-chunks/v1",
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed ^ 0xDEC0DE)
        decoder = WatTransformerDecoder(
            vocab_size=codec.FOUNDATION_VOCAB_SIZE,
            codec_sha256=p["codec"],
            d_model=m["d_model"],
            n_heads=m["heads"],
            n_decoder_layers=m["decoder_layers"],
            d_ff=m["ff"],
            dropout=m["dropout"],
            max_seq_len=p["decoder_positions"],
            pad_idx=codec.PAD_ID,
            position_style="sinusoidal",
            attention_backend=m["attention"],
            input_version="decoder-inputs/v1",
            token_scale=m["token_scale"],
            position_scale=m["position_scale"],
        )
    encoder, decoder = encoder.to(device), decoder.to(device)
    params = [*encoder.parameters(), *decoder.parameters()]
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in params):
        raise FloatingPointError("nonfinite or non-FP32 initialized research parameters")
    counts = {
        "encoder_parameters": sum(v.numel() for v in encoder.parameters()),
        "decoder_parameters": sum(v.numel() for v in decoder.parameters()),
        "total_parameters": sum(v.numel() for v in params),
        "active_parameters": sum(v.numel() for v in params if v.requires_grad),
        "device": str(next(encoder.parameters()).device),
        "precision": "float32",
        "effective_dropout": m["dropout"],
        "model_profile": p["identity"],
        "input_sha256": digest(inputs),
        "decoder_sha256": p["codec"],
        "qualified": False,
    }
    return encoder, decoder, counts
