"""Standard-library model contract shared by host and image constructors."""

import math

from .common import digest

PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)


def decoder_controls(*, token_scale=1.0, position_scale=1.0, dropout=0.1):
    """Closed, normalized controls for opt-in bootstrap-model/v2 experiments."""
    for name, value, low, high in (
        ("token_scale", token_scale, 0.0, 256.0),
        ("position_scale", position_scale, 0.0, 256.0),
        ("dropout", dropout, 0.0, 1.0),
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not low <= value <= high
            or (name == "token_scale" and value == 0)
            or (name == "dropout" and value == 1)
        ):
            raise ValueError(f"invalid decoder control: {name}")
    return dict(
        token_scale=float(token_scale), position_scale=float(position_scale), dropout=float(dropout)
    )


def bootstrap_profile(encoder, *, decoder_inputs=None):
    if encoder not in {"tri_stream", "hybrid", "integer_bytes"}:
        raise ValueError("explicit tri_stream, hybrid or integer_bytes encoder required")
    profile = dict(
        version="bootstrap-model/v1",
        encoder=encoder,
        d_model=256,
        heads=4,
        encoder_layers=4,
        decoder_layers=4,
        d_ff=1024,
        dropout=0.1,
        context_tokens=1024,
        position_style="sinusoidal",
        attention_backend="eager",
        objective="sft",
        primes=list(PRIMES),
        base_moduli=list(PRIMES),
        max_valuation=32,
        use_film=True,
        enable_summary_tokens=False,
        byte_width=4,
        decoder_initialization="independent-seed/v1",
    )
    if decoder_inputs is not None:
        if not isinstance(decoder_inputs, dict) or set(decoder_inputs) != {
            "token_scale",
            "position_scale",
            "dropout",
        }:
            raise ValueError("closed decoder controls required")
        selected = decoder_controls(**decoder_inputs)
        profile.update(
            version="bootstrap-model/v2",
            dropout=selected.pop("dropout"),
            decoder_inputs=dict(version="decoder-inputs/v1", **selected),
        )
    profile["identity_sha256"] = digest(profile)
    return profile


def bigint_profile(recipe):
    """Standard-library profile binding shared by orchestration and model roles."""
    from copy import deepcopy

    if recipe.get("version") not in {"bigint-bootstrap/v1","bigint-bootstrap/v2"} or tuple(
        recipe["execution"][k] for k in ("language", "codec", "runtime")
    ) != ("bigint-body/v1", "bigint-codec/v1", "bigint-wasm/v1"):
        raise ValueError("explicit bigint language/codec/runtime profile required")
    if recipe["inputs"]["profile"] not in {"hybrid-chunks/v1", "tri-stream/v1"}:
        raise ValueError("unsupported bigint input profile")
    profile = dict(
        version="bigint-model/v2" if recipe['version']=='bigint-bootstrap/v2' else "bigint-model/v1",
        architecture=deepcopy(recipe["model"]),
        features=deepcopy(recipe["inputs"]),
        execution=deepcopy(recipe["execution"]),
    )
    if recipe['version']=='bigint-bootstrap/v2':
        from oeis_learn.pipeline_validation.model_sizes import PROFILES,EXECUTION
        if recipe.get('model_profile')!=PROFILES['m100']:raise ValueError('registered sized profile required')
        profile.update(model_profile=recipe['model_profile'],numerics=EXECUTION)
    profile["identity_sha256"] = digest(profile)
    return profile
