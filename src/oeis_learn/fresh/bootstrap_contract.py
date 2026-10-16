"""Standard-library model contract shared by host and image constructors."""

from .common import digest

PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)


def bootstrap_profile(encoder):
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
    profile["identity_sha256"] = digest(profile)
    return profile
