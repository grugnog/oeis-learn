"""Frozen bounded pilot variants. Profiles are distinct from historical qualification."""

from oeis_learn.experiments.artifacts import compute_canonical_digest as digest


def pilot_profile(question=None, arm="baseline", *, binding_scope_cause=False):
    if question not in (
        None,
        "encoder",
        "self_training",
        "grpo",
        "search",
        "position",
        "capacity",
    ) or arm not in ("baseline", "challenger"):
        raise ValueError("one declared pilot question/arm required")
    large = question == "capacity" and arm == "challenger"
    profile = dict(
        version="bounded-pilot-model/v1",
        encoder="tri_stream" if question == "encoder" and arm == "baseline" else "integer_bytes",
        d_model=384 if large else 128,
        heads=6 if large else 4,
        encoder_layers=4 if large else 2,
        decoder_layers=6 if large else 2,
        d_ff=2048 if large else 512,
        dropout=0.0,
        context_tokens=1024,
        position_style=("prefix_ancestor" if binding_scope_cause else "rope")
        if question == "position" and arm == "challenger"
        else "sinusoidal",
        attention_backend="eager",
        objective="grpo" if question == "grpo" and arm == "challenger" else "sft",
        self_training=question == "self_training" and arm == "challenger",
        portfolio="portfolio32" if question == "search" and arm == "challenger" else "sampling32",
    )
    profile["identity_sha256"] = digest(profile)
    return profile


def construct_pilot_models(profile, device="cpu"):
    from oeis_learn.encoder.integer_bytes import IntegerByteEncoder
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.decoder.program_codec import FOUNDATION_VOCAB_SIZE, PAD_ID, codec_digest

    if profile.get("identity_sha256") != digest(profile, "identity_sha256"):
        raise ValueError("pilot profile identity mismatch")
    if profile not in [
        pilot_profile(q, a, binding_scope_cause=c)
        for q in (None, "encoder", "self_training", "grpo", "search", "position", "capacity")
        for a in ("baseline", "challenger")
        for c in (False, True)
    ]:
        raise ValueError("unknown pilot variant")
    kind = IntegerByteEncoder if profile["encoder"] == "integer_bytes" else TriStreamEncoder
    encoder = kind(
        d_model=profile["d_model"],
        n_heads=profile["heads"],
        n_encoder_layers=profile["encoder_layers"],
        d_ff=profile["d_ff"],
        dropout=profile["dropout"],
    ).to(device)
    decoder = WatTransformerDecoder(
        vocab_size=FOUNDATION_VOCAB_SIZE,
        d_model=profile["d_model"],
        n_heads=profile["heads"],
        n_decoder_layers=profile["decoder_layers"],
        d_ff=profile["d_ff"],
        dropout=profile["dropout"],
        max_seq_len=1024,
        pad_idx=PAD_ID,
        codec_sha256=codec_digest(),
        position_style=profile["position_style"],
        attention_backend=profile["attention_backend"],
    ).to(device)
    return encoder, decoder
