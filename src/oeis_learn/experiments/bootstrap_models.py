"""Explicit encoder comparison; the frozen qualified-model/v1 is unchanged."""

from oeis_learn.fresh.bootstrap_contract import PRIMES, bootstrap_profile


def construct_bootstrap_models(profile, *, device="cpu"):
    from oeis_learn.encoder.integer_bytes import IntegerByteEncoder
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
    from oeis_learn.encoder.hybrid import HybridEncoder
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.decoder.program_codec import FOUNDATION_VOCAB_SIZE, PAD_ID, codec_digest

    if profile != bootstrap_profile(profile.get("encoder")):
        raise ValueError("changed bootstrap model profile")
    common = dict(d_model=256, n_heads=4, n_encoder_layers=4, d_ff=1024, dropout=0.1)
    if profile["encoder"] == "integer_bytes":
        encoder = IntegerByteEncoder(**common, byte_width=4)
    else:
        cls = HybridEncoder if profile["encoder"] == "hybrid" else TriStreamEncoder
        encoder = cls(
            **common,
            primes=PRIMES,
            base_moduli=PRIMES,
            max_valuation=32,
            use_film=True,
            enable_summary_tokens=False,
        )
    import torch

    # Encoder parameter counts must not shift decoder initial weights between arms.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(torch.initial_seed() ^ 0xDEC0DE)
        decoder = WatTransformerDecoder(
            vocab_size=FOUNDATION_VOCAB_SIZE,
            d_model=256,
            n_heads=4,
            n_decoder_layers=4,
            d_ff=1024,
            dropout=0.1,
            max_seq_len=1024,
            pad_idx=PAD_ID,
            codec_sha256=codec_digest(),
            position_style="sinusoidal",
            attention_backend="eager",
        )
    counts = {
        "encoder": sum(p.numel() for p in encoder.parameters()),
        "decoder": sum(p.numel() for p in decoder.parameters()),
    }
    counts["total"] = sum(counts.values())
    return encoder.to(device), decoder.to(device), counts
