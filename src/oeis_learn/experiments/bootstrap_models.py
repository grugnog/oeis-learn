"""Explicit encoder comparison; the frozen qualified-model/v1 is unchanged."""

from oeis_learn.fresh.bootstrap_contract import PRIMES, bootstrap_profile


def construct_bootstrap_models(profile, *, device="cpu"):
    from oeis_learn.encoder.integer_bytes import IntegerByteEncoder
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
    from oeis_learn.encoder.hybrid import HybridEncoder
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.decoder.program_codec import FOUNDATION_VOCAB_SIZE, PAD_ID, codec_digest

    selected = None
    if profile.get("version") == "bootstrap-model/v2":
        inputs = profile.get("decoder_inputs", {})
        if (
            not isinstance(inputs, dict)
            or set(inputs) != {"version", "token_scale", "position_scale"}
            or inputs["version"] != "decoder-inputs/v1"
        ):
            raise ValueError("changed bootstrap model profile")
        selected = dict(
            token_scale=inputs["token_scale"],
            position_scale=inputs["position_scale"],
            dropout=profile.get("dropout"),
        )
    if profile != bootstrap_profile(profile.get("encoder"), decoder_inputs=selected):
        raise ValueError("changed bootstrap model profile")
    common = dict(d_model=256, n_heads=4, n_encoder_layers=4, d_ff=1024, dropout=profile["dropout"])
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
            dropout=profile["dropout"],
            max_seq_len=1024,
            pad_idx=PAD_ID,
            codec_sha256=codec_digest(),
            position_style="sinusoidal",
            attention_backend="eager",
            **(
                dict(
                    token_scale=selected["token_scale"],
                    position_scale=selected["position_scale"],
                    input_version="decoder-inputs/v1",
                )
                if selected is not None
                else {}
            ),
        )
    counts = {
        "encoder": sum(p.numel() for p in encoder.parameters()),
        "decoder": sum(p.numel() for p in decoder.parameters()),
    }
    counts["total"] = sum(counts.values())
    return encoder.to(device), decoder.to(device), counts


def construct_bigint_models(*, mask=None, ablation_mode="post_training", device="cpu"):
    """Frozen Phase 2 algorithm recipe; bigint_worker binds resolved profiles."""
    import torch
    from oeis_learn.encoder.hybrid import ChunkHybridEncoder
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder

    encoder = ChunkHybridEncoder(group_mask=mask, ablation_mode=ablation_mode)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(torch.initial_seed() ^ 0xDEC0DE)
        decoder = WatTransformerDecoder(
            vocab_size=codec.FOUNDATION_VOCAB_SIZE,
            codec_sha256=codec.codec_digest(),
            d_model=256,
            n_heads=4,
            n_decoder_layers=4,
            d_ff=1024,
            dropout=0,
            max_seq_len=1024,
            pad_idx=codec.PAD_ID,
            position_style="sinusoidal",
            attention_backend="eager",
            input_version="decoder-inputs/v1",
            token_scale=1,
            position_scale=16,
        )
    return encoder.to(device), decoder.to(device), encoder.ablation_report()


def construct_sizing_models(profile, *, device='cpu'):
    """Fresh hardware fixture; never reads/resizes a trained checkpoint."""
    import torch
    from oeis_learn.pipeline_validation.model_sizes import validate_profile,parameters
    from oeis_learn.encoder.hybrid import ChunkHybridEncoder
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.decoder import bigint_codec as codec
    p=validate_profile(profile)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(p['seed'])
        encoder=ChunkHybridEncoder(model_profile=p)
        torch.random.default_generator.manual_seed(p['seed'] ^ 0xDEC0DE)
        decoder=WatTransformerDecoder(vocab_size=codec.FOUNDATION_VOCAB_SIZE,codec_sha256=codec.codec_digest(),
            d_model=p['width'],n_heads=p['heads'],n_decoder_layers=p['decoder_layers'],d_ff=p['ff'],
            dropout=0,max_seq_len=1024,pad_idx=codec.PAD_ID,position_style='sinusoidal',
            attention_backend='eager',input_version='decoder-inputs/v1',token_scale=1,position_scale=16)
    counts=dict(encoder=sum(v.numel() for v in encoder.parameters()),decoder=sum(v.numel() for v in decoder.parameters()))
    counts.update(total=sum(counts.values()),active=sum(v.numel() for m in (encoder,decoder) for v in m.parameters() if v.requires_grad))
    if counts!=parameters(p):raise RuntimeError('actual sizing parameter count differs from prospective shape')
    return encoder.to(device),decoder.to(device),counts
