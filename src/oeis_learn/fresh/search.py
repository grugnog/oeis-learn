"""Bounded prefix-only selection and literal-neighbour grounding."""

from oeis_learn.decoder.program_codec import encode_body, decode_body
from oeis_learn.evaluation.search import typed_repairs


def grounded_variants(source, visible, limit=8):
    """Single-literal edits from visible values, not hidden targets or family hints.

    This deliberately bounded repair is not a general symbolic solver. Every
    proposal is independently executed; a prefix fit is not a proof.
    """
    words, variants = source.split(), list(typed_repairs(source, limit=limit // 2))
    values = list(dict.fromkeys([visible[0], visible[1] - visible[0], 0, 1, -1]))
    for index, op in enumerate(words[:-1]):
        if op not in {"i64.const", "i256.const"}:
            continue
        for value in values:
            if op == "i64.const" and not -(1 << 63) <= value < (1 << 63):
                continue
            changed = words[:]
            changed[index + 1] = str(value)
            try:
                variant = decode_body(encode_body(" ".join(changed)))
            except ValueError:
                continue
            if variant != source and variant not in variants:
                variants.append(variant)
            if len(variants) >= limit:
                return variants
    return variants


def select(attempts, *, verified):
    from .common import digest

    return next(
        (
            digest(r["source"])
            for r in attempts
            if r.get("complete")
            and r.get("source")
            and (not verified or r.get("prefix_fit") is True)
        ),
        None,
    )


def native_variants(source, visible, client, *, deadline, profile="wat_i256_checked_v1"):
    """Single fitter route. Unsupported retained candidates remain unsupported."""
    from oeis_learn.synthesis.admission import VisiblePrompt

    prompt = VisiblePrompt(
        tuple(str(i) for i in range(len(visible))), tuple(str(v) for v in visible)
    )
    return client.fit(
        source, prompt, profile=profile, policy="eligible-literals/v1", deadline=deadline
    )
