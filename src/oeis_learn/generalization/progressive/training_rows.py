"""Shared closed learner-row validation for preparation and installed training."""
from .contracts import seal


def validate_row(row, *, native_verified, short=False):
    from oeis_learn.decoder.bigint_codec import encode_body

    fields = {
        "version",
        "identity",
        "ordinal",
        "draw_identity",
        "source_policy",
        "source",
        "structure",
        "behavior",
        "indices",
        "values",
        "target_source",
        "tokens",
        "target_disposition",
        "membership",
        "native_verified",
        "control_events",
        "branch_events",
        "operations",
        "detector_limit",
    }
    if set(row) != fields:
        raise ValueError("closed training row; evaluation metadata excluded")
    if (
        seal(row) != row
        or row.get("version") != "progressive-training-row/v1"
        or row.get("membership") != "training"
    ):
        raise ValueError("sealed training rows only")
    if native_verified and row.get("native_verified") is not True:
        raise ValueError("native verification missing")
    if (
        len(row["values"]) != (4 if short else 100)
        or len(row["indices"]) != (4 if short else 100)
        or encode_body(row["target_source"]) != row["tokens"]
    ):
        raise ValueError("complete target/codec binding required")
    return row


def first_divergence(positive, negative, positives):
    """The negative logit must be available at the positive's teacher-forced context."""
    for i, (good, bad) in enumerate(zip(positive, negative)):
        if good != bad:
            if any(seq[:i + 1] == negative[:i + 1] for seq in positives):
                return None
            return dict(position=i, positive=good, negative=bad)
    return None


def verified_training_pairs(shards, *, excluded=()):
    """Mine only the registered strict training stream for same-index, prefix-matching alternatives."""
    from collections import defaultdict
    import re
    from oeis_learn.decoder.bigint_codec import decode_body
    from oeis_learn.generalization.targets import ground
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.reference_parser import parse
    from .shared_support import validated_rows

    rows = validated_rows(shards, native_verified=True)
    groups = defaultdict(list)
    for row in rows.values():
        groups[tuple(row["indices"]), tuple(row["values"][:20])].append(row)
    pairs = []
    for group in groups.values():
        if len({tuple(r["values"][20:]) for r in group}) < 2:
            continue
        verified = []
        for row in sorted(group, key=lambda r: r["draw_identity"]):
            decoded = decode_body(row["tokens"])
            target = parse(decoded, allow_parameters=True)
            if target != parse(row["target_source"], allow_parameters=True):
                raise ValueError("decoded training target differs")
            if "param.get" in row["target_source"]:
                # Preparation numbers literal occurrences in source order.
                constants = re.findall(r"\(int (-?(?:0|[1-9][0-9]*))\)", row["source"])
                assignment = {f"p{i}": v for i, v in enumerate(constants)}
                decoded = ground(decoded, assignment)
            if parse(decoded) != parse(row["source"]):
                raise ValueError("training target grounding differs from the verified program")
            result = evaluate(decoded, row["indices"])
            if result.outcome != "ok":
                continue
            if list(result.values) != row["values"]:
                raise ValueError("independent training continuation differs")
            verified.append(row)
        for pos in verified:
            if pos["draw_identity"] in excluded:
                continue
            same = [r["tokens"] for r in group if r["values"] == pos["values"]]
            for neg in verified:
                if neg["draw_identity"] in excluded or neg["values"] == pos["values"]:
                    continue
                label = first_divergence(pos["tokens"], neg["tokens"], same)
                if label is not None:
                    pairs.append(dict(positive_row=pos["draw_identity"], negative_row=neg["draw_identity"],
                                      family=pos["structure"], label=label))
                    break
    return dict(eligible_rows=len(pairs), eligible_families=len({p["family"] for p in pairs}),
                positions=len({p["label"]["position"] for p in pairs}), pairs=pairs)

def pair_seeds(registration, ordinal):
    from .contracts import digest
    import re
    if (not isinstance(registration, str) or not re.fullmatch("sha256:[0-9a-f]{64}", registration)
            or type(ordinal) is not int or ordinal not in (0, 1)):
        raise ValueError("registered generator pair and parent required")
    return {name: int(digest([registration, name, ordinal])[7:15], 16)
            for name in ("initialization", "training_families")}


def matched_stream(shards, layout, registration, ordinal):
    """Commit one outcome-independent cyclic order and its actual verified labels."""
    from .contracts import digest
    from .shared_support import validated_rows

    seeds = pair_seeds(registration, ordinal)
    rows = validated_rows(shards, native_verified=True)
    excluded = {probe["id"] for probe in layout["probes"].values()}
    if not excluded <= rows.keys():
        raise ValueError("missing held-out training probe")
    order = sorted(rows.keys() - excluded, key=lambda key: (digest([seeds["training_families"], key]), key))
    if not order:
        raise ValueError("empty matched training stream")
    support = verified_training_pairs(shards, excluded=excluded)
    return seal(dict(version="round028-matched-stream/v1", registration=registration,
                     pair=ordinal, seeds=seeds,
                     sources=sorted({r["source_policy"] for r in rows.values()}),
                     row_identities=digest([rows[key]["identity"] for key in order]),
                     order=order, **support))


def matched_batch(stream, cursor, count):
    """Address a full update by its committed example cursor on either arm."""
    if type(cursor) is not int or cursor < 0 or type(count) is not int or count < 1:
        raise ValueError("complete matched batch coordinates required")
    order = stream["order"]
    return [order[(cursor + offset) % len(order)] for offset in range(count)]
