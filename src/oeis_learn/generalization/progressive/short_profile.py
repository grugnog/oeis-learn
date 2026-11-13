"""Dedicated four-visible/nine-total finite task; authoritative terms never padded."""

from oeis_learn.synthesis.contracts import canonical_integer

PROFILE = "combinatorial-short/v1"
VISIBLE = 4
TOTAL = 9


def validate_terms(row):
    if (
        set(row) != {"indices", "values"}
        or row["indices"] != list(map(str, range(TOTAL)))
        or type(row["values"]) is not list
        or len(row["values"]) != TOTAL
    ):
        raise ValueError("exact n=0..8 short task required")
    for v in row["values"]:
        canonical_integer(v)
    return row


def visible(row):
    validate_terms(row)
    return dict(indices=row["indices"][:VISIBLE], values=row["values"][:VISIBLE])


def collate(rows):
    """One exact-length bucket, so input padding is unnecessary and prohibited."""
    if not 1 <= len(rows) <= 64:
        raise ValueError("bounded four-input bucket required")
    for r in rows:
        if (
            set(r) != {"indices", "values"}
            or r["indices"] != ["0", "1", "2", "3"]
            or len(r["values"]) != 4
        ):
            raise ValueError("four visible terms required")
        for v in r["values"]:
            canonical_integer(v)
    return dict(values=[list(r["values"]) for r in rows], position_mask=[[True] * 4 for _ in rows])


def admit(source, truth, *, member, client, purpose="research"):
    """Only a generated concrete program can become a learner target."""
    import time
    from .contracts import digest, seal
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.admission import reference_identity
    from oeis_learn.decoder.bigint_codec import encode_body

    validate_terms(truth)
    if purpose not in {"acceptance", "research"} or (purpose == "research" and client is None):
        raise ValueError("native source admission required")
    deadline = time.monotonic() + 4
    checked = evaluate(source, truth["indices"], deadline=deadline)
    if checked.outcome != "ok" or list(checked.values) != truth["values"]:
        return None
    native = client.execute(source, truth["indices"], deadline=deadline) if client else None
    if client and (native.get("outcome") != "ok" or native.get("values") != truth["values"]):
        raise ValueError("short native/reference disagreement")
    return seal(
        dict(
            version="progressive-short-target/v1",
            profile=PROFILE,
            member=member,
            source=source,
            tokens=encode_body(source),
            visible=visible(truth),
            truth=digest(truth),
            native=native,
            reference=reference_identity(),
            purpose=purpose,
            track="external_assisted",
        )
    )


def training_batch(targets, *, device):
    import torch
    from .contracts import seal
    from oeis_learn.decoder.bigint_codec import BOS_ID, PAD_ID, encode_body

    if not 1 <= len(targets) <= 64:
        raise ValueError("bounded short target bucket")
    for t in targets:
        if (
            set(t)
            != {
                "version",
                "identity",
                "profile",
                "member",
                "source",
                "tokens",
                "visible",
                "truth",
                "native",
                "reference",
                "purpose",
                "track",
            }
            or seal(t) != t
            or t["version"] != "progressive-short-target/v1"
            or t["profile"] != PROFILE
            or t["track"] != "external_assisted"
            or t["tokens"] != encode_body(t["source"])
            or t["native"] is None
        ):
            raise ValueError("original native-verified short target required")
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.admission import reference_identity
    from .contracts import digest

    for t in targets:
        observed = evaluate(t["source"], list(map(str, range(9))))
        if (
            t["reference"] != reference_identity()
            or observed.outcome != "ok"
            or t["native"].get("outcome") != "ok"
            or t["native"].get("values") != list(observed.values)
            or t["truth"]
            != digest(dict(indices=list(map(str, range(9))), values=list(observed.values)))
            or t["visible"] != dict(indices=["0", "1", "2", "3"], values=list(observed.values[:4]))
        ):
            raise ValueError("short target execution/visibility binding differs")
    bucket = collate([t["visible"] for t in targets])
    sequences = [[int(v) for v in row] for row in bucket["values"]]
    ids = [[BOS_ID] + t["tokens"] for t in targets]
    tokens = torch.full((len(ids), max(map(len, ids))), PAD_ID, dtype=torch.long, device=device)
    for i, row in enumerate(ids):
        tokens[i, : len(row)] = torch.tensor(row, device=device)
    return sequences, tokens
