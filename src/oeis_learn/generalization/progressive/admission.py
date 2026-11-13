"""Separate OEIS-derived finite target admission; no mechanical oracle bypass."""

import itertools
import time

from .contracts import digest, record
from .generator import structure_id


class AdmissionRejected(ValueError):
    code = "ADMISSION_REJECTED"


class AdmissionIntegrityError(ValueError):
    code = "ADMISSION_INTEGRITY"


def validate_member(member):
    from oeis_learn.synthesis.contracts import canonical_integer

    if (
        not isinstance(member, dict)
        or set(member) != {"id", "group", "role", "source_snapshot", "indices", "values"}
        or member["role"] != "training"
    ):
        raise ValueError("closed training-only source member required")
    if any(not isinstance(member[k], str) or not member[k] for k in ("id", "group")):
        raise ValueError("source identity/group required")
    from .contracts import _check, schema

    _check(schema("common")["$defs"]["hash"], member["source_snapshot"], schema("common"))
    if len(member["indices"]) != 100 or len(member["values"]) != 100:
        raise ValueError("exact hundred-term admission horizon required")
    for v in member["indices"] + member["values"]:
        if str(canonical_integer(v)) != v:
            raise ValueError("exact canonical source values required")
    indices = list(map(int, member["indices"]))
    if any(b != a + 1 for a, b in itertools.pairwise(indices)):
        raise ValueError("exact consecutive source offsets required")
    return member


def admit(
    member, source, *, client, created_at="2026-11-08T00:00:00Z", purpose="acceptance", budget=None
):
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.synthesis.admission import reference_identity
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.synthesis.reference import evaluate

    validate_member(member)
    if client is None:
        raise ValueError("actual native verification required")
    if budget:
        budget.consume(attempts=1, rejections=1)
    tokens = encode_body(source)
    limits = ResourceProfile()
    deadline = min(time.monotonic() + 4, budget.deadline) if budget else time.monotonic() + 4
    reference = evaluate(source, member["indices"], limits, deadline=deadline)
    native = client.execute(source, member["indices"], deadline=deadline)
    if reference.outcome == native.get("outcome") == "ok" and list(reference.values) != native.get(
        "values"
    ):
        raise AdmissionIntegrityError("native/reference admission disagreement")
    if (
        reference.outcome != "ok"
        or native.get("outcome") != "ok"
        or time.monotonic() >= deadline
        or list(reference.values) != member["values"]
        or native.get("values") != member["values"]
    ):
        raise AdmissionRejected(
            "final-source hundred-term native/reference mismatch or incomplete verification"
        )
    # Pairing reuses the established independent grounding/taint adapter; its
    # generic diagnostic indices start at zero, so shifted source rows retain
    # literal targets rather than silently reinterpret their source offsets.
    target_source = source
    target_tokens = tokens
    disposition = "literal_source"
    if member["indices"] == list(map(str, range(100))):
        from oeis_learn.pipeline_validation.progressive_preparation import verify_draw

        row, reason = verify_draw(
            {
                "source": source,
                "ordinal": 0,
                "draw_identity": digest(source),
                "source_policy": member["source_snapshot"],
                "structure": structure_id(source),
                "detector_limit": "finite admission only",
            },
            client=client,
            budget=budget,
        )
        if row is None:
            raise ValueError("paired target verification failed: " + str(reason))
        target_source = row["target_source"]
        target_tokens = row["tokens"]
        disposition = row["target_disposition"]
    from dataclasses import asdict

    return record(
        "admission",
        {
            "member": digest(member),
            "task": member["id"],
            "group": member["group"],
            "source_snapshot": member["source_snapshot"],
            "source": source,
            "source_hash": digest(source),
            "indices": member["indices"],
            "values_hash": digest(member["values"]),
            "structure": structure_id(source),
            "tokens": target_tokens,
            "target_source": target_source,
            "target_disposition": disposition,
            "native_verified": True,
            "native": client.runtime_identity("bigint-body/v1"),
            "visible_values": member["values"][:20],
            "reference": reference_identity(),
            "limits": digest(asdict(limits)),
            "validity": "finite_100_terms_only",
            "membership": "training",
            "status": "COMMITTED",
        },
        created_at=created_at,
        purpose=purpose,
        track="oeis_derived",
    )
