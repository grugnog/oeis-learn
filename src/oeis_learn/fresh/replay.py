"""Opt-in, independently admitted shorter targets for training prompts only."""

import time
from .common import digest


def restore(records, rows, archive):
    by_id = {row["program_id"]: row for row in rows}
    targets = {}
    if len(records) > 32:
        raise ValueError("replay cap exceeded")
    for record in records:
        packet = archive.validate_packet(record["packet"], record["key"])
        original = by_id.get(record["program_id"])
        if (
            original is None
            or not packet["entry"]["training_eligible"]
            or packet["entry"]["parent_sha256"] != original["program_id"]
            or packet["entry"]["conditioning_values"] != original["values"]
            or len(packet["tokens"]) >= len(original["tokens"])
        ):
            raise ValueError("replay is not an eligible shorter training target")
        targets[original["program_id"]] = packet["tokens"]
    return targets


def admit(report, rows, archive, runner, existing, *, seconds=20):
    from oeis_learn.decoder.program_codec import decode_body, encode_body
    from oeis_learn.sandbox.pipeline import digest_bytes
    from oeis_learn.experiments.artifacts import compute_canonical_digest
    from .diagnostics import execute_values

    deadline = time.monotonic_ns() + seconds * 10**9
    by_id = {row["program_id"]: row for row in rows}
    retained = {record["program_id"]: record for record in existing}

    class BoundedRunner:
        def run_single(self, source, indices, *, request_id):
            return runner.run_single(source, indices, request_id=request_id, deadline_ns=deadline)

    for candidate in report["splits"]["training"]["attempts"][:8]:
        if time.monotonic_ns() >= deadline or len(retained) >= 32:
            break
        row = by_id[candidate["program_id"]]
        if (
            not candidate["matched"]
            or not candidate["complete"]
            or len(encode_body(candidate["source"])) >= len(row["tokens"])
        ):
            continue
        source = candidate["source"]
        if not archive.holdouts.decision(source, list(map(int, row["values"])))[0]:
            continue
        expected, failure = execute_values(
            runner,
            decode_body(row["tokens"]),
            range(100),
            digest([row["program_id"], "replay-oracle"]),
            deadline,
        )
        if failure:
            continue
        evidence = runner.run_single(
            source,
            range(20),
            request_id=digest([row["program_id"], source, "replay-prefix"]),
            deadline_ns=deadline,
        ).to_dict()
        provenance = dict(
            version="search-hypothesis/v1",
            source_sha256=digest_bytes(source.encode()),
            body_tokens=encode_body(source),
            visible_prompt_sha256=compute_canonical_digest(row["values"]),
            execution=evidence,
            parent=None,
            **candidate["model_provenance"],
        )
        provenance["identity_sha256"] = compute_canonical_digest(provenance)
        key = archive.admit(
            source,
            list(map(int, row["values"])),
            list(map(int, expected)),
            origin="training_prefix_hypothesis",
            provenance={"search": provenance},
            runner=BoundedRunner(),
            request_id=digest([row["program_id"], source, "replay-admit"]),
            model_sha256=compute_canonical_digest(
                [provenance["encoder_sha256"], provenance["decoder_sha256"]]
            ),
            parent_sha256=row["program_id"],
        )
        packet = archive.get(key)
        retained[row["program_id"]] = dict(program_id=row["program_id"], packet=packet, key=key)
    result = [retained[key] for key in sorted(retained)]
    restore(result, rows, archive)
    return result
