"""Four-input training rows; nine-term admission transcripts stay with custodian."""

import copy
import re

from .contracts import digest, seal
from .short_profile import PROFILE

RULE = {
    "version": "progressive-short-mixture/v1",
    "fraction": [1, 4],
    "weighting": "charged-target-tokens",
    "overshoot": "whole-example",
    "library_update": "complete-task-boundary",
}


def prepare(package):
    result = copy.deepcopy(package)
    if "weights" in result or "full_state" in result:
        raise ValueError("short pilot preparation requires a fresh explicit child")
    for key in ("oeis_rules", "oeis_replay", "component_rules", "component_distillation"):
        result.pop(key, None)
    result["task_profile"] = PROFILE
    result["short_rules"] = RULE
    result["shards"] = [
        [seal(dict(r, indices=r["indices"][:4], values=r["values"][:4])) for r in shard]
        for shard in package["shards"]
    ]
    return seal(result)


def admitted_rows(targets, *, purpose):
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.synthesis.admission import reference_identity
    from oeis_learn.synthesis.reference import evaluate

    from .generator import structure_id

    rows = []
    for target in targets:
        if (
            seal(target) != target
            or target["profile"] != PROFILE
            or target["purpose"] != purpose
            or target["native"] is None
            or target["reference"] != reference_identity()
            or target["tokens"] != encode_body(target["source"])
        ):
            raise ValueError("original short admission required")
        observed = evaluate(target["source"], list(map(str, range(9))))
        if (
            observed.outcome != "ok"
            or target["native"].get("outcome") != "ok"
            or target["native"].get("values") != list(observed.values)
            or target["truth"]
            != digest({"indices": list(map(str, range(9))), "values": list(observed.values)})
            or target["visible"]
            != {"indices": ["0", "1", "2", "3"], "values": list(observed.values[:4])}
        ):
            raise ValueError("short admission execution binding differs")
        source = target["source"]
        rows.append(
            seal(
                {
                    "version": "progressive-training-row/v1",
                    "ordinal": len(rows),
                    "draw_identity": target["identity"],
                    "source_policy": digest(RULE),
                    "source": source,
                    "structure": structure_id(source),
                    "behavior": digest(list(observed.values)),
                    "indices": ["0", "1", "2", "3"],
                    "values": list(observed.values[:4]),
                    "target_source": source,
                    "tokens": encode_body(source),
                    "target_disposition": "literal_fallback",
                    "membership": "training",
                    "native_verified": True,
                    "control_events": {},
                    "branch_events": {},
                    "operations": sorted(set(re.findall(r"\(([a-z_.]+)", source))),
                    "detector_limit": "finite nine-term admitted short source",
                }
            )
        )
    if not 1 <= len(rows) <= 64:
        raise ValueError("bounded nonempty admitted short cut required")
    return rows


def mixture(base, source, *, cursor):
    if len(base) != 16:
        raise ValueError("frozen short mixture requires effective batch 16")
    for count in range(1, 17):
        chosen = [source[(cursor + i) % len(source)] for i in range(count)]
        rows = chosen + base[count:]
        if 4 * sum(len(r["tokens"]) for r in chosen) >= sum(len(r["tokens"]) for r in rows):
            return rows
    raise ValueError("short token mixture infeasible")
