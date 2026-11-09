"""Verified read-only full-state import closure and original cost provenance."""

import json
from pathlib import Path
from .contracts import load, verify_refs, reference, validate, seal
from oeis_learn.pipeline_validation.identity import confined


def inspect(root, spec):
    if (
        root is None
        or type(spec) is not dict
        or set(spec) != {"artifact", "state_root", "accounting"}
    ):
        raise ValueError("closed full-state source closure required")
    refs = [spec["artifact"], spec["accounting"]]
    raw, costs = verify_refs(root, refs, max_bytes=64 << 20)
    state = validate("state", json.loads(raw))
    costs = json.loads(costs)
    p = state["payload"]
    if (
        p["kind"] != "full_state"
        or seal(costs) != costs
        or costs.get("version") != "progressive-final-accounting/v1"
        or costs["task"] != p["task"]
        or costs["pending"]
        or costs.get("checkpoint") != state["identity"]
    ):
        raise ValueError("original committed full-state and final accounting required")
    base = confined(root, spec["state_root"])
    full = json.loads(verify_refs(base, [p["full_state"]], max_bytes=64 << 20)[0])
    from oeis_learn.generalization.state import restore

    restore(full, contract=p["compatibility"], cap=full["ledger"]["cap"], root=base)
    if full["ledger"]["pending"] is not None or full["ledger"]["updates"] != p["model_update"]:
        raise ValueError("complete matching imported model boundary required")
    from oeis_learn.generalization.contracts import artifact_refs

    for ref in [p["full_state"], *artifact_refs(full["states"])]:
        verify_refs(base, [ref], max_bytes=64 << 30)
        refs.append(reference(root, confined(base, ref["path"])))
    milestone = costs["milestone"]
    if (
        milestone["tokens"] < full["ledger"]["consumed"]
        or milestone["total_cost_us"] < 0
        or type(costs.get("useful_tokens")) is not int
        or not 0 <= costs["useful_tokens"] <= full["ledger"]["consumed"]
    ):
        raise ValueError("import accounting rolls back checkpoint exposure/cost")
    return state, costs, base, full, refs


def restore(root, spec, task):
    state, costs, base, full, refs = inspect(root, spec)
    p = state["payload"]
    init = task["payload"]["initializer"]
    if (
        init["mode"] != "full_state"
        or init["artifact"] != spec["artifact"]
        or init["compatibility"] != p["compatibility"]
        or state["purpose"] != task["purpose"]
        or state["track"] != task["track"]
    ):
        raise ValueError("incompatible imported full-state lineage")
    from oeis_learn.fresh.bigint_state import ProgressiveCheckpointStore

    store = ProgressiveCheckpointStore(
        base,
        compatibility=p["compatibility"],
        campaign=p["campaign"],
        task=p["task"],
        token_cap=full["ledger"]["cap"],
    )
    return state, costs, store.load(state)
