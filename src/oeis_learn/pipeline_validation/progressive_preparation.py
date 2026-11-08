"""Outcome-blind assembly adapters; no hidden model scoring or guessed inputs."""


def census(*args, **kwargs):
    from oeis_learn.generalization.progressive.cohorts import census as count

    return count(*args, **kwargs)


def publish_cohort(*args, **kwargs):
    from oeis_learn.generalization.progressive.cohorts import publish

    return publish(*args, **kwargs)


def verify_draw(value, *, client=None):
    """Custodian verification; reference-only diagnostic rows cannot train."""
    import re
    import time
    from collections import Counter

    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.generalization.targets import _control_taint, paired_learning_view
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.reference_parser import parse

    indices = list(map(str, range(100)))
    events = Counter()
    branches = Counter()
    deadline = time.monotonic() + 4
    result = evaluate(
        value["source"],
        indices,
        deadline=deadline,
        observer=lambda kind, v: events.update([f"{kind}:{v}"]),
        branch_observer=lambda path, v: branches.update([f"{path}:{v}"]),
    )
    if result.outcome != "ok":
        return None, "unsafe_reference:" + str(result.reason)
    if client is not None:
        native = client.execute(value["source"], indices, deadline=deadline)
        if native.get("outcome") != "ok" or time.monotonic() >= deadline:
            return None, "native_unresolved"
        if tuple(native.get("values", [])) != result.values:
            raise RuntimeError("native/reference disagreement in generated draw")
    assignment = {}

    def parameterize(match):
        name = f"p{len(assignment)}"
        assignment[name] = match.group(1)
        return "(param.get " + name + ")"

    parameterized = re.sub(r"\(int (-?(?:0|[1-9][0-9]*))\)", parameterize, value["source"])
    try:
        taint = (
            _control_taint(parse(parameterized, allow_parameters=True))
            if len(assignment) <= 8
            else None
        )
        view = paired_learning_view(
            value["source"],
            parameterized,
            assignment,
            indices=indices,
            sharing="distinct-occurrences",
        )
        if taint and view["target_source"] != value["source"]:
            raise RuntimeError("control-tainted parameter reached learner")
        if (
            view["final_source"] != value["source"]
            and evaluate(view["final_source"], indices).values != result.values
        ):
            raise RuntimeError("paired target grounding changed values")
    except ValueError as exc:
        return None, "target_not_encodable:" + type(exc).__name__
    row = seal(
        {
            "version": "progressive-training-row/v1",
            "ordinal": value["ordinal"],
            "draw_identity": value["draw_identity"],
            "source_policy": value["source_policy"],
            "source": value["source"],
            "structure": value["structure"],
            "behavior": digest(list(result.values)),
            "indices": indices,
            "values": list(result.values),
            "target_source": view["target_source"],
            "tokens": view["target_tokens"],
            "target_disposition": view["disposition"],
            "membership": "training",
            "native_verified": client is not None,
            "control_events": dict(events),
            "branch_events": dict(branches),
            "operations": sorted(set(re.findall(r"\(([a-z_.]+)", value["source"]))),
            "detector_limit": value["detector_limit"],
        }
    )
    return row, None


def prepare_stream(root, policy, *, client, shards=1, purpose="acceptance"):
    """Bounded native custodian producer; every rejected draw remains charged."""
    from oeis_learn.generalization.progressive.stream import StreamStore

    if client is None or type(shards) is not int or not 1 <= shards <= 64:
        raise ValueError("native client and finite shard reservation required")
    store = StreamStore(root, policy, purpose=purpose)
    result = []
    for _ in range(shards):
        while True:
            try:
                store.step(client=client)
            except TimeoutError:
                break
        state = store.state()
        if state["pending"] is not None:
            break
        if state["rows"] or state["rejected"]:
            result.append(store.seal_shard())
        if state["draws"] >= policy["max_draws"] or state["work_units"] >= policy["max_work_units"]:
            break
    return result
