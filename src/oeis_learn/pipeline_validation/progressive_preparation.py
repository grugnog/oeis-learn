"""Outcome-blind assembly adapters; no hidden model scoring or guessed inputs."""


def census(*args, **kwargs):
    from oeis_learn.generalization.progressive.cohorts import census as count

    return count(*args, **kwargs)


def publish_cohort(*args, **kwargs):
    from oeis_learn.generalization.progressive.cohorts import publish

    return publish(*args, **kwargs)


def verify_draw(value, *, client=None, budget=None):
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
    deadline = min(time.monotonic() + 4, budget.deadline) if budget else time.monotonic() + 4
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


def prepare_stream(root, policy, *, client, shards=1, purpose="acceptance", budget=None, return_receipt=False):
    """Bounded native custodian producer; every rejected draw remains charged."""
    from oeis_learn.generalization.progressive.stream import StreamStore

    if client is None or type(shards) is not int or not 1 <= shards <= 64:
        raise ValueError("native client and finite shard reservation required")
    import time
    began=time.monotonic_ns()
    store = StreamStore(root, policy, purpose=purpose)
    result = []
    for _ in range(shards):
        while True:
            try:
                store.step(client=client, budget=budget)
            except TimeoutError:
                break
        state = store.state()
        if state["pending"] is not None:
            break
        if state["rows"] or state["rejected"]:
            result.append(store.seal_shard())
        if state["draws"] >= policy["max_draws"] or state["work_units"] >= policy["max_work_units"]:
            break
    if return_receipt:
        from oeis_learn.generalization.progressive.contracts import seal,atomic_write,reference
        import uuid
        from pathlib import Path
        receipt=seal(dict(version='progressive-preparation-cost/v1',shards=[s['identity'] for s in result],
            elapsed_us=(time.monotonic_ns()-began)//1000,scope='observed producer invocation; host startup/cleanup accounted separately'))
        path=Path(root)/('preparation-'+uuid.uuid4().hex+'.json')
        atomic_write(path,receipt)
        return result,reference(root,path)
    return result


def prepare_oeis_pool(root, snapshot_ref, members, *, exposure, output, count=512):
    """Custodian projection from an explicit immutable bulk indexed source cut.

    The existing acquisition producer owns downloading/parsing. Missing exact
    offsets are deficits; stripped terms alone never imply an offset of zero.
    """
    import json

    from oeis_learn.generalization.progressive.admission import validate_member
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, verify_refs
    from oeis_learn.generalization.progressive.exposure import exposed
    from oeis_learn.pipeline_validation.identity import confined, seal

    if type(count) is not int or not 1 <= count <= 512:
        raise ValueError("finite frozen training pool required")
    payload = json.loads(verify_refs(root, [snapshot_ref], max_bytes=512 << 20)[0])
    if (
        not isinstance(payload, dict)
        or set(payload) != {"version", "release", "records"}
        or payload["version"] != "indexed-source/v1"
    ):
        raise ValueError("existing indexed-source/v1 bulk producer required")
    if not isinstance(members, list) or len(members) != count or len(set(members)) != count:
        raise ValueError("outcome-blind exact member roster required")
    from oeis_learn.fresh.bulk import indexed_terms
    from oeis_learn.pipeline_validation.identity import file_hash

    base = confined(root, snapshot_ref["path"]).parent
    records = {}
    for entry in payload["records"]:
        key = entry["record_id"]
        if key in records:
            raise ValueError("duplicate bulk member")
        if key not in members:
            continue
        for field in ("terms", "metadata"):
            if set(entry[field]) != {"path", "sha256"}:
                raise ValueError("exact original source reference required")
            if file_hash(confined(base, entry[field]["path"])) != entry[field]["sha256"]:
                raise ValueError("bulk source changed")
        metadata = json.loads(confined(base, entry["metadata"]["path"]).read_text())
        offset = metadata.get("offset")
        if type(offset) is not int or metadata.get("finite") or metadata.get("table"):
            continue
        values = indexed_terms(confined(base, entry["terms"]["path"]), offset, horizon=100)
        if len(values) != 100:
            continue
        records[key] = {
            "id": key,
            "group": key,
            "indices": [str(offset + i) for i in range(100)],
            "values": [values[offset + i] for i in range(100)],
        }
    if records:
        from oeis_learn.fresh.bigint_splits import overlap_groups

        groups = overlap_groups(list(records.values()))
        for key, r in records.items():
            r["group"] = groups[key]
    selected = []
    missing = []
    for key in members:
        if key not in records:
            missing.append(key)
            continue
        r = records[key]
        if set(r) != {"id", "group", "indices", "values"}:
            raise ValueError("bulk formula/metadata excluded from target preparation")
        row = dict(r, role="training", source_snapshot=snapshot_ref["sha256"])
        try:
            validate_member(row)
        except ValueError:
            missing.append(key)
            continue
        detector_keys = [row["group"], "source:" + row["id"], "behavior:" + digest(row["values"])]
        detector_keys += ["window:" + digest(row["values"][i : i + 20]) for i in range(81)]
        if any(exposed(exposure, key, endpoint="oeis", role="training") for key in detector_keys):
            missing.append(key)
            continue
        selected.append(row)
    if missing:
        return seal(
            {
                "version": "progressive-oeis-preparation/v1",
                "status": "INCOMPLETE",
                "missing": missing,
                "source": snapshot_ref,
            }
        )
    folder = confined(root, output, exists=False)
    if folder.exists():
        raise ValueError("fresh preparation output required")
    for row in selected:
        name = digest(row["id"]).split(":")[1] + ".json"
        atomic_write(
            folder / "visible" / name,
            seal(
                {
                    "version": "discovery-channel/v1",
                    "kind": "visible_prompt",
                    "role": "generator",
                    "data": {k: row[k] for k in ("id", "indices", "values")}
                    | {"indices": row["indices"][:20], "values": row["values"][:20]},
                }
            ),
        )
        atomic_write(
            folder / "private-admission" / name,
            seal(
                {
                    "version": "discovery-channel/v1",
                    "kind": "training_continuation",
                    "role": "admission",
                    "data": row,
                }
            ),
        )
    return seal(
        {
            "version": "progressive-oeis-preparation/v1",
            "status": "COMPLETE",
            "members": members,
            "source": snapshot_ref,
            "claim": "training_feedback_only",
        }
    )


def publish_training_input(
    root, output, shards, profile, update_policy, *, diagnostic=False, refresh_updates=8
):
    """Freeze one bounded learner-only package, with no external pointer chasing."""
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.pipeline_validation.identity import confined, seal
    from oeis_learn.pipeline_validation.progressive_training import PreparedSupport

    support = PreparedSupport(
        root, shards, arm="static", refresh_updates=refresh_updates, diagnostic=diagnostic
    )
    package = seal(
        {
            "version": "progressive-training-input/v1",
            "profile": profile,
            "update_policy": update_policy,
            "shards": support.rows,
            "layout": support.layout,
            "refresh_updates": refresh_updates,
            "scope": "DIAGNOSTIC_ONLY" if diagnostic else "NATIVE_VERIFIED",
            "diagnostic_fault": None,
        }
    )
    atomic_write(confined(root, output, exists=False), package)
    return package


def prepare_combinatorial(root, output, statistics, **limits):
    """Custodian-only bounded source preparation; no statistic code reaches learner."""
    from oeis_learn.generalization.progressive.combinatorial import prepare
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.pipeline_validation.identity import confined

    value = prepare(statistics, **limits)
    atomic_write(confined(root, output, exists=False), value)
    return value
