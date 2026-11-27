"""Outcome-blind assembly adapters; no hidden model scoring or guessed inputs."""


def census(*args, **kwargs):
    from oeis_learn.generalization.progressive.cohorts import census as count

    return count(*args, **kwargs)


def publish_cohort(*args, **kwargs):
    from oeis_learn.generalization.progressive.cohorts import publish

    return publish(*args, **kwargs)


def verify_draw(value, *, client=None, budget=None, measurements=None):
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
    if measurements is not None:
        measurements['reference_steps'] = result.steps
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


def prepare_velocity_start(profile, stream, *, microbatch, broad=None, device):
    """Prepare the actual random-weight fingerprint in a learner role from exposed inputs."""
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.progressive.contracts import _check, digest, schema, seal
    from oeis_learn.generalization.progressive.models import construct
    from oeis_learn.generalization.progressive.training_rows import (
        generator_common_policy,
        pair_seeds,
    )
    from oeis_learn.generalization.progressive.velocity_trials import validate_start
    from .progressive_training import resolve_update_policy

    if stream != seal(stream) or stream["version"] != "round028-matched-stream/v1":
        raise ValueError("sealed positive stream required")
    seeds = pair_seeds(stream["registration"], stream["pair"])
    order = stream["order"]
    if (
        seeds != stream["seeds"]
        or profile["model"]["seed"] != seeds["initialization"]
        or not order
        or sorted(set(order), key=lambda k: (digest([seeds["training_families"], k]), k)) != order
    ):
        raise ValueError("training order or initial model seed differs from its frozen randomness")
    _check(schema("common")["$defs"]["hash"], stream["row_identities"], {})
    broad_ref = None
    if broad is not None:
        if (
            broad != seal(broad)
            or broad["version"] != "round029-broad-stream/v1"
            or broad.get("ordinary_ref") != stream["identity"]
            or broad.get("slot_order") != digest(order)
            or broad["seed"] != seeds["training_families"]
            or len(broad["order"]) != len(order)
        ):
            raise ValueError("broader composition treatment belongs to a different ordinary stream")
        if broad["available"]:
            broad_ref = broad["identity"]
    ordinary = resolve_update_policy(
        profile,
        objective="ordinary",
        effective_batch=32,
        microbatch_size=microbatch,
        weight_decay=0,
        schedule_horizon_tokens=4_000_000,
    )
    encoder, decoder, _ = construct(profile, device=device)
    return validate_start(
        seal(
            dict(
                version="round029-initial-state/v1",
                seed_authority=stream["registration"],
                pair=stream["pair"],
                initialization=seeds["initialization"],
                stream_seed=seeds["training_families"],
                ordinary=stream["identity"],
                broad=broad_ref,
                initial_weights=model_snapshot(encoder, decoder),
                codec=profile["codec"],
                common_policy=generator_common_policy(profile, ordinary),
            )
        )
    )


def freeze_velocity_screen(root, *, preferred, fallback, source, design, development, retention):
    """Resolve historical seed-one provenance on the physical workspace before registering any look."""
    from oeis_learn.generalization.progressive.campaign import velocity_registration
    from oeis_learn.generalization.progressive.velocity_trials import select_start
    from .progressive_velocity_donor import inventory

    start = select_start(inventory(root), preferred, fallback)
    origin = start["origin"]
    return velocity_registration(
        source=source,
        design=design,
        development=development,
        retention=retention,
        initialization=origin["initialization"],
        stream_seed=origin["stream_seed"],
        ordinary=origin["ordinary"],
        broad=origin["broad"],
        training_start=start,
    )


def prepare_velocity_broad(
    *,
    root,
    stream,
    shards,
    law,
    ordinals,
    familiar,
    calibration,
    protected,
    indexes,
    client,
    budget=None,
):
    """Audit the original positives and prospective replacements before filling the same slots."""
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.generator import validate_policy
    from oeis_learn.generalization.progressive.program_search_panels import BANDS
    from oeis_learn.generalization.progressive.shared_support import validated_rows
    from oeis_learn.generalization.progressive.training_rows import pair_seeds
    from .progressive_research_inputs import checked_file

    seeds = pair_seeds(stream["registration"], stream["pair"])
    if (
        client is None
        or stream != seal(stream)
        or stream["version"] != "round028-matched-stream/v1"
        or seeds != stream["seeds"]
        or validate_policy(law)["seed"] != seeds["training_families"]
        or calibration != seal(calibration)
        or calibration["version"] != "round028-difficulty-map/v1"
        or not set(calibration["bins"].values()) <= set(BANDS) - {"unknown"}
    ):
        raise ValueError(
            "registered native verifier, seed stream and frozen calibration map required"
        )
    rows = validated_rows(shards, native_verified=True)
    order = stream["order"]
    if (
        not order
        or len(set(order)) != len(order)
        or not set(order) <= rows.keys()
        or sorted(order, key=lambda k: (digest([seeds["training_families"], k]), k)) != order
        or digest([rows[k]["identity"] for k in order]) != stream["row_identities"]
    ):
        raise ValueError("original verified positive row order differs")
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("each prospective composition draw has one ordinal")
    familiar = set(familiar)
    if not familiar <= {rows[k]["structure"] for k in order}:
        raise ValueError("familiar support must belong to the ordinary seed")
    from pathlib import Path
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.generalization.progressive.contracts import load, reference
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from .identity import confined
    from .preparation_recovery import retain

    root = Path(root)
    index_paths = [checked_file(root, ref) for ref in indexes]
    plan = seal(
        dict(
            version="round029-composition-plan/v1",
            ordinary=stream["identity"],
            law=digest(law),
            ordinals=list(ordinals),
            familiar=sorted(familiar),
            calibration=calibration["identity"],
            protected=digest(protected),
            indexes=indexes,
            runtime=client.runtime_identity("bigint-body/v1"),
            reference=client.reference_sha256,
            source=digest(source_snapshot()),
        )
    )
    area = confined(
        root,
        f"run-state/discovery/{SESSION}/preparation/composition/{plan['identity'][7:]}",
        exists=False,
    )
    area.mkdir(parents=True, exist_ok=True)
    with RunStore.owner(area):
        retain(area / "plan.json", plan)
        receipt = confined(area, "receipt.json", exists=False)
        if not receipt.exists():
            result, selected, audit = _measure_velocity_broad(
                stream, rows, law, calibration, protected, index_paths, plan, client, budget
            )

            def install(filename, value):
                path = confined(area, filename, exists=False)
                retain(path, value)
                return reference(area, path)

            row_refs = {
                key: install("rows/" + row["identity"][7:] + ".json", row)
                for key, row in selected.items()
            }
            retain(
                receipt,
                seal(
                    dict(
                        plan=plan["identity"],
                        stream=install("stream.json", result),
                        audit=install("audit.json", audit),
                        rows=row_refs,
                    )
                ),
            )
        published = load(receipt, maximum=receipt.stat().st_size)
        if published != seal(published) or published["plan"] != plan["identity"]:
            raise ValueError("composition receipt differs from frozen input plan")

        def reopen(ref):
            return load(checked_file(area, ref), maximum=ref["bytes"])

        result, audit = reopen(published["stream"]), reopen(published["audit"])
        selected = {key: reopen(ref) for key, ref in published["rows"].items()}
        if (
            result != seal(result)
            or audit != seal(audit)
            or result["audit"] != audit["identity"]
            or audit["plan"] != plan["identity"]
            or result["ordinary_ref"] != stream["identity"]
            or set(result["order"]) != selected.keys()
            or any(
                row != seal(row)
                or row["draw_identity"] != key
                or row["identity"] != audit["witnesses"][key]
                for key, row in selected.items()
            )
        ):
            raise ValueError("stored composition stream, selected witnesses or audit differs")
        return result, selected, audit


def _measure_velocity_broad(
    stream, rows, law, calibration, protected, index_paths, plan, client, budget
):
    from contextlib import ExitStack
    from oeis_learn.generalization.progressive.complexity import measure
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.generator import broader_stream, draw, draw_exclusion
    from oeis_learn.generalization.progressive.prepared_panels import composition
    from oeis_learn.generalization.progressive.program_search_panels import (
        descriptor,
        difficulty_band,
        fresh_groups,
    )
    from .progressive_research_inputs import ExposureIndex

    order = stream["order"]
    familiar = set(plan["familiar"])
    candidates, slots, witnesses, rejected = {}, {}, {}, []

    def witness(item, original=None):
        key = item["draw_identity"]
        if key in witnesses:
            if witnesses[key]["source"] != item["source"]:
                raise ValueError("different sources share a draw identity")
            return
        measurement = {}
        row, reason = verify_draw(item, client=client, budget=budget, measurements=measurement)
        if row is not None:
            reason = draw_exclusion(row, law)
            if reason:
                row = None
        if original is not None and row != original:
            raise ValueError("ordinary positive failed independent witness replay")
        if row is None:
            rejected.append(dict(draw=key, reason=reason))
            return
        case = dict(
            row,
            id=key,
            stratum="parameter" if row["structure"] in familiar else "training",
            difficulty=measure(row["source"], reference_steps=measurement["reference_steps"]),
        )
        band = difficulty_band(descriptor(case), calibration)
        candidates[key] = case
        witnesses[key] = row
        slots[key] = dict(
            draw=key,
            composition=composition(case, allow_parameters=True),
            tokens=len(row["tokens"]),
            band=band,
            familiar=row["structure"] in familiar,
        )

    for key in order:
        witness(rows[key], rows[key])
    for ordinal in plan["ordinals"]:
        witness(draw(law, ordinal))

    def eligible(keys, readers):
        groups, _ = fresh_groups(
            [candidates[k] for k in keys], protected, indexes=readers, velocity=True
        )
        return {r["id"] for members in groups.values() for r in members}

    with ExitStack() as stack:
        readers = [stack.enter_context(ExposureIndex(path, readonly=True)) for path in index_paths]
        originals = set(order)
        if eligible(order, readers) != originals:
            raise ValueError(
                "original ordinary stream overlaps protected families or lacks exposure provenance"
            )
        clean = eligible(candidates, readers)
    for key in sorted(candidates.keys() - originals - clean):
        rejected.append(dict(draw=key, reason="protected_family_or_unverified_history"))
    # A contaminated candidate bridge cannot invalidate an independently eligible ordinary slot.
    allowed = originals | clean
    support = [slots[k] for k in sorted(allowed) if slots[k]["band"] != "unknown"]
    result = broader_stream(
        [slots[k] for k in order],
        support,
        seed=stream["seeds"]["training_families"],
        ordinary_ref=stream["identity"],
    )
    audit = seal(
        dict(
            version="round029-composition-audit/v1",
            ordinary=stream["identity"],
            plan=plan["identity"],
            rejections=rejected,
            witnesses={k: v["identity"] for k, v in witnesses.items()},
            slots=slots,
            selection=result["identity"],
        )
    )
    result = seal(dict(result, audit=audit["identity"]))
    return result, {k: witnesses[k] for k in result["order"]}, audit


def prepare_velocity_learner(
    journal, method, profile, stream, shards, *, microbatch, audit=None, client=None
):
    """Project a registered recipe's actual ordered decoder targets and eligible donor prefixes."""
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.shared_support import validated_rows
    from oeis_learn.generalization.progressive.training_rows import matching_pairs
    from .progressive_velocity_matching import registered_update_policy

    trial = journal.velocity_trial(method)["binding"]
    if stream != seal(stream) or stream["identity"] != trial["stream_ref"]:
        raise ValueError("recipe requires its registered positive stream")
    rows = validated_rows(shards, native_verified=True)
    order = stream["order"]
    if not order or not set(order) <= rows.keys():
        raise ValueError("positive stream includes an unavailable verified row")
    if trial["recipe"]["stream"] == "ordinary":
        if (
            stream["version"] != "round028-matched-stream/v1"
            or stream["seeds"]["initialization"] != trial["initialization"]
            or stream["seeds"]["training_families"] != trial["stream_seed"]
            or stream["row_identities"] != digest([rows[k]["identity"] for k in order])
        ):
            raise ValueError("ordinary training rows differ from original provenance")
    elif (
        stream["version"] != "round029-broad-stream/v1"
        or not stream["available"]
        or audit is None
        or audit != seal(audit)
        or audit["identity"] != stream.get("audit")
        or any(rows[k]["identity"] != audit["witnesses"][k] for k in set(order))
    ):
        raise ValueError("broader composition requires its independently verified witnesses")
    policy = registered_update_policy(journal, profile, method, microbatch=microbatch)
    unique = list(dict.fromkeys(order))
    pairs = (
        matching_pairs(shards, unique, seed=trial["stream_seed"], client=client)
        if trial["recipe"]["matching"]
        else None
    )
    donors = (
        {k: rows[v["donor"]]["values"][:20] for k, v in pairs["pairs"].items()} if pairs else {}
    )
    if pairs and not donors:
        raise ValueError("matching treatment has no eligible training pairs")
    projection = seal(
        dict(
            version="round029-learner-input/v2",
            initial_weights=trial["initial_weights"],
            registration=trial["registration"],
            profile=profile,
            policy=policy,
            order=order,
            rows={k: dict(tokens=rows[k]["tokens"], values=rows[k]["values"][:20]) for k in unique},
            donors=donors,
        )
    )
    return projection, seal(
        dict(
            trial=trial["identity"],
            origin=trial["origin"],
            initial_weights=trial["initial_weights"],
            stream=stream["identity"],
            package=projection["identity"],
            matching_pairs=pairs,
        )
    )
