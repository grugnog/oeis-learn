"""Frozen complete-iteration bounds protect the finalization reserve."""


def margin(payload, *, research=False):
    value = payload.get("iteration_budget")
    if value is None:
        if research:
            raise ValueError("measured complete iteration budget required")
        return 0
    if set(value) != {"update_us", "probe_us", "checkpoint_us", "producer"}:
        raise ValueError("closed iteration budget required")
    phases = [value[k] for k in ("update_us", "probe_us", "checkpoint_us")]
    if any(type(n) is not int or n <= 0 for n in phases):
        raise ValueError("positive frozen iteration bounds required")
    return sum(phases)


def measured_producer(producer, *, source, images, compatibility):
    from .contracts import seal

    fields = {
        "version",
        "identity",
        "source",
        "image",
        "device",
        "status",
        "compatibility",
        "maximum_target_tokens",
        "effective_batch",
        "measurements",
        "measurement_scope",
        "checkpoint_bytes",
        "checkpoint_inodes",
    }
    if (
        set(producer) != fields
        or producer.get("version") != "progressive-iteration-calibration/v1"
        or not isinstance(producer["measurements"], dict)
        or set(producer["measurements"]) != {"update_us", "probe_us", "checkpoint_us"}
        or any(
            type(producer[k]) is not int or producer[k] <= 0
            for k in (
                "maximum_target_tokens",
                "effective_batch",
                "checkpoint_bytes",
                "checkpoint_inodes",
            )
        )
        or any(type(v) is not int or v <= 0 for v in producer["measurements"].values())
    ):
        raise ValueError("closed measured iteration/storage calibration required")
    if (
        seal(producer) != producer
        or producer["source"] != source
        or producer["image"] not in images
        or producer["device"] != "actual_HIP"
        or producer["status"] != "PASS"
        or producer["compatibility"] != compatibility
    ):
        raise ValueError("current exact prepared iteration calibration required")
    return producer


def validate_producer(root, plan, runtime, references):
    """Bind the conservative margin to a current, frozen original HIP case."""
    import json

    from .contracts import digest, seal, verify_refs

    ref = plan["iteration_budget"]["producer"]
    if not any(r["artifact"] == ref and r["kind"] == "qualification" for r in references):
        raise ValueError("original frozen iteration qualification required")
    from oeis_learn.pipeline_validation.identity import confined

    qref = runtime["payload"]["qualification"][0]
    q = json.loads(verify_refs(root, [qref])[0])
    source = json.loads(verify_refs(confined(root, qref["path"]).parent, [q["source_closure"]])[0])
    source_identity = digest(source)
    producer = json.loads(verify_refs(root, [ref])[0])
    if producer.get("version") == "progressive-iteration-calibration/v1":
        measured_producer(
            producer,
            source=source_identity,
            images=runtime["payload"]["scope"]["images"],
            compatibility=plan["initializer"]["compatibility"],
        )
        for phase in ("update_us", "probe_us", "checkpoint_us"):
            measured = producer["measurements"].get(phase)
            if (
                type(measured) is not int
                or measured <= 0
                or plan["iteration_budget"][phase] < 2 * measured
            ):
                raise ValueError("iteration phase below conservative measured bound")
        return producer
    if runtime.get("purpose") == "research":
        raise ValueError(
            "research requires exact prepared iteration calibration; generic short-case timing is insufficient"
        )
    if (
        seal(producer) != producer
        or producer.get("version") != "progressive-hip-case/v1"
        or producer.get("device") != "actual_HIP"
        or producer.get("status") != "PASS"
        or producer.get("source") != source_identity
    ):
        raise ValueError("current actual-device iteration producer required")
    matrices = []
    for item in references:
        if item["kind"] != "policy":
            continue
        value = json.loads(verify_refs(root, [item["artifact"]])[0])
        if isinstance(value, dict) and value.get("version") == "progressive-hip-matrix/v1":
            matrices.append(value)
    matched = []
    for matrix in matrices:
        if (
            seal(matrix) != matrix
            or matrix["source"] != producer["source"]
            or matrix["image"] not in runtime["payload"]["scope"]["images"]
        ):
            continue
        for case in matrix["cases"]:
            if producer["scope"] == digest(
                [matrix["source"], matrix["image"], matrix["engine"], matrix["hardware"], case]
            ):
                matched.append(case)
    if len(matched) != 1:
        raise ValueError("original iteration matrix/case scope required")
    case = matched[0]
    model = digest({k: v for k, v in case["profile"]["model"].items() if k != "seed"})
    if model != runtime["payload"]["scope"]["model"] or case["input_length"] != 20:
        raise ValueError("iteration model/input scope differs")
    measured = producer.get("measurements", {}).get("elapsed_us")
    if type(measured) is not int or measured <= 0:
        raise ValueError("measured complete iteration work required")
    for phase in ("update_us", "probe_us", "checkpoint_us"):
        # The entire measured update/restore/generate/checkpoint case bounds
        # each phase individually; four effective batches cover the 16/32/64
        # search space without extrapolating a single microbatch as a full update.
        if plan["iteration_budget"][phase] < 8 * measured:
            raise ValueError("iteration phase below conservative measured bound")
    return producer


def require_input_envelope(package, rows):
    """Newly admitted targets must fit the original calibrated tensor shape."""
    if not rows:
        return
    base = [row for shard in package["shards"] for row in shard]
    width = 4 if package.get("task_profile") == "combinatorial-short/v1" else 20
    if max(len(row["tokens"]) for row in rows) > max(len(row["tokens"]) for row in base):
        raise ValueError("admitted target exceeds frozen calibrated token envelope")
    for index in range(width):
        if max(abs(int(row["values"][index])).bit_length() for row in rows) > max(
            abs(int(row["values"][index])).bit_length() for row in base
        ):
            raise ValueError("admitted feature exceeds frozen calibrated numeric envelope")
