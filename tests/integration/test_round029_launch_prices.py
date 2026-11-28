"""CPU pricing replays positive-token bookkeeping at real durable boundaries."""

import pytest

from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.pipeline_validation import progressive_velocity_ledger_prices as prices


def test_bookkeeping_reopens_the_whole_update_that_crossed_each_look(tmp_path):
    order = ["a", "b", "c"]
    lengths = dict(a=2, b=3, c=4)
    frame = prices.frame(order, lengths, 16, looks=(50, 120))
    assert [(k["updates"], k["useful_tokens"]) for k in frame["cuts"]] == [(2, 95), (3, 144)]
    result = prices.measure(tmp_path, frame, order, lengths)
    assert result["updates"] == 3 and result["useful_tokens"] == 144
    assert result["price"]["tail_us"] >= result["price"]["expected_us"] > 0
    assert result["ledger_bytes"] > 0
    assert prices.verify(tmp_path, frame, order, lengths) == result
    saved = tmp_path / "fixture-accounting/budget.jsonl"
    saved.write_bytes(saved.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(ValueError):
        prices.verify(tmp_path, frame, order, lengths)


def test_paid_cpu_batches_resume_without_repaying_work(tmp_path):
    root = tmp_path
    folder = root / "run-state/discovery" / prices.SESSION
    folder.mkdir(parents=True)
    order, lengths = ["a", "b"], {"a": 2, "b": 3}
    frames = {str(b): prices.frame(order, lengths, b, looks=(50, 120)) for b in (16, 32)}
    plan = prices.plan_for("a" * 40, digest("fresh"), digest("cycles"), frames)
    first = prices.run(root, folder, plan, order, lengths, minutes=2, floor=0)
    assert set(first["batches"]) == {"16", "32"}
    assert first["charged_work"]["tokens"] == 0 and first["charged_work"]["attempts"] == 2
    assert first["remaining_us"] + first["charged_work"]["elapsed_us"] == 120_000_000
    assert prices.run(root, folder, plan, order, lengths, minutes=2, floor=0) == first
    assert prices.reopen(root, folder, plan, order, lengths, first["grant"]) == first
    prices.record(root, folder, plan, order, lengths, first)
    with pytest.raises(ValueError, match="both original"):
        prices.record(root, folder, plan, order, lengths, dict(first, errors={"32": "failed"}))
    assert len(list(folder.glob("preparation/update-accounting/*/operations/*/attempt-*"))) == 2


def test_denied_cpu_footprint_keeps_the_full_grant_available(tmp_path):
    root = tmp_path
    folder = root / "run-state/discovery" / prices.SESSION
    folder.mkdir(parents=True)
    lengths, order = {"a": 3}, ["a"]
    frames = {str(b): prices.frame(order, lengths, b, looks=(50, 120)) for b in (16, 32)}
    plan = prices.plan_for("a" * 40, digest("fresh"), digest("cycles"), frames)
    denied = prices.run(root, folder, plan, order, lengths, minutes=2, floor=1 << 60)
    assert denied["remaining_us"] == 120_000_000 and not any(denied["charged_work"].values())
    assert all(v["mounts"][0]["deficit_bytes"] > 0 for v in denied["errors"].values())
    assert not (folder / prices.MARKER).exists()
    completed = prices.run(root, folder, plan, order, lengths, minutes=2, floor=0)
    assert completed["grant"] == denied["grant"] and not completed["errors"]


def test_pinned_training_projection_and_whole_update_frames(tmp_path, monkeypatch):
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
    from oeis_learn.pipeline_validation import progressive_velocity_broad_host as broad
    from oeis_learn.pipeline_validation import progressive_velocity_checkpoint_host as checkpoints

    root = tmp_path
    folder = root / "run-state/discovery" / prices.SESSION
    grant_id = digest("paid")
    area = folder / "qualification" / grant_id[7:]
    cpu, gpu = area / "prepare", area / "measure"
    start = seal(dict(stream="exposed"))
    rows = {"a": dict(tokens=[0, 0]), "b": dict(tokens=[0, 0, 0])}
    positive = dict(rows=rows, order=["a", "b"] * 16)
    path = cpu / "native/projection.json"
    atomic_write(path, seal(dict(fresh_start=start["identity"], streams={"ordinary": positive})))
    methods = {"00": dict(batch=32), "06": dict(batch=16)}
    cells = {m: prices.batch_plan(["a", "b"], rows, {}, r["batch"]) for m, r in methods.items()}
    req = seal(dict(projection=reference(path.parent, path), design=dict(cells=cells)))
    producer = seal(dict(request=req["identity"]))
    atomic_write(area / "device-request.json", req)
    atomic_write(area / "grant.json", dict(identity=grant_id))
    atomic_write(gpu / "producer.json", producer)
    fresh = seal(dict(start=start, device=dict(producer=producer["identity"])))
    pointer = dict(grant=grant_id, receipt=fresh["identity"])
    monkeypatch.setattr(checkpoints, "origin", lambda _: pointer)
    monkeypatch.setattr(broad, "coordinates", lambda grant, phase: phase)
    monkeypatch.setattr(broad, "attempt", lambda area, phase, grant: area / phase)
    cycles = dict(methods=methods, origin=fresh["identity"], receipt=digest("cuts"))
    plan, order, lengths = prices.source_inputs(root, fresh, cycles, "a" * 40)
    assert order == ["a", "b"] and lengths == {"a": 2, "b": 3}
    assert plan["batches"]["16"]["cuts"][-1]["updates"] == 100000
    assert plan["batches"]["32"]["cuts"][-1]["updates"] == 50000
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        prices.source_inputs(root, fresh, cycles, "a" * 40)


def test_combined_command_keeps_pricing_and_returns_original_zip_when_conditioning_is_blocked(
    tmp_path, monkeypatch
):
    from tests.integration.test_round029_conditioning_prices import api
    from oeis_learn.generalization.progressive.contracts import seal

    read, publish = api["donor_launch_prices"], api["write_readiness_zip"]
    nested = tmp_path / "conditioning.zip"
    publish(
        nested,
        dict(
            scope=api["CONDITIONING"],
            status="BLOCKED",
            research_run_requested=False,
            scientific_verdict=None,
            files={},
        ),
    )
    folder = tmp_path / "run-state/discovery" / prices.SESSION
    folder.mkdir(parents=True)
    lengths, order = {"a": 3}, ["a"]
    frames = {str(b): prices.frame(order, lengths, b, looks=(50, 120)) for b in (16, 32)}
    plan = prices.plan_for("a" * 40, digest("fresh"), digest("cuts"), frames)
    monkeypatch.setattr(prices, "source_inputs", lambda *args: (plan, order, lengths))
    original = prices.run
    monkeypatch.setattr(
        prices,
        "run",
        lambda *args, minutes, floor, registered: original(*args, minutes=minutes, floor=0),
    )

    def action(_):
        prep = dict(checkpoint_receipt=digest("cuts"), fresh_receipt=digest("fresh"))
        marker = folder / prices.MARKER
        if marker.is_file():
            prep.update(accounting_receipt=prices.load(marker)["receipt"])
        return dict(
            inventory=dict(
                gpu_dispatches=0,
                capacity=dict(mounts=[]),
                prices=dict(
                    fresh_start={},
                    checkpoint_cycles=dict(receipt=digest("cuts")),
                    supported_wave=dict(checkpoint_receipt=digest("cuts")),
                    launch_preparation=prep,
                ),
            )
        )

    result = read(action, tmp_path, "a" * 40, "2", nested)
    assert result["status"] == "PARTIAL" and result["conditioning_status"] == "BLOCKED"
    assert result["files"]["accounting.json"]["charged_work"]["attempts"] == 2
    destination = tmp_path / "combined.zip"
    publish(destination, result)
    assert api["verify_launch_zip"](destination)["status"] == "PARTIAL"
    publish(destination, dict(result, status="COMPLETE"))
    with pytest.raises(ValueError):
        api["verify_launch_zip"](destination)
    # A problem reopening the CPU marker leaves the already paid byte-for-byte result available.
    retained = dict(result["files"], **{"preparation.json": {}})
    publish(destination, dict(result, files=retained, errors={"update-accounting": "ValueError"}))
    assert api["verify_launch_zip"](destination)["status"] == "PARTIAL"
    altered = dict(result["files"])
    cpu = dict(altered["accounting.json"])
    cpu["remaining_us"] += 1
    altered["accounting.json"] = seal(cpu)
    publish(destination, dict(result, files=altered))
    with pytest.raises(ValueError):
        api["verify_launch_zip"](destination)


def test_private_replay_clock_uses_the_successful_verifier_boundary(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write, seal
    from oeis_learn.generalization.progressive.velocity_records import operation_key
    from oeis_learn.pipeline_validation.progressive_velocity_launch import role_storage
    from oeis_learn.pipeline_validation.progressive_velocity_operations import operation_receipt
    from tests.integration.test_round029_conditioning_prices import api

    grant = dict(identity=digest("grant"), registration_or_parent_ref=digest("plan"))
    coords = api["conditioning_coordinates"](grant, "D-full", "search", 29)
    req = seal(dict(plan=digest("plan")))
    atomic_write(tmp_path / "inputs/D-full/search/request.json", req)
    logical = operation_key(coords, 0)["logical"]
    block = tmp_path / "operations" / logical[7:]
    out = block / "attempt-0000"
    out.mkdir(parents=True)
    original = operation_receipt(coords, req, grant, 0, seal(dict(elapsed_us=11)))
    atomic_write(block / "receipt.json", original)
    key = logical + "/dispatch/0"
    state = dict(committed=[key], windows={key: dict(event="COMMIT", end_us=30, charged_us=18)})
    elapsed, paths = role_storage(tmp_path, grant, state, coords)
    assert elapsed == 7 and paths[0] == out
    reader = dict(coords, role="verify-paid-search")
    repair = operation_receipt(reader, req, grant, 0, seal(dict(original=original)))
    repair_key = operation_key(reader, 0)["logical"]
    atomic_write(tmp_path / "operations" / repair_key[7:] / "receipt.json", repair)
    state["committed"].append(repair_key + "/dispatch/0")
    state["windows"][repair_key + "/dispatch/0"] = dict(event="COMMIT", end_us=52, charged_us=5)
    assert role_storage(tmp_path, grant, state, coords, recovered=original["identity"])[0] == 5
    with pytest.raises(ValueError):
        role_storage(tmp_path, grant, state, coords, recovered=digest("other search"))


def test_archive_review_recomputes_each_mount_and_checks_the_projected_fit(tmp_path):
    from copy import deepcopy
    from tests.integration.test_round029_conditioning_prices import api
    from oeis_learn.pipeline_validation.progressive_pilot_capacity import velocity_mounts

    owner = dict(owner="ordinary", scope="fixture cut", path="learner", observed=dict(bytes=45,inodes=4),
                 projected=dict(bytes=55,inodes=5), retained=[], revision="a"*40,
                 harm="lost cut", exhaustion="pause", rehearsal="T060")
    mount, = velocity_mounts(tmp_path,[owner],[dict(path="learner",bytes=7,inodes=2)],
                             measure=lambda path: (11,100,40))
    assert mount["deficit_bytes"] == 17
    capacity = dict(wave=[mount], first_ordinary_cut=[mount], missing=[],fits_with_reserve=False)
    review = api["review_mounts"]
    assert review(capacity)["first_ordinary_cut"][0]["deficit_bytes"] == 17
    for key, value in (("fits_with_reserve", True), ("missing", "unknown")):
        with pytest.raises(ValueError):
            review(dict(capacity, **{key:value}))
    for mutate in ("deficit_bytes","required_inodes","guard","owner"):
        altered=deepcopy(capacity)
        m=altered["wave"][0]
        if mutate=="guard":
            m["owners"][0]["guards"][0]["value"] += 1
        elif mutate=="owner":
            m["owners"].append(deepcopy(m["owners"][0]))
        else:
            m[mutate] += 1
        with pytest.raises(ValueError):
            review(altered)
