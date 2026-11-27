import json
from pathlib import Path
import subprocess
import sys


from oeis_learn.pipeline_validation.identity import digest
from oeis_learn.pipeline_validation.progressive_velocity_donor import reuse_layers


def test_only_the_changed_layer_needs_work():
    cut = dict(weights=digest("weights"), lineage=digest("seed+stream"), full_state=digest("full"))
    request = dict(
        lineage=cut["lineage"],
        weights=cut["weights"],
        inference=digest("prefix/profile/order"),
        selection=digest("selector"),
        truth=digest("truth"),
        scorer=digest("scorer"),
    )
    receipt = dict(request, choice=digest("choice"), score=digest("score"))
    assert reuse_layers(cut, receipt, request)["action"] == "reuse_score"
    assert reuse_layers(cut, receipt, dict(request, scorer=digest("new")))["action"] == "rescore"
    changed = reuse_layers(cut, receipt, dict(request, inference=digest("8 vs 1")))
    assert changed["action"] == "new_inference" and changed["eligible_for_parity"] and not changed["continuation"]
    assert not reuse_layers(dict(cut, full_state=None), receipt, request)["continuation"]
    assert (
        reuse_layers(cut, receipt, dict(request, lineage=digest("new seed")))["action"]
        == "rerun_control"
    )


def test_preflight_without_private_donor_is_successful_read_only(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-S",
        "-B",
        str(root / "scripts/foundation_run.py"),
        "--engine",
        "podman",
        "discovery",
        "round029",
        "--workspace",
        str(tmp_path),
        "--action",
        "preflight",
    ]
    for _ in range(2):
        result = json.loads(subprocess.check_output(cmd))
        assert result["mechanical_status"] == "INSPECTED" and result["scientific_verdict"] is None
        assert result["inventory"]["gpu_dispatches"] == 0
        assert result["inventory"]["prices"]["common_probe"]["expected_us"] is None
        assert len(result["inventory"]["cuts"]) == 7
    assert not list(tmp_path.iterdir())


def test_public_preflight_reports_only_bound_supply(tmp_path, velocity_supply_rows):
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
    from oeis_learn.pipeline_validation.progressive_velocity_supply import register_supply, measure_supply

    cmd = [sys.executable, "-S", "-B", str(Path(__file__).resolve().parents[2] / "scripts/foundation_run.py"),
           "--engine", "podman", "discovery", "round029", "--workspace", str(tmp_path), "--action", "preflight"]
    def inspect():
        before = {p.relative_to(tmp_path): p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
        result = json.loads(subprocess.check_output(cmd))
        assert before == {p.relative_to(tmp_path): p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
        assert result["inventory"]["gpu_dispatches"] == 0
        return result["inventory"]

    def register(kind, payload):
        path = tmp_path / (kind + "-" + digest(payload)[7:] + ".json")
        atomic_write(path, seal(payload))
        ref = dict(owner="custodian", format=payload["version"], artifact=reference(tmp_path, path))
        register_supply(tmp_path, kind, ref)
        return path

    initial = inspect()
    assert not list(tmp_path.iterdir())
    for kind in ("families", "matching_pairs"):
        assert initial[kind]["count"] is None and initial[kind]["reason"]

    def case(i, group, stratum):
        return dict(id=str(i), group=group, stratum=stratum, indices=list(map(str, range(100))),
                    values=[str(500 * i + n) for n in range(100)])
    old = case(4, "old", "oeis")
    register("families", dict(version="round029-census-input/v1", seed=29, indexes=[], historical=[old],
             candidates=[case(1, "shared", "oeis"), case(2, "shared", "oeis"),
                         case(3, "separate", "composition"), old],
             quotas={"development": {"oeis": 2}, "replication": {"composition": 1}}))
    measure_supply(tmp_path, "families")
    rows = velocity_supply_rows
    shard = tmp_path / "training-only.json"
    atomic_write(shard, rows)
    def pair_input(selected):
        return dict(version="round029-pair-input/v1", seed=29, shards=[reference(tmp_path, shard)],
                    order=[r["draw_identity"] for r in selected])
    pair_path = register("matching_pairs", pair_input(rows))
    pending = inspect()
    assert pending["families"]["by_role"] == {"development": {"oeis": {"cases": 2, "families": 1}},
                                              "replication": {"composition": {"cases": 1, "families": 1}}}
    assert pending["families"]["count"] == 2 and pending["families"]["excluded_cases"] == 1
    assert pending["families"]["provenance_reason"] is None
    assert pending["matching_pairs"]["count"] is None and pending["matching_pairs"]["reason"]
    measure_supply(tmp_path, "matching_pairs")
    measured = inspect()["matching_pairs"]
    assert (measured["positive_rows"], measured["count"], measured["eligible_families"],
            measured["excluded_rows"]) == (3, 2, 2, 1)
    assert measured["dispositions"] == {"eligible": 2, "no_differing_donor": 1}
    original = pair_path.read_bytes()
    pair_path.write_bytes(original.replace(b'"seed": 29', b'"seed": 30'))
    assert inspect()["matching_pairs"]["count"] is None
    pair_path.write_bytes(original)
    original_shard = shard.read_bytes()
    shard.write_bytes(original_shard + b" ")
    assert inspect()["matching_pairs"]["count"] is None
    shard.write_bytes(original_shard)
    register("matching_pairs", pair_input(rows[:2]))
    assert inspect()["matching_pairs"]["count"] is None  # New scope cannot reuse old counts.
    measure_supply(tmp_path, "matching_pairs")
    zero = inspect()["matching_pairs"]
    assert zero["count"] == zero["eligible_families"] == 0 and zero["excluded_rows"] == 2
    assert zero["dispositions"] == {"no_differing_donor": 2} and zero["reason"] is None


def test_registered_preflight_reuse_is_once_per_verified_score_and_local_to_look(tmp_path, monkeypatch):
    import copy
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.pipeline_validation import progressive_round029 as controller
    from oeis_learn.pipeline_validation import progressive_velocity_reuse as reuse

    target = dict(weights=digest("weights"), lineage=digest("seed and stream"), inference=digest("ordered profile"),
                  selection=digest("selection"), scorer=digest("scorer"), truth=digest("truth"))
    prior = seal(dict(target, choice=digest("one sealed choice"), score=digest("exact score")))
    blank = controller.inventory(tmp_path)
    for cut in blank["cuts"][:3]:
        cut.update(weights=target["weights"], lineage=target["lineage"], full_state=digest(cut["look"]) if cut["look"] else None,
                   continuation="requires_next_update_parity" if cut["look"] else False)
    blank["original_clocks"] = [seal(dict(block="ordinary-4000000", role="train", charged_us=123, wall_us=112, lease_us=91))]
    monkeypatch.setattr(controller, "inventory", lambda _: copy.deepcopy(blank))
    monkeypatch.setattr(reuse, "historical_probe", lambda *args: prior)
    requests = [dict(method="00", look=4_000_000, segment=0, score_attempt=0, request=vals) for vals in (
        target, target, dict(target, scorer=digest("changed score")), dict(target, inference=digest("eight attempts")),
        dict(target, lineage=digest("fresh seed")))]
    requests += [dict(requests[0], look=n) for n in (0, 1_000_000)]
    reuse.register_requests(tmp_path, requests)
    times = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    args = SimpleNamespace(action="preflight", authority=None, archive=None, qualification_archive=None, workspace=tmp_path)
    result, code = controller.execute(args, tmp_path)
    assert code == 0, result
    inv = result["inventory"]
    rows = inv["reuse"]["measurements"]
    assert [r["action"] for r in rows[:5]] == ["reuse_score", "reuse_score", "rescore", "new_inference", "rerun_control"]
    assert rows[0]["credited"] and not rows[1]["credited"] and inv["reuse"]["distinct_score_credits"] == 1
    assert [r["reason"] for r in rows[2:5]] == ["private truth or scorer differs", "search, selection or sealed choice differs",
                                                 "weights or training lineage unavailable"]
    assert rows[5]["continuation"] is False and rows[6]["look_work_us"] is None and rows[0]["look_work_us"] == 123
    assert inv["gpu_dispatches"] == inv["reuse"]["research_dispatches"] == 0
    assert times == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}


def test_preflight_opens_the_two_original_installed_role_boundaries(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, reference, seal
    from oeis_learn.pipeline_validation.pilot_clocks import clock_record, DESCRIPTIVE
    from oeis_learn.pipeline_validation.progressive_velocity_reuse import AREA, PARENT, historical_probe

    area = tmp_path / PARENT / AREA
    weights = tmp_path / "original.pt"
    weights.write_bytes(b"checkpoint fixture")
    cut = dict(weights=reference(tmp_path, weights)["sha256"], lineage=digest("original ordered stream"),
               source=digest("pinned source"), registration=digest("pilot"), images={k:digest(k) for k in ("hip","cpu")})
    key = "ordinary-1000000"
    row = dict(method="00", look=1_000_000, segment=0, score_attempt=0)
    truth = dict(id="case", indices=list(map(str, range(-5,95))), values=list(map(str, range(100))))
    prompt = dict(id="case", indices=truth["indices"][:20], values=truth["values"][:20])
    recipe = {"version": "frozen-fixture"}
    search = seal(dict(block=key, arm="ordinary", target=1_000_000, inference_seed=57, rows=[prompt], recipe=recipe,
                       weights=dict(reference(tmp_path, weights), path="inputs/weights.pt")))
    case_clock = clock_record(prompt="case", mode="correct", snapshot=digest("model"), start=1, scope_deadline=100,
                              finish=2, expected=1, started=1, terminal=1, policy=DESCRIPTIVE)
    clock = seal(dict(policy=DESCRIPTIVE, work=digest("model"), inference_seed=57, source=digest(recipe),
                      expected_portfolios=1, missing_portfolios=0, cost_complete=True, clocks=[case_clock]))
    candidates = seal(dict(version="discovery-candidates/v1", methods=["correct"], model=digest("model"),
                           checkpoint=cut["weights"], recipe=digest(recipe), prompts=digest([prompt]),
                           records={"case":{"selections":{"equal_budget":None}}}, clock=clock))

    def install(role, data, member, value, input_file, destination):
        image = cut["images"]["hip" if role == "search" else "cpu"]
        req = seal(dict(version="round028-pilot-role/v1", registration=cut["registration"], source=cut["source"],
                        role=role, data=data["identity"], image=image))
        folder = area / "roles" / req["identity"][7:]
        out = folder / "attempt-0000/output"
        atomic_write(area / "assignments" / (key+"-"+role+".json"), seal(dict(request=req["identity"])))
        atomic_write(folder / "request.json", req)
        atomic_write(folder / "data.json", data)
        path = out / (member+".json")
        atomic_write(path, value)
        producer = seal(dict(version="round028-pilot-output/v1", native=digest("native"), request=req["identity"],
                             result={member:reference(out, path)}))
        atomic_write(out / "producer.json", producer)
        mounts = [(folder / "request.json", "/work/inputs/request.json", False),
                  (folder / "data.json", "/work/inputs/data.json", False), (input_file, destination, False),
                  (out, "/work/output", True)]
        atomic_write(out / "inspect.json", [dict(Image=image, HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True,
                    Privileged=False), Mounts=[dict(Type="bind",Source=str(p),Destination=d,RW=w) for p,d,w in mounts])])
        atomic_write(out / "end.json", seal(dict(request=req["identity"], success=True, elapsed_us=42)))
        bound = dict(request=req["identity"], producer=reference(folder,out/"producer.json"),
                     inspection=reference(folder,out/"inspect.json"), timing=reference(folder,out/"end.json"))
        if role == "search":
            atomic_write(out / "lease.json", {"elapsed_us":32})
            bound["lease"] = reference(folder,out/"lease.json")
        atomic_write(folder / "boundary.json", seal(bound))
        return path

    produced = install("search", search, "candidates", candidates, weights, "/work/inputs/weights.pt")
    assert historical_probe(tmp_path,cut,row)["choice"] and historical_probe(tmp_path,cut,row)["score"] is None
    candidate_ref = reference(produced.parent, produced)
    score_input = seal(dict(block=key, rows=[truth], candidates=dict(candidate_ref,path="inputs/candidates.json")))
    scores = seal(dict(version="discovery-scoring/v1", candidates=candidate_ref["sha256"], denominator=1,
                       outcomes={"case":{"equal_budget":dict(success=False, complete=True,
                                  scoring=dict(complete=True), candidate_sha256=None)}}))
    install("score", score_input, "scores", scores, produced, "/work/inputs/candidates.json")
    valid = historical_probe(tmp_path,cut,row)
    assert reuse_layers(cut, valid, valid)["action"] == "reuse_score"
    weights.write_bytes(b"other fixture")
    import pytest
    with pytest.raises(ValueError):
        historical_probe(tmp_path,cut,row)


def test_registered_mount_plan_reports_only_affected_physical_deficit(tmp_path):
    import os
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
    from oeis_learn.pipeline_validation.progressive_research_resources import (
        register_velocity_capacity,
    )
    from oeis_learn.pipeline_validation.progressive_round029 import preflight

    fs = os.statvfs(tmp_path)
    owner = dict(
        owner="public_archive",
        path=".",
        scope="published files plus coexistence",
        observed=dict(bytes=128, inodes=3),
        projected=dict(bytes=160, inodes=4),
        retained=[],
        revision="b" * 40,
        harm="would consume the mount reserve",
        exhaustion="reduce the report batch",
        rehearsal="T060 peak coexistence",
    )
    pre = tmp_path / "mounts.json"
    atomic_write(
        pre,
        seal(
            dict(
                version="round029-mount-plan/v1",
                owners=[owner],
                floors=[dict(path=".", bytes=0, inodes=fs.f_favail + 1000)],
            )
        ),
    )
    register_velocity_capacity(
        tmp_path,
        dict(owner="custodian", format="round029-mount-plan/v1", artifact=reference(tmp_path, pre)),
    )
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    inv = preflight(tmp_path)["inventory"]
    m = inv["mount"]
    assert (m["required_bytes"], m["required_inodes"], m["deficit_bytes"]) == (320, 8, 0)
    assert m["deficit_inodes"] == max(0, 8 + m["reserved_inodes"] - m["available_inodes"]) > 0
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    pre.write_bytes(pre.read_bytes() + b" ")
    scoped = preflight(tmp_path)["inventory"]
    assert scoped["capacity"]["mounts"] is None and scoped["capacity"]["reason"]
    assert len(scoped["cuts"]) == 7 and scoped["gpu_dispatches"] == 0
