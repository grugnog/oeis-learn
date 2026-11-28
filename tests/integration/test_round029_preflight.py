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


def test_preflight_after_supported_wave_publication(tmp_path, monkeypatch):
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation import progressive_round029 as controller
    from oeis_learn.pipeline_validation import progressive_velocity_trial_preparation as trials

    folder = tmp_path / "run-state/discovery" / SESSION
    folder.mkdir(parents=True)
    (folder / "accepted.json").write_text("{}")
    sources = {key: digest(key) for key in trials.SOURCES}
    publication = dict(
        accepted=digest("accepted"), registration=digest("registration"), sources=sources,
        supported=["00", "04", "05", "06"], proposed_total_us=100,
        research_authority=None, receipt=digest("receipt"),
    )
    forecast = controller.velocity_forecast

    def quoted(*args):
        return dict(forecast(*args), launch_estimate={"sources": sources},
                    launch_options={"next_scope": "wave"})

    monkeypatch.setattr(controller, "velocity_forecast", quoted)
    monkeypatch.setattr(trials, "published", lambda *_: publication)
    value = controller.preflight(tmp_path)
    assert value["next_action"] == "review_qualification"
    assert value["inventory"]["trial_preparation"] == publication

    def invalid(*_):
        raise ValueError("accepted trial preparation differs from its original publication")

    monkeypatch.setattr(trials, "published", invalid)
    value = controller.preflight(tmp_path)
    assert "trial_preparation" not in value["inventory"]
    assert value["inventory"]["trial_preparation_error"] == (
        "accepted trial preparation differs from its original publication"
    )


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
    assert [rows[i]["costs"]["search"]["additional_us"] for i in (0, 2, 3)] == [0, 0, None]
    assert rows[2]["costs"]["score"]["additional_us"] is None
    assert [r["reason"] for r in rows[2:5]] == ["private truth or scorer differs", "search, selection or sealed choice differs",
                                                 "weights or training lineage unavailable"]
    assert rows[5]["continuation"] is False and rows[6]["look_work_us"] is None and rows[0]["look_work_us"] == 123
    assert inv["gpu_dispatches"] == inv["reuse"]["research_dispatches"] == 0
    assert times == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}


def test_missing_ordinary_1m_does_not_reprice_the_intact_4m_cut(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_round029 as host

    inventory = host.inventory(tmp_path)
    for row in inventory["cuts"]:
        if row["method"] == "00" and row["look"] != 1_000_000:
            row.update(weights=digest(row["look"]), lineage=digest("matched training"))
    monkeypatch.setattr(host, "inventory", lambda _: inventory)
    prices = host.preflight(tmp_path)["inventory"]["prices"]
    assert prices["missing_control_looks"] == [1_000_000]
    assert prices["additional_control_tokens"] == 1_000_000
    assert prices["wave"]["operations"]["ordinary_to_1000000_tokens"] == 1


def test_rejected_positive_stream_requires_a_new_control_and_matched_timings(tmp_path):
    from oeis_learn.generalization.progressive.campaign import velocity_policy
    from oeis_learn.generalization.progressive.program_search_costing import velocity_forecast
    from oeis_learn.pipeline_validation.progressive_round029 import preflight

    info = preflight(tmp_path)["inventory"]
    for cut in info["cuts"]:
        if cut["method"] == "00":
            cut.update(weights=digest(cut["look"]), lineage=digest("old stream"), training_to_look_us=75)
    info["reuse"].update(distinct_score_credits=3)
    info["prices"].update(
        timed_updates=dict(methods={"04": dict(expected_us=12, tail_us=13)},
                           matching_scope={"full": True}, calibration_us=7),
        broader_updates=dict(disposition="requires_fresh_disjoint_start", device=None),
    )
    prices = velocity_forecast(info, velocity_policy())
    assert prices["missing_control_looks"] == [0, 1_000_000, 4_000_000]
    assert prices["additional_control_tokens"] == 4_000_000 and prices["prior_score_credits"] == 0
    assert {"ordinary_to_4000000_tokens", "optimizer_to_4M_recipe_04",
            "ordinary_matching_calibration"} <= set(prices["wave"]["missing"])
    assert prices["timed_updates"]["methods"]["04"]["expected_us"] == 12
    fixed = dict(method="D", look=4_000_000, action="reuse_score", score=digest("fixed D"))
    info["reuse"]["measurements"].extend([fixed, fixed])
    rate = dict(expected_us=19, tail_us=24)
    info["prices"]["fresh_start"] = dict(
        disposition="fresh_ordinary_only",
        supply=dict(matching=dict(ordinary=dict(coverage=dict(full=False)))),
        device=dict(methods={"00": rate, "04": rate}, calibration=dict(elapsed_us=11)),
    )
    priced = velocity_forecast(info, velocity_policy())
    assert priced["unit_prices"]["ordinary_to_4000000_tokens"] == rate
    assert priced["unit_prices"]["optimizer_to_4M_recipe_04"] == rate
    assert {"optimizer_to_4M_recipe_01", "ordinary_matching_calibration"} <= set(
        priced["wave"]["missing"]
    )
    assert priced["additional_control_tokens"] == 4_000_000 and priced["prior_score_credits"] == 1


def test_preflight_prices_verified_exposed_clocks_even_without_private_freeze(
    tmp_path, monkeypatch
):
    from io import BytesIO
    from zipfile import ZipFile
    from oeis_learn.generalization.progressive.cohorts import STRATA
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.program_search_costing import PROXIES
    from oeis_learn.pipeline_validation import progressive_round029 as host
    from oeis_learn.pipeline_validation import progressive_velocity_donor as donor
    from tests.integration.test_round028_timing import measured

    def archive(files):
        stream = BytesIO()
        with ZipFile(stream, "w") as z:
            for name, value in files.items():
                z.writestr(name, value if isinstance(value, bytes) else json.dumps(value))
        return stream.getvalue()

    cell = dict(cases=32, scored=32, score_us=32)
    panels = seal(dict(historical_clocks={p: dict.fromkeys(STRATA, cell) for p in PROXIES}))
    blocks = measured()
    blocks["first_divergence-full"]["cells"]["oeis"]["max_search_us"] = 19
    timing = seal(
        dict(
            panel=panels["identity"],
            blocks=blocks,
            candidate="a" * 40,
            image_build_us=dict(cpu=200, hip=300),
        )
    )
    nested = archive({"round028.zip": archive({"timing.json": timing, "panels.json": panels})})
    pilot = tmp_path / "pilot.zip"
    original = dict(costs=[seal(dict(block="ordinary-0", role="search", charged_us=36, lease_us=30))],
                    looks=[dict(target=0)], design=dict(sizes={"0": [dict(size=1), dict(size=3)]}))
    pilot.write_bytes(archive({"pilot.json": original, "round028-parent.zip": nested}))
    scale = seal(
        dict(
            cases=160,
            no_selection=160,
            slowest_case_us=60_133_643,
            search_supervisor_us=6_142_074_000,
            score_supervisor_us=520,
            score_control=dict(supervisor_us=550),
        )
    )
    pre = tmp_path / "qualification.zip"
    pre.write_bytes(
        archive({"qualification.json": dict(scale=scale, registration=dict(candidate="b" * 40))})
    )
    monkeypatch.setattr(
        donor,
        "_archives",
        lambda *_: ({"pilot": pilot, "qualification": pre}, {"pilot": {}, "qualification": {}}),
    )
    verified = {"report": donor.PILOT_REPORT}
    monkeypatch.setattr(donor, "_verify_with_pin", lambda *_: verified)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    prices = host.preflight(tmp_path)["inventory"]["prices"]
    assert prices["exposed_028_timing"]["receipt"] == timing["identity"]
    workload = prices["original_028_workloads"][0]
    assert workload["cases"] == 4 and workload["roles"]["search"]["average_per_case_us"] == 9
    assert prices["followup"]["expected_us"] == 132 and prices["followup_reserve_us"] == 148
    assert prices["image_acquisition_us"] == {"cpu": 200, "hip": 300}
    assert prices["followup"]["uncached_tail_us"] == 648
    assert prices["common_probe"]["expected_us"] is None
    assert "development_family_census" in prices["common_probe"]["missing"]
    assert prices["provisional_guards"][0]["value"] == 2 * 6_142_074_000
    assert prices["provisional_guards"][1]["observed"] == 550
    by_role = {
        "development": {s: dict(cases=4, families=2) for s in STRATA},
        "development_retention": {"retention": dict(cases=3, families=2)},
    }
    family_supply = dict(
        count=10,
        by_role=by_role,
        eligible_cases=19,
        eligible_components=10,
        excluded_cases=0,
        deficits={},
        provenance_unavailable=0,
        provenance_reason=None,
        registration=digest("registration"),
        measurement=digest("census"),
        reason=None,
    )
    monkeypatch.setattr(
        host,
        "supply_inventory",
        lambda _: dict(
            families=family_supply, matching_pairs=dict(count=None, reason="missing training pairs")
        ),
    )
    priced = host.preflight(tmp_path)["inventory"]["prices"]
    assert priced["common_probe"]["expected_us"] == 2 * (54 + 8 * 12)
    assert priced["wave"]["operations"]["learner.one.search.oeis"] == 6 * 3 * 2
    assert priced["wave"]["operations"]["D.full.score.retention"] == 3
    assert priced["additional_control_tokens"] == 4_000_000 and priced["wave"]["tail_us"] is None
    assert "matched_full_bank_scoring" in priced["wave"]["missing"]
    assert priced["wave"]["known_tail_us"] > priced["common_probe"]["tail_us"]
    by_role["development"]["oeis"] = dict(cases=0, families=0)
    zero = host.preflight(tmp_path)["inventory"]["prices"]
    assert zero["followup_family_available"] is False and zero["followup_reserve_us"] is None
    assert zero["followup"]["known_tail_us"] == 0
    verified["report"] = digest("unrelated report")
    rejected = host.preflight(tmp_path)["inventory"]["prices"]
    assert rejected["exposed_028_timing"] is rejected["followup_reserve_us"] is None
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.iterdir()}


    from oeis_learn.pipeline_validation import progressive_velocity_broad_host as broad
    from oeis_learn.pipeline_validation import progressive_velocity_timing_host as ordinary
    rate = dict(expected_us=10, tail_us=12)
    paid = dict(
        measurement=digest("fixture ordinary timing"),
        methods={},
        sampled_methods={"02": rate},
        matching_scope={"full": False},
        calibration_us=5,
    )
    monkeypatch.setattr(ordinary, "timing_inventory", lambda *_: paid)
    previous = host.preflight(tmp_path)["inventory"]["prices"]
    assert {"optimizer_to_4M_recipe_02", "ordinary_matching_calibration"} <= set(
        previous["wave"]["missing"]
    )
    measurement = dict(
        ordinary_measurement=paid["measurement"],
        disposition="prepared",
        device={"methods": {"01": rate, "02": rate, "03": rate}, "calibration": {"elapsed_us": 7}},
    )
    monkeypatch.setattr(broad, "inventory", lambda *_: measurement)
    both = host.preflight(tmp_path)["inventory"]["prices"]
    assert not {
        "optimizer_to_4M_recipe_01",
        "optimizer_to_4M_recipe_02",
        "optimizer_to_4M_recipe_03",
        "ordinary_matching_calibration",
    }.intersection(both["wave"]["missing"])
    assert both["wave"]["operations"]["ordinary_matching_calibration"] == 1
    assert both["broader_updates"] == measurement
    monkeypatch.setattr(broad, "inventory", lambda *_: dict(measurement, ordinary_measurement=digest("other calibration")))
    stale = host.preflight(tmp_path)["inventory"]["prices"]
    assert "different ordinary calibration" in stale["broader_reason"]
    assert {"optimizer_to_4M_recipe_01", "optimizer_to_4M_recipe_03"} <= set(stale["wave"]["missing"])


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
                floors=[dict(path=".", bytes=0, inodes=fs.f_files + 1000)],
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


def test_combined_cpu_census_keeps_paid_pairs_across_family_failure(tmp_path, monkeypatch):
    from oeis_learn.generalization.progressive.contracts import load, reference, seal
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation import progressive_velocity_preparation as prep
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_velocity_qualification import (
        authorize_qualification,
    )
    from oeis_learn.pipeline_validation.progressive_velocity_supply import supply_inventory
    from oeis_learn.pipeline_validation.progressive_velocity_verify import private_result

    revision = "a" * 40
    folder = tmp_path / "run-state/discovery" / SESSION
    folder.mkdir(parents=True)
    (folder / "run.lock").touch()
    grant = load(authorize_qualification(tmp_path, 2, prep.preparation_parent(revision), {}))
    calls = []
    original = tmp_path / "original.json"
    retain(original, {"source": "exposed fixture"})
    pair = seal(
        dict(
            version="round029-training-pairs/v1",
            positive_rows=3,
            eligible_rows=0,
            eligible_families=0,
            pairs={},
            dispositions={"needs_native_replay": 3},
        )
    )
    families = seal(
        dict(
            version="round029-family-reservations/v1",
            quotas={"development": {"oeis": 2}},
            panels={"development": []},
            eligible=0,
            components=0,
            excluded=4,
            deficits={"development": {"oeis": 2}},
            provenance_unavailable=0,
            provenance_reason=None,
        )
    )

    def dispatch(root, source, out, kind, reserve):
        calls.append(kind)
        retain(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        if calls == ["matching_pairs", "families"]:
            raise ValueError("injected lost family worker")
        retain(out / "measurement.json", pair if kind == "matching_pairs" else families)
        retain(out / "inputs.json", seal(dict(kind=kind, originals=[reference(root, original)])))
        prep.seal_output(out, kind)

    monkeypatch.setattr(prep, "run_worker", dispatch)

    def run():
        return prep.qualify(None, tmp_path, folder, tmp_path, revision, grant, None)

    failed, code = run()
    assert code == 1 and failed["first_error"]["stage"] == "families"
    assert supply_inventory(tmp_path)["matching_pairs"]["dispositions"] == {
        "needs_native_replay": 3
    }
    assert supply_inventory(tmp_path)["families"]["count"] is None
    completed, code = run()
    assert code == 0 and calls == ["matching_pairs", "families", "families"]
    assert run() == (completed, 0)
    assert completed["charged_tokens"] == completed["research_dispatches"] == 0
    assert completed["remaining_authority"] + completed["elapsed_us"] == 120_000_000
    assert completed["first_error"] == failed["first_error"]
    verified, code = private_result(tmp_path, folder)
    assert code == 0 and verified["last_committed_ref"] == completed["last_committed_ref"]
    assert supply_inventory(tmp_path)["families"]["count"] == 0
    from oeis_learn.pipeline_validation.progressive_velocity_supply import _area
    pointer = load(_area(tmp_path, "matching_pairs") / "funded.json")
    counts, original_pairs = prep.funded_inventory(tmp_path, "matching_pairs", pointer, with_original=True)
    assert original_pairs == pair and counts == supply_inventory(tmp_path)["matching_pairs"]
    original.write_text("changed")
    assert all(r["count"] is None for r in supply_inventory(tmp_path).values())


def test_donor_census_opens_the_committed_stream_and_all_prior_roles(tmp_path, monkeypatch, velocity_supply_rows):
    import zipfile
    import pytest
    from oeis_learn.generalization.progressive.contracts import digest, reference, seal
    from oeis_learn.pipeline_validation import progressive_velocity_census as census
    from oeis_learn.pipeline_validation.preparation_recovery import retain

    root = tmp_path
    def save(path, data):
        retain(path, data)
        return reference(root, path)

    area = root / census.PARENT / census.AREA
    order = [r["draw_identity"] for r in velocity_supply_rows[:-1]]
    stream = seal(
        dict(
            order=order,
            seeds={"pair": 0, "training_families": 113},
            row_identities=digest([r["identity"] for r in velocity_supply_rows[:-1]]),
        )
    )
    def row(n):
        return dict(id=str(n), indices=list(map(str, range(100))), group="family-" + str(n),
                    values=[str(n * 1000 + i) for i in range(100)], stratum="oeis")
    old, calibration, fresh = row(1), row(2), row(3)
    allocations = dict(panels={"assessment": [old], "development": []})
    stock = seal(dict(version="round028-private-panel-supply/v1", registration=digest("panel"),
                      allocations=allocations, witnesses={old["id"]: old}))
    roster = save(root / "panel/supply.json", stock)
    cal_path = root / "panel/calibration-reservation.json"
    save(
        cal_path,
        seal(dict(registration=stock["registration"], witnesses={"family": digest(calibration)})),
    )
    data = dict(identity=digest("027 data"))
    census_id = digest("027 census")
    cut = digest(
        [census_id, data["identity"], stock["identity"], reference(cal_path.parent, cal_path)]
    )
    train_path = area / "training.json"
    save(
        train_path,
        seal(
            dict(
                version="round028-pilot-training/v1",
                stream=stream,
                rows=velocity_supply_rows,
                exposure_cut=cut,
            )
        ),
    )
    freeze = seal(
        dict(
            training=reference(area, train_path),
            stream=stream["identity"],
            seeds=stream["seeds"],
            roster_source=roster,
        )
    )
    save(area / "freeze.json", freeze)
    pilot, qualification = root / "pilot.zip", root / "qualification.zip"
    with zipfile.ZipFile(pilot, "w") as z:
        z.writestr("pilot.json", json.dumps(dict(design=dict(private_freeze=freeze["identity"]))))
    qualification.write_bytes(b"fixture qualification")
    monkeypatch.setattr(
        census, "_archives", lambda *_: ({"pilot": pilot, "qualification": qualification}, {})
    )
    monkeypatch.setattr(census, "_verify_with_pin", lambda *_: {"report": census.PILOT_REPORT})
    orig = seal(dict(payload={"references": []}))
    save(root / census.ORIGINAL / "prepare-research/frozen/campaign.json", orig)
    monkeypatch.setattr(census, "learned_search_policy", lambda: {"campaign": orig["identity"]})
    historical = save(root / "historical.json", {"historical": 1})
    monkeypatch.setattr(
        census,
        "read_inputs",
        lambda *_, velocity: dict(
            census=census_id,
            data=data,
            historical=[],
            candidates=[old, calibration, fresh],
            support={"structures": {}},
            originals_used=[historical],
        ),
    )
    pairs, pair_refs = census.measure(root, "matching_pairs")
    families, family_refs = census.measure(root, "families")
    assert pairs["seed"] == stream["seeds"]["training_families"]
    assert pairs["positive_rows"] == len(order) and pairs["order"] == digest(order)
    assert families["eligible"] == 1 and families["excluded"] == 2
    assert {p["task"] for members in families["panels"].values() for p in members} == {fresh["id"]}
    assert {r["path"] for r in pair_refs} < {r["path"] for r in family_refs}
    assert historical in family_refs and reference(root, cal_path) in family_refs
    train_path.write_text("{}")
    with pytest.raises(ValueError):
        census.measure(root, "matching_pairs")


def test_cpu_supervisor_preserves_original_missing_input_reason(tmp_path):
    import pytest
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.pipeline_validation.progressive_velocity_preparation import run_worker

    out = tmp_path / "attempt"
    source = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="pinned pilot and qualification archives required"):
        run_worker(tmp_path, source, out, "matching_pairs", 15_000_000)
    assert load(out / "cleanup.json") == dict(acknowledged=True, within_bound=True)


def test_reopened_complete_prices_select_the_smallest_eligible_boundary():
    from oeis_learn.pipeline_validation.progressive_velocity_launch import launch_options
    import copy

    def mounts(required):
        return [
            dict(
                device=1,
                available_bytes=18,
                required_bytes=required,
                reserved_bytes=5,
                deficit_bytes=max(0, required + 5 - 18),
                available_inodes=10,
                required_inodes=5,
                reserved_inodes=0,
                deficit_inodes=0,
            )
        ]

    preparation = dict(
        wave_space=dict(
            wave=mounts(20),
            first_ordinary_cut=mounts(14),
            ordinary_initialization=mounts(12),
            fits_with_reserve=False,
            missing=[],
        ),
        initialization_price=dict(expected_us=8, tail_us=10),
    )
    estimate = dict(
        expected_us=100,
        tail_us=140,
        followup_reserve_us=5,
        tail_with_reserve_us=145,
        missing=[],
        research_authority=None,
    )
    options = launch_options(estimate, preparation)
    assert options["next_scope"] == "ordinary_initialization"
    assert options["scopes"]["ordinary_initialization"]["tail_with_reserve_us"] == 15
    assert options["scopes"]["wave"]["mounts"][0]["deficit_bytes"] == 7
    failed = copy.deepcopy(preparation)
    failed["wave_space"]["ordinary_initialization"][0]["deficit_inodes"] = 1
    assert launch_options(estimate, failed)["next_scope"] is None
    partial = dict(
        estimate,
        missing=["paired_conditioning_sample"],
        expected_us=None,
        tail_with_reserve_us=None,
    )
    assert launch_options(partial, preparation)["next_scope"] == "ordinary_initialization"
