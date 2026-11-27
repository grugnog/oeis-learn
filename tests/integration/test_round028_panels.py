"""Supply, private scheduling, standalone pricing, and pre-score role boundaries."""

import json
import runpy
import sqlite3
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.program_search_costing import (
    MEASURES,
    PROXIES,
    clock_cells,
    forecast,
)
from oeis_learn.generalization.progressive.program_search_panel_report import verify_panels
from oeis_learn.generalization.progressive.program_search_panels import RESERVATIONS
from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
    encode,
    publish,
    verify_transport,
)
from oeis_learn.pipeline_validation import progressive_panel_sources as sources
from oeis_learn.pipeline_validation import progressive_panel_worker as worker
from oeis_learn.pipeline_validation import progressive_round028_panels as panels
from oeis_learn.pipeline_validation.pilot_clocks import DRAIN, clock_record
from oeis_learn.pipeline_validation.progressive_round028 import ARCHIVE
from oeis_learn.pipeline_validation.progressive_round028_receipts import recover
from tests.integration.test_round028_receipts import REVISION, SOURCE
from tests.integration.test_round028_receipts import original as original  # noqa: PLC0414 - shared pytest fixture
from tests.unit.test_progressive_research_inputs import source_snapshot


def summary(*, deficit):
    return {
        "history": {
            "original_reservations": 200,
            "corrected_reservations": 200,
            "native_training": 125,
            "older_exposure_rows": 130,
        },
        "supply": {
            "candidates": 2300,
            "historical_rows": 655,
            "eligible": 2000,
            "excluded": 300,
            "components": 1100,
        },
        "extension": {
            "native_draws": 0,
            "native_accepted": 0,
            "source_selected": 0,
            "source_remaining": None,
        },
        "calibration": {
            "reserved": 48,
            "observed": 0 if deficit else 48,
            "measured": 0 if deficit else 130,
            "exact": 0 if deficit else 81,
        },
        "calibrated": not deficit,
        "panels": {
            role: {
                s: {
                    "cases": 16 if deficit and s == "oeis" else 32,
                    "families": 8 if deficit and s == "oeis" else 16,
                    "bands": {
                        "unknown" if deficit or s == "oeis" else "easy": 16
                        if deficit and s == "oeis"
                        else 32
                    },
                }
                for s in (("retention",) if role.endswith("_retention") else STRATA)
            }
            for role in RESERVATIONS
        },
    }


def historical_clocks():
    return {
        p: {
            s: dict(
                dict.fromkeys(MEASURES, 0),
                cases=32,
                timed=32,
                closed=32,
                scored=32,
                portfolio_us=32_000_000,
                observed_max_us=1_000_000,
                score_us=3_200_000,
            )
            for s in STRATA
        }
        for p in PROXIES
    }


def test_portfolio_and_scorer_exclude_retention_worker_and_price_unfinished_clocks():
    model = digest("model")

    def clock(key, start, finish, terminal, deadline):
        return clock_record(
            prompt=key,
            mode="correct",
            snapshot=model,
            start=start,
            finish=finish,
            scope_deadline=deadline,
            expected=8,
            started=terminal,
            terminal=terminal,
            policy=DRAIN,
        )

    a, b = clock("a", 0, 3, 8, 100), clock("b", 10, 13, 2, 13)
    generation = {
        "model": model,
        "clock": seal({"policy": DRAIN, "clocks": [a, b]}),
        "records": {"a": {}, "b": {}, "old-other": {}},
    }
    scores = {
        "candidates": digest("generation"),
        "outcomes": {
            key: {"equal_budget": {"scoring": {"elapsed_seconds": 0.25}}}
            for key in ("a", "b", "old-other")
        },
    }
    cells = clock_cells(
        [(generation, scores, scores["candidates"])], {"a": "composition", "b": "composition"}
    )["composition"]
    assert (
        cells["cases"],
        cells["timed"],
        cells["closed"],
        cells["portfolio_us"],
        cells["score_us"],
    ) == (2, 2, 1, 6000000, 500000)
    saved = historical_clocks()
    saved[PROXIES[0]]["composition"].update(
        portfolio_us=6_000_000, open_portfolio_us=3_000_000, closed=31
    )
    train = {
        "combined_training_us": 500,
        "shift_us": 28_800_000_000,
        "evaluation_replication_reserve_us": 14_400_000_000,
    }
    budget = forecast(saved, train, dict.fromkeys(STRATA, 16))
    assert budget["rates_us"][PROXIES[0]]["composition"] == 2_068_750
    saved[PROXIES[1]]["oeis"]["scored"] = 31
    assert (
        forecast(saved, train, dict.fromkeys(STRATA, 16))["status"] == "ORIGINAL_CLOCKS_INCOMPLETE"
    )
    with pytest.raises(ValueError, match="parent"):
        clock_cells([(generation, scores, digest("other"))], {"a": "composition"})


def test_extends_indexed_source_without_repeating_an_original_member(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_source import project

    manifest = source_snapshot(tmp_path, count=5)
    initial = tmp_path / "original-source"
    old = project(tmp_path, manifest, initial, limit=2)
    candidates = reference(tmp_path, initial / old["candidates"]["path"])
    inputs = {"candidates": candidates, "source": old["snapshot"]}
    row = load(initial / old["candidates"]["path"])[0]
    # A successful continuation needs only the snapshot's remaining members.
    damaged = int(row["id"].rsplit("A", 1)[-1])
    (tmp_path / f"data/source/{damaged}.txt").write_text("already consumed")
    area = tmp_path / "new"
    area.mkdir()
    rows, costs = sources.extra_source(tmp_path, inputs, area)
    assert len(rows) == costs["selected"] == 3 and costs["remaining"] == 0
    assert all(r["indices"][0] == "-2" for r in rows)
    assert not {r["id"] for r in rows} & {
        r["id"] for r in load(initial / old["candidates"]["path"])
    }
    assert sources.extra_source(tmp_path, inputs, area) == (rows, costs)
    with (area / "extended-source/private/oeis-candidates.json").open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError):
        sources.extra_source(tmp_path, inputs, area)


def test_native_census_requires_original_ordinal_witness_and_training_reservoir(tmp_path):
    from oeis_learn.generalization.progressive.generator import structure_id

    source = "(program (return $index))"
    req = {"training_draws": 1, "evaluation_draws": 1}
    support = {"structures": {structure_id(source)}, "nodes": set(), "operators": set()}
    training = [seal({"source": source, "native_verified": True})]
    row = seal(
        {
            "id": "unscored",
            "source": source,
            "native_verified": True,
            "membership": "evaluation_candidate",
            "stratum": "parameter",
            "structure": structure_id(source),
        }
    )
    path = tmp_path / "census.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE draws(ordinal,status,row); CREATE TABLE native(value); CREATE TABLE binding(value);"
        )
        db.execute("INSERT INTO binding VALUES (?)", (json.dumps(req),))
        db.execute("INSERT INTO native VALUES ('original-native')")
        db.executemany(
            "INSERT INTO draws VALUES (?,?,?)",
            [(0, "TRAINING", json.dumps(training[0])), (1, "EVALUATION", json.dumps(row))],
        )
    census = {"native": "original-native", "counts": {"TRAINING": 1, "EVALUATION": 1}}
    assert sources.native_candidates(path, req, census, training, support) == [row]
    with pytest.raises(ValueError, match="reservoir"):
        sources.native_candidates(path, req, census, [dict(training[0], source="changed")], support)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE draws SET ordinal=3 WHERE status='EVALUATION'")
    with pytest.raises(ValueError, match="unfinished"):
        sources.native_candidates(path, req, census, training, support)


def test_calibrator_never_receives_truth_or_the_witness_and_keeps_incomplete_methods(tmp_path):
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.synthesis.reference import evaluate

    class Client:
        limits = ResourceProfile()

        def fit(self, source, visible, **kw):
            assert len(visible.indices) == len(visible.values) == 20
            return {"admitted": [], "complete": False, "outcome": "resource_exhausted"}

        def execute(self, source, indices, **kw):
            result = evaluate(source, indices)
            return {"outcome": result.outcome, "values": list(result.values)}

    client = Client()
    source = "(program (return $index))"
    sample = {
        "id": "disjoint",
        "source": source,
        "indices": list(map(str, range(100))),
        "values": list(map(str, range(100))),
        "stratum": "composition",
        "native_verified": True,
        "operations": ["index"],
        "difficulty": {"depth": 1, "nodes": 1, "work": 100},
    }
    requests = []

    def native(role, data, **kw):
        requests.append(role)
        if role == "choose":
            assert set(data) == {"identity", "prompts"}
            assert all(
                set(r) == {"id", "indices", "values"} and len(r["values"]) == 20
                for r in data["prompts"]
            )
            return worker.calibration_choices(data, client)
        return worker.calibration_scores(data, client)

    rows = panels.calibrate({"calibration-only": sample}, tmp_path, native)
    assert requests == ["choose", "score"]
    assert rows[0]["exact"] == 0 and rows[0]["measured"] < 3
    with pytest.raises(ValueError, match="witness"):
        panels.calibrate(
            {"calibration-only": dict(sample, source="(program (return (int 0)))")},
            tmp_path,
            native,
        )


@pytest.mark.parametrize("deficit", [False, True])
def test_collector_standalone_and_repeated_publication_without_private_mounts(
    original, monkeypatch, deficit
):
    o = original
    value = recover(o.workspace, o.folder, SOURCE, REVISION, o.store)
    destination = o.workspace / "run-state/discovery-tools"
    old = destination / "round028-receipts-original" / "round028-rehearsal.zip"
    old.parent.mkdir(parents=True)
    publish(old, value, o.folder / ARCHIVE)
    o.store.append(dict(o.store.state, route="MATCHED_GENERATOR"))
    monkeypatch.setattr(panels, "PARENT", value["identity"])
    monkeypatch.setattr(panels, "original_costs", lambda *a: historical_clocks())
    monkeypatch.setattr(panels, "read_inputs", lambda *a: {})
    calls = []

    def prepare(workspace, area, inputs, registration, native):
        calls.append("cpu")
        aggregate = summary(deficit=deficit)
        members = {p: [] for p in RESERVATIONS}
        witnesses = {}
        for p, cells in aggregate["panels"].items():
            for stratum, cell in cells.items():
                for i in range(cell["cases"]):
                    key = f"{p}-{stratum}-{i}"
                    family = f"{p}-{stratum}-{i // 2}"
                    members[p].append(
                        {
                            "id": key,
                            "group": family,
                            "component": family,
                            "stratum": stratum,
                            "band": next(iter(cell["bands"])),
                        }
                    )
                    witnesses[key] = {"id": key}
        atomic_write(
            area / "supply.json",
            seal(
                {
                    "registration": registration["identity"],
                    "allocations": {"panels": members},
                    "witnesses": witnesses,
                }
            ),
        )
        atomic_write(
            area / "calibration-reservation.json",
            seal(
                {
                    "registration": registration["identity"],
                    "witnesses": {str(i): digest(i) for i in range(48)},
                }
            ),
        )
        if not deficit:
            for name in ("difficulty-map", "calibration-observations"):
                atomic_write(area / (name + ".json"), seal({"name": name}))
        return aggregate

    monkeypatch.setattr(panels, "prepare", prepare)
    rev = "c" * 40

    def stage():
        return panels.panels_command(
            None,
            SOURCE,
            o.workspace,
            o.folder,
            rev,
            o.store,
            lambda *a, **kw: pytest.fail("unexpected device stage"),
        )

    raw, exit_code = stage()
    assert exit_code == 0 and raw["gpu_dispatches"] == 0 and raw == stage()[0] and calls == ["cpu"]
    expected = "EXTEND_FRESH_SUPPLY" if deficit else "MEASURE_CANDIDATE_EVALUATION_COST"
    assert raw["reason"] == expected
    collector = runpy.run_path(
        str(SOURCE / "specs/015-learned-program-search/validation/collect_audit.py")
    )["collect"]
    run = subprocess.run

    def cli(args, **kw):
        if "panels" in args:
            kw["stdout"].write(encode(raw) + b"\n")
            return SimpleNamespace(returncode=0)
        return run(args, **kw)

    monkeypatch.setattr(subprocess, "run", cli)
    assert collector(o.workspace, rev, "collect-panels") == 0
    exported = next(destination.glob("round028-panels-*/round028-panels.zip"))
    receipt = verify_transport(exported)
    assert receipt == raw["receipt"]
    if deficit:
        assert receipt["deficits"]["replication/oeis"] == {"cases": 16, "families": 8}
    assert receipt["historical_forecast"]["endpoint_us"] == 528_000_000
    with zipfile.ZipFile(exported) as archive:
        record = json.loads(archive.read("panels.json"))
        parent = archive.read("round028-rehearsal.zip")
    record["historical_forecast"]["pilot_us"] += 1
    bad = destination / "changed.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("panels.json", encode(seal(record)))
        archive.writestr("round028-rehearsal.zip", parent)
    with pytest.raises(ValueError):
        verify_panels(bad)

    private_supply = o.folder / "private/panel-supply" / rev / "supply.json"
    with private_supply.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        stage()


def test_complete_frozen_source_chain_includes_original_corrected_and_legacy_reservations(tmp_path):
    from oeis_learn.generalization.progressive.contracts import record
    from oeis_learn.generalization.progressive.generator import structure_id
    from oeis_learn.generalization.progressive.prepared_panels import LIMITS
    from oeis_learn.generalization.progressive.prepared_support import transfer_support
    from oeis_learn.pipeline_validation.progressive_research_bridge import request as census_request
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex

    def save(path, value):
        atomic_write(tmp_path / path, value)
        return reference(tmp_path, path)

    idx = list(map(str, range(100)))

    def task(name, k):
        return {"id": name, "indices": idx, "values": [str(k + i * i) for i in range(100)]}

    empty = save("legacy/empty.json", [])
    old = save("legacy/old.json", [task("historical", 800)])
    manifest = save(
        "registration/manifest.json",
        seal(
            {
                "version": "research-input-resolution-review/v1",
                "executable": False,
                "registration_applied": False,
                "research_ready": False,
                "retained": {"selected_arm": "D", "checkpoints": {"B": {}, "D": {}}},
                "prior_inputs": {
                    "originals": [{"name": "held", "role": "assessment", "ref": old, "count": 1}],
                    "probes": empty,
                    "observed": empty,
                },
                "fresh_002": {"targets": empty},
            }
        ),
    )
    for name in ("legacy/history.sqlite", "orig/training-exposure.sqlite"):
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        with ExposureIndex(p) as index:
            if name.startswith("legacy"):
                index.add(task("historical", 800), origin="old", role="assessment")
    data = seal(
        {
            "version": "progressive-registered-data/v1",
            "status": "COMPLETE",
            "manifest": manifest,
            "exposure": reference(tmp_path, "legacy/history.sqlite"),
            "recipe": empty,
            "candidates": save(
                "registration/source.json",
                [task(n, k) for n, k in (("retired", 1), ("repaired", 2), ("free", 3))],
            ),
        }
    )
    req = census_request(data)
    reqref = save("orig/request.json", req)
    src = "(program (return (add $index (int 1))))"
    training = seal(
        dict(
            task("learned", 900),
            source=src,
            structure=structure_id(src),
            native_verified=True,
            draw_identity=digest(src),
        )
    )
    package = seal(
        {
            "version": "progressive-prepared-support/v1",
            "layout": {"probes": {}},
            "policy": digest("policy"),
            "shards": [[training]],
        }
    )
    _, binding = transfer_support(package)
    variant = src.replace("int 1", "int 7")
    candidate = seal(
        dict(
            task("mechanical", 9999),
            source=variant,
            structure=structure_id(variant),
            stratum="parameter",
            membership="evaluation_candidate",
            native_verified=True,
        )
    )
    with sqlite3.connect(tmp_path / "orig/census.sqlite") as db:
        db.executescript(
            "CREATE TABLE draws(ordinal PRIMARY KEY,status,row); CREATE TABLE binding(value); CREATE TABLE native(value);"
        )
        db.execute("INSERT INTO binding VALUES (?)", (json.dumps(req),))
        db.execute("INSERT INTO native VALUES ('verified')")
        total = req["training_draws"] + req["evaluation_draws"]
        db.executemany("INSERT INTO draws VALUES (?,'REJECTED',NULL)", ((i,) for i in range(total)))
        db.execute(
            "UPDATE draws SET status='TRAINING',row=? WHERE ordinal=0", (json.dumps(training),)
        )
        db.execute(
            "UPDATE draws SET status='EVALUATION',row=? WHERE ordinal=?",
            (json.dumps(candidate), req["training_draws"]),
        )
    save("orig/training-reservoir.json", [training])

    def rosters(m, k):
        out = {p: [] for p in req["panels"] + req["retention_panels"]}
        out["assessment"] = [
            dict(task(m, k), stratum="oeis", group="assessment", role="assessment")
        ]
        return out

    roster = rosters("retired", 1)
    save("orig/panel-reservations.json", roster)
    corrected = rosters("repaired", 2)
    save("repaired/panel-reservations.json", corrected)
    sizes = {
        p + "/" + s: {"tasks": sum(r["stratum"] == s for r in rs)}
        for p, rs in roster.items()
        for s in (("parameter",) if p in req["retention_panels"] else STRATA)
    }
    names = (
        "training-reservoir.json",
        "census.sqlite",
        "training-exposure.sqlite",
        "panel-reservations.json",
    )
    original_census = seal(
        {
            "version": "progressive-research-census/v1",
            "diagnostic": False,
            "request": req["identity"],
            "native": "verified",
            "reserved_panel_rows": 1,
            "panels": sizes,
            "counts": {"TRAINING": 1, "EVALUATION": 1, "REJECTED": total - 2},
            "transfer_support": binding,
            "artifacts": [reference(tmp_path / "orig", n) for n in names],
        }
    )
    original_ref = save("orig/census.json", original_census)
    cens = seal(
        dict(
            original_census,
            panel_revision={"original": original_ref},
            artifacts=[
                reference(
                    tmp_path, ("repaired/" if n == "panel-reservations.json" else "orig/") + n
                )
                for n in names
            ],
        )
    )
    censref = save("repaired-census.json", cens)
    exposure = record(
        "exposure",
        {
            "previous": None,
            "detector_limits": LIMITS,
            "events": [],
            "indexes": [data["exposure"], reference(tmp_path, "orig/training-exposure.sqlite")],
        },
        created_at="2026-11-11T00:00:00Z",
        purpose="research",
        track="synthetic_strict",
        parents=[cens["identity"]],
    )
    eref = save("materials/private/exposure.json", exposure)
    m = seal(
        {
            "version": "progressive-research-materials/v1",
            "diagnostic": False,
            "status": "COMPLETE",
            "exposure": eref,
            "native": "verified",
            "census": censref,
            "request": reqref,
            "support": save("materials/private/support.json", package),
        }
    )
    save("materials/materials.json", m)
    refs = {
        n: {"role": "custodian", "kind": k, "artifact": a}
        for n, k, a in (
            ("exposure", "groups", eref),
            ("registered-research-data", "source", save("frozen/data.json", data)),
        )
    }
    result = sources.read_inputs(tmp_path, refs)
    assert {r["id"] for r in result["historical"]} == {
        "retired",
        "repaired",
        "learned",
        "historical",
    }
    groups, counts, _, _ = panels.preallocate(result, {"seed": 1, "calibration_limit": 48})
    assert counts["eligible"] == 2
    assert {r["id"] for members in groups.values() for r in members} == {"mechanical", "free"}
    with (tmp_path / "orig/panel-reservations.json").open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError):
        sources.read_inputs(tmp_path, refs)


def test_cpu_role_resumes_after_publication_interruption_and_checks_original_bytes(
    tmp_path, monkeypatch
):
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.pipeline_validation import progressive_handoff

    image = digest("test-image")
    host = SimpleNamespace(
        engine=SimpleNamespace(name="podman", image=lambda v: v),
        policy={"free_disk_floor_bytes": 0},
    )
    monkeypatch.setattr(panels, "candidate_image", lambda *args: (image, 4, True))
    dispatched = []

    def invoke(host, source, image, output, argv, *, gpu, seconds, readonly_inputs):
        assert gpu is False and len(readonly_inputs) == 2
        assert [p for _, p in readonly_inputs] == [
            "/work/inputs/request.json",
            "/work/inputs/data.json",
        ]
        req, data = [load(p) for p, _ in readonly_inputs]
        assert req["role"] == "choose" and req["data"] == data["identity"]
        output.mkdir()
        atomic_write(
            output / "producer.json",
            seal(
                {
                    "version": "round028-panel-output/v1",
                    "request": req["identity"],
                    "native": digest("current native"),
                    "result": {},
                }
            ),
        )
        mounts = [(p, d, False) for p, d in readonly_inputs] + [(output, "/work/output", True)]
        atomic_write(
            output / "container-inspect.json",
            [
                {
                    "Image": image,
                    "HostConfig": {
                        "NetworkMode": "none",
                        "ReadonlyRootfs": True,
                        "Privileged": False,
                    },
                    "Mounts": [
                        {"Type": "bind", "Source": str(p), "Destination": d, "RW": w}
                        for p, d, w in mounts
                    ],
                }
            ],
        )
        dispatched.append(1)

    monkeypatch.setattr(progressive_handoff, "stage", invoke)
    reg = seal({"source": digest("test-source")})

    def launch():
        native = panels.NativeRoles(
            None, SOURCE, tmp_path, "d" * 40, tmp_path, reg, lambda *a, **k: host
        )
        result = native("choose", seal({"prompts": []}), seconds=5)
        assert len(native.charges) == 1 and native.charges[0] > 0
        return result

    assert launch() == {}
    (boundary,) = tmp_path.glob("roles/*/boundary.json")
    boundary.unlink()
    assert launch() == {} and dispatched == [1]
    (producer,) = tmp_path.glob("roles/*/attempt-*/output/producer.json")
    with producer.open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError):
        launch()


@pytest.mark.parametrize("short", [False, True])
def test_prepare_binds_private_members_and_stops_before_calibration_on_a_deficit(
    tmp_path, monkeypatch, short
):
    from oeis_learn.generalization.progressive.program_search_panels import (
        allocate_roles,
        calibration_families,
    )

    groups = {}
    for stratum in STRATA:
        if short and stratum == "oeis":
            continue
        for i in range(100):
            family = digest([stratum, i])
            groups[family] = [
                {
                    "id": f"{stratum}-{i}-{j}",
                    "stratum": stratum,
                    "band": "unknown",
                    "native_verified": True,
                    "source": "(program (return $index))",
                    "values": list(map(str, range(100))),
                    "indices": list(map(str, range(100))),
                    "operations": ["read"],
                    "difficulty": {"nodes": 1, "depth": 1, "work": 100},
                }
                for j in range(2)
            ]
    samples = calibration_families(groups, 7, count=4)
    provisional = allocate_roles({k: v for k, v in groups.items() if k not in samples}, seed=7)
    n = sum(map(len, groups.values()))
    counts = {
        "candidates": n,
        "eligible": n,
        "excluded": 0,
        "components": len(groups),
        "historical_rows": 0,
    }
    monkeypatch.setattr(panels, "preallocate", lambda *args: (groups, counts, samples, provisional))
    monkeypatch.setattr(panels, "extra_source", lambda *args: ([], {"selected": 0, "remaining": 0}))
    roles = []

    def native(role, data, **kw):
        roles.append(role)
        if role == "choose":
            return {r["id"]: {} for r in data["prompts"]}
        assert set(data["choices"]) == {r["id"] for r in data["truth"]}
        return {
            r["id"]: {m: {"exact": False, "measured": True} for m in worker.METHODS}
            for r in data["truth"]
        }

    reg = seal({"seed": 7, "calibration_limit": 4})
    inputs = {
        "candidates": [],
        "data": {},
        "counts": {
            "original_reservations": 0,
            "corrected_reservations": 0,
            "native_training": 0,
            "older_exposure_rows": 0,
        },
    }
    result = panels.prepare(tmp_path, tmp_path, inputs, reg, native)
    assert roles == ([] if short else ["choose", "score"])
    assert result["calibrated"] is (not short)
    assert result["panels"]["replication"]["oeis"]["cases"] == (0 if short else 32)
    private = load(tmp_path / "supply.json", maximum=5 << 20)
    assert seal(private) == private and private["registration"] == reg["identity"]
    membership = [r for rs in private["allocations"]["panels"].values() for r in rs]
    assert len(membership) == len(private["witnesses"])
    assert set(private["witnesses"]) == {r["id"] for r in membership}
    assert not set(samples).intersection(r["component"] for r in membership)
