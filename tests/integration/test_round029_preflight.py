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
    assert changed["action"] == "new_inference" and changed["continuation"]
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
