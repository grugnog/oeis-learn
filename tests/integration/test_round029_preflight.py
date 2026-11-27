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
