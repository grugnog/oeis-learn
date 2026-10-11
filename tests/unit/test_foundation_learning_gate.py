"""Negative gate cases never infer accelerator readiness from a CPU pass."""

import pytest
from oeis_learn.evaluation.learning_gate import require_g8, check_junit


def test_incomplete_and_skipped_learning_evidence_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="complete"):
        require_g8({"gate": "G8"}, evidence_root=tmp_path, diagnostic=True)
    empty = tmp_path / "empty.xml"
    empty.write_text("<testsuite/>")
    with pytest.raises(ValueError, match="empty"):
        check_junit(empty)
    skipped = tmp_path / "skipped.xml"
    skipped.write_text("<testsuite><testcase><skipped/></testcase></testsuite>")
    with pytest.raises(ValueError, match="skipped"):
        check_junit(skipped)
    incomplete = tmp_path / "incomplete.xml"
    incomplete.write_text('<testsuite><testcase classname="fake"/></testsuite>')
    with pytest.raises(ValueError, match="coverage"):
        check_junit(incomplete, "cache")


def test_accelerator_evidence_cannot_omit_variants_or_use_a_tiny_proxy(tmp_path):
    from oeis_learn.evaluation.learning_gate import check_hardware_junit

    path = tmp_path / "gpu.xml"
    path.write_text(
        '<testsuite><testcase classname="tests.integration.test_foundation_learning_gpu" '
        'name="test_gpu_fp16_scaler_finite_optimizer_update[sinusoidal]"/></testsuite>'
    )
    with pytest.raises(ValueError, match="coverage"):
        check_hardware_junit(path)


@pytest.mark.parametrize(
    "mode",
    [
        "exit",
        "missing",
        "unreadable",
        "malformed",
        "incomplete",
        "skipped",
        "launch",
        "pass",
        "unrequested",
    ],
)
def test_hardware_execution_status_and_durable_diagnostics(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace
    import oeis_learn.evaluation.learning_gate as gate
    from oeis_learn.experiments.artifacts import load_json
    from oeis_learn.evaluation.readiness import release_readiness

    monkeypatch.setattr(gate, "SUITES", {})
    monkeypatch.setattr(gate, "implementation_identity", lambda: "sha256:" + "1" * 64)
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        path = tmp_path / "gpu.xml"
        if mode == "launch":
            raise OSError("cannot launch pytest")
        if mode == "unreadable":
            path.mkdir()
        elif mode != "missing":
            names = [
                f"test_gpu_fp32_cached_full_prefix[{n}-{p}]"
                for n in (1, 2, 513, 1024)
                for p in ("sinusoidal", "rope", "prefix_ancestor")
            ]
            names += [
                f"test_gpu_fp16_scaler_finite_optimizer_update[{p}]"
                for p in ("sinusoidal", "rope", "prefix_ancestor")
            ]
            cases = "".join(
                f'<testcase classname="tests.integration.test_foundation_learning_gpu" name="{n}">'
                + ("<skipped/>" if mode == "skipped" else "")
                + "</testcase>"
                for n in (names[:1] if mode == "incomplete" else names)
            )
            path.write_text(
                "<broken" if mode == "malformed" else "<testsuite>" + cases + "</testsuite>"
            )
        return SimpleNamespace(
            returncode=7 if mode == "exit" else 0, stdout="probe output", stderr="probe diagnostics"
        )

    monkeypatch.setattr(gate.subprocess, "run", execute)
    report = gate.run_qualification(tmp_path, hardware=mode != "unrequested")
    assert load_json((tmp_path / "g8.json").read_bytes()) == report
    expected = "PENDING" if mode == "unrequested" else "PASS" if mode == "pass" else "FAIL"
    assert report["hardware_status"] == expected
    if expected == "PENDING":
        assert report["hardware"] is None and not calls
    else:
        process = load_json((tmp_path / report["hardware"]["process"]["path"]).read_bytes())
        assert process["stdout"] == ("probe output" if mode != "launch" else "")
        assert process["returncode"] == (None if mode == "launch" else 7 if mode == "exit" else 0)
        assert bool(process["error"]) == (expected == "FAIL")
    if expected == "FAIL":
        with pytest.raises(ValueError, match="failed"):
            gate.require_g8(report, evidence_root=tmp_path, diagnostic=True)
        assert release_readiness(g8=(report, tmp_path))["gates"]["G8"]["status"] == "FAIL"


def test_failure_process_and_junit_are_both_in_transport_closure(tmp_path):
    from scripts.archive_phase11 import evidence_inputs
    from tests.unit.test_foundation_handoff import ci_evidence
    from oeis_learn.experiments.artifacts import canonical_bytes, compute_file_hash

    readiness = ci_evidence(tmp_path)
    root = tmp_path / "g8"
    (root / "gpu-process.json").write_text('{"error":"hardware exited 7"}')
    (root / "gpu.xml").write_text("<testsuite><testcase><failure/></testcase></testsuite>")
    report = dict(
        checks=dict(owning=dict(path="owning.xml")),
        hardware=dict(
            process=dict(
                path="gpu-process.json", sha256=compute_file_hash(root / "gpu-process.json")
            ),
            junit=dict(path="gpu.xml", sha256=compute_file_hash(root / "gpu.xml"), cases=None),
        ),
    )
    (root / "gate.json").write_bytes(canonical_bytes(report))
    paths = {r["path"] for r in evidence_inputs(tmp_path, readiness)["files"]}
    assert {"g8/gpu-process.json", "g8/gpu.xml"} <= paths
