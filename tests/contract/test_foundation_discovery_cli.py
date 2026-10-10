import json
import subprocess
import sys
from oeis_learn.discovery.certificates import make_certificate
from oeis_learn.sandbox.logical_ir import profile_identity


def test_public_fresh_certificate_cli(tmp_path):
    a = "(pure (return (add (index) (const 0))))"
    b = "(pure (return (index)))"
    certificate = make_certificate("polynomial_program", [a, b], [1, -1])
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(certificate))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(profile_identity("pure")))
    argv = [
        sys.executable,
        "-m",
        "oeis_learn.cli.main",
        "foundation",
        "check-certificate",
        "--certificate",
        str(path),
        "--profile",
        str(profile),
        "--output",
        str(tmp_path / "out"),
        "--json",
    ]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["result"] == "verified"
    assert (
        json.loads((tmp_path / "out" / "certificate-check.json").read_text())["wasm_theorem"]
        is False
    )
    certificate["statement"]["coefficients"][1] = "-2"
    path.write_text(json.dumps(certificate))
    failed = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    assert failed.returncode != 0


def test_discovery_missing_gate_rejected(tmp_path):
    report = tmp_path / "g9.json"
    report.write_text("{}")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "oeis_learn.cli.main",
            "foundation",
            "discover",
            "--archive",
            str(tmp_path / "archive"),
            "--discovery-gate-report",
            str(report),
            "--output",
            str(tmp_path / "out"),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert not (tmp_path / "archive").exists()
