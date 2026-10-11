"""CPU contract probes never claim an actual ready workstation."""

from types import SimpleNamespace
import json

import pytest

from oeis_learn.cli.foundation_preflight import PreflightConfig, main
from oeis_learn.evaluation.baseline_gate import validate_preflight
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest, compute_file_hash


def probe_fixture(tmp_path):
    for name in ("Dockerfile", "lock"):
        (tmp_path / name).write_text(name)
    config = dict(
        schema_version="preflight/v1",
        kind="preflight_config",
        base_image="rocm/pytorch:fixture",
        base_image_digest=digest("base"),
        final_image_id=digest("built"),
        dockerfile="Dockerfile",
        dependency_lock="lock",
        requested_device="hip",
        dtype="float32",
        updates=3,
        optimizer="adamw",
        visible_terms=20,
        body_tokens=1024,
        timeout_minutes=10,
        fixtures=[
            dict(kind=k)
            for k in ("zero", "alternating_signed256_extremes", "adjacent_large_integer_prefixes")
        ],
        model=dict(
            d_model=256, n_heads=4, n_encoder_layers=4, n_decoder_layers=4, d_ff=1024, dropout=0.1
        ),
    )
    cfg = PreflightConfig(config, str(tmp_path / "config.json"))
    env = dict(
        python="3.12.12",
        machine="x86_64",
        kfd=dict(exists=True),
        any_render_node=True,
        hip=dict(
            available=True,
            torch_version="2.9.1",
            hip_version="7.2.1",
            cuda_available=True,
            device_count=1,
            devices=[dict(name="AMD fixture", total_memory=128 << 30)],
        ),
    )
    runtime = dict(
        python=env["python"],
        machine=env["machine"],
        torch="2.9.1",
        hip="7.2.1",
        device="cuda",
        device_name="AMD fixture",
    )
    report = dict(
        schema_version="preflight-report/v2",
        status="ready",
        gate="passed",
        requested_device="hip",
        dtype="float32",
        visible_terms=20,
        body_tokens=1024,
        updates=3,
        no_cpu_fallback=True,
        no_driver_install=True,
        no_privileged_container=True,
        hip_version="7.2.1",
        device="cuda:0",
        device_name="AMD fixture",
        elapsed_total_s=1,
        peak_memory_bytes_after=1024,
        environment_report=env,
        runtime=runtime,
        effective_config=cfg.to_dict(),
        config_digest=digest(cfg.to_dict()),
        base_image_digest=config["base_image_digest"],
        final_image_id=config["final_image_id"],
        dependency_lock_sha256=compute_file_hash(tmp_path / "lock"),
        dockerfile_sha256=compute_file_hash(tmp_path / "Dockerfile"),
    )
    report["steps"] = [
        dict(
            update=i + 1,
            visible_terms=20,
            body_tokens=1024,
            dtype="float32",
            loss_finite=True,
            grad_finite=True,
            params_changed=1,
            max_abs_param_change=0.1,
            loss=1,
            device="cuda:0",
            device_name="AMD fixture",
        )
        for i in range(3)
    ]
    report["evidence_sha256"] = digest(report)
    provenance = dict(
        base_image_digest=report["base_image_digest"],
        final_image_id=report["final_image_id"],
        dependency_lock=report["dependency_lock_sha256"],
        environment_report=digest(env),
    )
    checkpoint = SimpleNamespace(runtime=runtime, contract=dict(run_provenance=provenance))
    context = dict(
        config=cfg,
        checkpoint=checkpoint,
        containment=dict(image=report["final_image_id"]),
        dependency_lock_sha256=report["dependency_lock_sha256"],
        dockerfile_sha256=report["dockerfile_sha256"],
    )
    return config, report, context


def test_preflight_identity_is_bound_to_selected_inputs(tmp_path):
    _, report, context = probe_fixture(tmp_path)
    validate_preflight(report, **context)


@pytest.mark.parametrize(
    "field",
    [
        "device_name",
        "config_digest",
        "environment_report",
        "base_image_digest",
        "final_image_id",
        "effective_config",
        "runtime",
        "dependency_lock_sha256",
        "dockerfile_sha256",
    ],
)
def test_missing_preflight_identity_is_rejected(tmp_path, field):
    _, report, context = probe_fixture(tmp_path)
    report.pop(field)
    report["evidence_sha256"] = digest(report, "evidence_sha256")
    with pytest.raises(ValueError, match="identity"):
        validate_preflight(report, **context)


@pytest.mark.parametrize(
    "field",
    [
        "base_image_digest",
        "final_image_id",
        "dependency_lock_sha256",
        "dockerfile_sha256",
        "config_digest",
        "device_name",
        "hip_version",
        "runtime",
        "environment_report",
        "effective_config",
        "step_device",
    ],
)
def test_rehashed_foreign_preflight_identity_is_rejected(tmp_path, field):
    _, report, context = probe_fixture(tmp_path)
    if field == "runtime":
        report[field]["torch"] = "foreign"
    elif field == "environment_report":
        report[field]["hip"]["devices"][0]["name"] = "foreign"
    elif field == "effective_config":
        report[field]["model"]["seed"] += 1
        report["config_digest"] = digest(report[field])
    elif field == "step_device":
        report["steps"][0]["device"] = "cuda:1"
    else:
        report[field] = (
            digest("foreign") if field.endswith(("digest", "id", "sha256")) else "foreign"
        )
    report["evidence_sha256"] = digest(report, "evidence_sha256")
    with pytest.raises(ValueError):
        validate_preflight(report, **context)


def test_emitter_uses_effective_config_and_real_file_identities(tmp_path, monkeypatch):
    import oeis_learn.cli.foundation_preflight as probe

    config, expected, _ = probe_fixture(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(probe, "inspect_host", lambda: expected["environment_report"])
    monkeypatch.setattr(
        probe,
        "run_updates",
        lambda *a: {
            k: expected[k]
            for k in (
                "device",
                "device_name",
                "hip_version",
                "steps",
                "elapsed_total_s",
                "peak_memory_bytes_after",
            )
        },
    )
    assert main(["--config", str(path), "--output", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out/preflight-report.json").read_text())
    assert report["schema_version"] == "preflight-report/v2"
    assert report["config_digest"] == digest(report["effective_config"])
    assert report["config_digest"] != report["base_image_digest"]
    assert report["dependency_lock_sha256"] == compute_file_hash(tmp_path / "lock")
    assert report["evidence_sha256"] == digest(report, "evidence_sha256")


def test_baseline_replays_selected_preflight_inputs_before_other_gates(tmp_path, monkeypatch):
    import oeis_learn.evaluation.baseline_gate as gate
    from oeis_learn.evaluation.foundation_cohort import write_json
    from oeis_learn.experiments.profiles import profile_digests

    config, report, context = probe_fixture(tmp_path)
    checkpoint = context["checkpoint"]
    checkpoint.manifest = dict(
        blob_sha256=digest("weights"), runtime_sha256=digest(checkpoint.runtime)
    )
    checkpoint.contract["run_id"] = "selected-run"
    checkpoint.contract["profiles"] = profile_digests()
    identity = digest("code")
    monkeypatch.setattr(gate, "implementation_identity", lambda: identity)
    config["final_image_id"] = digest("foreign-built-image")
    (tmp_path / "selected.yaml").write_text(json.dumps(config))
    g0 = dict(
        preflight=write_json(tmp_path, "preflight.json", report),
        preflight_config=dict(
            path="selected.yaml", sha256=compute_file_hash(tmp_path / "selected.yaml")
        ),
        dependency_lock=dict(path="lock", sha256=compute_file_hash(tmp_path / "lock")),
        dockerfile=dict(path="Dockerfile", sha256=compute_file_hash(tmp_path / "Dockerfile")),
        adoption={},
    )
    checks = {f"G{i}": {} for i in range(7)}
    checks["G0"] = g0
    checks["G2"]["containment"] = write_json(tmp_path, "containment.json", context["containment"])
    bundle = dict(
        version="baseline-gates/v1",
        binding=dict(
            checkpoint_sha256=checkpoint.manifest["blob_sha256"],
            contract_sha256=digest(checkpoint.contract),
            run_id="selected-run",
            code_sha256=identity,
            profiles=profile_digests(),
            runtime_sha256=checkpoint.manifest["runtime_sha256"],
        ),
        checks=checks,
    )
    bundle["report_sha256"] = digest(bundle)
    with pytest.raises(ValueError, match="identity mismatch"):
        gate.require_baseline(bundle, checkpoint, evidence_root=tmp_path)
