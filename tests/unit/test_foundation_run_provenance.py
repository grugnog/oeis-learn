"""Run provenance and real CLI transport boundaries, without GPU substitution."""

from types import SimpleNamespace
import json
import pytest
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest


def pins():
    return {
        k: "sha256:" + c * 64
        for k, c in zip(
            (
                "source_revision",
                "dependency_lock",
                "base_image_digest",
                "final_image_id",
                "environment_report",
            ),
            "abcde",
        )
    }


def setup_pool(tmp_path, monkeypatch):
    import oeis_learn.rl.foundation_sft as service

    pool = tmp_path / "pool"
    (pool / "trainer").mkdir(parents=True)
    for p, v in (
        (pool / "manifest.json", {"pool_id": "pool"}),
        (pool / "trainer/manifest.json", {}),
        (pool / "trainer/examples.json", {}),
    ):
        p.write_text(json.dumps(v))
    monkeypatch.setattr(service, "runtime_record", lambda device: {"device": device})
    monkeypatch.setattr(
        service,
        "load_pool",
        lambda *a, **k: SimpleNamespace(
            examples=[{"program_id": str(n)} for n in range(4)], pool_id="pool", view_id="view"
        ),
    )
    return service, pool


def test_provenance_bound_before_contract_and_cannot_override_data(tmp_path, monkeypatch):
    service, pool = setup_pool(tmp_path, monkeypatch)
    run = tmp_path / "run"
    result = service.prepare_run(
        "configs/foundation/wat_smoke.yaml",
        pool,
        run,
        device="cpu",
        diagnostic=True,
        run_provenance=pins(),
    )
    contract = json.loads((run / "checkpoints/contract.json").read_text())
    assert all(contract["run_provenance"][k] == v for k, v in pins().items())
    assert (
        contract["run_provenance"]["dataset_manifest"]
        and contract["run_provenance"]["effective_config"]
    )
    assert contract["contract_id"] == digest(contract, "contract_id")
    assert result["contract_sha256"] == digest(contract)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(extra="sha256:" + "f" * 64),
        lambda p: p.pop("environment_report"),
        lambda p: p.update(source_revision="main"),
    ],
)
def test_provenance_shape_and_hashes_closed(tmp_path, monkeypatch, change):
    service, pool = setup_pool(tmp_path, monkeypatch)
    p = pins()
    change(p)
    with pytest.raises(ValueError, match="provenance"):
        service.prepare_run(
            "configs/foundation/wat_smoke.yaml",
            pool,
            tmp_path / "run",
            device="cpu",
            diagnostic=True,
            run_provenance=p,
        )


@pytest.mark.parametrize(
    "argv,required",
    [
        (
            [
                "coordinated-smoke",
                "--config",
                "c",
                "--pool",
                "p",
                "--image",
                "i",
                "--output",
                "o",
                "--seconds",
                "20",
                "--resume-run",
                "r",
                "--run-provenance",
                "b",
            ],
            {"image": "i", "seconds": 20, "resume_run": "r", "run_provenance": "b"},
        ),
        (
            ["diagnose-overflow", "--strata", "s", "--output", "o", "--provenance", "p"],
            {"strata": "s", "provenance": "p"},
        ),
        (
            [
                "prepare-measurement",
                "--archive",
                "a",
                "--cohort",
                "c",
                "--gate-report",
                "g",
                "--learning-gate-report",
                "l",
                "--image",
                "i",
                "--output",
                "o",
                "--baseline-report",
                "b",
                "--baseline-checkpoint",
                "p",
            ],
            {"image": "i", "baseline_report": "b", "baseline_checkpoint": "p"},
        ),
    ],
)
def test_cli_forwards_every_selected_argument(monkeypatch, argv, required):
    import oeis_learn.cli.main as main
    import oeis_learn.cli.foundation as boundary

    observed = {}
    monkeypatch.setattr(boundary, "dispatch", lambda command, **kw: observed.update(kw) or 0)
    args = main.build_parser().parse_args(["foundation", *argv])
    assert main.handle_foundation(args) == 0
    assert all(observed.get(k) == v for k, v in required.items())


def test_coordinated_resume_rejects_configuration_override(tmp_path):
    from oeis_learn.evaluation.coordinated_smoke import coordinated_smoke

    previous = tmp_path / "previous"
    (previous / "checkpoints").mkdir(parents=True)
    (previous / "run.json").write_text("{}")
    (previous / "checkpoints/config.yaml").write_text("original")
    config = tmp_path / "config.yaml"
    config.write_text("changed")
    with pytest.raises(ValueError, match="override"):
        coordinated_smoke(
            config, tmp_path / "pool", "sha256:" + "a" * 64, tmp_path / "new", resume_run=previous
        )


def test_coordinated_image_provenance_must_match(tmp_path):
    from oeis_learn.evaluation.coordinated_smoke import coordinated_smoke

    with pytest.raises(ValueError, match="selected role image"):
        coordinated_smoke("c", "p", "sha256:" + "a" * 64, tmp_path / "new", run_provenance=pins())


def test_partial_role_reports_actual_completed_updates(tmp_path, monkeypatch):
    import oeis_learn.rl.foundation_sft as prepare
    import oeis_learn.tracking.foundation_controller as controller
    from oeis_learn.tracking.role_worker import smoke

    inputs = tmp_path / "input"
    output = tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    (inputs / "settings.json").write_text(
        json.dumps(dict(resume=False, seconds=600, run_provenance=pins(), stop_after=1))
    )
    observed = {}

    def fake_prepare(config, pool, run, **kw):
        observed.update(kw)
        run.mkdir()
        (run / "run.json").write_text("{}")

    monkeypatch.setattr(prepare, "prepare_run", fake_prepare)

    def fake_train(run, **kw):
        assert kw["stop_after"] == 1
        return dict(
            completed_update=1,
            run_id="actual",
            checkpoint_sha256="sha256:" + "a" * 64,
            checkpoint=str(run / "checkpoint-00000001.json"),
        )

    monkeypatch.setattr(controller, "run_training", fake_train)
    result = smoke(inputs, output)
    assert result["status"] == "paused" and result["completed_updates"] == 1
    assert observed["run_provenance"] == pins()
