"""Executable G3: strict generic bootstrap, replay and learner isolation."""

import json
import pytest
import yaml

from oeis_learn.data.generic_programs import default_config
from oeis_learn.data.program_pool import build_pool, load_pool, load_trainer_view
from oeis_learn.experiments.artifacts import load_json, compute_canonical_digest as digest
from oeis_learn.evaluation.foundation_cohort import freeze_cohort


def setup_inputs(root, **changes):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    source.mkdir()
    # Source-only holdout; neither its identity nor its terms enter the sampler.
    record = {
        "record_id": "fixture:private",
        "first_index": 4,
        "indices": list(range(4, 104)),
        "values": [str(1000003 + i * 7919) for i in range(100)],
        "metadata": {"name": "private sentinel"},
    }
    (source / "records.json").write_text(json.dumps([record]))
    cfg = root / "cohort.yaml"
    cfg.write_text(
        yaml.safe_dump(
            dict(
                schema_version="foundation/v1",
                profile="prefix20_total100_v1",
                seed=1,
                dev_count=1,
                final_count=0,
            )
        )
    )
    cohort = root / "cohort"
    freeze_cohort(source, cfg, cohort)
    config = default_config()
    config.update(target_count=4, max_attempts=256, workers=2)
    config.update(changes)
    cfg = root / "sampler.yaml"
    cfg.write_text(yaml.safe_dump(config))
    return cfg, cohort


def test_pool_and_trainer_view_are_complete_strict_and_separate(tmp_path):
    config, cohort = setup_inputs(tmp_path / "inputs")
    out = tmp_path / "pool"
    report = build_pool(config, cohort, out)
    assert report["status"] == "complete", report
    loaded = load_pool(out, expected_pool_id=report["pool_id"])
    assert len(loaded.examples) == report["published"] == 4
    # Independently check every selected output class against all admitted
    # candidates in the journal, including later shorter replacements.
    from oeis_learn.evaluation.foundation_cohort import read_ref

    archive = {}
    for path in (out / "programs").glob("*.json"):
        record = load_json(path.read_bytes())
        values = read_ref(out, record["outputs_ref"])["values"]
        archive.setdefault(digest(values), []).append(record)
    manifest = load_json((out / "manifest.json").read_bytes())
    counters = []
    for entry in manifest["entries"]:
        record = read_ref(out, entry["program"])
        best = min(
            archive[entry["output_fingerprint"]],
            key=lambda r: (len(r["body_tokens"]), r["source_sha256"], r["sample_counter"]),
        )
        assert record["program_id"] == best["program_id"]
        counters.append(record["sample_counter"])
    assert counters == sorted(counters)
    assert load_trainer_view(out / "trainer", expected_view_id=loaded.view_id) == loaded.examples
    assert build_pool(config, cohort, out) == report
    for example in loaded.examples:
        assert set(example) == {"program_id", "codec_profile", "visible_terms", "body_tokens"}
        assert len(example["visible_terms"]) == 20
    assert "fixture:private" not in (out / "trainer/examples.json").read_text()
    with pytest.raises(ValueError):
        load_pool(out, expected_pool_id="sha256:" + "0" * 64)
    with pytest.raises(FileNotFoundError):
        load_pool(tmp_path / "missing", expected_pool_id=report["pool_id"])
    (out / "trainer/examples.json").write_text("[]")
    with pytest.raises(ValueError):
        load_pool(out, expected_pool_id=report["pool_id"])


@pytest.mark.parametrize(
    "event", ["after_attempt", "after_chunk", "before_publish", "after_publish"]
)
def test_builder_crash_resume_keeps_counter_rng_order_and_dedup(tmp_path, event):
    config, cohort = setup_inputs(tmp_path / "inputs", target_count=2, max_attempts=64)
    baseline = build_pool(config, cohort, tmp_path / "baseline")
    assert baseline["status"] == "complete", baseline
    baseline_pool = load_pool(tmp_path / "baseline", expected_pool_id=baseline["pool_id"])

    def crash(where, root):
        if where == event:
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        build_pool(config, cohort, tmp_path / "resumed", _hook=crash)
    resumed = build_pool(config, cohort, tmp_path / "resumed")
    restored = load_pool(tmp_path / "resumed", expected_pool_id=resumed["pool_id"])
    assert restored.examples == baseline_pool.examples
    assert resumed["summary"] == baseline["summary"]
    assert build_pool(config, cohort, tmp_path / "resumed") == resumed
    # Raw elapsed/RSS evidence can differ across runs; program identity/order cannot.
    assert [x["program_id"] for x in restored.examples] == [
        x["program_id"] for x in baseline_pool.examples
    ]


def test_bounded_shortfall_never_publishes_pool_or_adds_teachers(tmp_path):
    config, cohort = setup_inputs(tmp_path / "inputs", target_count=64, max_attempts=1)
    out = tmp_path / "out"
    report = build_pool(config, cohort, out)
    assert report["status"] == "shortfall" and report["pool_id"] is None
    assert report["summary"]["attempted"] == 1 and report["published"] == 0
    assert not (out / "manifest.json").exists()
    assert report["summary"]["admitted"] + report["summary"]["rejected"] == 1


def test_pool_cli_exact_inputs_exit_codes_and_report(tmp_path, capsys):
    from oeis_learn.cli.main import cli

    config, cohort = setup_inputs(tmp_path / "inputs", target_count=64, max_attempts=1)
    args = [
        "foundation",
        "build-pool",
        "--config",
        str(config),
        "--cohort",
        str(cohort),
        "--output",
        str(tmp_path / "out"),
        "--json",
    ]
    assert cli(args) == 4
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "shortfall" and not report["qualified"]
    args[3] = str(tmp_path / "missing.yaml")
    assert cli(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_input"


@pytest.mark.parametrize("crash_point", ["after_chunk", "after_attempt"])
def test_expired_crash_budget_cannot_restart_sampling(tmp_path, monkeypatch, crash_point):
    from oeis_learn.data import program_pool

    config, cohort = setup_inputs(tmp_path / "inputs", target_count=64)
    out = tmp_path / "out"

    def crash(event, root):
        if event == crash_point:
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        build_pool(config, cohort, out, _hook=crash)
    start = load_json((out / "budget-start.json").read_bytes())
    monkeypatch.setattr(
        program_pool,
        "_clock",
        lambda: {
            **start,
            "monotonic_ns": start["monotonic_ns"] + 301_000_000_000,
            "utc_ns": start["utc_ns"] + 301_000_000_000,
        },
    )

    class ForbiddenWorkers:
        def __init__(self, *args, **kwargs):
            raise AssertionError("new execution after exhausted budget")

    monkeypatch.setattr(program_pool, "WorkerPool", ForbiddenWorkers)
    report = build_pool(config, cohort, out)
    assert report["reason"] == "wall_budget_exhausted"
    assert report["summary"]["attempted"] == 8 and not (out / "manifest.json").exists()
    summary = report["summary"]
    assert summary["pending_attempts"] == (7 if crash_point == "after_attempt" else 0)
    assert summary["admitted"] + summary["rejected"] + summary["pending_attempts"] == 8


def test_malformed_sampler_yaml_is_cli_input_error(tmp_path, capsys):
    from oeis_learn.cli.main import cli

    config = tmp_path / "invalid.yaml"
    config.write_text("schema_version: [unterminated")
    code = cli(
        [
            "foundation",
            "build-pool",
            "--config",
            str(config),
            "--cohort",
            "unused",
            "--output",
            str(tmp_path / "out"),
            "--json",
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_input"
