"""Real filesystem and CLI boundaries; diagnostics never masquerade as runs."""

import copy
import json
from pathlib import Path

import pytest

from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    ArtifactRegistry,
    atomic_write,
    canonical_bytes,
    compute_canonical_digest,
    compute_file_hash,
    verify_reference_graph,
)
from oeis_learn.experiments.models import FoundationValidationError, validate_artifact

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "specs/007-experiment-foundation/contracts/schema-examples.json"


@pytest.fixture
def examples():
    return json.loads(EXAMPLES.read_text())


def test_exact_bytes_hash_unicode_restart_and_corruption(tmp_path, examples):
    registry = ArtifactRegistry(tmp_path, diagnostic=True)
    artifact = examples[0]
    artifact["request_nonce"] = "診断用-fixture-nonce-0007"
    path = registry.store("visible-prompt", artifact).as_path()
    assert path.read_bytes() == canonical_bytes(artifact)
    digest = compute_file_hash(path)
    assert digest == compute_canonical_digest(artifact)
    restarted = ArtifactRegistry(tmp_path, diagnostic=True)
    assert restarted.lookup(digest) == ("visible-prompt", artifact)
    assert registry.store("visible-prompt", artifact).as_path() == path
    path.write_text("{}")
    with pytest.raises(FoundationValidationError, match="digest"):
        restarted.get("visible-prompt", digest)


@pytest.mark.parametrize(
    "value", ["../escape", "a/../b", "/absolute", "a//b", "./a", ".", "", "C:\\a", "a\\b", "a\x00b"]
)
def test_unsafe_paths_rejected(tmp_path, value):
    with pytest.raises(FoundationValidationError):
        ArtifactPath(tmp_path, value)


def test_symlink_escape_and_recheck_on_use(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    safe = ArtifactPath(root, "child/file.json")
    (root / "child").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FoundationValidationError):
        atomic_write(safe, b"{}")
    assert not (tmp_path / "file.json").exists()


def test_atomic_failure_cleanup_and_immutable_file(tmp_path, monkeypatch):
    path = ArtifactPath(tmp_path, "file.json")
    atomic_write(path, b"original")
    with pytest.raises(FoundationValidationError):
        atomic_write(path, b"different")
    assert path.as_path().read_bytes() == b"original"

    def fail(*args):
        raise OSError("injected disk failure")

    monkeypatch.setattr("oeis_learn.experiments.artifacts.os.link", fail)
    with pytest.raises(OSError):
        atomic_write(ArtifactPath(tmp_path, "second.json"), b"new")
    assert [p.name for p in tmp_path.iterdir()] == ["file.json"]


def test_reference_graph_missing_corrupt_and_cycles(tmp_path):
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    a, b = compute_file_hash(tmp_path / "a"), compute_file_hash(tmp_path / "b")
    refs = {a: {"path": "a", "references": [b]}, b: {"path": "b", "references": []}}
    verify_reference_graph(tmp_path, refs)
    refs[b]["references"] = [a]
    with pytest.raises(FoundationValidationError, match="cyclic"):
        verify_reference_graph(tmp_path, refs)
    refs[b]["references"] = []
    del refs[b]
    with pytest.raises(FoundationValidationError, match="missing"):
        verify_reference_graph(tmp_path, refs)
    refs[a]["references"] = []
    (tmp_path / "a").write_bytes(b"corrupt")
    with pytest.raises(FoundationValidationError, match="corrupt"):
        verify_reference_graph(tmp_path, refs)


def test_fixtures_and_kind_mismatch_never_enter_run_registry(tmp_path, examples):
    with pytest.raises(FoundationValidationError):
        ArtifactRegistry(tmp_path).store("visible-prompt", examples[0])
    registry = ArtifactRegistry(tmp_path, diagnostic=True)
    with pytest.raises(FoundationValidationError, match="kind"):
        registry.store("candidate-result", examples[0])
    with pytest.raises(FoundationValidationError):
        registry.store("../escape", examples[0])
    examples[1]["purpose"] = "model"
    examples[1]["checkpoint_sha256"] = "sha256:" + "1" * 64
    with pytest.raises(FoundationValidationError, match="conformance"):
        registry.store("candidate-result", examples[1])


def test_run_registry_verifies_reference_bytes_and_profile(tmp_path, examples):
    (tmp_path / "language.json").write_bytes(b'{"id":"test-profile"}')
    digest = compute_file_hash(tmp_path / "language.json")
    prompt = examples[0]
    prompt["language_profile"] = digest
    refs = {digest: {"path": "language.json", "references": []}}
    registry = ArtifactRegistry(
        tmp_path, references=refs, expected_profiles={"language_profile": digest}
    )
    path = registry.store("visible-prompt", prompt).as_path()
    assert registry.get("visible-prompt", compute_file_hash(path)) == prompt
    prompt["language_profile"] = "sha256:" + "f" * 64
    with pytest.raises(FoundationValidationError, match="mismatch"):
        registry.store("visible-prompt", prompt)
    (tmp_path / "language.json").write_bytes(b"changed")
    with pytest.raises(FoundationValidationError, match="corrupt"):
        registry.get("visible-prompt", compute_file_hash(path))


def test_checkpoint_manifest_is_json_and_never_rewrites_blob(tmp_path, examples):
    manifest = examples[2]
    payload = tmp_path / manifest["blob_path"]
    payload.write_bytes(b"opaque payload, registry does not unpickle")
    digest = compute_file_hash(payload)
    fields = [
        "contract_sha256",
        "pool_sha256",
        "codec_sha256",
        "runtime_sha256",
        "effective_config_sha256",
    ]
    for key in fields:
        manifest[key] = digest
    manifest["blob_sha256"] = digest
    manifest["next_sample_ids"] = [digest]
    refs = {digest: {"path": payload.name, "references": []}}
    registry = ArtifactRegistry(
        tmp_path, references=refs, expected_profiles={k: digest for k in fields}
    )
    path = registry.store("checkpoint", manifest).as_path()
    assert path.suffix == ".json"
    assert json.loads(path.read_text()) == manifest
    assert compute_file_hash(payload) == digest
    payload.write_bytes(b"tampered")
    with pytest.raises(FoundationValidationError):
        registry.get("checkpoint", compute_file_hash(path))


@pytest.mark.parametrize(
    "command", ["train", "resume", "inspect"]
)
def test_training_commands_reject_missing_required_inputs_without_writing(command, tmp_path):
    from oeis_learn.cli.main import cli
    with pytest.raises(SystemExit) as exc:
        cli(["foundation", command])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_cli_conformance_is_diagnostic_and_rejects_malformed_inputs(tmp_path, capsys):
    from oeis_learn.cli.main import cli

    assert cli(["foundation", "conformance", "--json", str(EXAMPLES)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["purpose"] == "diagnostic" and record["executed_evidence"] is False
    for bad in ["{}", "null", "[]", '{"kind":"candidate_result"}', "not json"]:
        path = tmp_path / "bad.json"
        path.write_text(bad)
        assert cli(["foundation", "conformance", "--json", str(path)]) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
    assert cli(["foundation", "conformance", "--json"]) == 2


def test_preflight_dispatch_preserves_argument_and_exit_contract(monkeypatch):
    import oeis_learn.cli.foundation_preflight as probe

    calls = []

    def fake(argv):
        calls.append(argv)
        return 7

    monkeypatch.setattr(probe, "main", fake)
    from oeis_learn.cli.main import cli

    assert cli(["foundation", "preflight", "--config", "probe.yaml", "--output", "out"]) == 7
    assert calls == [["--config", "probe.yaml", "--output", "out"]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt_index", True),
        ("attempt_index", 1.0),
        ("run_id", "not-a-uuid"),
        ("verified_terms", 101),
        ("proof_status", "PROVEN"),
        ("selected", 1),
    ],
)
def test_semantic_boundary_types(field, value, examples):
    result = examples[1]
    result[field] = value
    with pytest.raises(FoundationValidationError):
        validate_artifact(result)


def test_exact_integer_limits_and_wrong_metric_units(examples):
    for value in [str(-(2**255) - 1), str(2**255), "9" * 10000, "-0", "01", "1e3"]:
        prompt = copy.deepcopy(examples[0])
        prompt["observed_terms"][0] = value
        with pytest.raises(FoundationValidationError):
            validate_artifact(prompt)
    result = examples[1]
    result["usage"]["fuel"]["unit"] = "seconds"
    with pytest.raises(FoundationValidationError):
        validate_artifact(result)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_noncanonical_json_never_silently_overwrites_identity(text):
    from oeis_learn.experiments.artifacts import load_json

    with pytest.raises(FoundationValidationError):
        load_json(text)
